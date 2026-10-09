//! The wheel-odometry ingest: the stamp-indexed reading ring, the keyframe-interval integration
//! that turns two stamps into a kornia-slam [`WheelOdomFactor`], the per-frame motion prior, and
//! the camera-mount ring a tilting head needs. Free of copper runtime and graph types, like
//! [`crate::imu`].
//!
//! # What the wheels are used for
//!
//! * **Local BA.** Each keyframe pair gets one planar factor (`r = Log(dT^-1 T_wb_i^-1 T_wb_j)`,
//!   z, roll and pitch held stiff), next to the IMU edge when the inertial path is on.
//! * **The tracking prior.** The camera motion the wheels measured since the previous frame
//!   replaces the constant-velocity guess, which is wrong the moment the robot accelerates.
//! * **Coasting.** While vision is lost the same prior is the pose: dead reckoning on the wheels
//!   (and the gyro, when the inertial path is on), published as `Localization::Coasting`.
//!
//! # Why an edge can wait
//!
//! The readings cross a process link from the body, so the newest one is usually OLDER than the
//! camera frame that closes a keyframe interval. Refusing such an interval would refuse nearly
//! every one; an interval whose end is not covered yet stays pending and is integrated at a
//! later keyframe, when the readings have caught up. A factor is not stored in the map, so a late
//! one costs nothing but the BA solves it missed.
//!
//! A prior cannot wait: it is wanted at the frame. Past the newest reading it extrapolates at the
//! last reading pair's velocity, for at most `max_gap`. Holding the newest reading instead would
//! drop the stretch `[newest, frame]` from every frame's interval (the next one starts at the
//! frame stamp), and with a lag longer than a frame period integrate nothing at all.

use std::collections::VecDeque;

use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_slam::mapping::wheel_odom::{WheelOdomFactor, WheelOdomMeasurement};

use crate::error::VioError;
use crate::imu::rotation_residual;

/// Readings held. 8192 is ~80 s at 100 Hz: the ring must reach back to the last keyframe's stamp,
/// and a parked robot inserts none, so this is what bounds an idle stretch before that interval's
/// start is evicted and its edge refused.
pub const DEFAULT_WHEEL_BUFFER_CAPACITY: usize = 8192;

/// Mount samples held. Only a mount that MOVED is stored (see [`MOUNT_SAME_RAD`]), so a still
/// head costs no slots however long the robot parks, and the last keyframe's mount survives an
/// idle stretch the way its reading does. 256 changes is ~8 s of a head tilting at 30 Hz.
const MOUNT_CAPACITY: usize = 256;

/// A mount within this of the newest stored one is the same mount and is not stored. Far below
/// what a head-joint encoder resolves; it only keeps repeated identical samples out of the ring.
const MOUNT_SAME_RAD: f64 = 1e-4;
/// See [`MOUNT_SAME_RAD`].
const MOUNT_SAME_M: f64 = 1e-4;

/// Pending keyframe intervals kept waiting for readings. Past this the stream has stopped, not
/// lagged, and the oldest is refused as expired.
const MAX_PENDING_EDGES: usize = 8;

/// Consecutive frames outside the readings' span before the tracker says so, and how often it
/// repeats after that: ~2 s, then ~1 min, at the head's 15 Hz.
const UNCOVERED_WARN_AFTER: u64 = 30;
const UNCOVERED_WARN_EVERY: u64 = 900;

/// Integrated factors kept for local BA. The solve frees the newest three keyframes and drops a
/// factor with both ends fixed, so only the last few can act; 16 is slack for a late edge.
const MAX_FACTORS: usize = 16;

/// Variance floors, so a parked robot's zero-length interval is a stiff but finite constraint.
/// 1 mm and 1 mrad: unmeasured, chosen below the wheels' measured round-trip error (1 mm /
/// 0.36 deg over an in-place turn on XLeRobot).
const MIN_TRANSLATION_SIGMA_M: f64 = 1e-3;
const MIN_ROTATION_SIGMA_RAD: f64 = 1e-3;

/// One wheel-odometry reading: the cumulative base pose in the odometry frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelSample {
    /// Capture time, on the same clock as the frames' `tov`.
    pub stamp_ns: u64,
    /// Base position, metres.
    pub x: f64,
    /// Base position, metres.
    pub y: f64,
    /// Base heading, radians, wrapped or not.
    pub theta: f64,
}

impl WheelSample {
    fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.theta.is_finite()
    }
}

/// The wheel path's configuration. Build it with [`WheelOdomConfig::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct WheelOdomConfig {
    /// Base-to-camera transform (`X_cam = t_cb * X_base`), until a `CameraMount` arrives.
    pub t_cb: Pose3d,
    /// Translation noise, m per sqrt(m) of path: an interval of `d` metres gets variance
    /// `translation_noise^2 * d` on x and y.
    pub translation_noise: f64,
    /// Heading noise, rad per sqrt(rad) of turning: `rotation_noise^2 * |turned|` on theta.
    pub rotation_noise: f64,
    /// Out-of-plane translation sigma, metres: the planar stiffness on z.
    pub z_sigma: f64,
    /// Roll and pitch sigma, radians: the planar stiffness on tilt.
    pub tilt_sigma: f64,
    /// Huber threshold on the factor's 6-DOF chi-square; `None` keeps it quadratic.
    pub huber_chi2: Option<f64>,
    /// Longest gap between two readings an interval may span. A wheel stream that went silent
    /// for longer says nothing about the motion inside the hole; interpolating across it would.
    pub max_gap: Time,
    /// Readings held; see [`DEFAULT_WHEEL_BUFFER_CAPACITY`].
    pub buffer_capacity: usize,
}

impl WheelOdomConfig {
    /// Validates the mount and the noise model. `cam_in_base` is the camera's pose in the base
    /// frame (the URDF transform); the factor wants its inverse, taken here once.
    ///
    /// The out-of-plane sigmas default to kornia-slam's (1 cm, 0.01 rad), the gap to 0.5 s and
    /// the Huber kernel to off; set the fields afterwards and call [`WheelOdomConfig::validate`].
    pub fn new(
        cam_in_base: Pose3d,
        translation_noise: f64,
        rotation_noise: f64,
    ) -> Result<Self, VioError> {
        let cfg = Self {
            t_cb: cam_in_base.inverse(),
            translation_noise,
            rotation_noise,
            z_sigma: WheelOdomMeasurement::DEFAULT_Z_SIGMA,
            tilt_sigma: WheelOdomMeasurement::DEFAULT_TILT_SIGMA,
            huber_chi2: None,
            max_gap: Time::new::<second>(0.5),
            buffer_capacity: DEFAULT_WHEEL_BUFFER_CAPACITY,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Every check [`WheelOdomConfig::new`] runs, for a config edited after construction.
    pub fn validate(&self) -> Result<(), VioError> {
        validate_mount(&self.t_cb)?;
        for (field, value) in [
            ("translation_noise", self.translation_noise),
            ("rotation_noise", self.rotation_noise),
            ("z_sigma", self.z_sigma),
            ("tilt_sigma", self.tilt_sigma),
            ("max_gap_s", self.max_gap.get::<second>()),
            ("huber_chi2", self.huber_chi2.unwrap_or(1.0)),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(VioError::WheelParamInvalid { field, value });
            }
        }
        Ok(())
    }

    pub(crate) fn max_gap_ns(&self) -> u64 {
        (self.max_gap.get::<second>() * 1e9) as u64
    }

    /// The factor measurement for one integrated interval: a random walk in path length and in
    /// angle turned, floored so a zero-length interval stays finite.
    pub(crate) fn measurement(&self, d: &WheelDelta) -> WheelOdomMeasurement {
        let var_t =
            (self.translation_noise.powi(2) * d.path_m).max(MIN_TRANSLATION_SIGMA_M.powi(2));
        let var_r = (self.rotation_noise.powi(2) * d.path_rad).max(MIN_ROTATION_SIGMA_RAD.powi(2));
        WheelOdomMeasurement {
            delta: d.delta,
            covariance: [[var_t, 0.0, 0.0], [0.0, var_t, 0.0], [0.0, 0.0, var_r]],
            z_sigma: self.z_sigma,
            tilt_sigma: self.tilt_sigma,
        }
    }
}

/// Refuses a mount whose rotation is not a proper rotation, or whose lever arm is not finite.
pub(crate) fn validate_mount(t_cb: &Pose3d) -> Result<(), VioError> {
    if let Some((orthonormality_error, determinant)) = rotation_residual(&t_cb.rotation) {
        return Err(VioError::WheelMountNotRigid {
            orthonormality_error,
            determinant,
        });
    }
    for (field, value) in [
        ("cam_in_base_translation.x", t_cb.translation.x),
        ("cam_in_base_translation.y", t_cb.translation.y),
        ("cam_in_base_translation.z", t_cb.translation.z),
    ] {
        if !value.is_finite() {
            return Err(VioError::WheelParamInvalid { field, value });
        }
    }
    Ok(())
}

/// Planar base motion over one interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WheelDelta {
    /// `(dx, dy, dtheta)`: base at the end, in the base at the start. `dtheta` is the angle
    /// actually turned, unwrapped.
    pub delta: [f64; 3],
    /// Path length, metres: the sum of the reading-to-reading distances, not `|(dx, dy)|`, so a
    /// there-and-back interval is as uncertain as the distance driven.
    pub path_m: f64,
    /// Angle turned, radians, summed in absolute value for the same reason.
    pub path_rad: f64,
}

impl WheelDelta {
    /// The delta as a transform taking base-end coordinates to base-start.
    fn pose(&self) -> Pose3d {
        let [dx, dy, dtheta] = self.delta;
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, dtheta)).matrix(),
            Vec3F64::new(dx, dy, 0.0),
        )
    }
}

/// Why an interval could not be integrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WheelWindowError {
    /// The interval ends at or before it starts.
    NonPositiveInterval,
    /// No reading at or before the start: the stream began later, or the start was evicted.
    NotCovered,
    /// No reading at or after the end YET. The one recoverable refusal: retried later.
    Pending,
    /// Two consecutive readings inside the interval are further apart than `max_gap`.
    Gap,
}

/// Wheel-path counters. Diagnostics only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WheelStats {
    /// Readings accepted into the ring.
    pub accepted: u64,
    /// Readings refused because their stamp did not advance.
    pub out_of_order: u64,
    /// Readings refused for a non-finite pose.
    pub non_finite: u64,
    /// Readings evicted by the capacity bound before anything read them.
    pub evicted: u64,
    /// Readings the bus queue evicted because the tracker did not drain it in time, cumulative.
    pub queue_evicted: u64,
    /// Mount samples accepted, and refused (out of order or not rigid).
    pub mounts_accepted: u64,
    /// See [`WheelStats::mounts_accepted`].
    pub mounts_refused: u64,
    /// Keyframe intervals integrated into a factor.
    pub edges_added: u64,
    /// Intervals refused: start not covered, a gap, or a bad interval.
    pub edges_refused: u64,
    /// Intervals refused because the readings never caught up.
    pub edges_expired: u64,
    /// Frames whose tracking prior or coast came from the wheels.
    pub prior_frames: u64,
    /// Frames on which no reading covered the interval and the prior fell back.
    pub prior_unavailable: u64,
    /// Frames whose stamp lay outside `[oldest reading, newest reading + max_gap]`: the stream is
    /// silent or on another clock, and the wheels are not being used.
    pub frames_uncovered: u64,
}

/// A keyframe interval waiting for its readings.
#[derive(Debug, Clone, Copy)]
struct PendingEdge {
    from_idx: usize,
    from_ns: u64,
    to_idx: usize,
    to_ns: u64,
}

/// The tracker's wheel half: readings, mounts, pending intervals and the factors built from them.
pub(crate) struct WheelState {
    pub cfg: WheelOdomConfig,
    ring: VecDeque<WheelSample>,
    mounts: VecDeque<(u64, Pose3d)>,
    /// The newest mount stamp seen, stored or not: the order check for the next one.
    newest_mount_ns: Option<u64>,
    pending: Vec<PendingEdge>,
    factors: Vec<WheelOdomFactor>,
    /// Consecutive frames counted in [`WheelStats::frames_uncovered`].
    uncovered_streak: u64,
    pub stats: WheelStats,
}

impl WheelState {
    pub fn new(cfg: WheelOdomConfig) -> Self {
        Self {
            ring: VecDeque::with_capacity(cfg.buffer_capacity.min(1024)),
            mounts: VecDeque::new(),
            newest_mount_ns: None,
            pending: Vec::new(),
            factors: Vec::new(),
            uncovered_streak: 0,
            stats: WheelStats::default(),
            cfg,
        }
    }

    /// Appends readings, refusing any whose stamp does not advance or whose pose is not finite.
    pub fn push(&mut self, samples: impl IntoIterator<Item = WheelSample>) -> usize {
        let mut accepted = 0;
        for s in samples {
            if !s.is_finite() {
                self.stats.non_finite += 1;
                continue;
            }
            if self.ring.back().is_some_and(|b| s.stamp_ns <= b.stamp_ns) {
                self.stats.out_of_order += 1;
                continue;
            }
            self.ring.push_back(s);
            accepted += 1;
            while self.ring.len() > self.cfg.buffer_capacity {
                self.ring.pop_front();
                self.stats.evicted += 1;
            }
        }
        self.stats.accepted += accepted as u64;
        accepted
    }

    /// Records the base-to-camera transform valid from `stamp_ns`.
    pub fn push_mount(&mut self, stamp_ns: u64, t_cb: Pose3d) -> Result<(), VioError> {
        if let Err(e) = validate_mount(&t_cb) {
            self.stats.mounts_refused += 1;
            return Err(e);
        }
        if self.newest_mount_ns.is_some_and(|t| stamp_ns <= t) {
            self.stats.mounts_refused += 1;
            return Ok(());
        }
        self.newest_mount_ns = Some(stamp_ns);
        self.stats.mounts_accepted += 1;
        if self
            .mounts
            .back()
            .is_some_and(|(_, last)| same_mount(last, &t_cb))
        {
            return Ok(());
        }
        self.mounts.push_back((stamp_ns, t_cb));
        while self.mounts.len() > MOUNT_CAPACITY {
            self.mounts.pop_front();
        }
        Ok(())
    }

    /// The base-to-camera transform at `stamp_ns`: the newest mount at or before it, the oldest
    /// one for an earlier stamp, and the configured one before any mount arrived.
    pub fn t_cb_at(&self, stamp_ns: u64) -> Pose3d {
        let newer = self.mounts.partition_point(|(t, _)| *t <= stamp_ns);
        match newer {
            0 => self.mounts.front().map_or(self.cfg.t_cb, |(_, p)| *p),
            n => self.mounts[n - 1].1,
        }
    }

    /// Base motion over `[t0_ns, t1_ns]`.
    ///
    /// `hold_ns` lets the end (and the start) run past the newest reading by up to that much,
    /// extrapolated: a prior wants the motion now, while a BA factor (`hold_ns = 0`) must cover
    /// the interval.
    pub fn delta(
        &self,
        t0_ns: u64,
        t1_ns: u64,
        hold_ns: u64,
    ) -> Result<WheelDelta, WheelWindowError> {
        if t1_ns <= t0_ns {
            return Err(WheelWindowError::NonPositiveInterval);
        }
        // Last reading at or before the start.
        let first_after = self.ring.partition_point(|s| s.stamp_ns <= t0_ns);
        if first_after == 0 {
            return Err(WheelWindowError::NotCovered);
        }
        let a = first_after - 1;
        // First reading at or after the end, or the held newest one.
        let b = self.ring.partition_point(|s| s.stamp_ns < t1_ns);
        let held = b == self.ring.len();
        if held {
            let newest = self.ring.back().expect("a is a reading");
            if t1_ns - newest.stamp_ns > hold_ns {
                return Err(WheelWindowError::Pending);
            }
        }
        let b = b.min(self.ring.len() - 1);
        let max_gap = self.cfg.max_gap_ns();
        for i in a..b {
            if self.ring[i + 1].stamp_ns - self.ring[i].stamp_ns > max_gap {
                return Err(WheelWindowError::Gap);
            }
        }

        let start = if a + 1 == self.ring.len() {
            self.extrapolate(t0_ns)
        } else {
            self.interpolate(a, t0_ns)
        };
        let end = if held {
            self.extrapolate(t1_ns)
        } else {
            self.interpolate(b - 1, t1_ns)
        };
        // Walk start -> readings strictly inside -> end, summing the path and unwrapping theta.
        let (mut prev, mut turned, mut path_m, mut path_rad) = (start, 0.0, 0.0, 0.0);
        let inside = self
            .ring
            .range(a + 1..)
            .copied()
            .take_while(|s| s.stamp_ns < t1_ns)
            .filter(|s| s.stamp_ns > t0_ns);
        for s in inside.chain(std::iter::once(end)) {
            let dth = wrap(s.theta - prev.theta);
            turned += dth;
            path_rad += dth.abs();
            path_m += (s.x - prev.x).hypot(s.y - prev.y);
            prev = s;
        }
        let (dxw, dyw) = (end.x - start.x, end.y - start.y);
        let (sin, cos) = start.theta.sin_cos();
        Ok(WheelDelta {
            delta: [cos * dxw + sin * dyw, -sin * dxw + cos * dyw, turned],
            path_m,
            path_rad,
        })
    }

    /// The reading at `t_ns`, linearly interpolated between reading `i` and the next one.
    fn interpolate(&self, i: usize, t_ns: u64) -> WheelSample {
        let s0 = self.ring[i];
        let Some(s1) = self.ring.get(i + 1).copied() else {
            return s0;
        };
        if t_ns <= s0.stamp_ns {
            return s0;
        }
        if t_ns >= s1.stamp_ns {
            return s1;
        }
        let a = (t_ns - s0.stamp_ns) as f64 / (s1.stamp_ns - s0.stamp_ns) as f64;
        WheelSample {
            stamp_ns: t_ns,
            x: s0.x + a * (s1.x - s0.x),
            y: s0.y + a * (s1.y - s0.y),
            theta: s0.theta + a * wrap(s1.theta - s0.theta),
        }
    }

    /// The reading at `t_ns` at or past the newest one, continued at the last pair's velocity. A
    /// pair further apart than `max_gap` has no velocity worth trusting (the stream resumed after
    /// a hole), so the newest reading is held.
    fn extrapolate(&self, t_ns: u64) -> WheelSample {
        let n = self.ring.len();
        let s1 = self.ring[n - 1];
        if n < 2 || t_ns <= s1.stamp_ns {
            return s1;
        }
        let s0 = self.ring[n - 2];
        let dt = s1.stamp_ns - s0.stamp_ns;
        if dt > self.cfg.max_gap_ns() {
            return s1;
        }
        let a = (t_ns - s1.stamp_ns) as f64 / dt as f64;
        WheelSample {
            stamp_ns: t_ns,
            x: s1.x + a * (s1.x - s0.x),
            y: s1.y + a * (s1.y - s0.y),
            theta: s1.theta + a * wrap(s1.theta - s0.theta),
        }
    }

    /// The CAMERA motion the wheels measured over `[t0_ns, t1_ns]`, as the world-to-camera
    /// increment `M` with `T_cw(t1) = M * T_cw(t0)`, or `None` when no reading covers it.
    ///
    /// With `X_cam = T_cb X_base` at each end and the delta `dT` taking base-end to base-start
    /// coordinates, camera-end to camera-start is `T_cb(t0) dT T_cb(t1)^-1`; `M` is its inverse.
    pub fn camera_motion(&mut self, t0_ns: u64, t1_ns: u64) -> Option<Pose3d> {
        match self.delta(t0_ns, t1_ns, self.cfg.max_gap_ns()) {
            Ok(d) => {
                self.stats.prior_frames += 1;
                let c0_c1 = self
                    .t_cb_at(t0_ns)
                    .compose(&d.pose())
                    .compose(&self.t_cb_at(t1_ns).inverse());
                Some(c0_c1.inverse())
            }
            Err(_) => {
                self.stats.prior_unavailable += 1;
                None
            }
        }
    }

    /// Checks a frame stamp against the readings' span. Returns `frame - newest reading` in ns
    /// and the frames in the streak when the tracker should warn: a clock offset between the body (wheels) and the head
    /// (frames) otherwise turns the wheels off with nothing but the stop-time counters to say so.
    /// Silent before the first reading.
    pub fn check_coverage(&mut self, stamp_ns: u64) -> Option<(i128, u64)> {
        let (Some(oldest), Some(newest)) = (self.ring.front(), self.ring.back()) else {
            return None;
        };
        let covered = oldest.stamp_ns <= stamp_ns
            && stamp_ns <= newest.stamp_ns.saturating_add(self.cfg.max_gap_ns());
        if covered {
            self.uncovered_streak = 0;
            return None;
        }
        self.stats.frames_uncovered += 1;
        self.uncovered_streak += 1;
        let n = self.uncovered_streak;
        (n >= UNCOVERED_WARN_AFTER
            && (n - UNCOVERED_WARN_AFTER).is_multiple_of(UNCOVERED_WARN_EVERY))
        .then(|| (i128::from(stamp_ns) - i128::from(newest.stamp_ns), n))
    }

    /// Queues the interval a keyframe insertion closed, then integrates every queued interval
    /// whose readings have arrived. Returns the factors integrated by this call, for the motion
    /// consistency check ([`crate::consistency`]), which may [`drop_factor`](Self::drop_factor)
    /// one before any solve sees it.
    pub fn on_keyframe(
        &mut self,
        prev: Option<(usize, u64)>,
        to_idx: usize,
        to_ns: u64,
    ) -> Vec<WheelOdomFactor> {
        if let Some((from_idx, from_ns)) = prev {
            self.pending.push(PendingEdge {
                from_idx,
                from_ns,
                to_idx,
                to_ns,
            });
        }
        let mut still = Vec::new();
        let first_new = self.factors.len();
        for e in std::mem::take(&mut self.pending) {
            match self.delta(e.from_ns, e.to_ns, 0) {
                Ok(d) => {
                    self.stats.edges_added += 1;
                    self.factors.push(WheelOdomFactor {
                        from_idx: e.from_idx,
                        to_idx: e.to_idx,
                        measurement: self.cfg.measurement(&d),
                        t_cb_from: self.t_cb_at(e.from_ns),
                        t_cb_to: self.t_cb_at(e.to_ns),
                        huber_chi2: self.cfg.huber_chi2,
                    });
                }
                Err(WheelWindowError::Pending) => still.push(e),
                Err(_) => self.stats.edges_refused += 1,
            }
        }
        let expired = still.len().saturating_sub(MAX_PENDING_EDGES);
        self.stats.edges_expired += expired as u64;
        still.drain(..expired);
        self.pending = still;
        let added = self.factors[first_new..].to_vec();
        let excess = self.factors.len().saturating_sub(MAX_FACTORS);
        self.factors.drain(..excess);
        added
    }

    /// Removes the factor between keyframes `from_idx` and `to_idx`, if one is held: the motion
    /// check found the wheels disagreeing with the gyro and vision over it (slip).
    pub fn drop_factor(&mut self, from_idx: usize, to_idx: usize) -> bool {
        let before = self.factors.len();
        self.factors
            .retain(|f| !(f.from_idx == from_idx && f.to_idx == to_idx));
        self.factors.len() < before
    }

    /// The integrated factors, keyed by `Frame::idx` as `run_local_inertial_ba` wants them.
    pub fn factors(&self) -> &[WheelOdomFactor] {
        &self.factors
    }

    /// Drops readings and mounts no pending interval, keyframe or frame can still ask for.
    ///
    /// `keep_from` are the stamps a future lookup may start at (the last keyframe's, the last
    /// frame's); the reading at or before the oldest of those and the pending starts is kept.
    pub fn prune(&mut self, keep_from: impl IntoIterator<Item = Option<u64>>) {
        let oldest = keep_from
            .into_iter()
            .flatten()
            .chain(self.pending.iter().map(|e| e.from_ns))
            .min();
        let Some(oldest) = oldest else {
            return;
        };
        while self.ring.len() > 1 && self.ring[1].stamp_ns <= oldest {
            self.ring.pop_front();
        }
        while self.mounts.len() > 1 && self.mounts[1].0 <= oldest {
            self.mounts.pop_front();
        }
    }

    /// Forgets the intervals and factors of a map that was dropped: their keyframe indices name
    /// keyframes that no longer exist. The readings and mounts are kept, because the pose that
    /// seeds the next map is propagated on them.
    pub fn reset_map(&mut self) {
        self.pending.clear();
        self.factors.clear();
    }
}

/// Whether two mounts differ by less than [`MOUNT_SAME_RAD`] and [`MOUNT_SAME_M`].
fn same_mount(a: &Pose3d, b: &Pose3d) -> bool {
    let r = a.rotation.transpose() * b.rotation;
    let cos = ((r.x_axis.x + r.y_axis.y + r.z_axis.z) - 1.0) * 0.5;
    cos.clamp(-1.0, 1.0).acos() < MOUNT_SAME_RAD
        && (a.translation - b.translation).length() < MOUNT_SAME_M
}

/// `a` wrapped to `(-pi, pi]`.
fn wrap(a: f64) -> f64 {
    let w = a.rem_euclid(std::f64::consts::TAU);
    if w > std::f64::consts::PI {
        w - std::f64::consts::TAU
    } else {
        w
    }
}

/// `R` from nine ROW-major numbers, as a human writes a matrix (glam is column-major).
pub(crate) fn mat3_from_rows(rows: &[f64; 9]) -> Mat3F64 {
    Mat3F64::from_cols_array(rows).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn cfg() -> WheelOdomConfig {
        WheelOdomConfig::new(Pose3d::IDENTITY, 0.02, 0.05).expect("valid")
    }

    fn state(samples: &[(u64, f64, f64, f64)]) -> WheelState {
        let mut w = WheelState::new(cfg());
        w.push(samples.iter().map(|&(t, x, y, theta)| WheelSample {
            stamp_ns: t * MS,
            x,
            y,
            theta,
        }));
        w
    }

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    /// Straight drive at 1 m/s: the interval between two stamps that fall BETWEEN readings is
    /// the interpolated distance, in the start's base frame.
    #[test]
    fn test_integration_between_stamps_interpolates_both_ends() {
        let w = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.1, 0.0, 0.0),
            (200, 0.2, 0.0, 0.0),
        ]);
        let d = w.delta(50 * MS, 150 * MS, 0).expect("covered");
        assert!(close(d.delta, [0.1, 0.0, 0.0]), "{:?}", d.delta);
        assert!((d.path_m - 0.1).abs() < 1e-12);
    }

    /// The delta is in the START's base frame: driving along odom +y while facing +y is
    /// straight ahead, +x of the base.
    #[test]
    fn test_the_delta_is_expressed_in_the_start_base_frame() {
        let h = std::f64::consts::FRAC_PI_2;
        let w = state(&[(0, 1.0, 1.0, h), (100, 1.0, 1.5, h)]);
        let d = w.delta(0, 100 * MS, 0).expect("covered");
        assert!(close(d.delta, [0.5, 0.0, 0.0]), "{:?}", d.delta);
    }

    /// A heading that wraps through +-pi is the small turn it is, not a 2 pi spin, and an
    /// in-place turn has zero path length but its full turned angle.
    #[test]
    fn test_heading_unwraps_across_pi() {
        let pi = std::f64::consts::PI;
        let w = state(&[(0, 0.0, 0.0, pi - 0.1), (100, 0.0, 0.0, -pi + 0.1)]);
        let d = w.delta(0, 100 * MS, 0).expect("covered");
        assert!((d.delta[2] - 0.2).abs() < 1e-12, "{:?}", d.delta);
        assert!((d.path_rad - 0.2).abs() < 1e-12);
        assert_eq!(d.path_m, 0.0);
    }

    /// A there-and-back interval nets zero motion but keeps the uncertainty of what was driven.
    #[test]
    fn test_path_length_sums_the_readings_not_the_net_motion() {
        let w = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.3, 0.0, 0.0),
            (200, 0.0, 0.0, 0.0),
        ]);
        let d = w.delta(0, 200 * MS, 0).expect("covered");
        assert!(close(d.delta, [0.0, 0.0, 0.0]));
        assert!((d.path_m - 0.6).abs() < 1e-12);
        let m = w.cfg.measurement(&d);
        assert!((m.covariance[0][0] - 0.02f64.powi(2) * 0.6).abs() < 1e-15);
        // Parked: floored, not infinite information.
        let parked = w.cfg.measurement(&WheelDelta {
            delta: [0.0; 3],
            path_m: 0.0,
            path_rad: 0.0,
        });
        assert_eq!(parked.covariance[0][0], MIN_TRANSLATION_SIGMA_M.powi(2));
        assert!(parked.information().is_some());
    }

    #[test]
    fn test_uncovered_intervals_are_refused_and_a_late_end_is_pending() {
        let w = state(&[
            (100, 0.0, 0.0, 0.0),
            (200, 0.1, 0.0, 0.0),
            (1200, 1.1, 0.0, 0.0),
        ]);
        assert_eq!(
            w.delta(50 * MS, 150 * MS, 0),
            Err(WheelWindowError::NotCovered)
        );
        assert_eq!(
            w.delta(150 * MS, 150 * MS, 0),
            Err(WheelWindowError::NonPositiveInterval)
        );
        assert_eq!(
            w.delta(150 * MS, 1300 * MS, 0),
            Err(WheelWindowError::Pending)
        );
        // A 1 s hole is longer than the 0.5 s max_gap.
        assert_eq!(w.delta(150 * MS, 1100 * MS, 0), Err(WheelWindowError::Gap));
        // The prior may hold the newest reading for up to max_gap.
        let held = w.delta(1200 * MS, 1300 * MS, 500 * MS).expect("held");
        assert!(close(held.delta, [0.0; 3]));
    }

    /// An interval whose readings arrive after the keyframe that closed it is integrated at a
    /// later keyframe, with the mount valid at each END, not at the time it was built.
    #[test]
    fn test_a_pending_interval_completes_when_the_readings_catch_up() {
        let mut w = state(&[(0, 0.0, 0.0, 0.0), (100, 0.1, 0.0, 0.0)]);
        let tilted = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.2, 0.0));
        w.push_mount(150 * MS, tilted).expect("rigid");
        w.on_keyframe(Some((3, 50 * MS)), 4, 200 * MS);
        assert!(w.factors().is_empty(), "the end is not covered yet");
        w.push([WheelSample {
            stamp_ns: 250 * MS,
            x: 0.25,
            y: 0.0,
            theta: 0.0,
        }]);
        w.on_keyframe(None, 5, 260 * MS);
        let f = w.factors();
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].from_idx, f[0].to_idx), (3, 4));
        assert!(close(f[0].measurement.delta, [0.15, 0.0, 0.0]));
        // The oldest mount stands in for a stamp before it: it is nearer the truth than the
        // configured one once a head is publishing.
        assert_eq!(f[0].t_cb_from, tilted);
        assert_eq!(f[0].t_cb_to, tilted);
        assert_eq!(
            WheelState::new(cfg()).t_cb_at(0),
            Pose3d::IDENTITY,
            "configured until then"
        );
        assert_eq!(w.stats.edges_added, 1);
    }

    #[test]
    fn test_a_stream_that_stopped_expires_its_intervals() {
        let mut w = state(&[(0, 0.0, 0.0, 0.0)]);
        for k in 0..(MAX_PENDING_EDGES + 3) {
            let t = (k as u64 + 1) * 100 * MS;
            w.on_keyframe(Some((k, t - 100 * MS)), k + 1, t);
        }
        assert_eq!(w.stats.edges_expired, 3);
        assert_eq!(w.pending.len(), MAX_PENDING_EDGES);
    }

    #[test]
    fn test_out_of_order_and_non_finite_readings_are_refused() {
        let mut w = state(&[(100, 0.0, 0.0, 0.0)]);
        let s = |t: u64, x: f64| WheelSample {
            stamp_ns: t * MS,
            x,
            y: 0.0,
            theta: 0.0,
        };
        assert_eq!(
            w.push([s(100, 1.0), s(50, 1.0), s(200, f64::NAN), s(300, 1.0)]),
            1
        );
        assert_eq!((w.stats.out_of_order, w.stats.non_finite), (2, 1));
    }

    /// Camera looking along base +x (OpenCV axes: x right = base -y, y down = base -z). Driving
    /// forward 0.5 m moves the camera 0.5 m along its own +z, so a world point ahead gets 0.5 m
    /// closer in camera coordinates.
    #[test]
    fn test_camera_motion_maps_base_forward_onto_camera_forward() {
        let r_bc = mat3_from_rows(&[0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
        let cam_in_base = Pose3d::new(r_bc, Vec3F64::new(0.1, 0.0, 0.9));
        let mut w = WheelState::new(WheelOdomConfig::new(cam_in_base, 0.02, 0.05).unwrap());
        w.push([(0, 0.0), (100, 0.5)].map(|(t, x)| WheelSample {
            stamp_ns: t * MS,
            x,
            y: 0.0,
            theta: 0.0,
        }));
        let m = w.camera_motion(0, 100 * MS).expect("covered");
        let p_cam0 = Vec3F64::new(0.0, 0.0, 2.0);
        let p_cam1 = m.transform_point(&p_cam0);
        assert!(
            (p_cam1 - Vec3F64::new(0.0, 0.0, 1.5)).length() < 1e-12,
            "{p_cam1:?}"
        );
    }

    #[test]
    fn test_a_non_rigid_mount_is_refused() {
        let scaled = Pose3d::new(Mat3F64::IDENTITY * 2.0, Vec3F64::ZERO);
        assert!(matches!(
            WheelOdomConfig::new(scaled, 0.02, 0.05),
            Err(VioError::WheelMountNotRigid { .. })
        ));
        let mut w = WheelState::new(cfg());
        assert!(w.push_mount(1, scaled).is_err());
        assert_eq!(w.stats.mounts_refused, 1);
        assert!(matches!(
            WheelOdomConfig::new(Pose3d::IDENTITY, 0.0, 0.05),
            Err(VioError::WheelParamInvalid {
                field: "translation_noise",
                ..
            })
        ));
    }

    #[test]
    fn test_prune_keeps_the_reading_before_the_oldest_stamp_still_needed() {
        let mut w = state(&[
            (0, 0.0, 0.0, 0.0),
            (100, 0.1, 0.0, 0.0),
            (200, 0.2, 0.0, 0.0),
        ]);
        w.prune([Some(150 * MS), None]);
        assert_eq!(w.ring.front().unwrap().stamp_ns, 100 * MS);
        assert!(w.delta(150 * MS, 200 * MS, 0).is_ok());
    }

    /// A prior past the newest reading continues at the last pair's velocity, from both ends:
    /// a lag shorter than the interval, and one longer than it (start past the newest too).
    #[test]
    fn test_a_lagging_prior_extrapolates_instead_of_holding() {
        let w = state(&[(0, 0.0, 0.0, 0.0), (100, 0.1, 0.0, 0.0)]);
        let short = w
            .delta(50 * MS, 150 * MS, 500 * MS)
            .expect("within the hold");
        assert!(close(short.delta, [0.1, 0.0, 0.0]), "{:?}", short.delta);
        let long = w
            .delta(150 * MS, 250 * MS, 500 * MS)
            .expect("within the hold");
        assert!(close(long.delta, [0.1, 0.0, 0.0]), "{:?}", long.delta);
        // A factor still refuses to guess.
        assert_eq!(
            w.delta(50 * MS, 150 * MS, 0),
            Err(WheelWindowError::Pending)
        );
    }

    /// A still head stores one mount however long it publishes, so the mount at an old
    /// keyframe stamp survives; a moving one is stored per change.
    #[test]
    fn test_a_still_mount_takes_one_slot() {
        let mut w = WheelState::new(cfg());
        let tilted = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.2, 0.0));
        w.push_mount(MS, tilted).unwrap();
        for k in 2..(MOUNT_CAPACITY as u64 + 10) {
            w.push_mount(k * MS, Pose3d::IDENTITY).unwrap();
        }
        assert_eq!(w.mounts.len(), 2);
        assert_eq!(w.t_cb_at(MS), tilted);
        assert_eq!(w.stats.mounts_accepted, MOUNT_CAPACITY as u64 + 9);
        // Order is still checked against the newest sample, stored or not.
        w.push_mount(5 * MS, tilted).unwrap();
        assert_eq!(w.stats.mounts_refused, 1);
    }
}
