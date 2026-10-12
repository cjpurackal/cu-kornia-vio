//! Parked keyframes: stationary edges, and no time-based keyframes while parked.
//!
//! A parked robot used to insert a keyframe every `wheel_odom.parked_keyframe_interval_s` (5 s on
//! the robot), each joined to the last by an IMU edge. Keyframe culling merges IMU edges only up
//! to `keyframe_culling.max_merged_imu_span_s` (11 s): after one merge most parked edges spanned
//! ~10 s, two could not merge, `kept_for_edges` climbed and the map hit `max_keyframes` and was
//! dropped every ~20 minutes (robot, 2026-10-12, build 23a413bdf2e9: culling yields 61, 31, 15,
//! 7, 3, 2, 1, `kept_for_edges` 54 -> 113).
//!
//! With [`StationaryEdgeConfig`] on (`stationary_edges` in the RON), once the inertial path is
//! settled:
//!
//! * a keyframe a parked robot's policy asks for on its TIME bound (the frame-count bound,
//!   `keyframe_policy.max_interval_s`) is withheld while the robot stays parked, up to
//!   [`PARKED_REFRESH_NS`] (30 s) after the last keyframe; tracking-quality triggers (the tracked
//!   ratio dropping, parallax, an empty reference) still insert, so a changed scene is still
//!   mapped. The refresh keyframe is linked by a stationary factor, so the stop's gyro mean
//!   reaches VI-BA every 30 s and at most 30 s of it is lost when the stop ends in motion;
//! * a keyframe whose interval since the last one was verified still is joined to it by a
//!   kornia-slam `StationaryFactor` (zero relative pose, zero velocity, bias continuity and the
//!   mean gyro rate), not by an IMU edge. It has no span limit, and culling composes it across a
//!   removed keyframe indefinitely; a keyframe where it meets an IMU edge is culled too (the
//!   factor dropped, the IMU edge moved onto the still neighbour);
//! * a keyframe after a still stretch that ended in motion gets an IMU edge over the TAIL only
//!   (from [`STATIONARY_TAIL_MIN_S`] before the motion's start), its bias link widened for the
//!   elided time, which the edge records so a merge or a repropagation keeps it: given zero
//!   velocity and an unchanged pose up to the tail's start, that is exactly the edge from the
//!   keyframe;
//! * the IMU and wheel rings are pruned every frame to that anchor, so no keyframe for an hour
//!   holds a few hundred samples, not an hour of them.
//!
//! "Still" is the [`StationaryRun`]: from each keyframe on, the IMU samples are checked in chunks
//! against the stationary initialization's per-sample checks (accelerometer and gyroscope spread,
//! mean rate), and two failing chunks in a row close the run (one is a disturbance: a door,
//! footsteps); the wheel path stays under the parked floors per 5 s window, as the interval gate
//! judged it, and the net displacement since the keyframe under the same floors. A hole in the
//! IMU samples or a discontinuity in the wheel readings closes the run for good; a gap in the
//! wheel readings (a stale head/body link) is bridged once the IMU has verified the stretch. The
//! next keyframe restarts the run. A keyframe also needs the camera within 1 cm / 0.5 deg of the
//! last one AND of the first keyframe of the stationary chain to be linked as still, so a slow
//! creep cannot end up under one composed zero-motion factor.
//!
//! Before the inertial path is initialized, and while a refinement of it is pending, nothing
//! changes: the still window and the refinements are built from IMU edges, and the parked
//! interval gate keeps those edges inside the merge cap.

use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuMeasurement;
use kornia_slam::inertial::ImuWindowStats;
use kornia_slam::mapping::stationary::{StationaryFactor, StationaryGyro};

use crate::error::VioError;
use crate::imu::{ImuBuffer, ImuGates};
use crate::stationary::{StaticInitConfig, check_imu_stats};
use crate::wheel::{PARKED_PATH_M, PARKED_PATH_RAD, WheelState, WheelWindowError};

/// IMU kept before a stationary anchor and the shortest tail edge, seconds: enough samples for
/// the ring's coverage gates. Also the floor on [`StationaryEdgeConfig::min_span`].
pub const STATIONARY_TAIL_MIN_S: f64 = 0.5;
pub(crate) const STATIONARY_TAIL_MIN_NS: u64 = 500_000_000;

/// Samples and readings kept before the anchor on top of the tail, nanoseconds.
pub(crate) const PRUNE_MARGIN_NS: u64 = 500_000_000;

/// The shortest IMU chunk the run checks at once, nanoseconds: 50 samples at 200 Hz, so the
/// spread estimate is not one frame's 13 samples (a false closure costs a keyframe).
pub(crate) const RUN_CHUNK_NS: u64 = 250_000_000;

/// Fewest samples in a checked chunk.
const RUN_CHUNK_MIN_SAMPLES: usize = 10;

/// The window the wheel path floors apply to, nanoseconds: the 5 s interval
/// `WheelState::parked_between` judged them over, so an hour of encoder jitter does not add up to
/// motion. Creep slower than the floors per window is caught by the net displacement since the
/// keyframe instead.
pub(crate) const RUN_WHEEL_WINDOW_NS: u64 = 5_000_000_000;

/// How far past the verified end of a still run that is still open a time-bound keyframe stays
/// withheld, nanoseconds, beyond the wheel `max_gap`: the wheel readings cross the head/body link,
/// which goes stale rather than applying backpressure, and a stall is not motion. Past it nothing
/// vouches for the robot and the keyframe is inserted.
pub(crate) const SENSOR_LAG_HOLD_NS: u64 = 2_000_000_000;

/// Longest a parked robot goes without a keyframe, seconds: past it a time-bound keyframe is
/// inserted (and linked by a stationary factor), so the gyro mean of a long stop reaches the
/// estimator every interval instead of being lost when the stop ends in motion, and VI-BA
/// refines the bias meanwhile. Culling composes these factors, so they cost no map room.
pub(crate) const PARKED_REFRESH_NS: u64 = 30_000_000_000;

/// Most stationary factors held. Culling composes them and a map drop clears them, so this is
/// never reached with `max_keyframes` set; without it it bounds a map that is never culled.
pub(crate) const MAX_STATIONARY_FACTORS: usize = 256;

/// The `stationary_edges` configuration. Build with [`StationaryEdgeConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StationaryEdgeConfig {
    /// Sigma of the zero relative camera translation, metres.
    pub translation_sigma_m: f64,
    /// Sigma of the zero relative camera rotation, radians.
    pub rotation_sigma_rad: f64,
    /// Sigma of the zero velocity at each end, m/s.
    pub velocity_sigma_m_s: f64,
    /// Huber threshold on the factor's 12-row pose and velocity chi-square; `None` keeps it
    /// quadratic.
    pub huber_chi2: Option<f64>,
    /// The still time since the last keyframe under which a keyframe still gets the IMU edge of
    /// old: a robot that only paused does not change how it is linked.
    pub min_span: Time,
}

impl Default for StationaryEdgeConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl StationaryEdgeConfig {
    /// kornia-slam's `StationaryFactor` defaults (2 mm, 0.002 rad, 1 cm/s, Huber at the 95 %
    /// 12-DOF chi-square) and a 1 s minimum span.
    pub fn new() -> Self {
        Self {
            translation_sigma_m: StationaryFactor::DEFAULT_TRANSLATION_SIGMA_M,
            rotation_sigma_rad: StationaryFactor::DEFAULT_ROTATION_SIGMA_RAD,
            velocity_sigma_m_s: StationaryFactor::DEFAULT_VELOCITY_SIGMA_M_S,
            huber_chi2: Some(StationaryFactor::DEFAULT_HUBER_CHI2),
            min_span: Time::new::<second>(1.0),
        }
    }

    /// Every sigma finite and positive, a positive Huber threshold if any, and `min_span` at
    /// least [`STATIONARY_TAIL_MIN_S`].
    pub fn validate(&self) -> Result<(), VioError> {
        for (field, value) in [
            ("translation_sigma_m", self.translation_sigma_m),
            ("rotation_sigma_rad", self.rotation_sigma_rad),
            ("velocity_sigma_m_s", self.velocity_sigma_m_s),
            ("huber_chi2", self.huber_chi2.unwrap_or(1.0)),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(VioError::StationaryParamInvalid { field, value });
            }
        }
        let min_span = self.min_span.get::<second>();
        if !(min_span.is_finite() && min_span >= STATIONARY_TAIL_MIN_S) {
            return Err(VioError::StationaryParamInvalid {
                field: "min_span_s",
                value: min_span,
            });
        }
        Ok(())
    }

    pub(crate) fn min_span_ns(&self) -> u64 {
        (self.min_span.get::<second>() * 1e9) as u64
    }

    /// The factor between keyframes `from_idx` and `to_idx`.
    pub(crate) fn factor(
        &self,
        from_idx: usize,
        to_idx: usize,
        span_s: f64,
        gyro: Option<StationaryGyro>,
        calib: kornia_sensors::imu::ImuCalib,
    ) -> StationaryFactor {
        StationaryFactor {
            gyro,
            translation_sigma_m: self.translation_sigma_m,
            rotation_sigma_rad: self.rotation_sigma_rad,
            velocity_sigma_m_s: self.velocity_sigma_m_s,
            huber_chi2: self.huber_chi2,
            ..StationaryFactor::new(from_idx, to_idx, span_s, calib)
        }
    }
}

/// Parked-keyframe counters. Diagnostics only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StationaryStats {
    /// Keyframes linked to the last one by a stationary factor, lifetime.
    pub edges_added: u64,
    /// Stationary factors composed across a culled keyframe, lifetime.
    pub edges_merged: u64,
    /// Keyframes culled at a boundary between a stationary factor and an IMU edge (the factor
    /// dropped, the IMU edge moved onto the still neighbour), lifetime.
    pub boundaries_removed: u64,
    /// Keyframes after a still stretch linked by an IMU edge over the tail only, lifetime.
    pub tail_edges: u64,
    /// Time-bound keyframes withheld while parked, once per keyframe interval, lifetime.
    pub keyframes_withheld: u64,
    /// Still runs closed by the IMU (two failing chunks in a row, or a hole in the samples),
    /// lifetime.
    pub runs_closed_imu: u64,
    /// Still runs closed by the wheels (motion, or a discontinuity in the readings), lifetime.
    pub runs_closed_wheels: u64,
    /// Stationary factors held now.
    pub factors: usize,
    /// The robot is parked now: a still run from the last keyframe covers the last frame.
    pub parked: bool,
}

/// What closed a [`StationaryRun`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunClose {
    /// A chunk of IMU samples failed the stillness checks.
    Imu,
    /// The IMU samples have a hole (dropped, a gap, too few for the rate).
    ImuHole,
    /// The wheels' path since the keyframe passed the parked floors.
    Wheels,
    /// The wheel readings have a gap, a discontinuity, or do not cover the start.
    WheelGap,
}

/// The still stretch since one keyframe, verified as the samples and readings arrive.
#[derive(Debug, Clone)]
pub(crate) struct StationaryRun {
    /// `Frame::idx` of the keyframe the run starts at.
    pub kf_idx: usize,
    /// Its stamp.
    pub start_ns: u64,
    open: bool,
    /// The IMU is verified still up to here.
    imu_end_ns: u64,
    /// The wheels are verified still up to here.
    wheel_end_ns: u64,
    /// `sum(mean gyro * chunk span)` and the span, over the still chunks.
    gyro_dt_sum: Vec3F64,
    covered_s: f64,
    /// End of a chunk that failed the stillness checks, while the next one has not been judged:
    /// one failing chunk is a disturbance (a door, footsteps), two in a row close the run.
    suspect_end_ns: Option<u64>,
    /// Start of the current wheel window ([`RUN_WHEEL_WINDOW_NS`]) and the wheel path in it.
    window_start_ns: u64,
    path_m: f64,
    path_rad: f64,
    /// Net base motion since the keyframe, `(x, y, theta)` in the base at the keyframe.
    net: [f64; 3],
    /// Why it closed, once it did.
    pub closed_by: Option<RunClose>,
    /// The newest anchor the rings were pruned to; never decreases. Once past the keyframe the
    /// next edge can no longer start at the keyframe.
    pub committed_ns: u64,
}

impl StationaryRun {
    /// A run starting at keyframe `kf_idx`, stamped `start_ns`.
    pub fn new(kf_idx: usize, start_ns: u64) -> Self {
        Self {
            kf_idx,
            start_ns,
            open: true,
            imu_end_ns: start_ns,
            wheel_end_ns: start_ns,
            gyro_dt_sum: Vec3F64::ZERO,
            covered_s: 0.0,
            suspect_end_ns: None,
            window_start_ns: start_ns,
            path_m: 0.0,
            path_rad: 0.0,
            net: [0.0; 3],
            closed_by: None,
            committed_ns: start_ns,
        }
    }

    /// The newest stamp both sensors vouch for. Frozen once the run closes.
    pub fn end_ns(&self) -> u64 {
        self.imu_end_ns.min(self.wheel_end_ns)
    }

    /// Whether the run is still open.
    #[cfg(test)]
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Whether the run vouches for `stamp_ns`: open, and the stamp no further past the verified
    /// end than a wheel reading gap may be (the hold `WheelState::parked_between` allows).
    pub fn covers(&self, stamp_ns: u64, max_gap_ns: u64) -> bool {
        self.open && stamp_ns <= self.end_ns().saturating_add(max_gap_ns)
    }

    /// Whether a time-bound keyframe at `stamp_ns` stays withheld: the run is open (nothing has
    /// shown motion) and the stamp is within the wheel gap plus [`SENSOR_LAG_HOLD_NS`] of the
    /// verified end. Looser than [`Self::covers`], which decides what the keyframe is linked by.
    pub fn holds(&self, stamp_ns: u64, max_gap_ns: u64) -> bool {
        self.open
            && stamp_ns
                <= self
                    .end_ns()
                    .saturating_add(max_gap_ns)
                    .saturating_add(SENSOR_LAG_HOLD_NS)
    }

    /// The mean raw gyro rate over the still chunks, if any were checked.
    pub fn gyro(&self) -> Option<StationaryGyro> {
        (self.covered_s > 0.0).then(|| StationaryGyro {
            mean: self.gyro_dt_sum / self.covered_s,
            covered_s: self.covered_s,
        })
    }

    fn close(&mut self, why: RunClose) -> Option<RunClose> {
        self.open = false;
        self.closed_by = Some(why);
        Some(why)
    }

    /// Checks the IMU samples that arrived since the verified end (or since a suspect chunk),
    /// once at least [`RUN_CHUNK_NS`] of them are buffered. A chunk failing the stillness checks
    /// makes the run suspect without moving its end; the next chunk passing clears it (the
    /// disturbance is left out of the gyro mean), failing too closes the run at the end before
    /// the first. A hole closes it at once. Returns the reason when this call closed the run.
    pub fn advance_imu(
        &mut self,
        buffer: &ImuBuffer,
        gates: &ImuGates,
        cfg: &StaticInitConfig,
        epoch_ns: u64,
    ) -> Option<RunClose> {
        if !self.open {
            return None;
        }
        let from = self.suspect_end_ns.unwrap_or(self.imu_end_ns);
        let newest = buffer.newest_stamp_ns()?;
        if newest < from.saturating_add(RUN_CHUNK_NS) {
            return None;
        }
        let Ok(window) = buffer.window(from, newest, gates) else {
            return self.close(RunClose::ImuHole);
        };
        let samples: Vec<ImuMeasurement> = window.map(|s| s.to_measurement(epoch_ns)).collect();
        if samples.len() < RUN_CHUNK_MIN_SAMPLES {
            return self.close(RunClose::ImuHole);
        }
        let Some(stats) = ImuWindowStats::from_samples(&samples) else {
            return self.close(RunClose::ImuHole);
        };
        if check_imu_stats(&stats, cfg).is_err() {
            if self.suspect_end_ns.is_some() {
                return self.close(RunClose::Imu);
            }
            self.suspect_end_ns = Some(newest);
            return None;
        }
        let span_s = (newest - from) as f64 * 1e-9;
        self.gyro_dt_sum += stats.mean_gyro * span_s;
        self.covered_s += span_s;
        self.suspect_end_ns = None;
        self.imu_end_ns = newest;
        None
    }

    /// Adds the wheel motion since the verified end. The path floors apply per
    /// [`RUN_WHEEL_WINDOW_NS`] window, the same floors to the net displacement since the keyframe.
    /// A gap in the readings is bridged once the IMU has verified the stretch still (until then
    /// the end waits); a discontinuity closes the run. Returns the reason when this call closed
    /// it.
    pub fn advance_wheels(&mut self, wheels: &WheelState) -> Option<RunClose> {
        if !self.open {
            return None;
        }
        let newest = wheels.newest_stamp_ns()?;
        if newest <= self.wheel_end_ns {
            return None;
        }
        let d = match wheels.delta(self.wheel_end_ns, newest, 0) {
            Ok(d) => d,
            Err(WheelWindowError::Gap) if self.imu_end_ns < newest => return None,
            Err(WheelWindowError::Gap) => match wheels.delta_across_gaps(self.wheel_end_ns, newest)
            {
                Ok(d) => d,
                Err(_) => return self.close(RunClose::WheelGap),
            },
            Err(_) => return self.close(RunClose::WheelGap),
        };
        self.path_m += d.path_m;
        self.path_rad += d.path_rad;
        let [x, y, theta] = self.net;
        let (sin, cos) = theta.sin_cos();
        let [dx, dy, dtheta] = d.delta;
        self.net = [
            x + cos * dx - sin * dy,
            y + sin * dx + cos * dy,
            theta + dtheta,
        ];
        let net_m = self.net[0].hypot(self.net[1]);
        // A NaN path is not parked.
        if !(self.path_m <= PARKED_PATH_M
            && self.path_rad <= PARKED_PATH_RAD
            && net_m <= PARKED_PATH_M
            && self.net[2].abs() <= PARKED_PATH_RAD)
        {
            return self.close(RunClose::Wheels);
        }
        self.wheel_end_ns = newest;
        if newest - self.window_start_ns >= RUN_WHEEL_WINDOW_NS {
            self.window_start_ns = newest;
            self.path_m = 0.0;
            self.path_rad = 0.0;
        }
        None
    }
}

/// Where the edge into the next keyframe starts, and what it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EdgeAnchor {
    /// An IMU edge from the last keyframe's stamp: what every keyframe got before.
    Keyframe(u64),
    /// A stationary factor; the wheel interval (zero motion) starts at `anchor_ns`.
    Stationary {
        anchor_ns: u64,
        span_s: f64,
        gyro: Option<StationaryGyro>,
    },
    /// An IMU edge from `anchor_ns`; the robot was still from the keyframe to there.
    Tail { anchor_ns: u64, elided_s: f64 },
}

impl EdgeAnchor {
    /// The stamp the IMU and wheel intervals of the edge start at.
    pub fn start_ns(&self) -> u64 {
        match *self {
            Self::Keyframe(t) => t,
            Self::Stationary { anchor_ns, .. } | Self::Tail { anchor_ns, .. } => anchor_ns,
        }
    }
}

/// The anchor of the edge from keyframe `(kf_idx, kf_ns)` to a keyframe at `stamp_ns`, given the
/// run, whether stationary mode is active, and whether the camera is within the parked floors of
/// the keyframe.
///
/// `Keyframe` unless stationary mode is active, the run starts at that keyframe and the still
/// part (to `stamp_ns` while the run covers it, to its frozen end after) is at least `min_span`.
/// Then `Stationary` while the run covers `stamp_ns` and the camera is still, `Tail` otherwise,
/// both anchored `STATIONARY_TAIL_MIN` before `stamp_ns`, clamped into `[kf_ns, end]` and never
/// earlier than an anchor the rings were already pruned to. A run whose rings were pruned past
/// the keyframe never goes back to `Keyframe`: those samples are gone.
#[allow(clippy::too_many_arguments)]
pub(crate) fn edge_anchor(
    run: Option<&StationaryRun>,
    kf_idx: usize,
    kf_ns: u64,
    stamp_ns: u64,
    active: bool,
    camera_still: bool,
    min_span_ns: u64,
    max_gap_ns: u64,
) -> EdgeAnchor {
    let Some(run) = run.filter(|r| r.kf_idx == kf_idx && r.start_ns == kf_ns) else {
        return EdgeAnchor::Keyframe(kf_ns);
    };
    let elided = |anchor_ns: u64| (anchor_ns - kf_ns) as f64 * 1e-9;
    let fallback = if run.committed_ns > kf_ns {
        EdgeAnchor::Tail {
            anchor_ns: run.committed_ns,
            elided_s: elided(run.committed_ns),
        }
    } else {
        EdgeAnchor::Keyframe(kf_ns)
    };
    if !active {
        return fallback;
    }
    let covered = run.covers(stamp_ns, max_gap_ns);
    let still_to = if covered { stamp_ns } else { run.end_ns() };
    if still_to.saturating_sub(kf_ns) < min_span_ns {
        return fallback;
    }
    let anchor_ns = stamp_ns
        .saturating_sub(STATIONARY_TAIL_MIN_NS)
        .clamp(kf_ns, run.end_ns().max(kf_ns))
        .max(run.committed_ns);
    if covered && camera_still {
        EdgeAnchor::Stationary {
            anchor_ns,
            span_s: stamp_ns.saturating_sub(kf_ns) as f64 * 1e-9,
            gyro: run.gyro(),
        }
    } else {
        EdgeAnchor::Tail {
            anchor_ns,
            elided_s: elided(anchor_ns),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::RawImuSample;
    use crate::wheel::{WheelOdomConfig, WheelSample};
    use cu29::units::si::f64::Frequency;
    use cu29::units::si::frequency::hertz;
    use kornia_3d::pose::Pose3d;
    use kornia_sensors::imu::ImuCalib;

    const MS: u64 = 1_000_000;
    const T0: u64 = 1_000_000_000_000;
    const BIAS: [f64; 3] = [0.003, -0.002, 0.001];

    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 0.00012,
            accel_noise: 0.0019,
            gyro_bias_noise: 0.000002,
            accel_bias_noise: 0.00004,
        }
    }

    fn still_cfg() -> StaticInitConfig {
        StaticInitConfig::from_noise(&calib(), Frequency::new::<hertz>(200.0))
    }

    fn gates() -> ImuGates {
        ImuGates::new(Frequency::new::<hertz>(200.0))
    }

    /// 200 Hz samples over `[from, to)` ms after `T0`: gravity on z, the gyro bias, small
    /// deterministic noise, plus `extra(k)` (accel, gyro).
    fn imu(
        from_ms: u64,
        to_ms: u64,
        extra: impl Fn(u64) -> ([f64; 3], [f64; 3]),
    ) -> Vec<RawImuSample> {
        (from_ms / 5..to_ms / 5)
            .map(|k| {
                let j = k as f64;
                let n = [(1.7 * j).sin(), (2.9 * j).sin(), (4.3 * j).sin()];
                let (ea, eg) = extra(k);
                RawImuSample {
                    stamp_ns: T0 + k * 5 * MS,
                    gyro: [0, 1, 2].map(|i| BIAS[i] + 0.002 * n[i] + eg[i]),
                    accel: [0, 1, 2].map(|i| [0.0, 0.0, 9.81][i] + 0.02 * n[i] + ea[i]),
                }
            })
            .collect()
    }

    fn still(_: u64) -> ([f64; 3], [f64; 3]) {
        ([0.0; 3], [0.0; 3])
    }

    fn wheels(readings: &[(u64, f64)]) -> WheelState {
        let mut w =
            WheelState::new(WheelOdomConfig::new(Pose3d::IDENTITY, 0.01, 0.01).expect("valid"));
        w.push(readings.iter().map(|&(ms, x)| WheelSample {
            stamp_ns: T0 + ms * MS,
            x,
            y: 0.0,
            theta: 0.0,
        }));
        w
    }

    /// Feeds `samples` in 100 ms batches, advancing after each; the closure, if any.
    fn run_imu(run: &mut StationaryRun, samples: Vec<RawImuSample>) -> Option<RunClose> {
        let mut b = ImuBuffer::with_capacity(100_000);
        let mut closed = None;
        for chunk in samples.chunks(20) {
            b.push(chunk.iter().copied(), 0);
            closed = closed.or(run.advance_imu(&b, &gates(), &still_cfg(), T0));
        }
        closed
    }

    /// Still samples keep the run open to the newest one, with the gyro mean the bias and the
    /// covered time the chunks'; a gyro spread, a slow turn and an accelerometer spread over two
    /// chunks each close it, frozen before the first; one disturbed chunk does not; a hole
    /// closes it at once.
    ///
    /// Mutations (each turned this red): accumulating the chunk mean without its span
    /// (`covered_s += 1`), the covered-time assert; closing on the first failing chunk, the
    /// knock.
    #[test]
    fn the_imu_side_closes_on_each_kind_of_motion_and_on_a_hole() {
        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu(&mut run, imu(0, 10_005, still)), None);
        assert!(run.is_open());
        // Checked in chunks of at least 250 ms: the end is within one chunk of the newest sample.
        assert!(run.imu_end_ns > T0 + 9_750 * MS && run.imu_end_ns <= T0 + 10_000 * MS);
        let g = run.gyro().expect("chunks checked");
        let span = (run.imu_end_ns - T0) as f64 * 1e-9;
        assert!(
            (g.covered_s - span).abs() < 1e-9,
            "{} vs {span}",
            g.covered_s
        );
        for (i, b) in BIAS.iter().enumerate() {
            let m = [g.mean.x, g.mean.y, g.mean.z][i];
            assert!((m - b).abs() < 2e-4, "axis {i}: {m} vs {b}");
        }

        let at = |k0: u64| move |k: u64| k >= k0 && k < k0 + 60;
        let spin = at(1000);
        for (extra, why) in [
            (
                Box::new(move |k| ([0.0; 3], [0.0, 0.0, if spin(k) { 0.2 } else { 0.0 }]))
                    as Box<dyn Fn(u64) -> ([f64; 3], [f64; 3])>,
                RunClose::Imu,
            ),
            (
                Box::new(|k| ([0.0; 3], [0.0, if k >= 1000 { 0.03 } else { 0.0 }, 0.0])),
                RunClose::Imu,
            ),
            (
                Box::new(move |k| {
                    (
                        [if spin(k) { 0.5 * (k as f64).sin() } else { 0.0 }, 0.0, 0.0],
                        [0.0; 3],
                    )
                }),
                RunClose::Imu,
            ),
        ] {
            let mut run = StationaryRun::new(1, T0);
            assert_eq!(run_imu(&mut run, imu(0, 10_005, extra)), Some(why));
            assert!(!run.is_open());
            // The IMU side alone (no wheels here): `end_ns` is the sooner of the two.
            let end = run.imu_end_ns;
            assert!(
                (T0 + 4_500 * MS..=T0 + 5_000 * MS).contains(&end),
                "frozen at the chunk before the motion: {}",
                (end - T0) / MS
            );
            assert!(!run.covers(T0 + 5_000 * MS, 0));
        }

        // One disturbed chunk (a 50 ms knock inside the 5.4-5.7 s chunk) is not motion: the run
        // stays open, its end passes the knock, and the knock stays out of the gyro mean.
        let knock = |k: u64| {
            let on = (1_100..1_110).contains(&k);
            (
                [if on { 2.0 } else { 0.0 }, 0.0, 0.0],
                [0.0, 0.0, if on { 0.5 } else { 0.0 }],
            )
        };
        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu(&mut run, imu(0, 10_005, knock)), None);
        assert!(run.is_open());
        assert!(run.imu_end_ns > T0 + 9_700 * MS);
        let g = run.gyro().expect("chunks checked");
        assert!((g.mean.z - BIAS[2]).abs() < 2e-4, "{}", g.mean.z);

        // A 100 ms hole in the samples.
        let mut holed = imu(0, 4_000, still);
        holed.extend(imu(4_100, 8_000, still));
        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu(&mut run, holed), Some(RunClose::ImuHole));
    }

    /// The wheel side: jitter-free parked readings keep it open; a creep past the parked floor
    /// within one 5 s window closes it; so does a creep too slow for any window whose NET
    /// displacement passes the floor; encoder jitter (one tick back and forth on every reading)
    /// for ten minutes does not, though its summed path is metres.
    ///
    /// Mutations (each turned this red): dropping the net check (the slow creep stayed open);
    /// never resetting the window path (the jitter closed the run, as the run-long sum did).
    #[test]
    fn the_wheel_side_judges_the_path_per_window_and_the_net_since_the_keyframe() {
        let mut run = StationaryRun::new(1, T0);
        let parked: Vec<(u64, f64)> = (0..40).map(|i| (i * 250, 0.0)).collect();
        let w = wheels(&parked);
        assert_eq!(run.advance_wheels(&w), None);
        assert_eq!(run.wheel_end_ns, T0 + 39 * 250 * MS);

        // Fed one reading at a time, like the tracker; the closure, if any.
        let feed = |x: &dyn Fn(u64) -> f64, n: u64| {
            let mut run = StationaryRun::new(1, T0);
            let mut w = wheels(&[]);
            let mut closed = None;
            for i in 0..n {
                w.push([WheelSample {
                    stamp_ns: T0 + i * 250 * MS,
                    x: x(i),
                    y: 0.0,
                    theta: 0.0,
                }]);
                closed = closed.or(run.advance_wheels(&w));
            }
            closed
        };
        // 2 mm/s: 10 mm in a 5 s window.
        assert_eq!(feed(&|i| 0.0005 * i as f64, 40), Some(RunClose::Wheels));
        // 0.4 mm/s: 2 mm per window, 6 mm net after 15 s.
        assert_eq!(feed(&|i| 0.0001 * i as f64, 80), Some(RunClose::Wheels));
        // 0.1 mm back and forth every 250 ms: 2 mm per window, 0 net, 240 mm summed.
        assert_eq!(
            feed(&|i| if i % 2 == 0 { 0.0 } else { 0.0001 }, 2_400),
            None
        );
    }

    /// A gap in the wheel readings (a stale head/body link) waits for the IMU and is bridged once
    /// the IMU has verified the stretch still; motion across the gap still closes the run.
    ///
    /// Mutation: closing on every gap (the old `WheelGap`) turned the bridged case red.
    #[test]
    fn a_wheel_gap_is_bridged_once_the_imu_vouches_for_it() {
        let gap = wheels(&[(0, 0.0), (250, 0.0), (2_000, 0.0)]);
        let mut run = StationaryRun::new(1, T0);
        run.imu_end_ns = T0 + 1_000 * MS;
        assert_eq!(run.advance_wheels(&gap), None);
        assert!(run.is_open());
        assert_eq!(run.wheel_end_ns, T0, "the end waits for the IMU");
        run.imu_end_ns = T0 + 2_100 * MS;
        assert_eq!(run.advance_wheels(&gap), None);
        assert!(run.is_open());
        assert_eq!(run.wheel_end_ns, T0 + 2_000 * MS, "bridged");

        let moved = wheels(&[(0, 0.0), (250, 0.0), (2_000, 0.02)]);
        let mut run = StationaryRun::new(1, T0);
        run.imu_end_ns = T0 + 2_100 * MS;
        assert_eq!(run.advance_wheels(&moved), Some(RunClose::Wheels));
    }

    fn open_run(end_ms: u64) -> StationaryRun {
        let mut run = StationaryRun::new(3, T0);
        run.imu_end_ns = T0 + end_ms * MS;
        run.wheel_end_ns = T0 + end_ms * MS;
        run.gyro_dt_sum = Vec3F64::new(0.003, 0.0, 0.0) * (end_ms as f64 * 1e-3);
        run.covered_s = end_ms as f64 * 1e-3;
        run
    }

    /// The anchor: `Keyframe` when stationary mode is off, the run is another keyframe's or the
    /// still part is under `min_span`; `Stationary` while covered with the camera still; `Tail`
    /// after the run closed or the camera moved; clamped into `[kf, end]`, monotone through
    /// `committed_ns`, and never `Keyframe` again once the rings were pruned past the keyframe.
    #[test]
    fn the_edge_anchor_follows_the_run() {
        let (min_span, gap) = (1_000 * MS, 500 * MS);
        let at = |ms: u64| T0 + ms * MS;
        let run = open_run(9_800);
        let anchor = |run: &StationaryRun, kf: usize, stamp: u64, active: bool, cam: bool| {
            edge_anchor(Some(run), kf, T0, stamp, active, cam, min_span, gap)
        };
        assert_eq!(
            anchor(&run, 3, at(10_000), false, true),
            EdgeAnchor::Keyframe(T0)
        );
        assert_eq!(
            anchor(&run, 4, at(10_000), true, true),
            EdgeAnchor::Keyframe(T0)
        );
        assert_eq!(
            edge_anchor(None, 3, T0, at(10_000), true, true, min_span, gap),
            EdgeAnchor::Keyframe(T0)
        );
        // 0.9 s still: under the minimum span.
        assert_eq!(
            anchor(&open_run(900), 3, at(900), true, true),
            EdgeAnchor::Keyframe(T0)
        );

        let EdgeAnchor::Stationary {
            anchor_ns,
            span_s,
            gyro,
        } = anchor(&run, 3, at(10_000), true, true)
        else {
            panic!("covered and still");
        };
        assert_eq!(anchor_ns, at(9_500));
        assert!((span_s - 10.0).abs() < 1e-9);
        assert!((gyro.unwrap().mean.x - 0.003).abs() < 1e-12);

        assert!(matches!(
            anchor(&run, 3, at(10_000), true, false),
            EdgeAnchor::Tail { anchor_ns, .. } if anchor_ns == at(9_500)
        ));

        // Closed at 6 s: the anchor freezes at the end, whatever the stamp.
        let mut closed = run.clone();
        closed.close(RunClose::Imu);
        closed.imu_end_ns = at(6_000);
        closed.wheel_end_ns = at(6_000);
        let EdgeAnchor::Tail {
            anchor_ns,
            elided_s,
        } = anchor(&closed, 3, at(20_000), true, true)
        else {
            panic!("closed");
        };
        assert_eq!(anchor_ns, at(6_000));
        assert!((elided_s - 6.0).abs() < 1e-9);

        // Pruned to 9.5 s: off or short, it stays a tail from there.
        let mut committed = open_run(900);
        committed.committed_ns = at(400);
        assert!(matches!(
            anchor(&committed, 3, at(900), false, true),
            EdgeAnchor::Tail { anchor_ns, .. } if anchor_ns == at(400)
        ));
        let mut ahead = run.clone();
        ahead.committed_ns = at(9_700);
        assert_eq!(
            anchor(&ahead, 3, at(10_000), true, true).start_ns(),
            at(9_700)
        );
    }

    /// Startup validation: the defaults pass; a non-positive sigma or Huber threshold and a
    /// minimum span under the tail are refused, naming the field.
    #[test]
    fn the_config_validates_each_field() {
        StationaryEdgeConfig::new().validate().unwrap();
        let bad = |f: fn(&mut StationaryEdgeConfig)| {
            let mut c = StationaryEdgeConfig::new();
            f(&mut c);
            c.validate()
        };
        for (f, field) in [
            (
                (|c: &mut StationaryEdgeConfig| c.translation_sigma_m = 0.0)
                    as fn(&mut StationaryEdgeConfig),
                "translation_sigma_m",
            ),
            (|c| c.rotation_sigma_rad = f64::NAN, "rotation_sigma_rad"),
            (|c| c.velocity_sigma_m_s = -1.0, "velocity_sigma_m_s"),
            (|c| c.huber_chi2 = Some(0.0), "huber_chi2"),
            (|c| c.min_span = Time::new::<second>(0.4), "min_span_s"),
        ] {
            match bad(f) {
                Err(VioError::StationaryParamInvalid { field: got, .. }) => assert_eq!(got, field),
                other => panic!("{field}: {other:?}"),
            }
        }
        let mut quad = StationaryEdgeConfig::new();
        quad.huber_chi2 = None;
        quad.validate().unwrap();
        let f = quad.factor(1, 2, 30.0, None, calib());
        assert!(f.is_valid());
        assert_eq!(f.huber_chi2, None);
        assert_eq!((f.from_idx, f.to_idx), (1, 2));
    }
}
