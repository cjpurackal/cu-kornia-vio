//! The hard motion veto: a pose two independent motion sensors contradict is never published.
//!
//! The per-frame fusion ([`crate::fusion`]) weighs vision against the IMU and the wheels from
//! the LAST KEYFRAME, and the keyframe-level motion check ([`crate::consistency`]) compares
//! keyframes' tracked poses. Neither looks at the pose about to be published against what the
//! gyro and the wheels say happened since the LAST PUBLISHED pose, and both are blind to a pose
//! that moved after them: on 2026-10-12 21:53:38 a local BA at a keyframe insertion moved the
//! newest keyframe 2.18 m and 92 deg while the gyro turned 4.5 deg and the wheels 11 mm, and the
//! pose sync-back published it. The fusion's factors were anchored on that moved keyframe, so the
//! wheels agreed with vision there; the motion check had already passed the interval on the
//! pre-BA poses.
//!
//! This is the last gate, applied to every pose the tracker is about to adopt and publish
//! (the tracked or fused pose, a relocalization) and to the newest keyframe's local BA shift:
//!
//! * **rotation** against the gyro's prediction (rotation of the last published pose by the
//!   gyro over the interval), when the gyro covered the interval;
//! * **translation** against the wheels' prediction (camera centre carried by the wheels), when
//!   the wheels covered it.
//!
//! Each check runs only with its sensor fresh. With the wheels stale the veto is the gyro's
//! alone, on rotation; with the IMU uninitialized the gyro still integrates (at the initial
//! bias, a few mrad/s, nothing next to the bounds); with neither, nothing is checked and the
//! pose is vision's (there is no dead reckoning to fall back on anyway).
//!
//! The bounds are generous on purpose (a hard veto, not a filter): ~10x the worst per-frame
//! disagreement measured on the robot (gyro vs vision < 1 deg per frame, wheels vs vision a few
//! cm, the camera mount's lever arm and the head's play on its mast at most ~24 mm in a 4.5 deg
//! turn). They grow with the time since the last trusted pose (the dead reckoning drifts) and
//! with the motion the sensors measured since (wheel scale, gyro scale).
//!
//! **Cumulative, not only per frame.** A per-frame check alone re-anchors on every accepted
//! frame, so vision that drifts or freezes below the per-frame bound (a frozen view in a slow
//! turn: 1 deg per frame against a 10 deg bound) is never vetoed however far it walks. Every
//! trusted pose is therefore also kept as an ANCHOR for a while and carried forward by each
//! frame's gyro and wheel increment; a candidate is checked against every anchor's carried pose,
//! with the bounds grown by that anchor's age and the motion since. The rotation is checked over
//! [`MotionVetoConfig::rotation_window_s`] (the gyro does not slip), the translation over the
//! much shorter [`MotionVetoConfig::translation_window_s`]: over seconds a frozen view and wheels
//! spinning in place look alike, so the wheels only veto a translation that walks off fast. The
//! anchors also see every later write of the pose (a local BA's sync-back), which a check
//! anchored on the previous frame's pose cannot.
//!
//! After a veto the tracker publishes the wheel and gyro dead reckoning, and accepts vision again
//! only once it has agreed with that prediction (within [`AGREE_FRACTION`] of the bounds) for
//! [`MotionVetoConfig::recovery_frames`] frames in a row. If it never does within
//! [`MotionVetoConfig::recovery_timeout`], the map is dropped and re-anchored at the
//! dead-reckoned pose (the world stays continuous with wheels).

use kornia_3d::pose::Pose3d;

use crate::error::VioError;

/// Default rotation bound, degrees: vision's rotation against the gyro's over one frame.
pub const DEFAULT_ROTATION_DEG: f64 = 10.0;
/// Default translation bound, metres: vision's camera centre against the wheels' over one frame.
pub const DEFAULT_TRANSLATION_M: f64 = 0.3;
/// Default rotation bound growth per second since the last trusted pose, degrees per second:
/// ~5x a large uncorrected gyro bias (0.007 rad/s).
pub const DEFAULT_ROTATION_DEG_PER_S: f64 = 2.0;
/// Default translation bound growth per second since the last trusted pose, metres per second.
pub const DEFAULT_TRANSLATION_M_PER_S: f64 = 0.1;
/// Default bound growth per unit of measured motion since the last trusted pose (radians per
/// radian turned, metres per metre driven): wheel slip and scale, gyro scale.
pub const DEFAULT_MOTION_FRACTION: f64 = 0.2;
/// Default consecutive agreeing frames before vision is accepted again after a veto.
pub const DEFAULT_RECOVERY_FRAMES: u32 = 5;
/// Default time vision has to agree again after a veto before the map is re-anchored at the
/// dead-reckoned pose, seconds.
pub const DEFAULT_RECOVERY_TIMEOUT_S: f64 = 3.0;
/// Default horizon of the cumulative rotation check, seconds: a frozen view in a 0.1 rad/s turn
/// (5.7 deg/s against bounds growing 2 + 1.15 deg/s from 10 deg) is vetoed after ~3.9 s.
pub const DEFAULT_ROTATION_WINDOW_S: f64 = 5.0;
/// Default horizon of the cumulative translation check, seconds: catches a camera centre that
/// walks off faster than ~0.7 m/s with the wheels still, and never wheels spinning in place
/// below ~0.85 m/s (faster than the base drives).
pub const DEFAULT_TRANSLATION_WINDOW_S: f64 = 0.5;
/// Anchors kept at most (memory bound; ~34 s at 15 Hz).
const MAX_ANCHORS: usize = 512;

/// The fraction of the hard bounds vision must stay within to count as AGREEING while the
/// tracker re-localizes after a veto. Tighter than the veto itself: being under the veto only
/// means "not impossible", and the frames that end a recovery decide the pose from then on.
pub const AGREE_FRACTION: f64 = 0.5;

/// The `motion_veto` config block, tracker side. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVetoConfig {
    /// Rotation bound against the gyro, radians.
    pub rotation_rad: f64,
    /// Translation bound against the wheels, metres.
    pub translation_m: f64,
    /// Rotation bound growth per second since the last trusted pose, rad/s.
    pub rotation_rad_per_s: f64,
    /// Translation bound growth per second since the last trusted pose, m/s.
    pub translation_m_per_s: f64,
    /// Bound growth per unit of measured motion since the last trusted pose.
    pub motion_fraction: f64,
    /// Consecutive agreeing frames before vision is accepted again after a veto.
    pub recovery_frames: u32,
    /// Seconds vision has to agree again before the dead-reckoned re-anchor.
    pub recovery_timeout_s: f64,
    /// Horizon of the cumulative rotation check against older trusted poses, seconds.
    pub rotation_window_s: f64,
    /// Horizon of the cumulative translation check, seconds.
    pub translation_window_s: f64,
}

impl Default for MotionVetoConfig {
    fn default() -> Self {
        Self {
            rotation_rad: DEFAULT_ROTATION_DEG.to_radians(),
            translation_m: DEFAULT_TRANSLATION_M,
            rotation_rad_per_s: DEFAULT_ROTATION_DEG_PER_S.to_radians(),
            translation_m_per_s: DEFAULT_TRANSLATION_M_PER_S,
            motion_fraction: DEFAULT_MOTION_FRACTION,
            recovery_frames: DEFAULT_RECOVERY_FRAMES,
            recovery_timeout_s: DEFAULT_RECOVERY_TIMEOUT_S,
            rotation_window_s: DEFAULT_ROTATION_WINDOW_S,
            translation_window_s: DEFAULT_TRANSLATION_WINDOW_S,
        }
    }
}

impl MotionVetoConfig {
    /// Refuses a bound or a timeout that is not finite and positive, a negative growth, and
    /// zero recovery frames (which would accept the very next frame, i.e. recover in place).
    pub fn validate(&self) -> Result<(), VioError> {
        for (field, value) in [
            ("rotation_deg", self.rotation_rad),
            ("translation_m", self.translation_m),
            ("recovery_timeout_s", self.recovery_timeout_s),
            ("rotation_window_s", self.rotation_window_s),
            ("translation_window_s", self.translation_window_s),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(VioError::VetoParamInvalid { field, value });
            }
        }
        for (field, value) in [
            ("rotation_deg_per_s", self.rotation_rad_per_s),
            ("translation_m_per_s", self.translation_m_per_s),
            ("motion_fraction", self.motion_fraction),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(VioError::VetoParamInvalid { field, value });
            }
        }
        if self.recovery_frames == 0 {
            return Err(VioError::VetoParamInvalid {
                field: "recovery_frames",
                value: 0.0,
            });
        }
        Ok(())
    }

    /// The bounds `(rotation rad, translation m)` after `dt_s` seconds and `(turned rad,
    /// driven m)` of measured motion since the last trusted pose.
    pub fn bounds(&self, dt_s: f64, turned_rad: f64, driven_m: f64) -> (f64, f64) {
        let dt = dt_s.max(0.0);
        (
            self.rotation_rad + self.rotation_rad_per_s * dt + self.motion_fraction * turned_rad,
            self.translation_m + self.translation_m_per_s * dt + self.motion_fraction * driven_m,
        )
    }
}

/// Which motion sensors measured the interval being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Sensors {
    /// The gyro: the prediction's rotation is the gyro's.
    pub gyro: bool,
    /// The wheels: the prediction's translation is the wheels'.
    pub wheels: bool,
}

/// How far a candidate pose is from the motion sensors' prediction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Disagreement {
    /// Rotation between the two, radians; NaN when the gyro did not measure.
    pub rotation_rad: f64,
    /// Camera-centre distance, metres; NaN when the wheels did not measure.
    pub translation_m: f64,
    /// The rotation bound it was held to, radians.
    pub rotation_bound_rad: f64,
    /// The translation bound, metres.
    pub translation_bound_m: f64,
}

impl Disagreement {
    /// `candidate` and `prediction` (both world-to-camera) compared on what `sensors` measured,
    /// against `bounds`. A residual a MEASURED sensor yields as NaN (a non-finite pose) is stored
    /// as infinity, so it breaches; NaN stays the mark of a sensor that did not measure.
    pub fn measure(
        candidate: &Pose3d,
        prediction: &Pose3d,
        sensors: Sensors,
        bounds: (f64, f64),
    ) -> Self {
        let measured = |v: f64| if v.is_nan() { f64::INFINITY } else { v };
        let rotation_rad = if sensors.gyro {
            measured(crate::consistency::angle_between(
                &candidate.rotation,
                &prediction.rotation,
            ))
        } else {
            f64::NAN
        };
        let translation_m = if sensors.wheels {
            measured((candidate.inverse().translation - prediction.inverse().translation).length())
        } else {
            f64::NAN
        };
        Self {
            rotation_rad,
            translation_m,
            rotation_bound_rad: bounds.0,
            translation_bound_m: bounds.1,
        }
    }

    /// The larger of the two residuals as a fraction of its bound; 0 when neither sensor
    /// measured, infinite for a non-finite residual ([`Self::measure`] stores a measured NaN as
    /// infinity).
    pub fn ratio(&self) -> f64 {
        let r = |v: f64, bound: f64| {
            if v.is_nan() {
                0.0
            } else if v.is_infinite() {
                f64::INFINITY
            } else {
                v / bound
            }
        };
        r(self.rotation_rad, self.rotation_bound_rad)
            .max(r(self.translation_m, self.translation_bound_m))
    }

    /// Whether a measured quantity exceeds `fraction` of its bound.
    fn exceeds(&self, fraction: f64) -> bool {
        self.ratio() > fraction
    }

    /// Past the hard bounds: veto.
    pub fn vetoed(&self) -> bool {
        self.exceeds(1.0)
    }

    /// Within [`AGREE_FRACTION`] of them: agreeing, for a recovery.
    pub fn agrees(&self) -> bool {
        !self.exceeds(AGREE_FRACTION)
    }
}

/// The veto's lifetime counters. Diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VetoStats {
    /// Candidate poses vetoed (tracked, fused or relocalized), lifetime.
    pub vetoes: u64,
    /// Frames published from the dead-reckoned prediction because vision was vetoed or not yet
    /// re-accepted, lifetime.
    pub dead_reckoned: u64,
    /// Recoveries: vision accepted again after agreeing for the configured run, lifetime.
    pub relocalizations: u64,
    /// Recoveries that timed out into a map drop re-anchored at the dead-reckoned pose, lifetime.
    pub rebootstraps: u64,
    /// Local BA results refused because they moved a keyframe past the bounds, or the newest one
    /// away from the gyro and wheel dead reckoning, lifetime (each drops the map, re-anchored at
    /// the pre-BA pose).
    pub ba_refused: u64,
    /// Recoveries started by the keyframe-level motion check (vision disagreed with the wheels
    /// and the gyro over a keyframe interval: DEGRADED), lifetime.
    pub degraded_recoveries: u64,
}

/// What the tracker does with a candidate pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// Adopt and publish it.
    Accept,
    /// It agrees, but the tracker is still re-localizing after a veto: publish the prediction.
    Hold,
    /// Past the hard bounds: publish the prediction, never let it touch the map.
    Veto,
}

/// A trusted pose kept for the cumulative check, carried forward by every frame's dead reckoning
/// since.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Anchor {
    /// Its frame's stamp.
    pub stamp_ns: u64,
    /// The trusted pose carried by the gyro and the wheels to the latest frame (world-to-camera).
    pub carried: Pose3d,
    /// [`VetoState`]'s motion totals when it was trusted.
    pub turned_at: f64,
    pub driven_at: f64,
    /// Whether the gyro, and the wheels, measured EVERY frame since: an interval the motion
    /// model bridged is not a measurement.
    pub gyro: bool,
    pub wheels: bool,
}

/// The tracker's veto state between frames.
#[derive(Debug, Clone, Default)]
pub(crate) struct VetoState {
    /// Stamp of the last frame whose pose was trusted (vision accepted, or an anchor).
    pub trusted_ns: Option<u64>,
    /// Rotation and translation the prediction moved through since then.
    pub turned_rad: f64,
    pub driven_m: f64,
    /// Rotation and translation the prediction moved through since the last restart.
    pub turned_total: f64,
    pub driven_total: f64,
    /// Trusted poses within the cumulative windows, oldest first, all of `generation`.
    pub anchors: std::collections::VecDeque<Anchor>,
    /// The world the anchors are expressed in.
    pub generation: u64,
    /// Set while re-localizing after a veto: since when, and the agreeing run so far.
    pub recovering_since_ns: Option<u64>,
    pub agree_run: u32,
    pub stats: VetoStats,
}

impl VetoState {
    /// Adds one frame's predicted motion to what the bounds grow with.
    pub fn accumulate(&mut self, turned_rad: f64, driven_m: f64) {
        if turned_rad.is_finite() {
            self.turned_rad += turned_rad;
            self.turned_total += turned_rad;
        }
        if driven_m.is_finite() {
            self.driven_m += driven_m;
            self.driven_total += driven_m;
        }
    }

    /// Carries every anchor by one frame's dead-reckoned increment `delta` (world-to-camera,
    /// `delta.compose(previous)`), measured by the gyro and the wheels as flagged. Anchors of
    /// another world (`generation`: an alignment, a re-anchor) are forgotten.
    pub fn advance(&mut self, delta: &Pose3d, gyro: bool, wheels: bool, generation: u64) {
        if generation != self.generation {
            self.anchors.clear();
            self.generation = generation;
        }
        for a in &mut self.anchors {
            a.carried = delta.compose(&a.carried);
            a.gyro &= gyro;
            a.wheels &= wheels;
        }
    }

    /// The pose at `stamp_ns` is trusted: the bounds restart from it, it becomes an anchor of the
    /// cumulative check, and any recovery ends.
    pub fn trust(&mut self, stamp_ns: u64, pose: &Pose3d, generation: u64) {
        self.trusted_ns = Some(stamp_ns);
        self.turned_rad = 0.0;
        self.driven_m = 0.0;
        self.recovering_since_ns = None;
        self.agree_run = 0;
        if generation != self.generation {
            self.anchors.clear();
            self.generation = generation;
        }
        if self.anchors.len() >= MAX_ANCHORS {
            self.anchors.pop_front();
        }
        self.anchors.push_back(Anchor {
            stamp_ns,
            carried: *pose,
            turned_at: self.turned_total,
            driven_at: self.driven_total,
            gyro: true,
            wheels: true,
        });
    }

    /// Drops the anchors past both windows at `stamp_ns`.
    pub fn prune(&mut self, cfg: &MotionVetoConfig, stamp_ns: u64) {
        let horizon = cfg.rotation_window_s.max(cfg.translation_window_s);
        while self
            .anchors
            .front()
            .is_some_and(|a| stamp_ns.saturating_sub(a.stamp_ns) as f64 * 1e-9 > horizon)
        {
            self.anchors.pop_front();
        }
    }

    /// The worst disagreement (largest [`Disagreement::ratio`]) of `candidate` against the
    /// anchors' carried poses at `stamp_ns`, each on what measured every frame since it and
    /// within its window; `None` with no anchor to check. The anchor of this very frame (age 0)
    /// is included: it is what a later write of the pose (a BA sync-back) is held to.
    pub fn against_anchors(
        &self,
        cfg: &MotionVetoConfig,
        candidate: &Pose3d,
        stamp_ns: u64,
        generation: u64,
    ) -> Option<Disagreement> {
        if generation != self.generation {
            return None;
        }
        let mut worst: Option<Disagreement> = None;
        for a in &self.anchors {
            let age = stamp_ns.saturating_sub(a.stamp_ns) as f64 * 1e-9;
            let sensors = Sensors {
                gyro: a.gyro && age <= cfg.rotation_window_s,
                wheels: a.wheels && age <= cfg.translation_window_s,
            };
            if !sensors.gyro && !sensors.wheels {
                continue;
            }
            let bounds = cfg.bounds(
                age,
                self.turned_total - a.turned_at,
                self.driven_total - a.driven_at,
            );
            let d = Disagreement::measure(candidate, &a.carried, sensors, bounds);
            if worst.is_none_or(|w| d.ratio() > w.ratio()) {
                worst = Some(d);
            }
        }
        worst
    }

    /// Starts a recovery at `stamp_ns` without a per-frame veto (the keyframe-level motion check
    /// found vision disagreeing): from the next frame the dead reckoning is published until
    /// vision agrees for the configured run. No-op while already recovering.
    pub fn begin_recovery(&mut self, stamp_ns: u64) -> bool {
        if self.recovering_since_ns.is_some() {
            return false;
        }
        self.recovering_since_ns = Some(stamp_ns);
        self.agree_run = 0;
        self.stats.degraded_recoveries += 1;
        true
    }

    /// A frame in a recovery that vision could not place (no candidate to decide on): the
    /// agreeing run is broken, so `recovery_frames` stays CONSECUTIVE frames.
    pub fn note_unplaced(&mut self) {
        if self.recovering_since_ns.is_some() {
            self.agree_run = 0;
        }
    }

    /// Forgets everything but the counters (a new map, a new world).
    pub fn restart(&mut self) {
        *self = Self {
            stats: self.stats,
            ..Self::default()
        };
    }

    /// Seconds since the last trusted pose; `fallback_ns` (the previous frame) when none.
    pub fn since_trusted_s(&self, stamp_ns: u64, fallback_ns: Option<u64>) -> f64 {
        self.trusted_ns
            .or(fallback_ns)
            .map_or(0.0, |t| stamp_ns.saturating_sub(t) as f64 * 1e-9)
    }

    /// The decision on a candidate at `stamp_ns` that is `d` from the prediction. Updates the
    /// recovery run and the counters; the caller adopts the candidate on `Accept` and calls
    /// [`Self::trust`] then.
    pub fn decide(&mut self, cfg: &MotionVetoConfig, d: &Disagreement, stamp_ns: u64) -> Gate {
        let vetoed = d.vetoed();
        match self.recovering_since_ns {
            None if !vetoed => Gate::Accept,
            None => {
                self.stats.vetoes += 1;
                self.recovering_since_ns = Some(stamp_ns);
                self.agree_run = 0;
                Gate::Veto
            }
            Some(_) => {
                if vetoed {
                    self.stats.vetoes += 1;
                    self.agree_run = 0;
                    Gate::Veto
                } else if d.agrees() {
                    self.agree_run += 1;
                    if self.agree_run >= cfg.recovery_frames {
                        self.stats.relocalizations += 1;
                        Gate::Accept
                    } else {
                        Gate::Hold
                    }
                } else {
                    self.agree_run = 0;
                    Gate::Hold
                }
            }
        }
    }

    /// Whether a recovery has run past its timeout at `stamp_ns`.
    pub fn timed_out(&self, cfg: &MotionVetoConfig, stamp_ns: u64) -> bool {
        self.recovering_since_ns
            .is_some_and(|t| stamp_ns.saturating_sub(t) as f64 * 1e-9 > cfg.recovery_timeout_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    fn yaw(deg: f64) -> Mat3F64 {
        let (s, c) = deg.to_radians().sin_cos();
        Mat3F64::from_cols_array(&[c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0])
    }

    /// The incident's numbers: the gyro turned 4.5 deg, the wheels moved 11 mm, vision said
    /// 92 deg and 2.18 m. Vetoed with both sensors, with the gyro alone (wheels stale) and with
    /// the wheels alone (no IMU); never with neither.
    #[test]
    fn test_the_incident_is_vetoed_by_either_sensor() {
        let cfg = MotionVetoConfig::default();
        let prediction = Pose3d::new(yaw(4.5), Vec3F64::new(0.011, 0.0, 0.0));
        let vision = Pose3d::new(yaw(92.0), Vec3F64::new(2.18, 0.0, 0.0));
        let bounds = cfg.bounds(0.067, 4.5f64.to_radians(), 0.011);
        for (gyro, wheels, want) in [
            (true, true, true),
            (true, false, true),
            (false, true, true),
            (false, false, false),
        ] {
            let d = Disagreement::measure(&vision, &prediction, Sensors { gyro, wheels }, bounds);
            assert_eq!(d.vetoed(), want, "gyro {gyro} wheels {wheels}: {d:?}");
        }
    }

    /// A frame within the robot's measured per-frame disagreement (1 deg, 3 cm) is accepted
    /// and agrees; one at 7 deg is not vetoed but does not agree.
    #[test]
    fn test_normal_disagreement_is_accepted() {
        let cfg = MotionVetoConfig::default();
        let s = Sensors {
            gyro: true,
            wheels: true,
        };
        let b = cfg.bounds(0.067, 0.0, 0.0);
        let p = Pose3d::new(yaw(0.0), Vec3F64::new(0.0, 0.0, 0.0));
        let d = Disagreement::measure(
            &Pose3d::new(yaw(1.0), Vec3F64::new(0.03, 0.0, 0.0)),
            &p,
            s,
            b,
        );
        assert!(!d.vetoed() && d.agrees(), "{d:?}");
        let d = Disagreement::measure(
            &Pose3d::new(yaw(7.0), Vec3F64::new(0.0, 0.0, 0.0)),
            &p,
            s,
            b,
        );
        assert!(!d.vetoed() && !d.agrees(), "{d:?}");
    }

    /// The bounds grow with time and motion since the last trusted pose.
    #[test]
    fn test_the_bounds_grow_with_time_and_motion() {
        let cfg = MotionVetoConfig::default();
        let (r0, t0) = cfg.bounds(0.0, 0.0, 0.0);
        let (r1, t1) = cfg.bounds(2.0, 1.0, 1.0);
        assert!((r0 - 10f64.to_radians()).abs() < 1e-12 && (t0 - 0.3).abs() < 1e-12);
        assert!((r1 - (r0 + 4f64.to_radians() + 0.2)).abs() < 1e-12);
        assert!((t1 - (t0 + 0.2 + 0.2)).abs() < 1e-12);
    }

    /// A veto starts a recovery: frames that agree are held until `recovery_frames` in a row,
    /// a disagreeing frame restarts the run, and the timeout is measured from the veto.
    #[test]
    fn test_recovery_needs_a_run_of_agreeing_frames() {
        let cfg = MotionVetoConfig {
            recovery_frames: 3,
            ..MotionVetoConfig::default()
        };
        let bad = Disagreement {
            rotation_rad: 1.0,
            translation_m: 0.0,
            rotation_bound_rad: 0.17,
            translation_bound_m: 0.3,
        };
        let good = Disagreement {
            rotation_rad: 0.01,
            ..bad
        };
        let middling = Disagreement {
            rotation_rad: 0.12,
            ..bad
        };
        let mut s = VetoState::default();
        assert_eq!(s.decide(&cfg, &good, 0), Gate::Accept);
        assert_eq!(s.decide(&cfg, &bad, 1), Gate::Veto);
        assert_eq!(s.decide(&cfg, &good, 2), Gate::Hold);
        assert_eq!(
            s.decide(&cfg, &middling, 3),
            Gate::Hold,
            "not agreeing resets"
        );
        assert_eq!(s.decide(&cfg, &good, 4), Gate::Hold);
        assert_eq!(s.decide(&cfg, &good, 5), Gate::Hold);
        assert!(!s.timed_out(&cfg, 6));
        assert!(s.timed_out(&cfg, 1 + 3_000_000_001));
        assert_eq!(s.decide(&cfg, &good, 6), Gate::Accept);
        assert_eq!(s.stats.vetoes, 1);
        assert_eq!(s.stats.relocalizations, 1);
    }

    /// A candidate with NaN entries (a degenerate solve) breaches when the sensor measured; an
    /// unmeasured sensor's residual is still no verdict.
    ///
    /// Mutation: keeping the measured NaN as NaN (skipped like an unmeasured one) turned this
    /// red (accepted).
    #[test]
    fn test_a_nan_candidate_is_vetoed() {
        let cfg = MotionVetoConfig::default();
        let p = Pose3d::new(yaw(0.0), Vec3F64::new(0.0, 0.0, 0.0));
        let nan = Pose3d::new(
            Mat3F64::from_cols_array(&[f64::NAN; 9]),
            Vec3F64::new(f64::NAN, 0.0, 0.0),
        );
        let b = cfg.bounds(0.067, 0.0, 0.0);
        for (gyro, wheels) in [(true, true), (true, false), (false, true)] {
            let d = Disagreement::measure(&nan, &p, Sensors { gyro, wheels }, b);
            assert!(
                d.vetoed() && !d.agrees(),
                "gyro {gyro} wheels {wheels}: {d:?}"
            );
        }
        let d = Disagreement::measure(&nan, &p, Sensors::default(), b);
        assert!(!d.vetoed(), "{d:?}");
    }

    /// One frame of the cumulative check: carry the anchors by the sensors' increment, decide on
    /// `vision` against the worst of the per-frame prediction and the anchors, trust it on Accept.
    fn step(
        s: &mut VetoState,
        cfg: &MotionVetoConfig,
        stamp_ns: u64,
        delta: &Pose3d,
        prediction: &Pose3d,
        vision: &Pose3d,
    ) -> Gate {
        s.accumulate(
            crate::consistency::angle(&delta.rotation),
            delta.translation.length(),
        );
        s.advance(delta, true, true, 0);
        s.prune(cfg, stamp_ns);
        let sensors = Sensors {
            gyro: true,
            wheels: true,
        };
        let b = cfg.bounds(0.067, s.turned_rad, s.driven_m);
        let mut d = Disagreement::measure(vision, prediction, sensors, b);
        if let Some(a) = s.against_anchors(cfg, vision, stamp_ns, 0)
            && a.ratio() > d.ratio()
        {
            d = a;
        }
        let g = s.decide(cfg, &d, stamp_ns);
        if g == Gate::Accept {
            s.trust(stamp_ns, vision, 0);
        }
        g
    }

    const FRAME: u64 = 66_666_667;

    /// A frozen view in a 0.25 rad/s turn: 1 deg per frame against the gyro, under any per-frame
    /// bound, is vetoed by the anchors within ~1 s. A camera centre walking off 0.2 m per frame
    /// with the wheels still is vetoed on the second frame.
    ///
    /// Mutation: `against_anchors` returning `None` turned this red (never vetoed).
    #[test]
    fn test_the_anchors_veto_drift_under_the_per_frame_bound() {
        let cfg = MotionVetoConfig::default();
        let mut s = VetoState::default();
        s.trust(0, &Pose3d::IDENTITY, 0);
        let gyro = Pose3d::new(yaw(1.0), Vec3F64::new(0.0, 0.0, 0.0));
        let mut vetoed_at = None;
        for k in 1..60u64 {
            // Vision stays at the last accepted pose; the prediction is it turned 1 deg.
            let vision = Pose3d::IDENTITY;
            let prediction = gyro.compose(&vision);
            if step(&mut s, &cfg, k * FRAME, &gyro, &prediction, &vision) == Gate::Veto {
                vetoed_at = Some(k);
                break;
            }
        }
        let k = vetoed_at.expect("a frozen view in a turn was never vetoed");
        assert!(k <= 20, "vetoed after {k} frames");

        let mut s = VetoState::default();
        s.trust(0, &Pose3d::IDENTITY, 0);
        let still = Pose3d::IDENTITY;
        let mut vision = Pose3d::IDENTITY;
        let mut gates = Vec::new();
        for k in 1..4u64 {
            let prev = vision;
            vision = Pose3d::new(yaw(0.0), Vec3F64::new(-0.2 * k as f64, 0.0, 0.0));
            gates.push(step(&mut s, &cfg, k * FRAME, &still, &prev, &vision));
        }
        assert_eq!(gates[0], Gate::Accept, "{gates:?}");
        assert_eq!(gates[1], Gate::Veto, "{gates:?}");
    }

    /// Wheels spinning in place at 0.3 m/s (vision rightly still) for 5 s are never vetoed: the
    /// translation window is short, since over seconds slip and a frozen view look alike.
    #[test]
    fn test_wheel_spin_in_place_is_not_vetoed() {
        let cfg = MotionVetoConfig::default();
        let mut s = VetoState::default();
        s.trust(0, &Pose3d::IDENTITY, 0);
        let spin = Pose3d::new(yaw(0.0), Vec3F64::new(0.02, 0.0, 0.0));
        for k in 1..75u64 {
            let vision = Pose3d::IDENTITY;
            let prediction = spin.compose(&vision);
            assert_eq!(
                step(&mut s, &cfg, k * FRAME, &spin, &prediction, &vision),
                Gate::Accept,
                "frame {k}"
            );
        }
    }

    /// Anchors of another world are forgotten, not compared.
    #[test]
    fn test_anchors_are_per_world() {
        let cfg = MotionVetoConfig::default();
        let mut s = VetoState::default();
        s.trust(0, &Pose3d::IDENTITY, 3);
        let far = Pose3d::new(yaw(90.0), Vec3F64::new(0.0, 0.0, 0.0));
        assert!(s.against_anchors(&cfg, &far, FRAME, 3).unwrap().vetoed());
        assert!(s.against_anchors(&cfg, &far, FRAME, 4).is_none());
        s.advance(&Pose3d::IDENTITY, true, true, 4);
        assert!(s.anchors.is_empty());
    }

    /// The agreeing run is CONSECUTIVE: a frame vision could not place in between restarts it.
    ///
    /// Mutation: `note_unplaced` a no-op turned this red (accepted on the third agreeing frame).
    #[test]
    fn test_an_unplaced_frame_breaks_the_agreeing_run() {
        let cfg = MotionVetoConfig {
            recovery_frames: 3,
            ..MotionVetoConfig::default()
        };
        let bad = Disagreement {
            rotation_rad: 1.0,
            translation_m: 0.0,
            rotation_bound_rad: 0.17,
            translation_bound_m: 0.3,
        };
        let good = Disagreement {
            rotation_rad: 0.01,
            ..bad
        };
        let mut s = VetoState::default();
        assert_eq!(s.decide(&cfg, &bad, 1), Gate::Veto);
        assert_eq!(s.decide(&cfg, &good, 2), Gate::Hold);
        s.note_unplaced();
        assert_eq!(s.decide(&cfg, &good, 4), Gate::Hold);
        s.note_unplaced();
        assert_eq!(s.decide(&cfg, &good, 6), Gate::Hold);
        assert_eq!(s.decide(&cfg, &good, 7), Gate::Hold);
        assert_eq!(s.decide(&cfg, &good, 8), Gate::Accept);
    }

    #[test]
    fn test_validation() {
        assert!(MotionVetoConfig::default().validate().is_ok());
        for bad in [
            MotionVetoConfig {
                rotation_rad: 0.0,
                ..Default::default()
            },
            MotionVetoConfig {
                translation_m: f64::NAN,
                ..Default::default()
            },
            MotionVetoConfig {
                motion_fraction: -0.1,
                ..Default::default()
            },
            MotionVetoConfig {
                recovery_frames: 0,
                ..Default::default()
            },
            MotionVetoConfig {
                recovery_timeout_s: f64::INFINITY,
                ..Default::default()
            },
            MotionVetoConfig {
                rotation_window_s: 0.0,
                ..Default::default()
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
