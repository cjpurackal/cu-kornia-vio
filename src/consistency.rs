//! Motion consistency: the wheels, the gyro and vision, compared over one keyframe interval.
//!
//! # Why
//!
//! Each of the three can be wrong in a way that looks healthy on its own. Vision tracks the
//! robot's own body when the head points down at it (2026-10-09 loop test, head at 71 deg: the
//! wheels drove 0.4 m legs and +-38 deg turns, vision saw under 10 mm and 3 deg), and local
//! VI-BA then explains the gyro's real turn with a gyro bias equal to the turn rate. The wheels
//! slip. A wrong extrinsic or a diverged bias makes the gyro disagree with both.
//!
//! # What is decidable
//!
//! ROTATION, because all three measure it and two agreeing outvote one:
//!
//! | wheels-gyro | vision-gyro | wheels-vision | verdict                                     |
//! |-------------|-------------|---------------|---------------------------------------------|
//! | agree       | agree       | agree         | [`MotionVerdict::Consistent`]               |
//! | agree       | disagree    | disagree      | [`MotionVerdict::VisionDisagrees`]: frozen or self-tracking vision, ONLY when the wheels and the gyro both turned past the noise floor ([`IntervalMotion::base_turned`]); with both still it is vision noise and [`MotionVerdict::Inconclusive`]. Local BA is skipped while the interval is in its window, so it cannot invent a bias to explain it |
//! | disagree    | agree       | disagree      | [`MotionVerdict::WheelsDisagree`]: slip, OR the head turned on its joint relative to the base (the IMU is in the OAK head, so the gyro and vision share any head-on-base motion). The interval's wheel factor is dropped either way: it constrains the camera through a fixed mount, which that interval broke |
//! | disagree    | disagree    | agree         | [`MotionVerdict::GyroDisagrees`]: bias or extrinsic. Counted only |
//!
//! Anything else, or no gyro, is [`MotionVerdict::Inconclusive`].
//!
//! TRANSLATION is not decidable here: only the wheels and vision measure it, and the two ways
//! it disagrees on a straight run — the wheels slipped, or vision froze — look identical to a
//! gyro that saw no turn. The loop test's straight legs were the second; dropping the wheels
//! there would have dropped the one sensor that was right. A translation disagreement is
//! counted and logged, never acted on.
//!
//! # What is logged
//!
//! Every interval the wheels moved in ([`IntervalMotion::wheels_moved`]), with all three
//! sensors' rotations and both translations, whatever the verdict: a watched drive is read off
//! these `motion check:` lines. Parked intervals speak only when notable, throttled.

use kornia_3d::pose::Pose3d;
use kornia_algebra::Mat3F64;

/// Rotations closer than this always agree: the noise floor of a keyframe-interval rotation
/// (vision ~0.3 deg, the wheels' 0.36 deg round trip, the gyro far below either), with margin.
pub const ROTATION_ABS_TOL_RAD: f64 = 1.5 * std::f64::consts::PI / 180.0;

/// ...and so do rotations within this fraction of the largest of the three: wheel slip in a
/// turn and a scale error grow with the turn.
pub const ROTATION_REL_TOL: f64 = 0.25;

/// Translation is compared only once the wheels drove this far in the interval.
pub const TRANSLATION_MIN_M: f64 = 0.03;

/// A translation disagreement: vision's length outside this fraction of the wheels'. Counted
/// only, so set to flag the 2026-10-09 10 cm drive's shape (71 vs 99 mm, 28 %): the wheels'
/// straight-line error is ~1 %.
pub const TRANSLATION_REL_TOL: f64 = 0.2;

/// An interval the wheels moved in: logged whatever its verdict.
pub const MOVED_MIN_M: f64 = 0.01;
/// ...or turned this much, radians.
pub const MOVED_MIN_RAD: f64 = 0.5 * std::f64::consts::PI / 180.0;

/// What a keyframe interval's three rotations say about who is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionVerdict {
    /// All three agree.
    Consistent,
    /// The wheels and the gyro agree; vision does not.
    VisionDisagrees,
    /// Vision and the gyro agree; the wheels do not.
    WheelsDisagree,
    /// The wheels and vision agree; the gyro does not.
    GyroDisagrees,
    /// No two agree against the third, or there is no gyro to arbitrate.
    Inconclusive,
}

/// One keyframe interval as each sensor saw it: camera-at-end in camera-at-start, `T_c0c1`.
#[derive(Debug, Clone, Copy)]
pub struct IntervalMotion {
    /// The wheels' base delta through the mount at each end.
    pub wheels: Pose3d,
    /// The two keyframes' tracked (pre-BA) poses.
    pub vision: Pose3d,
    /// `R_c0c1` from the gyro; `None` without the inertial path or the interval's IMU edge.
    pub gyro: Option<Mat3F64>,
}

/// The rotation angle of `r`, radians.
pub fn angle(r: &Mat3F64) -> f64 {
    let cos = ((r.x_axis.x + r.y_axis.y + r.z_axis.z) - 1.0) * 0.5;
    cos.clamp(-1.0, 1.0).acos()
}

/// The angle between two rotations, radians.
pub fn angle_between(a: &Mat3F64, b: &Mat3F64) -> f64 {
    angle(&(a.transpose() * *b))
}

/// Whether two sensors' rotations over one interval disagree, by the rule
/// [`IntervalMotion::verdict`] applies: further apart than [`ROTATION_ABS_TOL_RAD`] and than
/// [`ROTATION_REL_TOL`] of the larger. What screens a wheel factor against the gyro when vision
/// cannot be compared over the interval.
pub fn rotations_disagree(a: &Mat3F64, b: &Mat3F64) -> bool {
    let tol = ROTATION_ABS_TOL_RAD.max(ROTATION_REL_TOL * angle(a).max(angle(b)));
    angle_between(a, b) > tol
}

impl IntervalMotion {
    /// The rotation verdict. See the module table.
    pub fn verdict(&self) -> MotionVerdict {
        let (w, v) = (self.wheels.rotation, self.vision.rotation);
        let Some(g) = self.gyro else {
            return MotionVerdict::Inconclusive;
        };
        let largest = angle(&w).max(angle(&v)).max(angle(&g));
        let tol = ROTATION_ABS_TOL_RAD.max(ROTATION_REL_TOL * largest);
        let agree = |a: &Mat3F64, b: &Mat3F64| angle_between(a, b) <= tol;
        match (agree(&w, &g), agree(&v, &g), agree(&w, &v)) {
            (true, true, _) | (_, true, true) | (true, _, true) => MotionVerdict::Consistent,
            // Frozen vision needs a turn to freeze through: the wheels and the gyro must BOTH
            // have turned past the noise floor. With both still, vision moving alone is vision
            // noise (a young map's PnP: 1.3 deg per keyframe on the parked robot, 2026-10-10),
            // never "frozen", and a parked robot can never be DEGRADED.
            (true, false, false) if !self.base_turned() => MotionVerdict::Inconclusive,
            (true, false, false) => MotionVerdict::VisionDisagrees,
            (false, true, false) => MotionVerdict::WheelsDisagree,
            (false, false, true) => MotionVerdict::GyroDisagrees,
            (false, false, false) => MotionVerdict::Inconclusive,
        }
    }

    /// Whether the wheels AND the gyro both turned past [`ROTATION_ABS_TOL_RAD`]: the motion
    /// evidence [`MotionVerdict::VisionDisagrees`] requires.
    pub fn base_turned(&self) -> bool {
        self.gyro.is_some_and(|g| {
            angle(&self.wheels.rotation) >= ROTATION_ABS_TOL_RAD
                && angle(&g) >= ROTATION_ABS_TOL_RAD
        })
    }

    /// Whether the wheels moved in the interval: [`MOVED_MIN_M`] or [`MOVED_MIN_RAD`].
    pub fn wheels_moved(&self) -> bool {
        self.wheels.translation.length() >= MOVED_MIN_M
            || angle(&self.wheels.rotation) >= MOVED_MIN_RAD
    }

    /// Whether vision's translation length is outside [`TRANSLATION_REL_TOL`] of the wheels',
    /// once the wheels drove [`TRANSLATION_MIN_M`]. Counted, never acted on.
    pub fn translation_disagrees(&self) -> bool {
        let w = self.wheels.translation.length();
        let v = self.vision.translation.length();
        w >= TRANSLATION_MIN_M && (v - w).abs() > TRANSLATION_REL_TOL * w
    }
}

/// Consistency counters, one per verdict plus what was done about them. Diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConsistencyStats {
    /// Intervals with a wheel factor that were checked.
    pub checked: u64,
    /// [`MotionVerdict::Consistent`].
    pub consistent: u64,
    /// [`MotionVerdict::VisionDisagrees`]; each one holds local BA off while in its window.
    pub vision_disagrees: u64,
    /// [`MotionVerdict::WheelsDisagree`]; each one's wheel factor was dropped.
    pub wheels_disagree: u64,
    /// [`MotionVerdict::GyroDisagrees`].
    pub gyro_disagrees: u64,
    /// [`MotionVerdict::Inconclusive`].
    pub inconclusive: u64,
    /// Translation length disagreements (counted only; see the module doc).
    pub translation_disagrees: u64,
    /// Local BA solves skipped because an interval vision disagreed on was in the window.
    pub ba_skipped: u64,
    /// Times tracking turned DEGRADED (a vision-suspect interval entered the BA window).
    pub degraded_entries: u64,
    /// Wheel factors dropped, outside [`ConsistencyStats::checked`], because they contradicted
    /// the gyro over an interval vision could not be compared on (a map's first local BA
    /// between its ends, or an alignment).
    pub wheels_dropped_on_gyro: u64,
}

impl ConsistencyStats {
    pub(crate) fn count(&mut self, verdict: MotionVerdict, translation_disagrees: bool) {
        self.checked += 1;
        self.translation_disagrees += u64::from(translation_disagrees);
        *match verdict {
            MotionVerdict::Consistent => &mut self.consistent,
            MotionVerdict::VisionDisagrees => &mut self.vision_disagrees,
            MotionVerdict::WheelsDisagree => &mut self.wheels_disagree,
            MotionVerdict::GyroDisagrees => &mut self.gyro_disagrees,
            MotionVerdict::Inconclusive => &mut self.inconclusive,
        } += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{SO3F64, Vec3F64};

    fn yaw_pose(deg: f64, forward_m: f64) -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, deg.to_radians(), 0.0)).matrix(),
            Vec3F64::new(0.0, 0.0, forward_m),
        )
    }

    fn gyro(deg: f64) -> Option<Mat3F64> {
        Some(yaw_pose(deg, 0.0).rotation)
    }

    fn motion(w: f64, v: f64, g: Option<Mat3F64>) -> IntervalMotion {
        IntervalMotion {
            wheels: yaw_pose(w, 0.0),
            vision: yaw_pose(v, 0.0),
            gyro: g,
        }
    }

    /// The loop test's turn: wheels and gyro 8.6 deg (0.3 rad/s over 0.5 s), vision frozen.
    #[test]
    fn frozen_vision_in_a_turn_is_vision_disagreeing() {
        assert_eq!(
            motion(8.6, 0.2, gyro(8.4)).verdict(),
            MotionVerdict::VisionDisagrees
        );
    }

    /// Parked: the wheels and the gyro still, a young map's vision 2.5 deg off. Vision is the odd
    /// one out but nothing turned, so it cannot be frozen.
    #[test]
    fn vision_moving_alone_on_a_still_base_is_not_frozen() {
        let m = motion(0.0, 2.5, gyro(0.1));
        assert!(!m.base_turned());
        assert_eq!(m.verdict(), MotionVerdict::Inconclusive);
        // The same three with the base turning: vision froze.
        let turn = motion(8.0, 2.5, gyro(8.1));
        assert!(turn.base_turned());
        assert_eq!(turn.verdict(), MotionVerdict::VisionDisagrees);
    }

    #[test]
    fn two_rotations_disagree_past_the_same_bands() {
        let r = |deg: f64| yaw_pose(deg, 0.0).rotation;
        assert!(rotations_disagree(&r(8.6), &r(0.0)));
        assert!(!rotations_disagree(&r(0.3), &r(-0.5)));
        assert!(!rotations_disagree(&r(40.0), &r(34.0)));
    }

    #[test]
    fn wheels_turning_alone_are_slipping() {
        assert_eq!(
            motion(8.6, 0.1, gyro(0.0)).verdict(),
            MotionVerdict::WheelsDisagree
        );
    }

    #[test]
    fn a_gyro_alone_is_a_bias_or_an_extrinsic() {
        assert_eq!(
            motion(0.1, 0.0, gyro(6.0)).verdict(),
            MotionVerdict::GyroDisagrees
        );
    }

    /// The noise floor and the relative band: parked, and a 40 deg turn 15 % apart.
    #[test]
    fn noise_and_proportional_error_are_consistent() {
        assert_eq!(
            motion(0.3, -0.5, gyro(0.6)).verdict(),
            MotionVerdict::Consistent
        );
        assert_eq!(
            motion(40.0, 34.0, gyro(38.0)).verdict(),
            MotionVerdict::Consistent
        );
    }

    #[test]
    fn a_parked_interval_did_not_move_and_a_drive_or_turn_did() {
        assert!(!motion(0.2, 0.0, None).wheels_moved());
        assert!(motion(1.0, 0.0, None).wheels_moved());
        let drive = IntervalMotion {
            wheels: yaw_pose(0.0, 0.03),
            vision: yaw_pose(0.0, 0.0),
            gyro: None,
        };
        assert!(drive.wheels_moved());
    }

    #[test]
    fn without_a_gyro_nothing_is_decided() {
        assert_eq!(
            motion(8.6, 0.0, None).verdict(),
            MotionVerdict::Inconclusive
        );
    }

    /// Translation is only counted, past the wheels' minimum.
    #[test]
    fn translation_compares_lengths_past_the_minimum() {
        let m = |w: f64, v: f64| IntervalMotion {
            wheels: yaw_pose(0.0, w),
            vision: yaw_pose(0.0, v),
            gyro: None,
        };
        assert!(m(0.4, 0.01).translation_disagrees());
        assert!(!m(0.4, 0.35).translation_disagrees());
        // The 10 cm drive: 71 mm of vision against 98.9 mm of wheels is flagged.
        assert!(m(0.0989, 0.071).translation_disagrees());
        assert!(!m(0.0989, 0.095).translation_disagrees());
        assert!(!m(0.01, 0.0).translation_disagrees());
    }
}
