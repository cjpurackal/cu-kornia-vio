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
//! | agree       | disagree    | disagree      | [`MotionVerdict::VisionDisagrees`]: frozen or self-tracking vision. Local BA is skipped while the interval is in its window, so it cannot invent a bias to explain it |
//! | disagree    | agree       | disagree      | [`MotionVerdict::WheelsDisagree`]: slip. The interval's wheel factor is dropped |
//! | disagree    | disagree    | agree         | [`MotionVerdict::GyroDisagrees`]: bias or extrinsic. Counted only |
//!
//! Anything else, or no gyro, is [`MotionVerdict::Inconclusive`].
//!
//! TRANSLATION is not decidable here: only the wheels and vision measure it, and the two ways
//! it disagrees on a straight run — the wheels slipped, or vision froze — look identical to a
//! gyro that saw no turn. The loop test's straight legs were the second; dropping the wheels
//! there would have dropped the one sensor that was right. A translation disagreement is
//! counted and logged, never acted on.

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

/// A translation disagreement: vision's length outside this fraction of the wheels'.
pub const TRANSLATION_REL_TOL: f64 = 0.5;

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
            (true, false, false) => MotionVerdict::VisionDisagrees,
            (false, true, false) => MotionVerdict::WheelsDisagree,
            (false, false, true) => MotionVerdict::GyroDisagrees,
            (false, false, false) => MotionVerdict::Inconclusive,
        }
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
        assert!(!m(0.01, 0.0).translation_disagrees());
    }
}
