//! Tracker diagnostics: what the estimator knew about each frame it processed, for a status topic,
//! a black-box recorder and an offline replay to read.
//!
//! None of this is part of the estimate. [`VioDiagnostics`] says WHY a frame was or was not placed
//! (the pose stream alone cannot: an untracked frame carries no payload at all), and [`VioFrame`]
//! is the exact stereo pair the tracker solved, with its features, so a capture shows what the
//! tracker saw rather than what the camera produced while the solve was busy.

use bincode::de::Decoder;
use bincode::error::DecodeError;
use bincode::{Decode, Encode};
use cu29::pool::CuHandle;
use cu29::prelude::Reflect;
#[allow(unused_imports)]
use cu29::prelude::bevy_reflect;
use serde::{Deserialize, Serialize};

use crate::StereoPair;

/// The tracker's state after one processed frame.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub enum TrackingState {
    /// No world yet: the first map (or the first after a requested reset) is not anchored.
    #[default]
    Initializing,
    /// The pose was measured against the map.
    Tracking,
    /// Vision failed on a frame of an established world, and nothing carried the pose.
    Lost,
    /// The world is being re-anchored after a loss or a map drop: a re-bootstrap frame, or a
    /// frame waiting to bootstrap a map in a world that existed.
    Recovering,
    /// The pose was propagated on the wheels (vision failed, or a map drop's seeded anchor).
    Coasting,
}

impl TrackingState {
    /// The label a status topic and a recorder trigger compare on.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initializing => "initializing",
            Self::Tracking => "tracking",
            Self::Lost => "lost",
            Self::Recovering => "recovering",
            Self::Coasting => "coasting",
        }
    }
}

/// Why the frame was not measured. [`FailureReason::None`] on a measured frame.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub enum FailureReason {
    /// The frame was measured, or a map drop's seeded anchor (no vision failure).
    #[default]
    None,
    /// Bootstrap: fewer stereo points than the bootstrap minimum (a dark or textureless view).
    TooFewStereoPoints,
    /// The map refused the bootstrap keyframe.
    MapRefused,
    /// Tracking: too few map points matched by projection.
    LowProjectionMatches,
    /// Tracking: PnP did not converge.
    PnpFailed,
    /// Tracking: PnP converged on too few inliers.
    LowPnpInliers,
    /// Tracking: too few matches against the reference keyframe.
    LowReferenceMatches,
    /// Tracking: too few 2D-3D correspondences against the reference keyframe.
    LowReferenceCorrespondences,
    /// A map was anchored at a dead-reckoned pose after a loss: a coast, reported unmeasured.
    DeadReckonedAnchor,
    /// Per-frame fusion: vision disagreed with the IMU and wheel prediction past the gate, and
    /// the frame published the prediction as a coast.
    VisionRejected,
}

impl FailureReason {
    /// The label a status topic carries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TooFewStereoPoints => "too_few_stereo_points",
            Self::MapRefused => "map_refused",
            Self::LowProjectionMatches => "low_projection_matches",
            Self::PnpFailed => "pnp_failed",
            Self::LowPnpInliers => "low_pnp_inliers",
            Self::LowReferenceMatches => "low_reference_matches",
            Self::LowReferenceCorrespondences => "low_reference_correspondences",
            Self::DeadReckonedAnchor => "dead_reckoned_anchor",
            Self::VisionRejected => "vision_rejected",
        }
    }
}

/// How far the staged inertial initialization has got.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub enum InertialStage {
    /// No `inertial` block: the tracker is visual (and wheel) only.
    #[default]
    Off,
    /// Configured, not initialized yet (parked without a still window, or the solve refused).
    Uninitialized,
    /// Initialized from a still window; the refinements wait for motion.
    Static,
    /// Initialized by the motion solve.
    Initial,
    /// The first refinement (~5 s in) has run.
    Refine1,
    /// The second refinement (~15 s in) has run.
    Refine2,
}

impl InertialStage {
    /// The label a status topic carries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Uninitialized => "uninitialized",
            Self::Static => "static",
            Self::Initial => "initial",
            Self::Refine1 => "refine_1",
            Self::Refine2 => "refine_2",
        }
    }

    /// Whether the inertial path has been initialized.
    pub fn initialized(self) -> bool {
        !matches!(self, Self::Off | Self::Uninitialized)
    }
}

/// Who disagreed over a keyframe interval: the wheels, the gyro and vision compared.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub enum MotionVerdict {
    /// No moving interval checked yet (no wheels, or parked since start).
    #[default]
    Unchecked,
    /// All three agree.
    Consistent,
    /// The wheels and the gyro agree, vision does not: frozen or self-tracking vision.
    VisionDisagrees,
    /// Vision and the gyro agree, the wheels do not: slip, OR the head turned on its joint.
    WheelsDisagree,
    /// The wheels and vision agree, the gyro does not: a bias or an extrinsic.
    GyroDisagrees,
    /// No two agree against the third, or no gyro to arbitrate.
    Inconclusive,
}

impl MotionVerdict {
    /// The label a status topic carries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unchecked => "unchecked",
            Self::Consistent => "consistent",
            Self::VisionDisagrees => "vision_disagrees",
            Self::WheelsDisagree => "wheels_disagree",
            Self::GyroDisagrees => "gyro_disagrees",
            Self::Inconclusive => "inconclusive",
        }
    }

    /// Whether one sensor was outvoted by the other two.
    pub fn is_fault(self) -> bool {
        matches!(
            self,
            Self::VisionDisagrees | Self::WheelsDisagree | Self::GyroDisagrees
        )
    }
}

/// The last keyframe interval the wheels MOVED in, as each sensor saw it.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct MotionCheck {
    /// Moving intervals checked so far, lifetime. A change means `verdict` is a new interval's.
    pub checks: u64,
    /// That interval's rotation verdict.
    pub verdict: MotionVerdict,
    /// Vision's translation length disagreed with the wheels' (counted only).
    pub translation_disagrees: bool,
    /// The wheels' rotation over the interval, degrees.
    pub wheels_deg: f32,
    /// The gyro's, degrees; NaN without one.
    pub gyro_deg: f32,
    /// Vision's (the keyframes' tracked poses), degrees.
    pub vision_deg: f32,
    /// The wheels' translation, millimetres.
    pub wheels_mm: f32,
    /// Vision's translation, millimetres.
    pub vision_mm: f32,
    /// Vision's base height change over the interval minus the wheels' (zero, planar),
    /// millimetres: one sample of the out-of-plane error the wheel factor's `z_sigma` covers.
    pub oop_z_mm: f32,
    /// Vision's relative base roll over the interval, degrees (`tilt_sigma`'s sample).
    pub oop_roll_deg: f32,
    /// Vision's relative base pitch over the interval, degrees (`tilt_sigma`'s sample).
    pub oop_pitch_deg: f32,
    /// The path the wheels drove over the interval, metres (what `z_slope` scales with).
    pub path_m: f32,
}

/// The per-frame fusion's verdict on one frame: vision's pose against the IMU and wheel
/// prediction from the last keyframe. All zero when the fusion is off.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct FusionCheck {
    /// Frames fused (a tracked frame with at least one motion factor), lifetime.
    pub frames: u64,
    /// Frames whose vision the gate rejected (the prediction was published), lifetime.
    pub rejects: u64,
    /// Frames whose wheel factor was left out because the wheels disagreed with the gyro (slip,
    /// or the head turning on its mast), lifetime.
    pub wheel_dropped: u64,
    /// This frame's Mahalanobis distance between vision and the prediction; 0 with `dof` 0 when
    /// the fusion did not run on it.
    pub chi2: f32,
    /// Its degrees of freedom: 6 when a motion factor constrains the whole pose, 3 when only the
    /// gyro's rotation does.
    pub dof: u8,
    /// This frame's vision was rejected and the prediction published.
    pub rejected: bool,
    /// The fusion's cost on this frame, microseconds.
    pub cost_us: u32,
    /// Rejection runs that reached their limit, after which vision was accepted and the fusion
    /// suspended until the next keyframe, lifetime. A count that keeps climbing means the
    /// prediction is wrong, not vision.
    pub reject_runs_expired: u64,
    /// Frames whose IMU factor was left out because the IMU disagreed with the wheels and vision
    /// sided with the wheels, lifetime.
    pub imu_dropped: u64,
}

/// What the tracker knew about one processed frame. The time of validity is the frame's capture
/// stamp, on the carrying `CuMsg`'s `tov`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct VioDiagnostics {
    /// The frame's sequence number (the left eye's `CuImage::seq`).
    pub seq: u64,
    /// Frames the tracker processed, lifetime.
    pub frames: u64,
    /// Frames the camera produced that the tracker never saw (refused while a solve ran, or
    /// dropped upstream), lifetime: the sum of the gaps in `seq`.
    pub skipped_frames: u64,
    /// The state after this frame.
    pub state: TrackingState,
    /// Why it was not measured.
    pub failure: FailureReason,
    /// This frame inserted a keyframe.
    pub keyframe: bool,
    /// This frame was placed by relocalization against the existing map.
    pub relocalized: bool,
    /// Keypoints detected in the left image.
    pub keypoints: u32,
    /// Keypoints with a stereo depth.
    pub stereo_matches: u32,
    /// PnP inliers (0 when tracking failed).
    pub inliers: u32,
    /// Consecutive frames not measured.
    pub consecutive_failures: u32,
    /// Live map points.
    pub map_points: u32,
    /// Keyframes in the current map.
    pub keyframes: u32,
    /// Maps dropped (size bound, sustained loss, refused biases), lifetime.
    pub map_drops: u64,
    /// The tracker world this frame belongs to; see `VioStatus::reset_epoch`.
    pub reset_epoch: u64,
    /// Maps started over by any reset (requested, tracking loss, size bound), lifetime: bumps on
    /// every new map and on nothing else, unlike `reset_epoch` (also an inertial alignment).
    pub map_generation: u64,
    /// Wall-clock cost of the solve, milliseconds.
    pub solve_ms: f32,
    /// An interval vision disagreed on is in the local BA window: the pose may be frozen.
    pub vision_degraded: bool,
    /// The inertial initialization's progress.
    pub inertial_stage: InertialStage,
    /// The current gyro bias, rad/s (zero with the inertial path off).
    pub gyro_bias: [f32; 3],
    /// The current accelerometer bias, m/s^2.
    pub accel_bias: [f32; 3],
    /// The last moving interval's motion check.
    pub motion: MotionCheck,
    /// The per-frame fusion's verdict on this frame.
    pub fusion: FusionCheck,
}

/// [`FeaturePoint::class`] bit: the keypoint has a stereo depth.
pub const FEATURE_STEREO: u8 = 1;
/// [`FeaturePoint::class`] bit: matched to a map point and kept by PnP's final gate.
pub const FEATURE_INLIER: u8 = 2;
/// [`FeaturePoint::class`] bit: matched to a map point and rejected by PnP's final gate.
pub const FEATURE_OUTLIER: u8 = 4;

/// One left-image keypoint of a processed frame, in pixels of the rectified left eye.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct FeaturePoint {
    /// Column, pixels.
    pub x: f32,
    /// Row, pixels.
    pub y: f32,
    /// `FEATURE_*` bits.
    pub class: u8,
}

/// The stereo pair the tracker solved, with its left keypoints.
///
/// The eyes are the camera's own pool handles: forwarding this costs a refcount, and every holder
/// keeps one camera buffer checked out until it lets go.
#[derive(Default, Debug, Clone, Encode, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false, no_field_bounds)]
pub struct VioFrame {
    /// The pair, exactly as the tracker received it.
    pub pair: StereoPair,
    /// Its left keypoints, or `None` when the producer's pool was empty.
    #[reflect(ignore)]
    pub features: Option<CuHandle<Vec<FeaturePoint>>>,
}

impl Decode<()> for VioFrame {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            pair: Decode::decode(decoder)?,
            features: Decode::decode(decoder)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::config::standard;

    #[test]
    fn test_diagnostics_round_trip() {
        let d = VioDiagnostics {
            seq: 7,
            frames: 6,
            skipped_frames: 1,
            state: TrackingState::Lost,
            failure: FailureReason::LowPnpInliers,
            keypoints: 400,
            stereo_matches: 120,
            inertial_stage: InertialStage::Refine1,
            gyro_bias: [0.01, -0.02, 0.003],
            map_generation: 5,
            motion: MotionCheck {
                checks: 3,
                verdict: MotionVerdict::WheelsDisagree,
                gyro_deg: f32::NAN,
                ..Default::default()
            },
            ..Default::default()
        };
        let bytes = bincode::encode_to_vec(d, standard()).expect("encode");
        let (back, used): (VioDiagnostics, usize) =
            bincode::decode_from_slice(&bytes, standard()).expect("decode");
        assert_eq!(used, bytes.len());
        assert_eq!(back.state, TrackingState::Lost);
        assert_eq!(back.failure, FailureReason::LowPnpInliers);
        assert_eq!(back.gyro_bias, d.gyro_bias);
        assert_eq!(back.map_generation, 5);
        assert!(back.motion.gyro_deg.is_nan());
        assert_eq!(back.motion.verdict, MotionVerdict::WheelsDisagree);
    }

    #[test]
    fn test_only_an_outvoted_sensor_is_a_motion_fault() {
        for (v, fault) in [
            (MotionVerdict::Unchecked, false),
            (MotionVerdict::Consistent, false),
            (MotionVerdict::Inconclusive, false),
            (MotionVerdict::VisionDisagrees, true),
            (MotionVerdict::WheelsDisagree, true),
            (MotionVerdict::GyroDisagrees, true),
        ] {
            assert_eq!(v.is_fault(), fault, "{v:?}");
        }
    }

    #[test]
    fn test_feature_point_is_a_pool_element() {
        fn assert_element<T: cu29::pool::ElementType>() {}
        assert_element::<FeaturePoint>();
    }
}
