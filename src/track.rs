//! Stereo visual-odometry tracking core.
//!
//! This module is the **composition root** over `kornia-slam`'s public API.
//! `kornia-slam` ships the algorithms (stereo matching, map-projection PnP,
//! bundle adjustment, the map itself) but its own orchestration,
//! `kornia_slam::SlamSystem`, keeps its tracking state (`SystemState`,
//! `SystemMode`) and inertial schedule crate-private and cannot be driven piece
//! by piece. `kornia-slam` is read-only for us, so the *logic* of that
//! orchestration is re-implemented here for a live stereo camera, over the
//! map's public mutation API. Nothing is vendored.
//!
//! # Pose convention
//!
//! Every pose in this module — [`TrackedPose::pose_world_to_cam`], the poses
//! stored on keyframes, everything `kornia_slam` returns — is
//! **world-to-camera** (`kornia_3d::pose::Pose3d`): `p_cam = R * p_world + t`.
//!
//! A viewer or a `TransformStamped` almost always wants the **inverse**:
//! camera-in-world, i.e. where the camera *is*. Use
//! [`TrackedPose::cam_in_world`]. Publishing `pose_world_to_cam` directly is the
//! classic "frustum flies backwards" bug.
//!
//! The world frame is fixed at the first successfully bootstrapped frame's pose,
//! which is identity. Scale is **metric**, taken from the stereo baseline — no
//! Sim(3) gauge freedom, which is why the synthetic test asserts absolute
//! trajectory error without any alignment step.
//!
//! # Input contract
//!
//! [`Tracker::process_stereo`] takes a **rectified, row-aligned** pair and a
//! zero-distortion [`PinholeCamera`] describing it — build both with
//! `kornia_3d::stereo::StereoRectifier`. `kornia_slam::stereo::unproject_stereo`
//! back-projects raw keypoint coordinates and the keyframe-growth pass builds
//! `K^-1` by hand from `fx/fy/cx/cy`; both silently assume zero distortion and
//! neither checks. [`Tracker`] checks, and returns
//! [`VioError::DistortedCamera`].
//!
//! # Frame indices
//!
//! `kornia_slam` keys keyframes by `Frame::idx` (`Map::get_keyframe` is a linear
//! scan over it) and `KeyframePolicy::should_insert` measures the keyframe gap in
//! those indices. The tracker therefore owns the counter: it stamps every frame
//! with its own **received-frame count**, ignoring any `idx` on a caller-supplied
//! [`Frame`]. Dropped camera frames consequently do *not* inflate the gap and
//! force a keyframe, which is the behaviour a live variable-rate stream wants;
//! the upstream example, reading a dataset, keys off source indices instead.
//!
//! # The inertial path, and why it is off
//!
//! [`TrackerConfig::inertial`] is `None` unless the caller supplies a full
//! camera-to-IMU extrinsic and the four IMU noise densities, and with it `None`
//! this module is byte-for-byte the stereo-only tracker it has always been: no
//! buffer is allocated, no IMU factor is ever inserted into the map, local BA stays on
//! `run_local_ba`, and [`Tracker::push_imu`] refuses.
//!
//! It is off by default because an OAK-D answers `imu_to_camera_extrinsics` with
//! "IMU calibration data is not available on device yet", so without a measured
//! extrinsic there is no value that would be anything but a guess. The
//! static gravity read pins roll and pitch (chip +y is up, camera +y is down);
//! yaw about gravity is unresolved and needs a motion-correlated hand-eye solve.
//! A guessed extrinsic is not a degraded answer: `ImuInitializer` solves for
//! gravity in the frame the extrinsic names, and `apply_initialization` then
//! ROTATES THE WHOLE MAP onto it, so a wrong extrinsic produces a self-consistent
//! trajectory that is tipped over, with every health field reading green.
//!
//! # What is deliberately absent
//!
//! No loop closure. No transport, no camera I/O, no threads — this is a pure
//! function of the frames and samples you feed it, so it is unit-testable.

use std::collections::HashSet;

use cu29::prelude::*;
use cu29::units::si::f64::{Length, Time};
use cu29::units::si::frequency::hertz;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, TriangulationConfig, triangulate_matched_points};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::features::{
    OrbDetector, OrbMatchConfig, hamming_distance, match_orb_descriptors,
};
use kornia_imgproc::resize::resize_fast_mono;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, PreintegratedImu};
use kornia_slam::Frame;
use kornia_slam::TrackingStatus;
use kornia_slam::initialization::two_view::TwoViewInitConfig;
use kornia_slam::initialization::{
    AlignedTrackingState, ImuInitResult, ImuInitializer, ImuWindowStats, StationaryInitError,
};
use kornia_slam::mapping::bundle_adjustment::{local_window, run_local_ba, run_local_inertial_ba};
use kornia_slam::mapping::culling::cull_landmarks;
use kornia_slam::mapping::growth::accepted_pair_claims;
use kornia_slam::mapping::map::{
    ImuFactor, Keyframe, LandmarkSeed, Map, MapInsertion, ORB_N_LEVELS, ORB_SCALE_FACTOR,
    ObservationKey,
};
use kornia_slam::mapping::wheel_odom::WheelOdomFactor;
use kornia_slam::stereo::{StereoMatchConfig, compute_stereo_matches, unproject_stereo};
use kornia_slam::tracking::local_map::landmarks_in_frustum;
use kornia_slam::tracking::pose_estimation::map_projection::{
    MapProjectionConfig, MapProjectionRejectReason,
};
use kornia_slam::tracking::pose_estimation::pnp::{self, PnpConfig};
use kornia_slam::tracking::{
    KeyframePolicy, LocalMapSelectionConfig, MapProjectionEstimator, TrackingLossRecoveryPolicy,
    select_local_map_points,
};

use crate::consistency::{ConsistencyStats, IntervalMotion, MotionVerdict};
use crate::covariance::{self, Cov6};
use crate::error::VioError;
use crate::fusion::{
    DEFAULT_EXTRINSIC_SIGMA_M, DEFAULT_EXTRINSIC_SIGMA_RAD, FusionConfig, FusionOutcome,
    FusionStats, IncrementalPreintegration, KF_GYRO_BIAS_SIGMA, MAX_CONSECUTIVE_REJECTS,
    MotionFactor, Predictor, UNINITIALIZED_GYRO_BIAS_SIGMA, VISION_ROTATION_SIGMA_FLOOR,
    VISION_TRANSLATION_SIGMA_FLOOR, VisualObs,
};
use crate::imu::{
    ImuBuffer, ImuWindowError, InertialConfig, InertialStats, RawImuSample, WindowExcitation,
    preintegration_is_finite, ts_sec, window_excitation,
};
use crate::stationary::{NotStill, StaticInitConfig, check_camera, check_imu, check_wheels};
use crate::wheel::{WheelOdomConfig, WheelSample, WheelState, WheelStats, WheelWindowError};
use cu_stereo_payloads::ImuSample;
use cu_stereo_payloads::{
    FEATURE_INLIER, FEATURE_OUTLIER, FEATURE_STEREO, FailureReason, FeaturePoint, InertialStage,
    MotionCheck, MotionVerdict as VerdictLabel,
};

/// Close-depth threshold in baselines: the near/far split is `35 * baseline`
/// metres (the stereo close/far split of Campos et al., ORB-SLAM3, IEEE T-RO
/// 2021). Points nearer than that are back-projected straight from stereo at
/// every keyframe; farther ones are left to multi-view triangulation.
pub const CLOSE_DEPTH_BASELINES: f64 = 35.0;

/// Minimum stereo-depth keypoints required to bootstrap a metric map from a
/// single frame (single-frame stereo initialization, as in Campos et al.,
/// ORB-SLAM3, IEEE T-RO 2021). This is the *only*
/// bootstrap gate: no parallax check, no two-view estimation, no post-BA health
/// check — metric scale comes from the baseline, not from motion.
pub const DEFAULT_MIN_BOOTSTRAP_STEREO_POINTS: usize = 50;

/// Relocalization-on-loss: match the lost frame's descriptors against candidate keyframes'
/// map-point associations and PnP back onto the EXISTING map, instead of surrendering the
/// map to a re-bootstrap. Thresholds are provisional — mirrored from the tracking-path
/// gates where analogous; revisit against a measured loss corpus. When that corpus exists,
/// candidate retrieval should move to `kornia_slam::loop_closure::place_recognition` (BoW, handles the
/// post-drift/kidnap cases distance ordering cannot) and the matcher to
/// `match_orb_descriptors` (ratio + orientation gates); the estimator itself is
/// upstream-shaped (`kornia_slam::tracking::pose_estimation`) once measured.
#[derive(Debug, Clone)]
pub struct RelocConfig {
    /// Candidate keyframes per attempt, ordered by camera-centre distance to the last known
    /// pose (the overwhelmingly common loss is a blip near the frontier).
    pub max_candidates: usize,
    /// Hamming gate for descriptor pairing (the projection matcher's lost-mode gate).
    pub max_hamming: u32,
    /// Minimum 2D-3D pairs before PnP is attempted for a candidate.
    pub min_matches: usize,
    /// Minimum PnP inliers to accept a relocalization — the acceptance gate applied here,
    /// after `solve_pnp` returns (`PnpConfig::min_inliers` is not read by that solver).
    pub min_inliers: usize,
    /// Prior-reprojection gate handed to PnP. The prior is the CANDIDATE KEYFRAME's own
    /// pose, which can be far from the truth — wide by design; the tight final threshold
    /// still does the accepting.
    pub prior_reproj_threshold_px: f64,
    /// Keep attempting relocalization this long before the loss falls through to the
    /// re-bootstrap (or the map drop, with [`TrackerConfig::reset_map_on_loss`]).
    pub patience: Time,
}

impl Default for RelocConfig {
    fn default() -> Self {
        Self {
            max_candidates: 12,
            max_hamming: 60,
            min_matches: 20,
            min_inliers: 30,
            prior_reproj_threshold_px: 200.0,
            patience: Time::new::<second>(3.0),
        }
    }
}

/// How many recent keyframes the growth and fuse passes consider as
/// "covisible" neighbours (recency is the covisibility proxy; there is no
/// covisibility graph in the frontend yet).
pub const DEFAULT_MAX_COVISIBLE_KEYFRAMES: usize = 10;

/// Sample every Nth keyframe for the `map growth` line; a line per keyframe is ~1 MB/h.
const KF_GROWTH_SAMPLE: usize = 25;

/// Minimum epipolar-consistent matches before the growth pass will triangulate.
const MIN_GROWTH_MATCHES: usize = 15;
/// 1-DOF chi-square gate at 95% for point-to-epipolar-line distance, scaled per
/// octave (the epipolar check of Campos et al., ORB-SLAM3, IEEE T-RO 2021).
const EPIPOLAR_CHI2: f64 = 3.84;
/// Fuse pass: projection search radius, in pixels.
const FUSE_SEARCH_RADIUS_PX: f32 = 7.0;
/// Fuse pass: descriptor distance ceiling.
const FUSE_MAX_HAMMING: u32 = 50;
/// How many trailing keyframes `kornia_slam::mapping::bundle_adjustment::run_local_ba`
/// leaves free (its private
/// `MAX_ACTIVE_KFS`). Mirrored here only to know when at least one pose is
/// *fixed*, i.e. when the BA gauge is anchored — see the call site. If upstream
/// changes its constant, BA here simply starts one keyframe early or late; it
/// can never make the solve unanchored, because the guard is `>`.
const BA_ACTIVE_KEYFRAMES: usize = 3;

/// Tuning for [`Tracker`].
///
/// Construct with [`TrackerConfig::new`], which needs the metric stereo
/// baseline: `bf = fx * baseline` sets metric depth, and `35 * baseline` sets
/// the near/far densification split, so a wrong baseline corrupts scale and the
/// split at once. There is no `Default` on purpose — a guessed baseline is a
/// silent metric error.
///
/// Not `Debug`/`Clone`: `kornia_imgproc::features::OrbDetector` implements
/// neither, and wrapping it would hide which detector is actually in use.
pub struct TrackerConfig {
    /// Metric stereo baseline. Take it from
    /// `StereoRectifier::baseline()`, *not* from depthai's
    /// `getBaselineDistance()` (which defaults to design values in centimetres).
    pub baseline: Length,
    /// ORB detector used for both eyes.
    ///
    /// `downscale` / `n_scales` are pinned to `kornia_slam::mapping::map`'s
    /// `ORB_SCALE_FACTOR` / `ORB_N_LEVELS` by [`TrackerConfig::new`]: the stereo
    /// coordinate mapping, the scale-invariance gates and the epipolar sigma all
    /// read those constants independently, and a mismatch is silent.
    pub orb: OrbDetector,
    /// Keyframe insertion heuristics.
    pub keyframe_policy: KeyframePolicy,
    /// Map-projection tracker thresholds (projection search, PnP).
    pub map_projection: MapProjectionConfig,
    /// Supplies the descriptor matcher + triangulation thresholds used by the
    /// keyframe growth pass. (Two-view *initialization* is never run: stereo
    /// bootstraps from one frame.)
    pub two_view_init: TwoViewInitConfig,
    /// Grace period before a sustained tracking failure resets to bootstrap.
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    /// Run local bundle adjustment after each keyframe insertion.
    pub enable_local_ba: bool,
    /// Close-depth threshold. `None` disables per-keyframe stereo
    /// densification. [`TrackerConfig::new`] sets `Some(35 * baseline)`.
    pub stereo_close_depth: Option<Length>,
    /// Bootstrap gate; see [`DEFAULT_MIN_BOOTSTRAP_STEREO_POINTS`].
    pub min_bootstrap_stereo_points: usize,
    /// Relocalize against the existing map while lost, before any re-bootstrap. `None`
    /// disables (pre-relocalization behaviour).
    pub relocalization: Option<RelocConfig>,
    /// On a sustained tracking loss, drop the WHOLE map (full [`reset`](Tracker::reset)) instead
    /// of re-bootstrapping into it. The append-to-map path leaves `current_keyframe_idx` at
    /// `None`, so the next local-map build returns every non-culled point and per-frame
    /// tracking cost jumps with total map size — measured on a Jetson Orin with an OAK-D at
    /// 640x400: 3 frames processed in 15 s after a loss, a spiral that never re-establishes.
    /// Dropping the map bounds cost and memory; a consumer re-anchors via
    /// [`world_generation`](Tracker::world_generation). Default false: accumulating across
    /// losses is the upstream behaviour, and what offline evaluation expects.
    pub reset_map_on_loss: bool,
    /// Drop the map once it holds this many keyframes. `None` is unbounded.
    ///
    /// kornia-slam never removes a keyframe — `cull_landmarks` only retires map points, and
    /// `keyframes`/`map_points` are `Vec`s that only push — so insertion cost rises without
    /// limit. Measured on a Jetson Orin with an OAK-D at 640x400: 62 ms at 25 keyframes, 260 ms
    /// at 137, 427 ms at 156, then a silent stall (one thread at 99.9 %, 757 MB RSS, zero poses
    /// after 2.5 h).
    /// `reset_map_on_loss` bounds this ONLY when tracking fails; tracking cleanly in good light
    /// never trips it, so the better the conditions the sooner VIO dies.
    pub max_keyframes: Option<usize>,
    /// Neighbour window for grow/fuse; see [`DEFAULT_MAX_COVISIBLE_KEYFRAMES`].
    pub max_covisible_keyframes: usize,
    /// Inertial front end. `None` — what [`TrackerConfig::new`] sets — is not a degraded mode
    /// but the ONLY mode until the camera-to-IMU extrinsic is measured; see the module docs.
    ///
    /// `Option` rather than a `bool` beside a defaulted `InertialConfig` on purpose: there is
    /// no `Default for InertialConfig`, and its only constructor demands the extrinsic and the
    /// noise densities, so "inertial on with a guessed extrinsic" is not a state this type can
    /// represent.
    pub inertial: Option<InertialConfig>,
    /// Wheel odometry. `None` — what [`TrackerConfig::new`] sets — leaves every path below as it
    /// was: no wheel factor, the constant-velocity prior, no published coast, and a map drop
    /// that restarts the world at identity. See [`crate::wheel`] for what `Some` changes.
    pub wheel_odom: Option<WheelOdomConfig>,
    /// Per-frame fused tracking ([`crate::fusion`]). `None` — what [`TrackerConfig::new`] sets —
    /// publishes PnP's pose, the IMU and the wheels seeding it only. It fuses whatever of
    /// [`Self::inertial`] and [`Self::wheel_odom`] is on, and with neither it changes nothing.
    pub per_frame_fusion: Option<FusionConfig>,
}

impl TrackerConfig {
    /// Builds a config for a rectified pair with the given metric baseline.
    pub fn new(baseline: Length) -> Self {
        // Upstream's `SlamConfig::default` tightens two triangulation
        // thresholds relative to kornia-3d's defaults; both feed the growth pass.
        let mut two_view_init = TwoViewInitConfig::default();
        two_view_init.triangulation_config.max_midpoint_gap = 0.25;
        two_view_init.triangulation_config.max_reprojection_error = 3.0;

        Self {
            baseline,
            orb: OrbDetector {
                n_keypoints: 1000,
                downscale: ORB_SCALE_FACTOR as f32,
                n_scales: ORB_N_LEVELS,
                ..OrbDetector::default()
            },
            keyframe_policy: KeyframePolicy::default(),
            // Upstream's `MapProjectionConfig::default()`, unmodified.
            //
            // Lowering `pnp.outlier_rounds` from 4 to 2 does not show a reproducible saving: one
            // A/B read 13.6 ms (est_ms median 83.8 -> 70.2, 500 frames each), but four repeats of
            // the rounds=2 config gave medians of 70.2, 90.7, 91.1 and 88.5 — a ~20 ms run-to-run
            // spread that is larger than the effect, and which puts the rounds=4 baseline (83.8)
            // BELOW the rounds=2 cluster. Matched settle windows minutes apart were not enough:
            // something drifts between runs (map state, scene, or the board's thermal state) that
            // dominates.
            //
            // Fewer refit rounds is still plausibly cheaper, but it costs outlier
            // re-classification on a tracker whose reliability is already the weak link, and an
            // unproven saving does not buy that.
            //
            // To settle it properly, `outlier_rounds` needs to be runtime-configurable so the two
            // can be A/B'd many times within one session without a rebuild, or the measurement
            // needs to control for map size rather than only for settle time.
            map_projection: MapProjectionConfig::default(),
            two_view_init,
            tracking_loss_recovery: TrackingLossRecoveryPolicy::default(),
            enable_local_ba: true,
            stereo_close_depth: Some(baseline * CLOSE_DEPTH_BASELINES),
            min_bootstrap_stereo_points: DEFAULT_MIN_BOOTSTRAP_STEREO_POINTS,
            relocalization: Some(RelocConfig::default()),
            reset_map_on_loss: false,
            max_keyframes: None,
            max_covisible_keyframes: DEFAULT_MAX_COVISIBLE_KEYFRAMES,
            inertial: None,
            wheel_odom: None,
            per_frame_fusion: None,
        }
    }
}

/// The outcome of one frame: a measured pose, or none.
///
/// Untracked is a controlled outcome of a healthy tracker, not a failure, so it is a variant here
/// rather than an error or an `Option` inside the `Result`: the `Err` side of
/// [`Tracker::process_stereo`] is reserved for malformed input and kornia errors.
#[derive(Debug, Clone)]
pub enum TrackStatus {
    /// The pose was measured: the frame was tracked, or a keyframe was accepted.
    Tracked(TrackedPose),
    /// The frame could not be placed. The tracker still coasts an extrapolated pose under the
    /// constant-velocity model, available from [`Tracker::last_pose`], but it is unverified and
    /// must not be published as a measurement.
    Untracked,
    /// Wheel odometry only: vision did not place the frame, and the pose is the last one
    /// propagated on the wheels (and the gyro, with the inertial path on). Also the frame a map
    /// drop re-anchored on, whose pose is the same propagation, and a frame whose vision the
    /// per-frame fusion rejected ([`crate::fusion`]), whose pose is the IMU and wheel prediction.
    Coasting(TrackedPose),
    /// Wheel odometry only: vision did not place the frame and no wheel reading covered it. The
    /// pose is the tracker's extrapolation, carried so a consumer can see the state, not use it.
    NotLocalized(TrackedPose),
}

/// A MEASURED pose for one input frame.
///
/// Only a measured frame produces one: a frame the tracker could not place is
/// [`TrackStatus::Untracked`], so there is no "untracked" state to check here as well.
#[derive(Debug, Clone)]
pub struct TrackedPose {
    /// Capture time of the frame this pose belongs to, exactly as handed to
    /// [`Tracker::process_stereo`].
    pub stamp: CuTime,
    /// **World-to-camera** pose. See the module docs before publishing it.
    pub pose_world_to_cam: Pose3d,
    /// Whether a keyframe was inserted for this frame. Per-frame cost is bimodal on it.
    pub keyframe: bool,
}

impl TrackedPose {
    /// Camera-in-world: the transform a viewer / `TransformStamped` wants.
    pub fn cam_in_world(&self) -> Pose3d {
        self.pose_world_to_cam.inverse()
    }
}

/// Snapshot of tracker health, for the periodic log. Diagnostics only — none of
/// these is an acceptance criterion (an odometry front end can publish 4x the true
/// translation at a perfect RMSE and inlier fraction).
#[derive(Debug, Clone, Copy)]
pub struct TrackerStats {
    /// Frames handed to the tracker so far.
    pub frames: u64,
    /// Keypoints detected in the most recent left image.
    pub keypoints: usize,
    /// Left keypoints with a valid stereo depth in the most recent frame.
    pub stereo_matches: usize,
    /// PnP inliers on the most recent frame (0 when tracking failed).
    pub inliers: usize,
    /// Non-culled map points. Culling is logical, so this can shrink while
    /// `Map::num_map_points()` only grows.
    pub map_points: usize,
    /// Keyframes in the map. Never culled, so this only grows — including
    /// across tracking losses, which do *not* clear the map on the stereo path.
    pub keyframes: usize,
    /// Status of the most recent frame; `None` before the first frame.
    pub status: Option<TrackingStatus>,
    /// Current pipeline mode.
    pub mode: SystemMode,
    /// Consecutive frames that failed to track (0 while healthy).
    pub consecutive_failures: u32,
}

impl Default for TrackerStats {
    fn default() -> Self {
        Self {
            frames: 0,
            keypoints: 0,
            stereo_matches: 0,
            inliers: 0,
            map_points: 0,
            keyframes: 0,
            status: None,
            mode: SystemMode::Bootstrap,
            consecutive_failures: 0,
        }
    }
}

/// Which stage the tracker is in.
///
/// Owned here since kornia-slam made its own `SystemMode` crate-private. It mirrors that enum
/// minus `ImuInit`: kornia-slam's `SlamSystem` parks in `ImuInit` and stops tracking until the
/// inertial initialization succeeds, whereas this tracker keeps tracking visually throughout and
/// runs the initializer as a side effect of keyframe insertion, so it never entered that mode
/// (the arm that handled it was an unreachable recovery path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMode {
    /// Waiting for a usable stereo frame to anchor on: before the first map, or after a tracking
    /// loss. After a loss the old map is still served; the re-anchor appends to it
    /// ([`TrackerConfig::reset_map_on_loss`] off, the default) or starts a fresh one (on).
    Bootstrap,
    /// Track against the existing map and insert keyframes when needed.
    Tracking,
}

/// Mutable tracker state carried across frames.
///
/// Owned here since kornia-slam made its own `SystemState` crate-private. The fields and
/// [`SystemState::reset`] are kornia-slam's (main `a9f8c52`, `system/state.rs`, unchanged since
/// `c94e2a8`), minus the monocular `bootstrap_frame`, which a single-frame stereo bootstrap never holds.
#[derive(Debug, Clone)]
struct SystemState {
    pose_world_to_cam: Pose3d,
    /// Constant-velocity motion model, `Pose3d::between(previous, current)`.
    velocity: Option<Pose3d>,
    /// Metric body velocity in the world frame (m/s); valid once `imu_initialized`, when it
    /// seeds each new keyframe (see `Tracker::seeded_keyframe`).
    velocity_world: Vec3F64,
    /// Timestamp of the previous processed frame. Written for parity with kornia-slam's state,
    /// where it bounds the per-frame preintegration window; nothing here reads it.
    #[allow(dead_code)]
    last_frame_timestamp_sec: f64,
    /// Whether visual-inertial initialization has been applied (selects inertial local BA).
    imu_initialized: bool,
    /// Timestamp (sec) at which inertial initialization was applied.
    #[cfg_attr(not(test), allow(dead_code))]
    imu_init_timestamp_sec: Option<f64>,
    /// Reference keyframe tracking runs against (ORB-SLAM3's `mpReferenceKF`).
    current_keyframe_idx: Option<usize>,
    /// Most recently inserted keyframe (ORB-SLAM3's `mpLastKeyFrame`); spaces keyframe
    /// insertion and anchors the next IMU edge.
    last_keyframe_idx: Option<usize>,
    /// Timestamp (sec) of the first frame in the current run of tracking failures, or `None`
    /// while tracking is healthy.
    lost_since_sec: Option<f64>,
    mode: SystemMode,
}

impl SystemState {
    fn new() -> Self {
        Self {
            pose_world_to_cam: Pose3d::IDENTITY,
            velocity: None,
            velocity_world: Vec3F64::ZERO,
            last_frame_timestamp_sec: 0.0,
            imu_initialized: false,
            imu_init_timestamp_sec: None,
            current_keyframe_idx: None,
            last_keyframe_idx: None,
            lost_since_sec: None,
            mode: SystemMode::Bootstrap,
        }
    }

    /// Back to bootstrap, KEEPING `pose_world_to_cam` (the re-bootstrap re-anchors there,
    /// appending to the kept map) and `last_frame_timestamp_sec`. Clears the metric IMU state:
    /// the map it applied to is about to be re-anchored, so inertial initialization has to run
    /// again.
    fn reset(&mut self) {
        self.mode = SystemMode::Bootstrap;
        self.current_keyframe_idx = None;
        self.last_keyframe_idx = None;
        self.velocity = None;
        self.lost_since_sec = None;
        self.imu_initialized = false;
        self.imu_init_timestamp_sec = None;
        self.velocity_world = Vec3F64::ZERO;
    }

    /// Resumes tracking from an applied inertial initialization: kornia-slam's
    /// `SystemState::adopt_inertial_initialization` for the `Some` case. The caller handles the
    /// `None` case, where kornia-slam would still mark the state initialized.
    fn adopt_inertial_initialization(&mut self, aligned: AlignedTrackingState) {
        self.velocity_world = aligned.velocity_world;
        self.pose_world_to_cam = aligned.pose_world_to_cam;
        self.velocity = None;
        self.imu_initialized = true;
    }
}

/// The tracker's inertial half: the sample ring, the initializer, and the estimates
/// `kornia_slam`'s inertial APIs keep OUTSIDE the map.
///
/// `Map` stores per-keyframe velocity and bias, but the *current* linearization bias, the
/// gravity direction and the staged initialization's progress live nowhere in the library —
/// `SlamSystem` owns them privately, and this tracker does not use `SlamSystem`. This struct is
/// that ownership.
struct InertialState {
    cfg: InertialConfig,
    buffer: ImuBuffer,
    initializer: ImuInitializer,
    /// Linearization point for the NEXT edge. Refreshed from the newest keyframe after every
    /// local BA: leaving it at the bootstrap value makes every edge linearize at a stale bias,
    /// which drives the inertial BA's 0.02 repropagation threshold to fire on every factor, every
    /// solve.
    bias: ImuBias,
    /// Gravity in the world frame. `(0, 0, -9.81)` until `apply_initialization` gravity-aligns
    /// the map, after which it is `(0, +9.81, 0)` — OpenCV Y-down, matching
    /// `ViBaParams::default().gravity`. Handing `run_local_inertial_ba` the pre-init value
    /// would rotate the inertial residual 90 degrees against the map it is constraining.
    gravity_world: Vec3F64,
    /// `Frame::idx` of the keyframe the initialization window starts at. The window never
    /// shrinks, which is why the retry below is throttled.
    start_kf_idx: Option<usize>,
    /// Timestamp of that keyframe: the start of the initialization window, and the clock the
    /// two refinement stages are scheduled on.
    window_start_sec: Option<f64>,
    last_init_attempt_sec: Option<f64>,
    /// When [`Tracker::log_init_window`] last spoke, and the verdict it reported. Its OWN clock,
    /// because `last_init_attempt_sec` is not assigned until `ready` first answers true and so
    /// throttles nothing in the case that actually runs long: a parked robot.
    last_window_log: Option<(f64, bool)>,
    /// Set when `enable_inertial_ba` is off and `try_initialize` has returned an accepted
    /// result for the current window that was then NOT applied. The dry run exists to answer
    /// one question — would this extrinsic pass the initializer's gates, and what gravity and
    /// bias would it produce — and one acceptance answers it. Without this latch the initial-solve
    /// branch would keep paying a 200-iteration LM over a never-shrinking window every
    /// `init_retry` for the rest of the session, because `state.imu_initialized` (the
    /// thing that normally ends that loop) is only ever set by `apply_initialization`.
    dry_run_accepted: bool,
    /// When the stationary initialization's refusal line last spoke; see [`should_log_throttled`].
    last_static_log: Option<(f64, bool)>,
    /// Set by a map drop that kept the world (wheels carried the pose, or a loss re-bootstrapped
    /// into the kept map) while the inertial path was initialized. The world frame did not
    /// change, so neither did gravity in it, and the next bootstrap adopts the alignment instead
    /// of re-initializing; see [`Tracker::carried_velocity`] and
    /// [`Tracker::adopt_carried_alignment`].
    ///
    /// No velocity is carried with it. The tracker's velocity is synced only at keyframes (~1.4 s
    /// apart on the robot), a loss adds its grace period, and a bootstrap that keeps failing
    /// (a dark room) can defer the adoption by minutes: a velocity from before the drop says
    /// nothing about the robot when the next map starts. The anchor's velocity is measured at
    /// the bootstrap instead, from the wheels, or the alignment is not adopted.
    carried_alignment: bool,
    /// Set when a stationary initialization was applied: the newest keyframe of its still
    /// window. While set the refinements wait. On parked data they cannot separate the
    /// accelerometer bias from tilt any better than the still window did, yet an accepted one
    /// rotates the world again and bumps the generation (every consumer wipes its trail), and
    /// each stage runs once, so both would be spent before the robot moved. Cleared, with the
    /// refinement window re-armed at that moment, once the camera has moved
    /// [`STATIC_REFINE_MOTION_M`] or turned [`STATIC_REFINE_MOTION_RAD`] from that keyframe.
    static_hold: Option<usize>,
    /// When the map last refused a stationary initialization: the still window is not searched
    /// again for `init_retry`, as the motion solve is throttled after its own refusals. Without
    /// it a refused alignment re-ran the search and the apply, and warned, on every keyframe.
    static_map_refused_at: Option<f64>,
    /// Test-only fault injection: empties the stationary result's velocities so the map refuses
    /// it (`MissingVelocities`), the one way to reach the map-refusal path from a real window.
    #[cfg(test)]
    refuse_static_alignment: bool,
    /// The stage of the last APPLIED initialization, for diagnostics; `None` until one, and
    /// after a reset (a carried alignment then reads as the motion solve's).
    applied_stage: Option<InitStage>,
    /// Counters, plus the staged initialization's progress (`first_refinement_done` / `second_refinement_done` mark the
    /// first and second refinement, live here and nowhere else, and are read by the ladder and
    /// published as-is).
    stats: InertialStats,
}

impl InertialState {
    fn new(cfg: InertialConfig) -> Self {
        Self {
            buffer: ImuBuffer::with_capacity(cfg.buffer_capacity),
            initializer: ImuInitializer::new(cfg.init.clone()),
            bias: cfg.initial_bias,
            // Mirrors `SlamSystem`'s pre-initialization value. It is never read before
            // `apply_initialization` overwrites it, but a zero vector here would be a silent
            // "no gravity" if that invariant ever slipped.
            gravity_world: Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE),
            start_kf_idx: None,
            window_start_sec: None,
            last_init_attempt_sec: None,
            last_window_log: None,
            dry_run_accepted: false,
            last_static_log: None,
            carried_alignment: false,
            static_hold: None,
            static_map_refused_at: None,
            applied_stage: None,
            #[cfg(test)]
            refuse_static_alignment: false,
            stats: InertialStats::default(),
            cfg,
        }
    }

    /// Back to the pre-bootstrap state, keeping the config and the lifetime counters.
    ///
    /// `keep_bias` carries the estimated IMU bias across the reset. Gyro and accelerometer
    /// bias are properties of the SENSOR, not of the map, so a map that was dropped for being
    /// too large says nothing about them — and re-estimating from
    /// [`InertialConfig::initial_bias`] makes the next initial solve land wide and the first
    /// refinement that corrects it rotate the whole world. Measured on a Jetson Orin with an
    /// OAK-D at 640x400: gravity_y 8.94 at the initial solve then 9.81 at the first refinement,
    /// i.e. one large world rotation per map cycle.
    ///
    /// `gravity_world` is NOT carried here: it is expressed in the world frame, and the bootstrap
    /// that follows a reset defines a new one. `Tracker::reset_inner` restores it, and sets
    /// [`Self::carried_alignment`], when the drop keeps the world.
    fn reset(&mut self, keep_bias: bool) {
        // The sample buffer is NOT cleared. Its samples are stamped readings of the sensor, not
        // of the map, and the task drains the IMU queue BEFORE the frame the next map anchors
        // on is processed, so the ring already holds that frame's ~60 ms of latency past its
        // stamp. Cleared, they were gone and the new map's first edge started that far before
        // the oldest sample left: refused as a gap after every drop (2026-10-09 loop test,
        // "IMU gap of 52-64 ms"), which left the next local BA window with no IMU tie to the
        // carried bias. The bootstrap's `arm_inertial_window` prunes everything before its own
        // stamp, and the drop marks are stamped, so nothing older can bound an interval.
        if !keep_bias {
            self.bias = self.cfg.initial_bias;
        }
        self.gravity_world = Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
        self.start_kf_idx = None;
        self.window_start_sec = None;
        self.last_init_attempt_sec = None;
        // Cleared, so the first window after a reset is described rather than throttled against
        // the window it replaced.
        self.last_window_log = None;
        self.dry_run_accepted = false;
        self.last_static_log = None;
        self.carried_alignment = false;
        self.static_hold = None;
        self.static_map_refused_at = None;
        self.applied_stage = None;
        self.stats.first_refinement_done = false;
        self.stats.second_refinement_done = false;
    }
}

/// Whether a throttled line should speak, given when it last did and what it then said.
///
/// A flipped `verdict` ALWAYS speaks: the transition is the one line worth keeping, and a plain
/// time throttle is exactly what would drop it. Otherwise once per `retry_sec`. Split out so the
/// throttle is testable without a map behind it.
///
/// Three callers. `log_init_window` passes `ready`, where the transition out of "never ready" is
/// the only line anyone waits for. The tracking-loss line passes `map_established`, where losing
/// an ESTABLISHED map and failing to bootstrap one are different faults a time throttle would
/// blend into one stream. The BA-refusal line passes a constant, so it is a plain time throttle.
fn should_log_throttled(
    last: Option<(f64, bool)>,
    now_sec: f64,
    verdict: bool,
    retry_sec: f64,
) -> bool {
    match last {
        None => true,
        Some((_, was)) if was != verdict => true,
        Some((then, _)) => now_sec - then >= retry_sec,
    }
}

/// The stages of the staged inertial initialization: the initial solve, then the refinements
/// scheduled at 5 s and 15 s into the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitStage {
    /// From a still window ([`crate::stationary`]) rather than the motion solve.
    Static,
    Initial,
    Refine1,
    Refine2,
}

impl InitStage {
    /// The stage's label in the structured log.
    fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Initial => "initial",
            Self::Refine1 => "refine_1",
            Self::Refine2 => "refine_2",
        }
    }
}

impl std::fmt::Display for InitStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A still window [`Tracker::still_window`] found.
#[derive(Debug, Clone, Copy)]
struct StillWindow {
    /// `Frame::idx` of its first keyframe.
    first_kf: usize,
    /// Seconds of contiguous IMU edges it spans.
    span_s: f64,
    stats: ImuWindowStats,
}

/// Why the stationary initialization did not run on a keyframe.
#[derive(Debug, Clone, Copy, thiserror::Error)]
enum StaticRefusal {
    #[error("not still: {0}")]
    NotStill(NotStill),
    #[error("refused by kornia-slam: {0}")]
    Init(StationaryInitError),
}

/// How often the tracking-loss line may repeat while nothing about the fault changes.
///
/// Not cosmetic. Measured on a Jetson Orin with an OAK-D at 640x400 over a 7 h run with a healthy
/// camera: 100,444 losses at 3.88/s, ~26 % of all frames. Unthrottled, that volume is enough to
/// fill an embedded disk. 5 s is ~1 line per 19 losses; `suppressed` carries the rest, so
/// throttling never hides the volume.
const LOSS_LOG_RETRY_SEC: f64 = 5.0;

/// Camera translation from the still window that ends a stationary initialization's hold on the
/// refinements, metres. Two orders over a parked keyframe's drift (0.2 mm), and enough motion for
/// the accelerometer bias to begin separating from tilt.
const STATIC_REFINE_MOTION_M: f64 = 0.2;

/// Camera rotation from the still window that ends the same hold, radians (10 deg).
const STATIC_REFINE_MOTION_RAD: f64 = 0.174_532_925_199_432_95;

/// Camera travel since the last keyframe under which a parked robot withholds keyframes, metres:
/// the stationary initialization's camera floor, ~50x a parked keyframe's drift (0.2 mm).
const PARKED_CAMERA_TRANSLATION_M: f64 = 0.01;

/// The rotation counterpart of [`PARKED_CAMERA_TRANSLATION_M`]: 0.5 deg, an order over stereo
/// attitude noise and well under a head re-tilt or the mast's ~7 deg pan play.
const PARKED_CAMERA_ROTATION_RAD: f64 = 0.008_726_646_259_971_648;

/// IMU ring headroom past a parked keyframe interval, seconds. The ring holds from the last
/// keyframe to the newest drained sample, which leads the frame by its lag plus the solve (140 ms
/// worst lag at 15 fps, p99 ~200 ms after a keyframe solve, multi-second outliers), and a device
/// running fast of its nominal rate fills it sooner.
const PARKED_IMU_RING_MARGIN_S: f64 = 2.0;

/// The tracker's IMU ring capacity: the configured one, grown to span a parked keyframe interval
/// plus [`PARKED_IMU_RING_MARGIN_S`]. A 10 s interval in the default 2048 (10.24 s at 200 Hz)
/// evicted the start of the parked edge whenever the IMU led the frame by ~0.25 s, and the edge
/// and the per-frame IMU prediction were refused as a leading gap.
fn imu_ring_capacity(inertial: &InertialConfig, wheel: Option<&WheelOdomConfig>) -> usize {
    let Some(interval) = wheel.and_then(|w| w.parked_keyframe_interval) else {
        return inertial.buffer_capacity;
    };
    let span_s = interval.get::<cu29::units::si::time::second>() + PARKED_IMU_RING_MARGIN_S;
    let need = (inertial.gates.nominal_rate.get::<hertz>() * span_s).ceil();
    // Validated finite and positive; the cast saturates on anything absurd.
    inertial.buffer_capacity.max(need as usize)
}

/// Wheel-odometry span the anchor velocity of a carried alignment is measured over: a few wheel
/// readings, short enough to be the velocity at the bootstrap and not an average over a stop.
const CARRY_VELOCITY_WINDOW_NS: u64 = 200_000_000;

/// Replace a world-to-camera pose's rotation while leaving the camera CENTRE where it is.
///
/// `Pose3d` is world-to-camera (`X_cam = R X_world + t`), so the centre is `C = -R^T t` and
/// writing `rotation` alone moves it by ~theta * |t| — metres once the robot is any distance
/// from the map origin, and worst exactly while turning, which is when a rotation prior is for.
/// Extracted so a test can exercise THIS, rather than re-deriving the same two lines beside it.
fn rotate_about_centre(pose: Pose3d, rotation: Mat3F64) -> Pose3d {
    let centre = pose.inverse().translation;
    Pose3d::from_rt(rotation, -(rotation * centre))
}

/// Stereo visual-odometry tracker.
///
/// Feed it rectified stereo pairs with [`Tracker::process_stereo`]; it
/// bootstraps a metric map from the first usable frame and then tracks against
/// it, inserting keyframes and running local BA as needed. See the module docs
/// for the pose convention and the input contract.
pub struct Tracker {
    camera: PinholeCamera,
    config: TrackerConfig,
    stereo_config: StereoMatchConfig,
    estimator: MapProjectionEstimator,
    map: Map,
    state: SystemState,
    /// Stamp of the previous frame, for the gyro prior's integration interval.
    last_frame_stamp_ns: Option<u64>,
    /// Frames the gyro prior could not be built for. Counted because the fallback — the
    /// constant-velocity model — is silent and plausible: a prior that never once applied
    /// looks exactly like one that applied perfectly.
    gyro_prior_unavailable: u64,
    /// Set when an insertion took the map past `max_keyframes`; acted on next frame.
    pending_map_reset: bool,
    /// Set when a loss re-bootstrap into the KEPT map could not anchor on the loss frame itself.
    /// The next frame that does anchor is then the same dead-reckoned coast, so it is reported
    /// `Skipped` as the immediate re-bootstrap is. Cleared by every map drop: a fresh world at
    /// identity is a measurement.
    coast_rebootstrap_pending: bool,
    /// Map writes (keyframe or landmark insertions, observation links) kornia-slam refused, over
    /// the tracker's lifetime — a map drop does not zero it. Each is logged where it happened. A
    /// refusal leaves nothing behind but a thinner map, so this count is the only trace of one.
    map_writes_refused: u64,
    /// Local BA results `Map::apply_ba_update` refused, over the tracker's lifetime. A refusal
    /// leaves the map as it was before the solve, which looks exactly like a BA that ran.
    ba_updates_refused: u64,
    /// When the BA-refusal line last spoke; see [`should_log_throttled`]. The lifetime count
    /// it prints carries the volume the throttle drops.
    last_ba_refusal_log: Option<(f64, bool)>,
    /// Monotonic count of frames received; stamped onto `Frame::idx`.
    next_frame_idx: usize,
    /// First timestamp seen; all internal times are seconds relative to it, so
    /// f64 keeps full nanosecond resolution regardless of the caller's epoch.
    epoch_ns: Option<u64>,
    last_keyframe_timestamp_sec: Option<f64>,
    /// The same stamp in the caller's nanosecond clock. Kept alongside the seconds value rather
    /// than reconstructed from it: the IMU interval is queried in ns and integrated in seconds,
    /// and round-tripping through the f64 seconds would move the interval boundary by hundreds
    /// of nanoseconds — enough to include or exclude a sample at the edge, which is exactly the
    /// kind of difference `from_measurements` absorbs without a word.
    last_keyframe_stamp_ns: Option<u64>,
    /// `None` unless [`TrackerConfig::inertial`] was supplied; see the module docs.
    inertial: Option<InertialState>,
    stats: TrackerStats,
    /// When the tracking-loss line last spoke, and what `map_established` was then. Drives
    /// [`should_log_throttled`].
    last_loss_log: Option<(f64, bool)>,
    /// Loss events swallowed by that throttle since it last spoke. Reported on the next line
    /// that does, so the throttle cannot turn 19 losses into the appearance of one.
    losses_suppressed: u64,
    /// Bumped every time the pose chain's frame of reference stops being trustworthy: a
    /// requested [`reset`](Self::reset), or an internal loss re-bootstrap (which re-anchors at
    /// a dead-reckoned coast, not a measurement). Consumers that map tracker poses into
    /// another frame must re-derive their transform when this changes.
    world_generation: u64,
    /// Live (not culled) map points, and the `(keyframe count, world generation)` they were
    /// counted at; see the refresh in `process_frame`.
    live_map_points: usize,
    live_map_points_key: (usize, u64),
    /// `None` unless [`TrackerConfig::wheel_odom`] was supplied.
    wheel: Option<WheelState>,
    /// Whether the current frame's prior came from the wheels. Reset at the top of every frame.
    frame_on_wheels: bool,
    /// Whether a world exists that a pose can stay continuous in: set by every bootstrap,
    /// cleared by a requested [`reset`](Self::reset). With wheels it is what lets a map drop
    /// carry the pose into the next map instead of restarting at identity.
    world_anchored: bool,
    /// Whether the current map was seeded at a carried pose rather than at identity. Its
    /// inertial alignment then pivots on the robot, not on the world origin; see
    /// [`Self::apply_inertial_init`].
    map_carried: bool,
    /// Each recent keyframe's TRACKED pose at insertion, before any BA moved it, with the map
    /// state it was tracked against: vision's own word for the motion check
    /// ([`crate::consistency`]). A BA-refined keyframe already leans on the gyro and the wheels
    /// it is being compared with.
    kf_tracked_poses: std::collections::VecDeque<(usize, TrackedIn, Pose3d)>,
    /// Whether local BA has solved the current map at least once. Until then the map is YOUNG:
    /// its keyframes were tracked against the bootstrap's points alone, and the first solve
    /// re-pivots them (on the parked robot its edges started 0.7-1.3 deg off the gyro,
    /// 2026-10-10), so a pose tracked before it and one tracked after are not in one frame: the
    /// interval between them is not checked. Cleared with every new map.
    map_settled: bool,
    /// Keyframe intervals `(from, to)` vision disagreed with the wheels and the gyro on. Local
    /// BA is skipped while one ends at a free keyframe: it would explain frozen vision with a
    /// gyro bias.
    vision_suspect: std::collections::VecDeque<(usize, usize)>,
    consistency: ConsistencyStats,
    /// When a motion-consistency line last spoke; see [`should_log_throttled`].
    last_consistency_log: Option<(f64, bool)>,
    /// A vision-suspect interval is in the local BA window: tracking is DEGRADED. Edge-logged.
    vision_degraded: bool,
    /// The gyro bias at the last interval all three sensors agreed on: what the motion check
    /// integrates the gyro at. The CURRENT bias may already hold part of what it is checking (a
    /// bias local BA absorbed under the limit, ~1.9 deg per keyframe interval at 0.05 rad/s),
    /// which would make the gyro side with frozen vision.
    consistent_gyro_bias: Option<Vec3F64>,
    /// Local BA updates refused for an implausible bias in a row; see
    /// [`BIAS_REFUSALS_BEFORE_DROP`].
    bias_refusals_in_a_row: u32,
    /// This frame's diagnostics, reset at the top of every frame; see [`Tracker::frame_report`].
    frame_report: FrameReport,
    /// Maps dropped (size bound, sustained loss, refused biases), lifetime.
    map_drops: u64,
    /// Maps discarded by ANY reset (requested, loss, size bound, refused biases), lifetime: each
    /// one means the next bootstrap starts a new map. Not `world_generation`, which also moves on
    /// an inertial alignment and a kept-map re-bootstrap, neither of which starts a map.
    map_generation: u64,
    /// Whether a map was anchored since construction or the last requested reset: what tells
    /// "initializing" from "recovering" on a frame that could not be placed.
    anchored_since_reset: bool,
    /// The last keyframe interval the wheels moved in, as the motion check saw it.
    last_motion: MotionCheck,
    /// The last frame's left keypoints, classified. Reused: cleared and refilled per frame.
    features: Vec<FeaturePoint>,
    /// Per-frame fusion counters; see [`crate::fusion`].
    fusion: FusionStats,
    /// The keyframe-to-frame preintegration the per-frame fusion extends frame by frame.
    fusion_preint: Option<IncrementalPreintegration>,
    /// The keyframe the per-frame fusion is suspended on: set when a rejection run expires or a
    /// relocalization moves the pose, since every frame predicted from that keyframe would carry
    /// the same stale prediction. Frame indices only grow, so the next keyframe ends it.
    fusion_suspended_on: Option<usize>,
    /// The last tracked frame's PnP pose and the pose it was given (fused, or the same):
    /// `(frame idx, vision, published)`. What the keyframe motion check records as vision's word.
    vision_pose: Option<(usize, Pose3d, Pose3d)>,
    /// The published pose's hold on the end of a rejection run; see [`PublishHold`].
    publish_hold: Option<PublishHold>,
    /// Set by the frame that ends a rejection run: where its coast would have gone. Turned into
    /// [`PublishHold`] when the frame's pose is published (`held_pose`).
    publish_hold_target: Option<Pose3d>,
    /// Each keyframe's pose covariance when it was inserted, keyed by `Frame::idx`: the world
    /// left perturbation of its `T_wc` ([`crate::covariance`]). The spanning tree a frame's
    /// covariance is chained from: a frame's is its base keyframe's plus its own relative one.
    /// Cleared with the map, purged of keyframes the map no longer holds (`record_covariance`).
    kf_cov: std::collections::HashMap<usize, Cov6>,
    /// This frame's covariance, same convention; `None` for a frame with no pose to publish. See
    /// [`Tracker::frame_covariance`].
    frame_cov: Option<Cov6>,
    /// What `tracking_step` measured this frame: the keyframe the frame's covariance chains from,
    /// and the frame's covariance relative to it in the camera tangent. Reset every frame.
    frame_rel: Option<(Option<usize>, Cov6)>,
    /// What a coast grows from: the last measured frame's stamp and covariance, re-based every
    /// [`COAST_REBASE_NS`] of coasting and at every anchor. Kept across a map drop that carries
    /// the world, so the next map's first keyframe starts from where the last one left the
    /// uncertainty (never from zero, which would read as perfectly localised); cleared whenever
    /// the world restarts at identity.
    coast_origin: Option<CoastOrigin>,
    /// Coasted frames whose interval since the coast origin the wheels did not measure (a gap
    /// or a discontinuity in the readings, readings pruned under an old origin, a factor that
    /// would not invert); each grew its covariance by [`CoastOrigin::unmeasured`]. Lifetime.
    coast_unmeasured: u64,
}

/// How long a coast's covariance is integrated from one origin before it re-bases on its own
/// value. The wheel interval it integrates must stay in the readings ring (it is held there,
/// `prune`), and a minutes-long blind coast would otherwise pin minutes of readings. Each re-base
/// restarts the interval's floors and its systematic heading term, which a 1 s chunk barely
/// feels next to the per-metre growth it adds; the mount term does not restart, it is carried
/// as the total motion since the measured origin ([`CoastOrigin`]).
const COAST_REBASE_NS: u64 = 1_000_000_000;

/// How fast a coast's covariance grows over an interval the wheels did not measure, 1 sigma per
/// second of it, metres and radians: a base driven blind at a brisk walk, turning a radian a
/// second. A worst case chosen above anything the robot is driven at, not a measured figure;
/// what matters is that the growth is never lost while the pose keeps moving.
const COAST_UNMEASURED_M_PER_S: f64 = 0.5;
/// See [`COAST_UNMEASURED_M_PER_S`].
const COAST_UNMEASURED_RAD_PER_S: f64 = 1.0;
/// Unmeasured coast frames ([`Tracker::coast_unmeasured`]): the first is logged, then one in
/// this many, ~1 min apart at the head's 15 Hz while it persists.
const COAST_UNMEASURED_WARN_EVERY: u64 = 900;

/// What a coast grows from ([`Tracker::coast_covariance`]). The wheels' measurement noise is
/// independent from one interval to the next and accumulates chunk by chunk; the camera-to-base
/// mount term is not: a mount error is systematic, so its effect on the residual depends only on
/// the TOTAL base motion since the last measured origin, `(I - Ad(dT^-1)) Sigma_e (...)^T` of that
/// one `dT` ([`crate::fusion::extrinsic_covariance`]). Summed per chunk it would depend on where
/// the re-bases fell rather than on the motion: a long turn's term (which grows with the whole
/// angle) came out as a root sum of small ones, and a turn driven back never cancelled.
#[derive(Debug, Clone, Copy)]
struct CoastOrigin {
    /// Where the current chunk is integrated from.
    stamp_ns: u64,
    /// The covariance at `stamp_ns`, as published: `accumulated` plus the mount term of
    /// `since_measured`. World left perturbation, like every covariance here.
    cov: Cov6,
    /// The last measured (or anchored) covariance plus every chunk's wheel noise since, moved
    /// into the world at its own end; no mount term.
    accumulated: Cov6,
    /// Base motion from the last measured origin to `stamp_ns`: the chunks' deltas composed.
    since_measured: Pose3d,
}

impl CoastOrigin {
    /// A measured origin, or one taken as such (an anchor, an unmeasured interval): nothing of
    /// its covariance is left to recompute.
    fn measured(stamp_ns: u64, cov: Cov6) -> Self {
        Self {
            stamp_ns,
            cov,
            accumulated: cov,
            since_measured: Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, 0.0)),
        }
    }

    /// Advanced to `stamp_ns` by a chunk the wheels measured: `noise` the measurement's
    /// covariance and `delta` its base motion (a right perturbation of the END base pose),
    /// `t_wb` that end pose in the world, `extrinsic` the mount's `[metres, radians]` sigma.
    fn advanced(
        &self,
        stamp_ns: u64,
        noise: &Cov6,
        delta: &Pose3d,
        t_wb: &Pose3d,
        extrinsic: [f64; 2],
    ) -> Self {
        let since_measured = self.since_measured.compose(delta);
        let ad = covariance::pose_adjoint(t_wb);
        let accumulated = covariance::add(&self.accumulated, &covariance::congruence(&ad, noise));
        let mount = crate::fusion::extrinsic_covariance(&since_measured, extrinsic);
        Self {
            stamp_ns,
            cov: covariance::add(&accumulated, &covariance::congruence(&ad, &mount)),
            accumulated,
            since_measured,
        }
    }

    /// Advanced to `stamp_ns` over an interval nothing measured: the covariance so far (its
    /// mount term included, which can no longer be recomputed from a total motion nobody saw)
    /// plus [`COAST_UNMEASURED_M_PER_S`] / [`COAST_UNMEASURED_RAD_PER_S`] for the time elapsed,
    /// isotropic in the world. Becomes a measured origin: the next chunk restarts the mount
    /// term from here. Linear in the interval's own length; consecutive unmeasured re-bases add
    /// their variances, so a blind run of many chunks grows as their root sum.
    fn unmeasured(&self, stamp_ns: u64) -> Self {
        let dt = stamp_ns.saturating_sub(self.stamp_ns) as f64 * 1e-9;
        let (vm, vr) = (
            (COAST_UNMEASURED_M_PER_S * dt).powi(2),
            (COAST_UNMEASURED_RAD_PER_S * dt).powi(2),
        );
        let mut cov = self.cov;
        for k in 0..6 {
            cov[k * 7] += if k < 3 { vm } else { vr };
        }
        Self::measured(stamp_ns, cov)
    }
}

/// How fast the published pose closes on the tracker's after a rejection run, metres and radians
/// per second of frame time: 2.5 mm and 0.24 deg a frame at the head's 12 Hz. A run that ended
/// 60 mm off (2026-10-10, the live turns) closes in 2 s instead of in one 67 ms frame.
pub const PUBLISH_CLOSE_M_PER_S: f64 = 0.03;
/// See [`PUBLISH_CLOSE_M_PER_S`].
pub const PUBLISH_CLOSE_RAD_PER_S: f64 = 0.05;

/// A rejection run publishes the motion sensors' coast; the frame that ends it (vision accepted
/// again, or the run expiring) puts the tracker back on vision, which may be centimetres from
/// where the coast drifted. Published as is, that difference is one frame's jump. Instead the
/// published pose is `offset * T_cw`, the offset (camera frame, a left perturbation, so an
/// inertial re-alignment of the world does not touch it) set so the end frame publishes exactly
/// where the coast would have, then shrunk at [`PUBLISH_CLOSE_M_PER_S`] /
/// [`PUBLISH_CLOSE_RAD_PER_S`]. The map, the keyframes and the next frame's prior all use the
/// tracker's pose; only the output is held. Dropped with the map (a new map generation) and on
/// a relocalization, both of which are jumps a consumer is told about.
///
/// Only the END of a run is held: the frame that starts one publishes the prediction as is,
/// since holding it would publish the very vision the gate just refused.
#[derive(Debug, Clone, Copy)]
struct PublishHold {
    offset: Pose3d,
    at_sec: f64,
    map_generation: u64,
}

/// Why the current frame was or was not measured, beyond its [`TrackStatus`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameReport {
    /// The FIRST cause of the frame not being measured; `None` on a measured frame.
    pub failure: FailureReason,
    /// Placed by relocalization against the existing map.
    pub relocalized: bool,
    /// A map was anchored on this frame at a dead-reckoned pose (a loss re-bootstrap).
    pub reanchored: bool,
    /// Per-frame fusion rejected this frame's vision: the pose is the IMU and wheel prediction
    /// from the last keyframe, and the frame left the map untouched.
    pub vision_rejected: bool,
    /// Per-frame fusion ran on this frame (it had a motion factor to fuse with).
    pub fused: bool,
}

/// Consecutive local BA updates refused for an implausible bias (`BiasLimit`) before the map is
/// dropped. One refusal leaves the map as it was; a run of them means the window's keyframes
/// hold a conflict (or a bias) the solve keeps resolving the same impossible way, and without a
/// drop BA stays off until `max_keyframes` (75-140 s on the robot). The drop keeps the plausible
/// bias and re-initializes.
const BIAS_REFUSALS_BEFORE_DROP: u32 = 3;

/// Whether two poses differ in any bit.
fn pose_differs(a: &Pose3d, b: &Pose3d) -> bool {
    a.rotation != b.rotation || a.translation != b.translation
}

/// Whether the IMU's position rows may enter the per-frame fusion at this inertial stage. They
/// predict from the gravity direction, which a motion initialization only gets to within ~1 deg
/// at its second refinement (23 deg at the initial solve on EuRoC V101); the rows' sigma has no
/// gravity term, so before that they would reject every frame and starve the keyframes the
/// refinements need. A still window measures gravity directly.
fn imu_position_trusted(stage: InertialStage) -> bool {
    matches!(stage, InertialStage::Static | InertialStage::Refine2)
}

/// The wheels' rotation covariance (the `[roll, pitch, yaw]` block of a 6x6 row-major base-frame
/// covariance) in the camera at the interval's END, `t_cb_to`: the wheel factor's error is a
/// right-side perturbation there, as is the gyro's it is voted against.
fn wheel_rotation_cov_in_camera(cov6: &[f64], t_cb_to: &Pose3d) -> Mat3F64 {
    let c = |r: usize, k: usize| cov6[(3 + r) * 6 + 3 + k];
    let cov_base = Mat3F64::from_cols_array(&[
        c(0, 0),
        c(1, 0),
        c(2, 0),
        c(0, 1),
        c(1, 1),
        c(2, 1),
        c(0, 2),
        c(1, 2),
        c(2, 2),
    ]);
    let r_cb = t_cb_to.rotation;
    r_cb * cov_base * r_cb.transpose()
}

/// Keyframe tracked poses kept for the motion check: the pending wheel intervals (8) plus the
/// factors that can still act (16), with slack.
const KF_TRACKED_POSES: usize = 32;

/// The map state a keyframe's tracked pose was measured against; two poses are comparable only
/// when they were measured against the same one, in the current world generation. See
/// `Tracker::interval_sensors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrackedIn {
    world_generation: u64,
    map_settled: bool,
}

impl Tracker {
    /// Creates a tracker for a **rectified, zero-distortion** camera.
    ///
    /// The camera and baseline are not validated here (the signature has no
    /// `Result`); they are checked on every `process_*` call, which returns
    /// [`VioError::DistortedCamera`] / [`VioError::InvalidBaseline`]. A bad
    /// configuration is also logged at `error` level right here so it shows up
    /// at startup rather than only in a per-frame error.
    pub fn new(camera: PinholeCamera, config: TrackerConfig) -> Self {
        let baseline_m = config.baseline.get::<meter>();
        let close_depth_m = config.stereo_close_depth.map(|d| d.get::<meter>());
        let stereo_config = StereoMatchConfig::new(
            baseline_m as f32,
            camera.fx as f32,
            ORB_SCALE_FACTOR as f32,
            ORB_N_LEVELS,
        );

        // Startup log mirrors upstream's: a wrong baseline is invisible in the
        // output but corrupts metric scale and the near/far split at once.
        info!(
            "cu-kornia-vio stereo tracker configured: fx={} fy={} cx={} cy={} baseline_m={} bf={} close_depth_m={}",
            camera.fx, camera.fy, camera.cx, camera.cy, baseline_m, stereo_config.bf, close_depth_m
        );
        if let Err(e) = validate_camera(&camera, baseline_m) {
            error!(
                "cu-kornia-vio tracker configuration is invalid: {}",
                e.to_string()
            );
        }

        let inertial = config.inertial.clone().map(|mut c| {
            c.buffer_capacity = imu_ring_capacity(&c, config.wheel_odom.as_ref());
            InertialState::new(c)
        });
        if let Some(inert) = &inertial {
            // Loud at startup, because from here on the only externally visible difference is
            // that the world gets rotated once, silently, when initialization succeeds.
            warning!(
                "cu-kornia-vio INERTIAL path enabled: poses are only as good as the configured \
                 imu_t_bc, which nothing downstream can validate (gyro_noise={} accel_noise={} \
                 gyro_bias_noise={} accel_bias_noise={} rate_hz={} inertial_ba={})",
                inert.cfg.calib.gyro_noise,
                inert.cfg.calib.accel_noise,
                inert.cfg.calib.gyro_bias_noise,
                inert.cfg.calib.accel_bias_noise,
                inert.cfg.gates.nominal_rate.get::<hertz>(),
                inert.cfg.enable_inertial_ba
            );
        }

        let wheel = config.wheel_odom.clone().map(WheelState::new);
        if let Some(w) = &wheel {
            warning!(
                "cu-kornia-vio WHEEL ODOMETRY enabled: local BA, the tracking prior and the \
                 pose through a vision loss now lean on the wheels and the configured mount \
                 (translation_noise={} rotation_noise={} z_sigma={} tilt_sigma={})",
                w.cfg.translation_noise,
                w.cfg.rotation_noise,
                w.cfg.z_sigma,
                w.cfg.tilt_sigma
            );
        }

        Self {
            camera,
            stereo_config,
            estimator: MapProjectionEstimator::new(config.map_projection.clone()),
            config,
            map: Map::new(),
            state: SystemState::new(),
            last_frame_stamp_ns: None,
            gyro_prior_unavailable: 0,
            pending_map_reset: false,
            coast_rebootstrap_pending: false,
            map_writes_refused: 0,
            ba_updates_refused: 0,
            last_ba_refusal_log: None,
            next_frame_idx: 0,
            epoch_ns: None,
            last_keyframe_timestamp_sec: None,
            last_keyframe_stamp_ns: None,
            inertial,
            stats: TrackerStats::default(),
            last_loss_log: None,
            losses_suppressed: 0,
            world_generation: 0,
            live_map_points: 0,
            live_map_points_key: (0, 0),
            wheel,
            frame_on_wheels: false,
            world_anchored: false,
            map_carried: false,
            kf_tracked_poses: std::collections::VecDeque::new(),
            map_settled: false,
            vision_suspect: std::collections::VecDeque::new(),
            consistency: ConsistencyStats::default(),
            last_consistency_log: None,
            vision_degraded: false,
            consistent_gyro_bias: None,
            bias_refusals_in_a_row: 0,
            frame_report: FrameReport::default(),
            map_drops: 0,
            map_generation: 0,
            anchored_since_reset: false,
            last_motion: MotionCheck::default(),
            features: Vec::new(),
            fusion: FusionStats {
                last_chi2: f64::NAN,
                ..FusionStats::default()
            },
            fusion_preint: None,
            fusion_suspended_on: None,
            vision_pose: None,
            publish_hold: None,
            publish_hold_target: None,
            kf_cov: std::collections::HashMap::new(),
            frame_cov: None,
            frame_rel: None,
            coast_origin: None,
            coast_unmeasured: 0,
        }
    }

    /// Feeds one **rectified** stereo pair.
    ///
    /// Returns [`TrackStatus::Tracked`] when the pose is *measured* (tracked, or a
    /// keyframe was accepted) and [`TrackStatus::Untracked`] when the frame could not be
    /// tracked. On `Untracked` the tracker still advances an extrapolated pose under
    /// the constant-velocity model — available from [`Tracker::last_pose`] — but
    /// it is unverified and must not be published as a measurement.
    ///
    /// The frame on which a tracking loss re-bootstraps is `Untracked` too, even
    /// though it *does* create a keyframe: the new map is anchored at the
    /// dead-reckoned pose, so publishing it would be publishing a coast as a
    /// measurement. That holds when the re-bootstrap lands on a later frame
    /// because the loss frame itself was unusable.
    ///
    /// `stamp` is the pair's capture time, on the same clock as the IMU samples. It drives the
    /// tracking-loss grace period and the inertial windows, and is returned unmodified on
    /// [`TrackedPose::stamp`].
    pub fn process_stereo(
        &mut self,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
        stamp: CuTime,
    ) -> Result<TrackStatus, VioError> {
        validate_camera(&self.camera, self.config.baseline.get::<meter>())?;
        if left.width() != right.width() || left.height() != right.height() {
            return Err(VioError::StereoSizeMismatch {
                left_w: left.width(),
                left_h: left.height(),
                right_w: right.width(),
                right_h: right.height(),
            });
        }

        let t0 = std::time::Instant::now();
        let features = self.config.orb.detect_and_extract_u8(left)?;
        let right_features = self.config.orb.detect_and_extract_u8(right)?;
        let orb_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Both pyramids come from the same builder, so the SAD correlation
        // compares like with like. NOTE: this is a *single-resize* pyramid
        // (every level resamples the full-res image), which is what
        // `kornia_slam::stereo` assumes when it maps coordinates by
        // `u * 1.2^-octave`. `OrbDetector` builds a *successive*-downscale
        // pyramid internally, whose true scale at level o drifts from 1.2^-o.
        // Do not swap in a successive pyramid here without re-deriving that
        // mapping — and the mapping lives in read-only kornia-slam.
        let left_pyramid = build_u8_pyramid(left)?;
        let right_pyramid = build_u8_pyramid(right)?;
        let pyr_ms = t0.elapsed().as_secs_f64() * 1e3 - orb_ms;

        let stereo = compute_stereo_matches(
            &left_pyramid,
            &right_pyramid,
            &features,
            &right_features,
            &self.stereo_config,
        );
        let stereo_ms = t0.elapsed().as_secs_f64() * 1e3 - orb_ms - pyr_ms;

        let keypoint_colors = sample_keypoint_colors(left, &features.keypoints_xy);
        let image_size = ImageSize {
            width: left.width(),
            height: left.height(),
        };

        let frame = Frame {
            idx: 0, // replaced by `process_frame`
            features,
            pose_world_to_cam: Pose3d::IDENTITY, // replaced by `process_frame`
            image_size,
            keypoint_colors,
            u_right: stereo.u_right,
            depth: stereo.depth,
            keypoints_undist: Vec::new(), // filled by `ensure_undistorted`
        };

        let result = self.process_frame(frame, stamp);
        // Per-stage split at debug level: the frontend (ORB + pyramids + stereo) is fixed
        // cost per frame, while `track_ms` (colour sampling + Frame build + process_frame,
        // dominated by process_frame) scales with the map — which one saturates a live loop
        // decides whether the fix is a feature budget or a map bound. The four fields sum
        // to the whole call.
        let track_ms = t0.elapsed().as_secs_f64() * 1e3 - orb_ms - pyr_ms - stereo_ms;
        debug!(
            "process_stereo stage split: orb_ms={} pyr_ms={} stereo_ms={} track_ms={}",
            orb_ms, pyr_ms, stereo_ms, track_ms
        );
        result
    }

    /// Feeds a frame whose features and per-keypoint stereo depths were built
    /// elsewhere — the seam the synthetic tests drive.
    ///
    /// `frame.idx` and `frame.pose_world_to_cam` are **ignored and overwritten**
    /// (see the module docs on frame indices). Every per-keypoint array must be
    /// index-parallel with `features.keypoints_xy`, with `-1.0` sentinels in
    /// `u_right` / `depth` for keypoints without a stereo match; a fully empty
    /// `depth` means "monocular", which this stereo-only tracker can never
    /// bootstrap from and reports as [`TrackingStatus::Skipped`].
    pub fn process_frame(
        &mut self,
        mut frame: Frame,
        stamp: CuTime,
    ) -> Result<TrackStatus, VioError> {
        validate_camera(&self.camera, self.config.baseline.get::<meter>())?;
        validate_frame_arrays(&frame)?;
        let stamp_ns = stamp.as_nanos();

        self.frame_report = FrameReport::default();
        self.frame_rel = None;
        self.frame_cov = None;
        self.apply_pending_map_reset();

        frame.idx = self.next_frame_idx;
        self.next_frame_idx += 1;

        // Relative seconds: an absolute nanosecond epoch (~1.7e18) would leave
        // f64 with ~400 ns of resolution. Only differences matter here.
        let epoch = *self.epoch_ns.get_or_insert(stamp_ns);
        let ts_sec = stamp_ns.saturating_sub(epoch) as f64 * 1e-9;

        // Fill the per-frame undistortion cache once; tracking, BA gathering,
        // growth and fuse all read from it. Must happen before anything else.
        frame.ensure_undistorted(&self.camera);

        self.stats.frames += 1;
        self.stats.keypoints = frame.features.keypoints_xy.len();
        self.stats.stereo_matches = frame.depth.iter().filter(|&&d| d > 0.0).count();
        self.stats.inliers = 0;
        self.features.clear();
        self.features.extend(
            frame
                .features
                .keypoints_xy
                .iter()
                .enumerate()
                .map(|(i, &[x, y])| FeaturePoint {
                    x,
                    y,
                    class: if frame.depth.get(i).is_some_and(|&d| d > 0.0) {
                        FEATURE_STEREO
                    } else {
                        0
                    },
                }),
        );

        // No `ImuInit` arm: see [`SystemMode`] for why this tracker never parks there.
        let frame_idx = frame.idx;
        self.frame_on_wheels = false;
        // Set when this frame anchors a map at a propagated pose rather than a measured one.
        let mut coast_anchor = false;
        let status = match self.state.mode {
            SystemMode::Bootstrap => {
                // With wheels, a world that already exists is CONTINUED: the pose a map drop
                // carried over (or a loss left behind) keeps moving on the wheels while this
                // waits for a frame to anchor on, and the new map is built where the robot is
                // now, not where it was when vision went.
                let seeded = self.wheel.is_some() && self.world_anchored;
                if seeded {
                    self.state.pose_world_to_cam = self.predict_pose(stamp_ns).0;
                }
                let status = self.bootstrap_stereo(frame, ts_sec, stamp_ns);
                coast_anchor = seeded && status == TrackingStatus::KeyframeAccepted;
                if status == TrackingStatus::KeyframeAccepted
                    && std::mem::take(&mut self.coast_rebootstrap_pending)
                {
                    // The deferred half of the loss path's re-bootstrap; see the field.
                    self.frame_report.reanchored = true;
                    self.note_failure(FailureReason::DeadReckonedAnchor);
                    warning!(
                        "re-bootstrapped at the extrapolated pose on frame {}; reporting Skipped \
                         because that pose is dead-reckoned, not measured",
                        frame_idx
                    );
                    TrackingStatus::Skipped
                } else {
                    status
                }
            }
            SystemMode::Tracking => self.tracking_step(frame, ts_sec, stamp_ns),
        };

        // EVERY path, not just `tracking_step`: a frame handled by `bootstrap_stereo` that left
        // this stale would make the first tracking frame afterwards integrate the gyro across
        // the whole bootstrap gap — seconds of rotation applied as if it were one frame.
        self.last_frame_stamp_ns = Some(stamp_ns);

        self.stats.status = Some(status);
        self.stats.mode = self.state.mode;
        // `map_points` and `keyframes` are deliberately NOT filled in here.
        // `num_active_map_points()` is an O(every map point ever created) scan
        // and this runs on every frame, for two numbers the health log reads
        // once every 30 — an unbounded per-frame cost for a log field. They are
        // computed on demand in [`Tracker::stats`] instead.
        //
        // The live-point count a consumer publishes per frame is cached instead. Every point this
        // tracker creates or culls does so inside a keyframe insertion (bootstrap included), and
        // every map drop bumps the world generation, so the scan runs only when one of those two
        // moved: once per keyframe, not once per frame.
        let key = (self.map.keyframes().len(), self.world_generation);
        if key != self.live_map_points_key {
            self.live_map_points = self.map.num_active_map_points();
            self.live_map_points_key = key;
        }

        // A still window reaches back past the last keyframe: up to its own length, plus the one
        // edge that crosses its start. Twice the window covers that for edges up to a window long.
        let still_window_start = self
            .inertial
            .as_ref()
            .filter(|_| !self.state.imu_initialized)
            .and_then(|i| i.cfg.static_init)
            .zip(self.last_keyframe_stamp_ns)
            .map(|(c, t)| t.saturating_sub((2.0 * c.window.get::<second>() * 1e9) as u64));
        // A carried alignment measures its anchor velocity over the wheel readings that END at
        // the next bootstrap, so they are kept from that window's length before this frame. Kept
        // always, not only while a carry is pending: the drop that arms one can land on the very
        // frame that bootstraps, before any prune sees the flag. A few readings.
        let carry_velocity_start = self
            .last_frame_stamp_ns
            .map(|t| t.saturating_sub(CARRY_VELOCITY_WINDOW_NS));
        // The coast covariance integrates the wheels from its origin, which is re-based at least
        // every `COAST_REBASE_NS` while coasting; an older one (the wheels stopped covering) is
        // not worth holding readings for.
        let coast_start = self
            .coast_origin
            .map(|o| o.stamp_ns)
            .filter(|&t| stamp_ns.saturating_sub(t) <= 2 * COAST_REBASE_NS);
        if let Some(w) = self.wheel.as_mut() {
            w.prune([
                self.last_keyframe_stamp_ns,
                self.last_frame_stamp_ns,
                still_window_start,
                carry_velocity_start,
                coast_start,
            ]);
            if let Some((offset_ns, frames)) = w.check_coverage(stamp_ns) {
                warning!(
                    "wheel odometry: frame {} lies outside the readings for {} frames; frame -                      newest reading = {} ms. The stream is silent or its stamps are on another                      clock than the frames'; the wheels are not being used",
                    frame_idx,
                    frames,
                    (offset_ns / 1_000_000) as i64
                );
            }
        }

        let pose = TrackedPose {
            stamp,
            pose_world_to_cam: self.held_pose(ts_sec),
            keyframe: status == TrackingStatus::KeyframeAccepted,
        };
        self.record_covariance(status, frame_idx, stamp_ns);
        match status {
            TrackingStatus::Skipped => {
                self.stats.consecutive_failures = self.stats.consecutive_failures.saturating_add(1);
                // Without wheels, or before any world exists, an unplaced frame says nothing.
                Ok(
                    match (
                        self.wheel.is_some() && self.world_anchored,
                        self.frame_on_wheels,
                    ) {
                        (false, _) => TrackStatus::Untracked,
                        (true, true) => TrackStatus::Coasting(TrackedPose {
                            keyframe: false,
                            ..pose
                        }),
                        (true, false) => TrackStatus::NotLocalized(TrackedPose {
                            keyframe: false,
                            ..pose
                        }),
                    },
                )
            }
            TrackingStatus::Tracked | TrackingStatus::KeyframeAccepted => {
                // A frame whose vision the fusion rejected is the IMU and wheel prediction: a
                // coast, not a measurement, so it neither clears the failure count nor leaves
                // as Tracked.
                if self.frame_report.vision_rejected {
                    self.stats.consecutive_failures =
                        self.stats.consecutive_failures.saturating_add(1);
                    return Ok(TrackStatus::Coasting(TrackedPose {
                        keyframe: false,
                        ..pose
                    }));
                }
                self.stats.consecutive_failures = 0;
                Ok(if coast_anchor {
                    TrackStatus::Coasting(pose)
                } else {
                    TrackStatus::Tracked(pose)
                })
            }
        }
    }

    /// Health snapshot; see [`TrackerStats`].
    ///
    /// `map_points` / `keyframes` are counted here rather than on every frame:
    /// both are linear scans over the whole map, so paying them per frame would
    /// make frame time grow with trajectory length for the sake of a log line.
    /// Call this at the log cadence, not per frame.
    pub fn stats(&self) -> TrackerStats {
        TrackerStats {
            map_points: self.map.num_active_map_points(),
            keyframes: self.map.keyframes().len(),
            ..self.stats
        }
    }

    /// The last frame's counters WITHOUT the two whole-map scans [`stats`](Self::stats) runs:
    /// `map_points` and `keyframes` read zero here. Cheap enough to call per frame.
    pub fn frame_stats(&self) -> TrackerStats {
        self.stats
    }

    /// Rotation the gyro says the CAMERA turned through over `[t0_ns, t1_ns]`.
    ///
    /// Returns `R_c0_c1`: `R_world_to_cam` at t1 is `delta.transpose() * R_world_to_cam(t0)`.
    /// Derivation, because the transposes are where this goes silently wrong —
    /// `dR_wb/dt = R_wb [w]x` gives `R_wb1 = R_wb0 dR_b`, and with `X_body = R_BC X_cam`
    /// (`imu_t_bc`), `dR_c = R_BC^T dR_b R_BC`.
    ///
    /// Integrated first-order per sample and re-orthonormalised at the end: at 200 Hz one step
    /// is ~1e-4 rad, far below where the small-angle error matters over a 66 ms frame interval.
    fn gyro_camera_delta(&mut self, t0_ns: u64, t1_ns: u64) -> Option<Mat3F64> {
        let inert = self.inertial.as_ref()?;
        let bias = inert.bias.gyro;
        let r_bc = inert.cfg.imu_t_bc.rotation;
        // Straight off the borrowed window, holding the previous sample: no per-frame copy of
        // the interval.
        let integrated = inert
            .buffer
            .window(t0_ns, t1_ns, &inert.cfg.gates)
            .ok()
            .and_then(|mut window| {
                let mut prev = *window.next()?;
                let mut d = Mat3F64::IDENTITY;
                let mut pairs = 0usize;
                for &s in window {
                    pairs += 1;
                    let dt = (s.stamp_ns.saturating_sub(prev.stamp_ns)) as f64 * 1e-9;
                    if dt > 0.0 {
                        let w = Vec3F64::new(prev.gyro[0], prev.gyro[1], prev.gyro[2]) - bias;
                        let k = Mat3F64::from_cols_array(&[
                            0.0, w.z, -w.y, //
                            -w.z, 0.0, w.x, //
                            w.y, -w.x, 0.0,
                        ]);
                        d += d * k * dt;
                    }
                    prev = s;
                }
                (pairs >= 1).then_some(d)
            });
        let Some(d) = integrated else {
            self.gyro_prior_unavailable += 1;
            return None;
        };
        // Gram-Schmidt: the first-order steps drift off SO(3), and `estimate_pose` is handed
        // this as a rotation.
        let c = d.to_cols_array();
        let x = Vec3F64::new(c[0], c[1], c[2]).normalize();
        let y0 = Vec3F64::new(c[3], c[4], c[5]);
        let y = (y0 - x * x.dot(y0)).normalize();
        let z = x.cross(y);
        let d = Mat3F64::from_cols_array(&[x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]);

        Some(r_bc.transpose() * d * r_bc)
    }

    /// Arm the map bound if this insertion took the map to [`TrackerConfig::max_keyframes`].
    ///
    /// Split out from the insertion path so the predicate is reachable from a test: every
    /// conjunct here is a way to silently disable the bound or to fire it every frame.
    fn arm_map_bound(&mut self, inserted: bool) {
        if let Some(cap) = self.config.max_keyframes
            && inserted
            && self.map.keyframes().len() >= cap
        {
            self.pending_map_reset = true;
        }
    }

    /// Drop the map if the last insertion tripped [`TrackerConfig::max_keyframes`].
    ///
    /// DEFERRED from the keyframe that tripped it: resetting there would wipe the state the
    /// caller is still reading that frame's pose out of. Bumps `world_generation` like any
    /// other reset, so a consumer re-anchors on `reset_epoch`.
    fn apply_pending_map_reset(&mut self) {
        if !std::mem::take(&mut self.pending_map_reset) {
            return;
        }
        warning!(
            "map bound reached: dropping the map (TrackerConfig::max_keyframes) keyframes={} cap={}",
            self.map.keyframes().len(),
            self.config.max_keyframes
        );
        // Keeps the IMU bias: the map hit a size cap, the sensor did not change.
        self.reset_keeping_calibration();
    }

    /// Frames the gyro rotation prior could not be built for. See the field.
    pub fn gyro_prior_unavailable(&self) -> u64 {
        self.gyro_prior_unavailable
    }

    /// Coasted frames whose interval the wheels did not measure, which grew their covariance at
    /// the worst-case rate instead. See the field.
    pub fn coast_unmeasured(&self) -> u64 {
        self.coast_unmeasured
    }

    /// Map writes kornia-slam refused, over the tracker's lifetime. See the field.
    pub fn map_writes_refused(&self) -> u64 {
        self.map_writes_refused
    }

    /// Local BA results kornia-slam refused, over the tracker's lifetime. See the field.
    pub fn ba_updates_refused(&self) -> u64 {
        self.ba_updates_refused
    }

    /// The tracker's current world-to-camera pose, measured or extrapolated.
    ///
    /// After a frame that returned [`TrackStatus::Untracked`] this is the constant-velocity
    /// prediction, not a measurement.
    pub fn last_pose(&self) -> Pose3d {
        self.state.pose_world_to_cam
    }

    /// Read-only access to the map (points, keyframes) for viewers and tests.
    pub fn map(&self) -> &Map {
        &self.map
    }

    /// Current pipeline mode.
    pub fn mode(&self) -> SystemMode {
        self.state.mode
    }

    /// The current world generation; see the field doc. Monotone, starts at 0.
    ///
    /// A successful visual-inertial initialization bumps this too — but only with
    /// `enable_inertial_ba` on: `apply_initialization` scales and rotates the whole map to put
    /// gravity on world +y, so poses before and after it are in different frames. With the
    /// flag off the initialization is a dry run and this counter does not move.
    pub fn world_generation(&self) -> u64 {
        self.world_generation
    }

    /// Live (not culled) map points, as of the last frame. Cheap: counted once per keyframe
    /// insertion or map drop, not per call. [`Tracker::stats`] recounts on demand.
    pub fn live_map_points(&self) -> usize {
        self.live_map_points
    }

    /// The live (not culled) points among the newest `cap` map-point slots, oldest first, each
    /// as `(map-point index, world-frame position in metres)`.
    ///
    /// Newest, not all: kornia-slam's map is append-only, so the tail is the frontier being built
    /// now, and walking only the tail keeps the cost at O(`cap`) however long the map has grown.
    /// The result is a complete view of that window, meant to REPLACE the previous one: a point
    /// culled or fused since is simply absent, and after a [`Tracker::world_generation`] change
    /// (map dropped, or kept in a moved world) the window describes the new world. The index is
    /// the point's position in [`Map::map_points`], stable while the generation holds.
    pub fn newest_live_map_points(
        &self,
        cap: usize,
    ) -> impl Iterator<Item = (usize, [f64; 3])> + '_ {
        let mps = self.map.map_points();
        let first = mps.len().saturating_sub(cap);
        mps[first..]
            .iter()
            .enumerate()
            .filter(|(_, mp)| !mp.culled)
            .map(move |(i, mp)| {
                let p = mp.position;
                (first + i, [p.x, p.y, p.z])
            })
    }

    /// Hands raw IMU samples to the tracker, to be preintegrated into the edge between the next
    /// two keyframes.
    ///
    /// `dropped_cumulative` is the producer's CUMULATIVE dropped-sample count (never reset, so
    /// only its increase is a new hole) and `aligned` says whether the samples were already
    /// rotated into the camera frame. Returns how many samples were accepted; samples whose
    /// stamp does not advance are rejected and counted rather than silently sorted later.
    ///
    /// Each sample's `tov` must be on the SAME clock as the `stamp` handed to
    /// [`Tracker::process_stereo`] — for an OAK-D stereo pair and IMU they are, by construction.
    ///
    /// # Errors
    ///
    /// [`VioError::InertialDisabled`] when no extrinsic is configured, and
    /// [`VioError::InertialSamplesAligned`] when the samples are already in the camera frame.
    /// Both are wiring mistakes that otherwise present as a plausible, wrong map: the first
    /// means nobody is reading what the caller is pushing, the second means `imu_t_bc` would
    /// rotate the samples a second time.
    pub fn push_imu(
        &mut self,
        samples: impl IntoIterator<Item = ImuSample>,
        dropped_cumulative: u64,
        aligned: bool,
    ) -> Result<usize, VioError> {
        self.push_raw_imu(
            samples.into_iter().map(RawImuSample::from),
            dropped_cumulative,
            aligned,
        )
    }

    /// [`Tracker::push_imu`] for samples already unpacked, which is what the bus queue holds.
    pub(crate) fn push_raw_imu(
        &mut self,
        samples: impl IntoIterator<Item = RawImuSample>,
        dropped_cumulative: u64,
        aligned: bool,
    ) -> Result<usize, VioError> {
        let Some(inert) = self.inertial.as_mut() else {
            return Err(VioError::InertialDisabled);
        };
        if aligned {
            return Err(VioError::InertialSamplesAligned);
        }
        Ok(inert.buffer.push(samples, dropped_cumulative))
    }

    /// Inertial counters, or `None` when the inertial path is off.
    ///
    /// The refusal counters are the externally visible number: a refused interval leaves the
    /// keyframe pair visual-only and no other trace anywhere, so a rising refusal rate is the
    /// only symptom of an IMU stream that has stopped covering its intervals.
    pub fn inertial_stats(&self) -> Option<InertialStats> {
        self.inertial.as_ref().map(|i| {
            let factors = self.map.imu_factors();
            InertialStats {
                buffer: i.buffer.stats(),
                initialized: self.state.imu_initialized,
                // Walks every factor, which is fine for a stop line and would not be for a
                // per-frame call: `Map::imu_factors` is push-only (see `crate::imu`).
                retained_factors: factors.len(),
                retained_samples: factors.iter().map(|f| f.raw_samples.len()).sum(),
                ..i.stats
            }
        })
    }

    /// Hands wheel-odometry readings to the tracker. Returns how many were accepted; readings
    /// whose stamp does not advance or whose pose is not finite are refused and counted.
    ///
    /// Each reading's stamp must be on the SAME clock as the frames' `stamp`. A frame outside
    /// the readings' span for ~2 s is logged with the offset (see [`WheelStats::frames_uncovered`]).
    ///
    /// # Errors
    ///
    /// [`VioError::WheelOdomDisabled`] when [`TrackerConfig::wheel_odom`] is `None`.
    pub fn push_wheel(
        &mut self,
        samples: impl IntoIterator<Item = WheelSample>,
    ) -> Result<usize, VioError> {
        let w = self.wheel.as_mut().ok_or(VioError::WheelOdomDisabled)?;
        Ok(w.push(samples))
    }

    /// Records where the camera sits on the base from `stamp`: `cam_in_base` is the camera's
    /// pose in the base frame (`X_base = cam_in_base * X_cam`). Until the first one, the
    /// configured mount is used. A stamp that does not advance is ignored and counted.
    ///
    /// # Errors
    ///
    /// [`VioError::WheelOdomDisabled`], or [`VioError::WheelMountNotRigid`] for a rotation that
    /// is not one.
    pub fn push_mount(&mut self, stamp: CuTime, cam_in_base: Pose3d) -> Result<(), VioError> {
        let w = self.wheel.as_mut().ok_or(VioError::WheelOdomDisabled)?;
        w.push_mount(stamp.as_nanos(), cam_in_base.inverse())
    }

    /// The bus-side half of [`Tracker::push_mount`]: already base-to-camera.
    pub(crate) fn push_mount_t_cb(&mut self, stamp_ns: u64, t_cb: Pose3d) -> Result<(), VioError> {
        let w = self.wheel.as_mut().ok_or(VioError::WheelOdomDisabled)?;
        w.push_mount(stamp_ns, t_cb)
    }

    /// Records the bus queue's cumulative eviction count, for the stop line.
    pub(crate) fn note_wheel_queue_evicted(&mut self, evicted: u64) {
        if let Some(w) = self.wheel.as_mut() {
            w.stats.queue_evicted = evicted;
        }
    }

    /// Wheel counters, or `None` when the wheel path is off.
    pub fn wheel_stats(&self) -> Option<WheelStats> {
        self.wheel.as_ref().map(|w| w.stats)
    }

    /// Drops the map and all state, returning to bootstrap at identity.
    ///
    /// Note this is *stronger* than what a tracking loss does by default: there a loss
    /// resets the tracker state (`SystemState::reset`) and re-bootstraps into the **same** map
    /// at the preserved
    /// pose, so keyframes and old map points accumulate — unless
    /// [`TrackerConfig::reset_map_on_loss`] routes losses here too. This is the only way to
    /// actually bound memory.
    pub fn reset(&mut self) {
        self.reset_inner(false);
    }

    /// Reset for a dropped MAP — the size bound or a tracking loss — keeping the IMU
    /// calibration, because nothing about the sensor changed. [`Tracker::reset`] is the only
    /// path that starts the bias over, and it means "different robot or re-calibration", not
    /// "the map went away". See [`InertialState::reset`].
    fn reset_keeping_calibration(&mut self) {
        self.reset_inner(true);
    }

    fn reset_inner(&mut self, keep_calibration: bool) {
        // A map DROP (the size bound, a loss with `reset_map_on_loss`) with wheels carries the
        // pose into the next map, so the world frame stays continuous across it: the next
        // bootstrap anchors where the robot is, propagated on the wheels from here. A requested
        // reset still means a new world at identity.
        let carry = keep_calibration && self.wheel.is_some() && self.world_anchored;
        let carried_pose = self.state.pose_world_to_cam;
        // A bias outside the plausible range is NEVER carried: not as the calibration and not
        // with the alignment. The next map starts from `initial_bias` and initializes again
        // (the still window when parked), which is what a drop did before the carry existed.
        let bias_plausible = self.inertial.as_ref().is_none_or(|inert| {
            let ok = inert.cfg.bias_limit.admits(&inert.bias);
            if !ok && keep_calibration {
                warning!(
                    "inertial: not carrying an IMU bias outside the plausible range into the \
                     next map; it re-initializes: gyro_bias={} accel_bias={}",
                    inert.bias.gyro.length(),
                    inert.bias.accel.length()
                );
            }
            ok
        });
        if !bias_plausible
            && keep_calibration
            && let Some(inert) = self.inertial.as_mut()
        {
            inert.stats.bias_rejected += 1;
        }
        let keep_calibration_bias = keep_calibration && bias_plausible;
        // A carried world keeps its gravity, so an initialized inertial path carries too.
        let carry_alignment = carry && self.state.imu_initialized && bias_plausible;
        let carried_stamp = self.last_frame_stamp_ns;
        if let Some(w) = self.wheel.as_mut() {
            w.reset_map();
        }
        // Their keyframe indices name keyframes of the dropped map.
        self.kf_tracked_poses.clear();
        self.kf_cov.clear();
        // A carried world keeps what a coast grows from, so the next map's first keyframe
        // inherits the uncertainty this one ended with. A world that restarts at identity starts
        // from nothing: the old covariance is in the old world's axes and would only stack up.
        if !carry {
            self.coast_origin = None;
        }
        self.map_settled = false;
        self.vision_suspect.clear();
        self.vision_degraded = false;
        self.bias_refusals_in_a_row = 0;
        self.map = Map::new();
        self.state = SystemState::new();
        if carry {
            self.state.pose_world_to_cam = carried_pose;
        } else {
            self.world_anchored = false;
        }
        self.map_carried = carry;
        self.last_keyframe_timestamp_sec = None;
        self.last_keyframe_stamp_ns = None;
        // `start_kf_idx` names a keyframe that no longer exists — `ImuInitializer` would
        // silently skip every factor whose endpoints it cannot find rather than report an empty
        // problem. The buffered SAMPLES survive (see `InertialState::reset`): the next anchor
        // prunes what is older than itself.
        if let Some(inert) = self.inertial.as_mut() {
            let gravity_world = inert.gravity_world;
            inert.reset(keep_calibration_bias);
            if carry_alignment {
                inert.gravity_world = gravity_world;
                inert.carried_alignment = true;
            }
        }
        self.stats = TrackerStats::default();
        // Cleared, so the first loss after a reset is described rather than throttled against a
        // loss of the map it replaced — the same rule `last_window_log` follows. The suppressed
        // count goes with it: it counts losses of a map that no longer exists.
        self.last_loss_log = None;
        self.losses_suppressed = 0;
        // The map this armed against is gone. Left set, a requested reset landing while the
        // bound was armed would drop the fresh map on the next frame and bump the generation
        // a SECOND time, for no reason a consumer could see.
        self.pending_map_reset = false;
        self.coast_rebootstrap_pending = false;
        // Kept across a carried drop: it starts the wheel interval that propagates the pose.
        self.last_frame_stamp_ns = if carry { carried_stamp } else { None };
        self.world_generation += 1;
        self.map_generation += 1;
        if keep_calibration {
            self.map_drops += 1;
        } else {
            self.anchored_since_reset = false;
        }
        // `next_frame_idx` and `epoch_ns` deliberately keep counting: frame
        // indices must stay unique for `Map::get_keyframe`, and restarting the
        // clock would corrupt the loss grace period.
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────

    /// Single-frame metric initialization from stereo depth (as in Campos et al.,
    /// ORB-SLAM3, IEEE T-RO 2021).
    ///
    /// The map is built at the *current* odometry pose, not at identity, so a
    /// re-bootstrap after a tracking loss continues the same trajectory. The
    /// returned pose is the input pose unchanged: bootstrap never moves the
    /// camera.
    fn bootstrap_stereo(
        &mut self,
        mut curr_frame: Frame,
        timestamp_sec: f64,
        stamp_ns: u64,
    ) -> TrackingStatus {
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let cam_points = unproject_stereo(&curr_frame, &self.camera);
        if cam_points.len() < self.config.min_bootstrap_stereo_points {
            debug!(
                "bootstrap_stereo: too few stereo points, frame dropped: frame={} stereo_points={} needed={}",
                curr_frame.idx,
                cam_points.len(),
                self.config.min_bootstrap_stereo_points
            );
            self.note_failure(FailureReason::TooFewStereoPoints);
            return TrackingStatus::Skipped;
        }

        // Captured before `from_frame` consumes the frame.
        let pose_inv = curr_frame.pose_world_to_cam.inverse();
        let curr_idx = curr_frame.idx;
        let seeds: Vec<LandmarkSeed> = cam_points
            .iter()
            .map(|(desc_idx, p_cam)| LandmarkSeed {
                position: pose_inv.transform_point(p_cam),
                color: keypoint_color(&curr_frame, *desc_idx),
                reference: ObservationKey {
                    keyframe_idx: curr_idx,
                    feature_idx: *desc_idx,
                },
            })
            .collect();

        // A carried alignment initializes the anchor keyframe itself, so `seeded_keyframe` gives
        // it the velocity measured now. With none measurable the alignment is dropped and the
        // still window or the motion solve initializes again, as a loss always did before the
        // carry existed.
        let carried = if self.inertial.as_ref().is_some_and(|i| i.carried_alignment) {
            let v = self.carried_velocity(stamp_ns, curr_frame.pose_world_to_cam);
            if v.is_none() {
                self.drop_carried_alignment(curr_frame.idx);
            }
            v
        } else {
            None
        };
        if let Some(v) = carried {
            self.state.imu_initialized = true;
            self.state.velocity_world = v;
        }
        // The keyframe enters the map FIRST: kornia-slam links a landmark only to a keyframe the
        // map holds, and derives the landmark's scale geometry from it on the spot.
        if let Err(e) = self.map.insert_keyframe(self.seeded_keyframe(curr_frame)) {
            if carried.is_some() {
                self.state.imu_initialized = false;
                self.state.velocity_world = Vec3F64::ZERO;
            }
            self.map_writes_refused += 1;
            self.note_failure(FailureReason::MapRefused);
            error!(
                "bootstrap_stereo: the map refused the bootstrap keyframe, frame dropped: frame={} error={}",
                curr_idx,
                e.to_string()
            );
            return TrackingStatus::Skipped;
        }
        let inserted = self.insert_landmarks(seeds, "bootstrap");

        info!(
            "bootstrap_stereo: metric map created from one stereo frame: frame={} points={}",
            curr_idx, inserted
        );

        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        self.state.velocity = None;
        self.state.mode = SystemMode::Tracking;
        self.world_anchored = true;
        self.anchored_since_reset = true;
        self.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.last_keyframe_stamp_ns = Some(stamp_ns);
        self.arm_inertial_window(curr_idx, timestamp_sec, stamp_ns);
        self.adopt_carried_alignment(carried, timestamp_sec);

        TrackingStatus::KeyframeAccepted
    }

    /// A keyframe for `frame`, seeded with the current IMU bias and, once initialized, the
    /// current velocity — as kornia-slam's `SlamSystem` seeds its own.
    ///
    /// `Keyframe::from_frame` zeroes both, and the sync-back after each insertion copies the
    /// newest keyframe's bias into the NEXT edge's linearization point. Unseeded, a keyframe no
    /// alignment or VI-BA wrote would zero that point: the bias a map drop kept, or
    /// `InertialConfig::initial_bias`, lost at the first keyframe. Seeded, VI-BA also starts the
    /// newest keyframe from the current estimate rather than from rest.
    fn seeded_keyframe(&self, frame: Frame) -> Keyframe {
        let mut kf = Keyframe::from_frame(frame);
        if let Some(inert) = &self.inertial {
            kf.imu_bias = inert.bias;
            if self.state.imu_initialized {
                kf.velocity_world = self.state.velocity_world;
            }
        }
        kf
    }

    /// Anchors the inertial initialization window on the keyframe the map was just built from.
    ///
    /// Every gate in `ImuInitializer` is measured from `start_kf_idx` forward, and a stale one
    /// would keep counting keyframes and IMU seconds from a map that no longer exists — the
    /// readiness test would pass while `try_initialize` silently skipped every factor whose
    /// endpoints it could not find. No-op with the inertial path off.
    fn arm_inertial_window(&mut self, kf_idx: usize, timestamp_sec: f64, stamp_ns: u64) {
        let Some(inert) = self.inertial.as_mut() else {
            return;
        };
        inert.start_kf_idx = Some(kf_idx);
        inert.window_start_sec = Some(timestamp_sec);
        inert.last_init_attempt_sec = None;
        // A new window is a new question — for the dry run, and for the line that describes it,
        // which would otherwise stay throttled against the window this one replaces. A loss with
        // `reset_map_on_loss` off (the default) re-arms here without any `InertialState::reset`.
        inert.dry_run_accepted = false;
        inert.last_window_log = None;
        inert.static_hold = None;
        inert.static_map_refused_at = None;
        inert.stats.first_refinement_done = false;
        inert.stats.second_refinement_done = false;
        // Samples older than the anchor can never bound an interval again.
        inert.buffer.prune_before(stamp_ns);
    }

    /// Attempts to relocalize a lost frame against the EXISTING map: 2D-3D descriptor
    /// matching against candidate keyframes' map-point associations, then PnP with the
    /// candidate keyframe's own pose as the prior. Read-only; on success the caller
    /// re-enters normal tracking from the returned pose — same world, same map-point
    /// indices, no generation bump.
    ///
    /// Cost: ~(associated kf descriptors x frame descriptors) Hamming ops per candidate,
    /// a few ms each — and it runs only on lost frames, which are otherwise wasted.
    fn try_relocalize(&self, frame: &Frame) -> Option<(Pose3d, usize, usize)> {
        let cfg = self.config.relocalization.as_ref()?;
        let kfs = self.map.keyframes();
        let descs = &frame.features.descriptors;
        if kfs.is_empty() || descs.is_empty() {
            return None;
        }
        // Candidates: nearest camera centres to the last known pose, best first.
        let here = self.state.pose_world_to_cam.inverse().translation;
        let mut order: Vec<(usize, f64)> = kfs
            .iter()
            .enumerate()
            .map(|(i, kf)| {
                let c = kf.frame.pose_world_to_cam.inverse().translation;
                let d = c - here;
                (i, d.dot(d))
            })
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        order.truncate(cfg.max_candidates);

        let mps = self.map.map_points();
        for (kf_i, _) in order {
            let kf = &kfs[kf_i];
            // Pair each of the keyframe's map-point descriptors with its best frame
            // keypoint, then keep one map point per keypoint (best Hamming wins).
            let mut pairs: Vec<(usize, usize, u32)> = Vec::new();
            for (d_idx, slot) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = slot else { continue };
                let Some(mp) = mps.get(*mp_idx) else { continue };
                if mp.culled {
                    continue;
                }
                let kd = &kf.frame.features.descriptors[d_idx];
                let (mut best, mut best_kp) = (u32::MAX, usize::MAX);
                for (kp_idx, fd) in descs.iter().enumerate() {
                    let d = hamming_distance(kd, fd);
                    if d < best {
                        best = d;
                        best_kp = kp_idx;
                    }
                }
                if best <= cfg.max_hamming {
                    pairs.push((best_kp, *mp_idx, best));
                }
            }
            pairs.sort_by_key(|&(kp, _, d)| (kp, d));
            pairs.dedup_by_key(|&mut (kp, _, _)| kp);
            if pairs.len() < cfg.min_matches {
                continue;
            }

            let mut points_world = Vec::with_capacity(pairs.len());
            let mut points_image = Vec::with_capacity(pairs.len());
            for &(kp, mp_idx, _) in &pairs {
                // The same coordinates (and distortion fallback) the tracking-path PnP uses.
                let Some(px) = frame.undistorted_xy(kp, &self.camera) else {
                    continue;
                };
                points_world.push(mps[mp_idx].position);
                points_image.push(kornia_algebra::Vec2F32::new(px[0], px[1]));
            }
            let pnp_config = PnpConfig {
                prior_reproj_threshold_px: cfg.prior_reproj_threshold_px,
                ..self.config.map_projection.pnp.clone()
            };
            if let Some((pose, inliers)) = pnp::solve_pnp(
                &points_world,
                &points_image,
                &self.camera,
                &kf.frame.pose_world_to_cam,
                &pnp_config,
            ) && inliers >= cfg.min_inliers
            {
                return Some((pose, inliers, kf.frame.idx));
            }
        }
        None
    }

    // ── Tracking ─────────────────────────────────────────────────────────────

    /// The pose prior for a frame at `stamp_ns`, from the current pose, and the gyro rotation
    /// it used. Sets `frame_on_wheels`.
    ///
    /// Translation (and, without a gyro, rotation) comes from the wheels when a reading covers
    /// the interval, else from the constant-velocity model. The wheels measured this interval;
    /// constant velocity guesses it from the last one, which is wrong whenever the robot
    /// accelerates, and while vision is lost this prior IS the pose.
    fn predict_pose(&mut self, stamp_ns: u64) -> (Pose3d, Option<Mat3F64>) {
        let pose_before = self.state.pose_world_to_cam;
        let wheel_motion = match (self.last_frame_stamp_ns, self.wheel.as_mut()) {
            (Some(t0), Some(w)) => w.camera_motion(t0, stamp_ns),
            _ => None,
        };
        self.frame_on_wheels = wheel_motion.is_some();

        // Constant-velocity motion model. `velocity` is stored below as
        // `Pose3d::between(&pose_before, &estimate.pose)` and applied here as
        // `velocity.compose(&pose_before)`; the two halves are a matched pair —
        // do not re-derive one from an assumed convention. The wheel increment is built in the
        // same world-to-camera form (`WheelState::camera_motion`).
        let mut candidate_pose = match wheel_motion {
            Some(m) => m.compose(&pose_before),
            None => self
                .state
                .velocity
                .map(|v| v.compose(&pose_before))
                .unwrap_or(pose_before),
        };

        // GYRO PRIOR on the rotation only. The constant-velocity model above predicts the next
        // ROTATION from the last one, which is wrong the moment angular rate changes — exactly
        // when the platform turns, and exactly when tracking was being lost. The gyro
        // measured that turn at 200 Hz, and it beats the wheels too, which slip in a turn.
        // Translation stays on the wheels or constant velocity: predicting it from the IMU
        // needs double-integrated acceleration plus gravity, which is a different error budget.
        let gyro_delta = self
            .last_frame_stamp_ns
            .and_then(|t0| self.gyro_camera_delta(t0, stamp_ns));
        if let Some(d) = gyro_delta {
            candidate_pose =
                rotate_about_centre(candidate_pose, d.transpose() * pose_before.rotation);
        }
        (candidate_pose, gyro_delta)
    }

    fn tracking_step(&mut self, frame: Frame, timestamp_sec: f64, stamp_ns: u64) -> TrackingStatus {
        let image_size = frame.image_size;
        // Bound before any path that can move `frame` into `bootstrap_stereo`.
        let frame_idx = frame.idx;
        let pose_before = self.state.pose_world_to_cam;
        let (candidate_pose, gyro_delta) = self.predict_pose(stamp_ns);

        // Widen the search in proportion to how long we have been failing: a
        // pose compounded over seconds of loss carries far more uncertainty
        // than a single-frame prediction.
        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let search_scale = self.estimator.config().search_scale_for(currently_lost_for);

        let t_est = std::time::Instant::now();
        let result = self.estimator.estimate_pose(
            &frame,
            &candidate_pose,
            &pose_before,
            &self.map,
            &self.camera,
            self.state.current_keyframe_idx,
            search_scale,
            // No KLT/flow front-end here: no pre-seeded correspondences.
            None,
        );
        let est_ms = t_est.elapsed().as_secs_f64() * 1e3;
        let t_rest = std::time::Instant::now();

        let mut fusion_us = 0u32;
        let (mut status, matches, tracked_inliers, vision_rejected) = match result {
            Ok(estimate) => {
                // GYRO vs VISION, over the same interval. The transposes in
                // `gyro_camera_delta` cannot be checked by inspection — a wrong one yields a
                // rotation of the right MAGNITUDE pointing the wrong way, which still tracks
                // in a slow scene. This prints the disagreement: near 0 deg means the
                // convention holds, and roughly twice the frame's own rotation means it does
                // not.
                if let Some(d) = gyro_delta {
                    // `between(a, b)` is the T with `T.compose(a) == b`, so its rotation is
                    // `R_cw1 * R_cw0^T` = `d^T`. The residual is therefore `d * visual`, which
                    // is identity when they agree. Comparing `d^T * visual` instead computes
                    // `d^T d^T` and reads as TWICE the true rotation — small when parked and
                    // ~9 deg while turning.
                    let visual = Pose3d::between(&pose_before, &estimate.pose).rotation;
                    let resid = d * visual;
                    let cos = ((resid.x_axis.x + resid.y_axis.y + resid.z_axis.z) - 1.0) * 0.5;
                    let gyro_vs_visual_deg = cos.clamp(-1.0, 1.0).acos().to_degrees();
                    debug!(
                        "gyro prior residual: gyro_vs_visual_deg={}",
                        gyro_vs_visual_deg
                    );
                }
                // `Estimate::matches` is the FULL correspondence set, not the
                // inlier set — `solve_pnp` returns `(Pose3d, usize)` and throws
                // the mask away. Everything downstream of here writes those
                // correspondences into the map permanently (keyframe
                // associations, the landmarks' observation records, and the
                // `n_found` counter that `cull_landmarks` keys off), and `run_local_ba` uses
                // `BaParams::default()`, whose kernel is `Identity` with
                // `robust_scale_sq = INFINITY` — plain L2, no Huber. So a
                // correspondence that PnP itself rejected would enter BA at
                // full weight and, because its `n_found` rises in lockstep with
                // `n_visible`, would keep a perfect found-ratio and never be
                // culled. Re-apply PnP's own final gate here, before any of
                // those writes, so a rejected correspondence never reaches
                // the map's bookkeeping.
                let matches = self.reject_pnp_outliers(&frame, &estimate.pose, &estimate.matches);

                // PER-FRAME FUSION: the pose re-solved with the IMU and the wheels from the last
                // keyframe, or their prediction when vision disagrees with them past the gate.
                let mut fused =
                    self.fuse_frame(&frame, &estimate.pose, &matches, &candidate_pose, stamp_ns);
                fusion_us = self.fusion.last_cost_us;
                let was_coasting = self.fusion.consecutive_rejects > 0;
                if fused.is_some_and(|o| o.rejected)
                    && self.fusion.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS
                {
                    // Vision has stayed self-consistent against the prediction for longer than a
                    // coast should be trusted: keep vision's pose and matches (so this frame can
                    // become the keyframe the next predictions start from), and suspend the
                    // fusion until a keyframe is inserted. Every frame until then would be
                    // predicted from the same stale keyframe and rejected again.
                    self.fusion.consecutive_rejects = 0;
                    self.fusion.reject_runs_expired += 1;
                    self.fusion_suspended_on = self.state.last_keyframe_idx;
                    warning!(
                        "per-frame fusion: vision rejected for {} frames in a row; accepting \
                         frame {} and suspending the fusion until the next keyframe chi2={} dof={}",
                        MAX_CONSECUTIVE_REJECTS,
                        frame.idx,
                        self.fusion.last_chi2,
                        self.fusion.last_dof
                    );
                    fused = None;
                }
                let rejected = fused.is_some_and(|o| o.rejected);
                if rejected {
                    self.fusion.rejects += 1;
                    self.fusion.consecutive_rejects += 1;
                    if fused.is_some_and(|o| o.rotation_kept) {
                        self.fusion.rotation_kept += 1;
                    }
                } else {
                    self.fusion.consecutive_rejects = 0;
                }
                let pose = fused.map_or(estimate.pose, |o| o.pose);
                // This frame's covariance relative to the keyframe it chains from: the fusion's
                // (its posterior, or the prediction's for a rejected frame), else vision's own.
                let relative = match fused {
                    Some(o) => o.covariance_cw,
                    None => crate::fusion::vision_covariance(
                        &self.camera,
                        &self.visual_obs(&frame, &matches),
                        &estimate.pose,
                    ),
                };
                if let Some(rel) = relative {
                    // A rejected frame is the prediction from the last keyframe and nothing else.
                    let base = if rejected {
                        self.state.last_keyframe_idx
                    } else {
                        self.chain_base(&matches)
                    };
                    self.frame_rel = Some((base, rel));
                }
                if was_coasting && !rejected {
                    // Where the coast would have gone on this frame: the last one carried by
                    // this frame's motion prior (the gyro and the wheels since it), which is
                    // continuous with what was published whichever predictors the fusion used.
                    self.publish_hold_target = Some(candidate_pose);
                }
                self.vision_pose = Some((frame.idx, estimate.pose, pose));
                self.state.pose_world_to_cam = pose;
                self.state.velocity = Some(Pose3d::between(&pose_before, &pose));
                if rejected {
                    // The IMU and the wheels outvoted this frame's matches: published as a coast
                    // (process_frame), and none of the matches reaches the map, so a bad frame
                    // cannot become a keyframe or a landmark observation.
                    self.frame_report.vision_rejected = true;
                    self.note_failure(FailureReason::VisionRejected);
                    self.classify_matches(&estimate.matches, &[]);
                    debug!(
                        "track: vision rejected by per-frame fusion frame={} chi2={} dof={}",
                        frame.idx, self.fusion.last_chi2, self.fusion.last_dof
                    );
                    (TrackingStatus::Tracked, Vec::new(), estimate.inliers, true)
                } else {
                    self.classify_matches(&estimate.matches, &matches);
                    debug!(
                        "track: ok frame={} matches={} kept={} inliers={}",
                        frame.idx,
                        estimate.matches.len(),
                        matches.len(),
                        estimate.inliers
                    );
                    (TrackingStatus::Tracked, matches, estimate.inliers, false)
                }
            }
            Err(reason) => {
                // Carry the *predicted* pose forward instead of freezing at
                // `pose_before`: freezing desyncs position from velocity and
                // compounds one bad frame into a full loss. `velocity` is left
                // untouched, so this is a constant-velocity coast, or a wheel one when
                // `predict_pose` had readings.
                self.state.pose_world_to_cam = candidate_pose;
                debug!(
                    "track: rejected frame={} reason={}",
                    frame.idx,
                    reject_reason_str(reason)
                );
                self.note_failure(failure_reason(reason));
                (TrackingStatus::Skipped, Vec::new(), 0, false)
            }
        };
        self.stats.inliers = tracked_inliers;

        if status == TrackingStatus::Tracked && !vision_rejected {
            // Visibility bookkeeping over the local map only; a full-map scan
            // here would grow with trajectory length.
            let current_kf = self
                .state
                .current_keyframe_idx
                .and_then(|ki| self.map.get_keyframe(ki));
            // `default()` is behaviour-preserving: 10/10/15/4 are the exact literals the
            // `Map` method this replaced hardcoded, including its `< 4` fallback.
            let local_indices = select_local_map_points(
                &self.map,
                &matches,
                current_kf,
                LocalMapSelectionConfig::default(),
            );
            // Evaluated at `candidate_pose`, NOT at the estimated pose: visibility
            // is counted before refinement, matching the ORB-SLAM3 reference
            // implementation. It biases `n_visible`, hence `cull_landmarks`'s found_ratio.
            let visible = landmarks_in_frustum(
                &self.map,
                &local_indices,
                &self.camera,
                &candidate_pose,
                image_size,
            );
            self.map.update_observation_counts(&visible, &matches);

            let bookkeep_ms = t_rest.elapsed().as_secs_f64() * 1e3;
            let t_kf = std::time::Instant::now();
            let inserted = self.try_insert_keyframe(
                &frame,
                timestamp_sec,
                stamp_ns,
                tracked_inliers,
                &matches,
            );
            if inserted {
                status = TrackingStatus::KeyframeAccepted;
            }
            let kf_ms = t_kf.elapsed().as_secs_f64() * 1e3;
            // Splits `track_ms` from the `process_stereo stage split` line one level further.
            // `estimate_pose` is projection matching + PnP; `bookkeep` is the local-map
            // visibility accounting that feeds culling; `kf` is insertion (grow/fuse/BA).
            debug!(
                "tracking_step split: est_ms={} fusion_us={} bookkeep_ms={} kf_ms={}",
                est_ms, fusion_us, bookkeep_ms, kf_ms
            );

            // MAP GROWTH, at info so it survives a production log filter that drops debug.
            // Nothing culls keyframes, so insertion cost rises with `kfs` until the graph stalls
            // silently: measured on a Jetson Orin with an OAK-D at 640x400, 757 MB RSS and zero
            // poses after 2.5 h of clean tracking. This is the curve that sets a keyframe bound.
            //
            // SAMPLED ONLY, with no "slow insertion" escape: past the knee EVERY insertion is
            // slow, so such an escape degenerates to a line per keyframe on exactly the
            // unbounded run it is meant to diagnose.
            if inserted && self.map.keyframes().len().is_multiple_of(KF_GROWTH_SAMPLE) {
                info!(
                    "map growth: kfs={} map_points={} kf_ms={}",
                    self.map.keyframes().len(),
                    self.map.num_map_points(),
                    kf_ms
                );
            }

            self.arm_map_bound(inserted);
        }

        if status == TrackingStatus::Skipped {
            let lost_since = *self.state.lost_since_sec.get_or_insert(timestamp_sec);
            let recently_lost_for = timestamp_sec - lost_since;
            // `imu_confident` is always false: the inertial path does not feed loss recovery.
            let grace_period_sec = self.config.tracking_loss_recovery.grace_period_sec(false);
            let map_established = self.map.keyframes().len()
                > self.config.tracking_loss_recovery.min_keyframes_for_grace;

            // Relocalization first: a PnP pose against the EXISTING map is a measurement —
            // strictly better than the widening projection search (which needs the coast
            // prior to be roughly right) and than any re-bootstrap (which moves the world).
            if map_established && let Some((pose, inliers, kf_idx)) = self.try_relocalize(&frame) {
                warning!(
                    "relocalized against the existing map: frame={} inliers={} kf={} lost_for_sec={}",
                    frame_idx,
                    inliers,
                    kf_idx,
                    recently_lost_for
                );
                self.state.pose_world_to_cam = pose;
                // The jump invalidates the constant-velocity model; rebuild from the
                // next tracked pair.
                self.state.velocity = None;
                self.state.current_keyframe_idx = Some(kf_idx);
                // The last keyframe's prediction does not span the jump: no fusion until the
                // next keyframe.
                self.fusion_suspended_on = self.state.last_keyframe_idx;
                self.publish_hold = None;
                self.publish_hold_target = None;
                self.state.lost_since_sec = None;
                self.state.last_frame_timestamp_sec = timestamp_sec;
                self.frame_report.relocalized = true;
                self.frame_report.failure = FailureReason::None;
                // Placed against that keyframe's own points; `try_relocalize` keeps no matches to
                // weigh, so vision's floors stand in for the PnP fit's spread.
                let mut rel = [0.0; 36];
                for k in 0..3 {
                    rel[k * 7] = VISION_TRANSLATION_SIGMA_FLOOR.powi(2);
                    rel[(k + 3) * 7] = VISION_ROTATION_SIGMA_FLOOR.powi(2);
                }
                self.frame_rel = Some((Some(kf_idx), rel));
                // Measured (PnP inliers), so it may leave as Tracked — unlike the
                // re-bootstrap coast below, which must not.
                return TrackingStatus::Tracked;
            }
            // While relocalization is enabled, give it its patience window before
            // surrendering the map — the grace period alone was tuned for blind coasting.
            let give_up_sec = self
                .config
                .relocalization
                .as_ref()
                .map_or(grace_period_sec, |r| {
                    grace_period_sec.max(r.patience.get::<second>())
                });

            if !map_established || recently_lost_for >= give_up_sec {
                if should_log_throttled(
                    self.last_loss_log,
                    timestamp_sec,
                    map_established,
                    LOSS_LOG_RETRY_SEC,
                ) {
                    // `stats()` and not the `self.stats` fields: it is the bundle, and its doc
                    // asks to be called at the log cadence rather than per frame — which is
                    // exactly where the throttle has put this. It also folds in `map_points`,
                    // the other whole-map scan, which a hand-picked field list left out.
                    let s = self.stats();
                    warning!(
                        "tracking lost: resetting to bootstrap: frame={} lost_for_sec={} \
                         map_established={} map_cleared={} keypoints={} stereo_matches={} \
                         inliers={} consecutive_failures={} keyframes={} map_points={} \
                         suppressed={}",
                        frame.idx,
                        recently_lost_for,
                        map_established,
                        self.config.reset_map_on_loss,
                        s.keypoints,
                        s.stereo_matches,
                        s.inliers,
                        s.consecutive_failures,
                        s.keyframes,
                        s.map_points,
                        self.losses_suppressed
                    );
                    self.last_loss_log = Some((timestamp_sec, map_established));
                    self.losses_suppressed = 0;
                } else {
                    self.losses_suppressed += 1;
                }
                if self.config.reset_map_on_loss {
                    // Full reset (map dropped): bounds the per-frame cost the append-to-map
                    // arm below lets grow without limit; see the config field doc for the
                    // measurements. Bumps world_generation.
                    //
                    // KEEPS THE IMU BIAS, for the same reason the `max_keyframes` drop does: a
                    // map thrown away says nothing about the sensor. This is the reset a live
                    // graph takes most often, so it is the one that most needs the bias.
                    self.reset_keeping_calibration();
                } else {
                    // `SystemState::reset` clears mode/keyframe pointers but keeps the map
                    // and `pose_world_to_cam`: the re-bootstrap APPENDS at the preserved
                    // pose — a coast, not a measurement, hence the generation bump — and
                    // with `current_keyframe_idx == None` the next local-map build returns
                    // every non-culled point.
                    // The kept map is the same world, gravity included: an initialized inertial
                    // path carries into the re-anchor rather than rotating the world again.
                    let carried = self.state.imu_initialized;
                    self.state.reset();
                    if let (true, Some(inert)) = (carried, self.inertial.as_mut()) {
                        inert.carried_alignment = true;
                    }
                    self.world_generation += 1;
                }
                let recovered = self.bootstrap_stereo(frame, timestamp_sec, stamp_ns);
                // An unusable loss frame defers the re-anchor to the next usable one, which is
                // the same coast. A dropped map needs no flag: its fresh world starts at identity.
                self.coast_rebootstrap_pending =
                    !self.config.reset_map_on_loss && recovered != TrackingStatus::KeyframeAccepted;
                // The pose the new map was just anchored at is `candidate_pose`
                // — pure constant-velocity extrapolation over every frame since
                // tracking failed, which at 30 fps and a 0.5 s grace is up to
                // ~15 compoundings with no correction. It is a coast, not a
                // measurement, so it MUST NOT leave here as `KeyframeAccepted`:
                // `process_frame` maps that to `TrackStatus::Tracked`, which the
                // task publishes and which also resets
                // `consecutive_failures` to 0 — a fabricated pose, published as
                // measured, with every health field reading green. Report
                // `Skipped` regardless of whether the map was rebuilt (or, via
                // `coast_rebootstrap_pending`, on the later frame that rebuilds it);
                // the next frame that actually tracks against it is the first
                // measured pose again.
                if recovered == TrackingStatus::KeyframeAccepted {
                    self.frame_report.reanchored = true;
                    warning!(
                        "re-bootstrapped at the extrapolated pose on frame {}; reporting Skipped \
                         because that pose is dead-reckoned, not measured",
                        frame_idx
                    );
                }
                // Early return, so `last_frame_timestamp_sec` is not updated on
                // this frame — harmless, nothing here reads it (kornia-slam's own
                // state bounds a per-frame IMU window with it), but do not "fix"
                // it silently if a reader is ever added.
                return TrackingStatus::Skipped;
            }
        } else if !vision_rejected {
            // A rejected frame is a coast: it neither starts nor ends a loss.
            self.state.lost_since_sec = None;
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        status
    }

    /// Per-frame fused tracking ([`crate::fusion`]) for a frame vision placed at `vision_pose`.
    ///
    /// Builds the motion factors from the last keyframe — the IMU preintegrated at the keyframe's
    /// bias (position rows once the inertial path is initialized), and the wheels through the
    /// mount at each end — and hands them with the kept matches to the gate and the solve.
    /// `None` when the fusion is off, there is no keyframe to anchor on, or no factor could be
    /// built (no IMU coverage, no wheel reading): the frame keeps vision's pose.
    ///
    /// The wheel factor is left out when the wheels disagree with the gyro over the interval:
    /// [`crate::consistency`]'s verdict (vision and the gyro outvote the wheels), or the two
    /// predictors' own Mahalanobis vote ([`crate::fusion::wheels_agree_with_gyro`]), which also
    /// catches the head turning on its mast by less than the verdict's 1.5 deg floor.
    fn fuse_frame(
        &mut self,
        frame: &Frame,
        vision_pose: &Pose3d,
        matches: &[(usize, usize)],
        prediction_seed: &Pose3d,
        stamp_ns: u64,
    ) -> Option<FusionOutcome> {
        let cfg = self.config.per_frame_fusion?;
        self.fusion.last_chi2 = f64::NAN;
        self.fusion.last_dof = 0;
        self.fusion.last_cost_us = 0;
        let t0 = std::time::Instant::now();
        let kf_ns = self.last_keyframe_stamp_ns.filter(|&t| t < stamp_ns)?;
        let kf_idx = self.state.last_keyframe_idx?;
        if self.fusion_suspended_on == Some(kf_idx) {
            return None;
        }
        let position_rows = imu_position_trusted(self.inertial_diag().0);
        let kf = self.map.get_keyframe(kf_idx)?;
        let kf_pose = kf.frame.pose_world_to_cam;
        let (kf_bias, kf_velocity) = (kf.imu_bias, kf.velocity_world);

        let mut factors: Vec<MotionFactor> = Vec::with_capacity(2);
        // The gyro's camera rotation over the interval and its camera-frame covariance, for the
        // wheels' vote below.
        let mut gyro_cam: Option<(Mat3F64, Mat3F64)> = None;
        if let (Some(inert), Some(epoch)) = (self.inertial.as_ref(), self.epoch_ns)
            && let Ok(window) = inert.buffer.window(kf_ns, stamp_ns, &inert.cfg.gates)
        {
            let pim = IncrementalPreintegration::extend(
                &mut self.fusion_preint,
                kf_idx,
                self.world_generation,
                kf_bias,
                inert.cfg.calib,
                window.map(|s| s.to_measurement(epoch)),
                ts_sec(kf_ns, epoch),
                ts_sec(stamp_ns, epoch),
            );
            let initialized = self.state.imu_initialized;
            let rate = self
                .fusion_preint
                .as_ref()
                .map_or(0.0, |c| c.last_rate(kf_bias));
            let bias_sigma = if initialized {
                KF_GYRO_BIAS_SIGMA
            } else {
                UNINITIALIZED_GYRO_BIAS_SIGMA
            };
            if pim.dt > 0.0
                && preintegration_is_finite(&pim)
                && let Some(f) = MotionFactor::imu(
                    &pim,
                    inert.cfg.imu_t_bc,
                    &kf_pose,
                    kf_velocity,
                    inert.gravity_world,
                    initialized && position_rows,
                    bias_sigma,
                    rate,
                )
            {
                let r_bc = inert.cfg.imu_t_bc.rotation;
                let d = f.dim;
                let c = &f.covariance;
                let cov_b = Mat3F64::from_cols_array(&[
                    c[0],
                    c[d],
                    c[2 * d],
                    c[1],
                    c[d + 1],
                    c[2 * d + 1],
                    c[2],
                    c[d + 2],
                    c[2 * d + 2],
                ]);
                gyro_cam = Some((
                    r_bc.transpose() * pim.delta_rotation * r_bc,
                    r_bc.transpose() * cov_b * r_bc,
                ));
                factors.push(f);
            }
        }
        if let Some(w) = self.wheel.as_ref()
            && let Ok(d) = w.delta(kf_ns, stamp_ns, w.cfg.max_gap_ns())
        {
            let m = w.cfg.measurement(&d);
            let (t_from, t_to) = (w.t_cb_at(kf_ns), w.t_cb_at(stamp_ns));
            let extrinsic = [cfg.extrinsic_sigma_m, cfg.extrinsic_sigma_rad];
            if let Some(f) = MotionFactor::wheel(&m, t_from, t_to, &kf_pose, extrinsic) {
                let disagree = gyro_cam.is_some_and(|(gyro, gyro_cov)| {
                    let wheels = t_from.compose(&m.delta_pose()).compose(&t_to.inverse());
                    let verdict = IntervalMotion {
                        wheels,
                        vision: kf_pose.compose(&vision_pose.inverse()),
                        gyro: Some(gyro),
                    }
                    .verdict();
                    let cov_cam = wheel_rotation_cov_in_camera(&f.covariance, &t_to);
                    verdict == MotionVerdict::WheelsDisagree
                        || !crate::fusion::wheels_agree_with_gyro(
                            &wheels.rotation,
                            &cov_cam,
                            &gyro,
                            &gyro_cov,
                            cfg.chi2_gate,
                        )
                });
                if disagree {
                    self.fusion.wheel_dropped += 1;
                } else {
                    factors.push(f);
                }
            }
        }
        if factors.is_empty() {
            return None;
        }

        let visual = self.visual_obs(frame, matches);
        let out = crate::fusion::fuse(
            &self.camera,
            &visual,
            &factors,
            vision_pose,
            prediction_seed,
            cfg.chi2_gate,
        );
        self.fusion.last_cost_us = u32::try_from(t0.elapsed().as_micros()).unwrap_or(u32::MAX);
        if let Some(o) = &out {
            self.fusion.frames += 1;
            match o.dropped {
                Some(Predictor::Imu) => self.fusion.imu_dropped += 1,
                Some(Predictor::Wheel) => self.fusion.wheel_dropped += 1,
                None => {}
            }
            self.fusion.last_chi2 = o.chi2;
            self.fusion.last_dof = u8::try_from(o.dof).unwrap_or(u8::MAX);
            self.frame_report.fused = true;
        }
        out
    }

    /// The per-frame fusion's counters; see [`crate::fusion`]. All zero with it off.
    pub fn fusion_stats(&self) -> FusionStats {
        self.fusion
    }

    /// The kept matches as the fusion's observations: the landmark, the undistorted pixel, and
    /// the octave's sigma.
    fn visual_obs(&self, frame: &Frame, matches: &[(usize, usize)]) -> Vec<VisualObs> {
        let inv_scale = 1.0 / ORB_SCALE_FACTOR;
        let mps = self.map.map_points();
        matches
            .iter()
            .filter_map(|&(mp_idx, kp_idx)| {
                let mp = mps.get(mp_idx).filter(|mp| !mp.culled)?;
                let xy = frame.undistorted_xy(kp_idx, &self.camera)?;
                let octave = frame.features.octaves.get(kp_idx).copied().unwrap_or(0);
                Some(VisualObs {
                    point_world: mp.position,
                    pixel: [f64::from(xy[0]), f64::from(xy[1])],
                    inv_sigma: inv_scale.powi(i32::from(octave)),
                })
            })
            .collect()
    }

    /// The keyframe a measured frame's covariance chains from: the last keyframe, unless most of
    /// the frame's matches are landmarks whose reference keyframe is better localised than it
    /// (a smaller covariance trace), in which case the one of those keyframes most of them refer
    /// to.
    ///
    /// A HEURISTIC, documented as one. A frame tracked against old landmarks is placed relative
    /// to where those landmarks were mapped, so its uncertainty is that keyframe's plus its own,
    /// not the newest keyframe's plus its own. Without it a parked robot, which inserts a keyframe
    /// every eight frames, would chain them one onto the next and grow its covariance for as long
    /// as it stood still; with it the parked covariance stays at the keyframe the scene was mapped
    /// from, and a robot that drives back over its own map stops growing. A real answer is the
    /// marginal of a joint solve over the window, which nothing here runs.
    fn chain_base(&self, matches: &[(usize, usize)]) -> Option<usize> {
        let last = self.state.last_keyframe_idx?;
        let Some(last_cov) = self.kf_cov.get(&last) else {
            return Some(last);
        };
        let trace = |c: &Cov6| (0..6).map(|k| c[k * 7]).sum::<f64>();
        let last_trace = trace(last_cov);
        let mps = self.map.map_points();
        let mut votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut older = 0usize;
        for &(mp_idx, _) in matches {
            let Some(mp) = mps.get(mp_idx) else {
                continue;
            };
            if mp.keyframe_idx != last
                && self
                    .kf_cov
                    .get(&mp.keyframe_idx)
                    .is_some_and(|c| trace(c) < last_trace)
            {
                older += 1;
                *votes.entry(mp.keyframe_idx).or_default() += 1;
            }
        }
        if 2 * older <= matches.len() {
            return Some(last);
        }
        // Most votes; the older keyframe on a tie, so the choice does not depend on hash order.
        votes
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(k, _)| k)
            .or(Some(last))
    }

    /// The coast at `stamp_ns`: its covariance (`cov`) is the origin's, plus the wheels' over
    /// the interval since (the propagated measurement), plus the mount term of the whole base
    /// motion since the last measured origin, moved into the world at the coasted pose
    /// ([`CoastOrigin`]). The origin's alone without wheels, and zero when nothing was ever
    /// measured (a world being defined, not estimated). `true` with an interval the wheels did
    /// not measure, which grows by [`CoastOrigin::unmeasured`] instead: a coast whose pose keeps
    /// moving never hands back the origin's covariance, nor re-bases on it.
    ///
    /// The wheel measurement is a right perturbation of the END base pose, so in the world it is
    /// `Ad(T_wb)` of it, `T_wb = T_wc T_cb` at the end; the origin's left perturbation carries
    /// through the composition unchanged.
    fn coast_covariance(&self, stamp_ns: u64, pose_cw: &Pose3d) -> (CoastOrigin, bool) {
        let Some(origin) = self.coast_origin else {
            return (CoastOrigin::measured(stamp_ns, [0.0; 36]), false);
        };
        let Some(w) = self.wheel.as_ref() else {
            return (CoastOrigin { stamp_ns, ..origin }, false);
        };
        let d = match w.delta(origin.stamp_ns, stamp_ns, w.cfg.max_gap_ns()) {
            Ok(d) => d,
            // No time since the origin: nothing to grow.
            Err(WheelWindowError::NonPositiveInterval) => {
                return (CoastOrigin { stamp_ns, ..origin }, false);
            }
            Err(_) => return (origin.unmeasured(stamp_ns), true),
        };
        let t_cb_to = w.t_cb_at(stamp_ns);
        let extrinsic = self.config.per_frame_fusion.map_or(
            [DEFAULT_EXTRINSIC_SIGMA_M, DEFAULT_EXTRINSIC_SIGMA_RAD],
            |c| [c.extrinsic_sigma_m, c.extrinsic_sigma_rad],
        );
        let measurement = w.cfg.measurement(&d);
        // The noise alone (a zero mount sigma adds nothing); the mount term is the origin's.
        let Some(f) = MotionFactor::wheel(
            &measurement,
            w.t_cb_at(origin.stamp_ns),
            t_cb_to,
            pose_cw,
            [0.0, 0.0],
        ) else {
            return (origin.unmeasured(stamp_ns), true);
        };
        let noise: Cov6 = std::array::from_fn(|k| f.covariance[k]);
        let t_wb = pose_cw.inverse().compose(&t_cb_to);
        (
            origin.advanced(
                stamp_ns,
                &noise,
                &measurement.delta_pose(),
                &t_wb,
                extrinsic,
            ),
            false,
        )
    }

    /// Counts an unmeasured coast frame and says so, rate-limited.
    fn note_coast_unmeasured(&mut self, frame_idx: usize, stamp_ns: u64) {
        self.coast_unmeasured += 1;
        let n = self.coast_unmeasured;
        if n == 1 || n.is_multiple_of(COAST_UNMEASURED_WARN_EVERY) {
            let since_ms = self
                .coast_origin
                .map_or(0, |o| stamp_ns.saturating_sub(o.stamp_ns) / 1_000_000);
            warning!(
                "vio coast: frame {}: the wheels did not measure the {} ms since the coast origin (a gap or \
                 discontinuity in the readings, or readings already pruned); its covariance grows at the \
                 worst-case rate instead ({} such frames so far)",
                frame_idx,
                since_ms,
                n
            );
        }
    }

    /// This frame's covariance, from what `tracking_step` measured or from the coast, and the
    /// keyframe bookkeeping that goes with it. Runs once per frame, after the status is known.
    fn record_covariance(&mut self, status: TrackingStatus, frame_idx: usize, stamp_ns: u64) {
        let pose_cw = self.state.pose_world_to_cam;
        // Inserted by tracking, by a bootstrap, or by the re-bootstrap a loss reports Skipped.
        let keyframe = self.state.last_keyframe_idx == Some(frame_idx)
            && self.last_keyframe_stamp_ns == Some(stamp_ns);
        let measured = status != TrackingStatus::Skipped && !self.frame_report.vision_rejected;
        let cov = if let Some((base, rel)) = self.frame_rel.take() {
            let base_cov = base
                .and_then(|k| self.kf_cov.get(&k).copied())
                .or(self.coast_origin.map(|o| o.cov))
                .unwrap_or([0.0; 36]);
            let c = covariance::add(
                &base_cov,
                &covariance::world_from_camera_tangent(&rel, &pose_cw),
            );
            if measured {
                self.coast_origin = Some(CoastOrigin::measured(stamp_ns, c));
            }
            Some(c)
        } else if keyframe {
            // A map anchored on this frame at a propagated pose (a carried drop, a loss
            // re-bootstrap) or at the start of a world: the coast's covariance, which is zero for
            // a world that starts here. The anchor is the next coast's origin.
            let (o, unmeasured) = self.coast_covariance(stamp_ns, &pose_cw);
            if unmeasured {
                self.note_coast_unmeasured(frame_idx, stamp_ns);
            }
            self.coast_origin = Some(CoastOrigin::measured(stamp_ns, o.cov));
            Some(o.cov)
        } else if status == TrackingStatus::Skipped
            && self.wheel.is_some()
            && self.world_anchored
            && self.frame_on_wheels
        {
            let (o, unmeasured) = self.coast_covariance(stamp_ns, &pose_cw);
            if unmeasured {
                self.note_coast_unmeasured(frame_idx, stamp_ns);
            }
            if let Some(t0) = self.coast_origin.map(|o| o.stamp_ns)
                && stamp_ns.saturating_sub(t0) >= COAST_REBASE_NS
            {
                self.coast_origin = Some(o);
            }
            Some(o.cov)
        } else {
            None
        };
        if keyframe && let Some(c) = cov {
            self.kf_cov.insert(frame_idx, c);
            // The removal hook: a keyframe the map no longer holds (nothing removes one today; a
            // keyframe culling would) must not stay a chain base.
            if self.kf_cov.len() > self.map.keyframes().len() {
                let live: HashSet<usize> =
                    self.map.keyframes().iter().map(|k| k.frame.idx).collect();
                self.kf_cov.retain(|k, _| live.contains(k));
            }
        }
        self.frame_cov = cov;
    }

    /// The last frame's pose covariance: 6x6 row-major over `[rho; omega]`, the world left
    /// perturbation of its camera-in-world pose ([`crate::covariance`]), the convention
    /// `VioPose::covariance` carries. `None` for a frame with no pose to publish (untracked, not
    /// localized).
    ///
    /// What it is: the frame's covariance relative to the keyframe it was tracked from (the
    /// per-frame fusion's posterior, the prediction's when vision was rejected, or vision's own),
    /// added to that keyframe's at its insertion, which was chained the same way back to the
    /// map's first keyframe; a coast adds the wheels' propagated covariance to the last measured
    /// frame's; a map drop carries it into the next map. What it is not: a joint marginal (local
    /// BA's corrections are not fed back), so it is an approximation that grows with the chain.
    pub fn frame_covariance(&self) -> Option<Cov6> {
        self.frame_cov
    }

    /// The tracker world's up axis: against the gravity the inertial path aligned to once it is
    /// initialized, else the bootstrap camera's `-y` (optical: `y` points down the image), which
    /// is up only as far as the camera was level.
    pub fn up_axis(&self) -> [f64; 3] {
        match self.inertial.as_ref() {
            Some(i) if self.state.imu_initialized => {
                let g = i.gravity_world;
                [-g.x, -g.y, -g.z]
            }
            _ => [0.0, -1.0, 0.0],
        }
    }

    /// The pose to publish for the frame at `now_sec`: the tracker's, behind the hold that is
    /// still closing ([`PublishHold`]), which this shrinks by the time since it last did.
    ///
    /// A frame that ended a rejection run left its coast target in `publish_hold_target`; the
    /// hold is set HERE, against the tracker's pose as the frame leaves it — a keyframe's local
    /// BA may have moved the pose after the gate (by centimetres, through the same wheel factor
    /// that started the run), and that move must be held too. A hold still closing carries on
    /// from where it is.
    fn held_pose(&mut self, now_sec: f64) -> Pose3d {
        let tracker = self.state.pose_world_to_cam;
        if let Some(target) = self.publish_hold_target.take() {
            let prior = self
                .publish_hold
                .filter(|h| h.map_generation == self.map_generation)
                .map_or(Pose3d::IDENTITY, |h| h.offset);
            let offset = prior.compose(&target).compose(&tracker.inverse());
            self.publish_hold = Some(PublishHold {
                offset,
                at_sec: now_sec,
                map_generation: self.map_generation,
            });
            return offset.compose(&tracker);
        }
        let Some(hold) = self.publish_hold.as_mut() else {
            return tracker;
        };
        if hold.map_generation != self.map_generation {
            self.publish_hold = None;
            return tracker;
        }
        let dt = (now_sec - hold.at_sec).max(0.0);
        let xi = crate::fusion::se3_log(&hold.offset);
        let rho = (xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]).sqrt();
        let omega = (xi[3] * xi[3] + xi[4] * xi[4] + xi[5] * xi[5]).sqrt();
        // The fraction to close this frame: whichever of the two rates binds.
        let frac = |len: f64, rate: f64| {
            if len > 0.0 {
                rate * dt / len
            } else {
                f64::INFINITY
            }
        };
        let k = frac(rho, PUBLISH_CLOSE_M_PER_S)
            .min(frac(omega, PUBLISH_CLOSE_RAD_PER_S))
            .min(1.0);
        if k >= 1.0 {
            self.publish_hold = None;
            return tracker;
        }
        let keep = 1.0 - k;
        hold.offset = crate::fusion::se3_exp(&xi.map(|v| v * keep));
        hold.at_sec = now_sec;
        hold.offset.compose(&tracker)
    }

    /// Keeps only the correspondences that reproject within PnP's own final
    /// inlier threshold at the estimated pose.
    ///
    /// The threshold is `PnpConfig::final_reproj_threshold_px` **unscaled**:
    /// `search_scale` widens the projection search and the PnP *prior* gate
    /// during a loss, but upstream leaves the final gate alone precisely
    /// because it is the one that decides what is true, so widening it here
    /// would let the map absorb garbage exactly when tracking is weakest.
    ///
    /// A correspondence whose map point or undistorted keypoint cannot be read
    /// is dropped: it could not have contributed to the solve either.
    fn reject_pnp_outliers(
        &self,
        frame: &Frame,
        pose: &Pose3d,
        matches: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let th = self.estimator.config().pnp.final_reproj_threshold_px;
        let th_sq = th * th;
        matches
            .iter()
            .copied()
            .filter(|&(mp_idx, curr_idx)| {
                let Some(mp) = self.map.map_points().get(mp_idx) else {
                    return false;
                };
                if mp.culled {
                    return false;
                }
                let Some(kp) = frame.undistorted_xy(curr_idx, &self.camera) else {
                    return false;
                };
                self.camera
                    .reprojection_error_sq_world(pose, &mp.position, kp[0] as f64, kp[1] as f64)
                    .is_some_and(|err_sq| err_sq <= th_sq)
            })
            .collect()
    }

    /// Whether to withhold a keyframe the policy asked for because the robot is parked.
    ///
    /// kornia-slam inserts one every `max_frames_between` frames however still the scene, which
    /// filled `max_keyframes` and dropped the map every ~75 s on a parked robot (2026-10-11). All
    /// must hold to withhold it: the interval is configured and not yet elapsed since the last
    /// keyframe; tracking is strong (a weak reference is the policy's scene-change trigger, and a
    /// person walking past a parked robot still gets its keyframe); the wheels' PATH since the last
    /// keyframe is under the parked floors; and the camera is within
    /// [`PARKED_CAMERA_TRANSLATION_M`] / [`PARKED_CAMERA_ROTATION_RAD`] of it, so a head re-tilt or
    /// a pan the wheels cannot see is motion. Anything missing (no wheels, no readings, no
    /// keyframe) inserts as before, and so does an inertial path not yet initialized: the still
    /// window is built from keyframe IMU edges, and withholding them kept a robot parked 9 s
    /// uninitialized into its first drive (replay of a 2026-10-11 capture). So does a fusion
    /// suspended on the last keyframe, which waits for the next one.
    fn parked_keyframe_suppressed(
        &mut self,
        stamp_ns: u64,
        tracked_inliers: usize,
        n_ref_map_points: usize,
    ) -> bool {
        let Some(w) = self.wheel.as_ref() else {
            return false;
        };
        let Some(interval) = w.cfg.parked_keyframe_interval else {
            return false;
        };
        if self.inertial.is_some() && !self.state.imu_initialized {
            return false;
        }
        let (Some(kf_ns), Some(kf_idx)) =
            (self.last_keyframe_stamp_ns, self.state.last_keyframe_idx)
        else {
            return false;
        };
        // A fusion suspended on this keyframe resumes only at the next one; withheld, it stayed
        // off for the rest of the interval.
        if self.fusion_suspended_on == Some(kf_idx) {
            return false;
        }
        let interval_ns = (interval.get::<cu29::units::si::time::second>() * 1e9) as u64;
        if stamp_ns.saturating_sub(kf_ns) >= interval_ns {
            return false;
        }
        let strong = n_ref_map_points > 0
            && tracked_inliers as f64
                >= n_ref_map_points as f64 * self.config.keyframe_policy.ref_ratio;
        if !strong || !w.parked_between(kf_ns, stamp_ns) {
            return false;
        }
        let Some(kf) = self.map.get_keyframe(kf_idx) else {
            return false;
        };
        let (p0, p1) = (kf.frame.pose_world_to_cam, self.state.pose_world_to_cam);
        let moved = (p1.inverse().translation - p0.inverse().translation).length();
        let turned = crate::consistency::angle_between(&p0.rotation, &p1.rotation);
        // A NaN pose is not parked.
        if !(moved <= PARKED_CAMERA_TRANSLATION_M && turned <= PARKED_CAMERA_ROTATION_RAD) {
            return false;
        }
        if let Some(w) = self.wheel.as_mut() {
            w.stats.parked_keyframes_skipped += 1;
        }
        true
    }

    /// Canonical order: insert -> associate -> densify close stereo -> grow(xN)
    /// -> IMU edge -> fuse -> local BA -> pose sync-back -> cull.
    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        stamp_ns: u64,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        // Bound BEFORE the block below overwrites both: the IMU edge spans from the PREVIOUS
        // keyframe, and both halves of that interval are about to be replaced in place.
        let prev_edge = self
            .state
            .last_keyframe_idx
            .zip(self.last_keyframe_stamp_ns);

        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .map(|kf| kf.num_associated_points())
            .unwrap_or(0);

        if !self.config.keyframe_policy.should_insert(
            frame.idx,
            self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }
        if self.parked_keyframe_suppressed(stamp_ns, tracked_inliers, n_ref_map_points) {
            return false;
        }

        // The reference keyframe must exist before we can triangulate against it.
        if self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .is_none()
        {
            return false;
        }

        let curr_kf = self.seeded_keyframe(Frame {
            idx: frame.idx,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
            u_right: frame.u_right.clone(),
            depth: frame.depth.clone(),
            // Carry the undistortion cache over — dropping it would silently
            // re-undistort per consumer for the rest of this keyframe's life.
            keypoints_undist: frame.keypoints_undist.clone(),
        });

        // Recency as a covisibility proxy, newest first. Taken BEFORE the insertion below, so
        // `curr_kf` itself is not among them.
        let neighbor_kf_indices: Vec<usize> = self
            .map
            .keyframes()
            .iter()
            .rev()
            .take(self.config.max_covisible_keyframes)
            .map(|kf| kf.frame.idx)
            .collect();

        // The keyframe enters the map FIRST, with no associations: kornia-slam links a feature
        // to a landmark only on a keyframe the map already holds, and every link below
        // (tracked matches, stereo densification, growth, fuse) goes through the map so both
        // sides of the association stay in step. Nothing between here and the old insertion
        // point read the keyframe list, so the neighbour set and every pass see what they did.
        if let Err(e) = self.map.insert_keyframe(curr_kf) {
            self.map_writes_refused += 1;
            error!(
                "keyframe: the map refused the keyframe, not inserted: frame={} error={}",
                frame.idx,
                e.to_string()
            );
            return false;
        }

        // One landmark per feature and one feature per landmark, first claim wins — upstream's
        // `keyframe_insertion` resolves tracked links the same way. Duplicates are reachable:
        // `estimate_pose`'s reference-keyframe fallback matches one-directionally, so two
        // landmarks can claim one feature, and `reject_pnp_outliers` keeps both. Dropped here,
        // they never reach the map as a refusal.
        let claims = accepted_pair_claims(matches);
        if claims.len() < matches.len() {
            debug!(
                "keyframe: duplicate tracked claims dropped: frame={} dropped={} matches={}",
                frame.idx,
                matches.len() - claims.len(),
                matches.len()
            );
        }
        let mut link_refused = 0u64;
        for (mp_idx, curr_idx) in claims.into_iter().map(|i| matches[i]) {
            if let Err(e) = self.map.link_observation(frame.idx, curr_idx, mp_idx) {
                link_refused += 1;
                debug!(
                    "keyframe: tracked match not linked: frame={} feature={} map_point={} error={}",
                    frame.idx,
                    curr_idx,
                    mp_idx,
                    e.to_string()
                );
            }
        }
        if link_refused > 0 {
            self.map_writes_refused += link_refused;
            warning!(
                "keyframe: the map refused tracked-match links: frame={} refused={} matches={}",
                frame.idx,
                link_refused,
                matches.len()
            );
        }

        // Stereo densification. Must run after association (so the
        // "already tracked" filter is meaningful) and before grow (so grow's
        // "unassociated only" filter sees these as taken).
        if let Some(close_depth_threshold) =
            self.config.stereo_close_depth.map(|d| d.get::<meter>())
            && frame.is_stereo()
        {
            let n_close = self.add_close_stereo_points(frame.idx, close_depth_threshold);
            debug!(
                "keyframe: stereo densification frame={} close_points={}",
                frame.idx, n_close
            );
        }

        let match_config = self.config.two_view_init.match_config;
        let triangulation_config = self.config.two_view_init.triangulation_config.clone();

        let mut total_grown = 0usize;
        for &nb_kf_idx in &neighbor_kf_indices {
            total_grown += self.grow_map_points_from_keyframe_pair(
                nb_kf_idx,
                frame.idx,
                match_config,
                &triangulation_config,
            );
        }

        self.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.last_keyframe_stamp_ns = Some(stamp_ns);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        // Inertial edge + initialization ladder: after growth, before fuse, where it has always
        // been. Both endpoints of the edge are in the map, which `Map::apply_insertion` refuses
        // an IMU factor without and `ImuInitializer` silently skips a factor without. `None` —
        // the configuration this repo ships — means every line below behaves exactly as before.
        let inertial_ba = self.on_keyframe_inertial(prev_edge, frame.idx, stamp_ns);

        // Fuse before local BA, deliberately: BA then sees the extra
        // reprojection constraints.
        let n_fused = self.fuse_into_neighbors(frame.idx, &neighbor_kf_indices);

        // Local BA only once the gauge is anchored. `run_local_ba` frees the
        // last `BA_ACTIVE_KEYFRAMES` poses and fixes the rest, so at 2 or 3
        // keyframes it fixes NOTHING — `bundle_adjust_schur` only requires
        // `n_free_poses > 0`, so the whole map, keyframe 0 included, is free to
        // translate and rotate, held near the initialization by LM damping
        // alone. On noisy input that rigidly shifts the world frame away from
        // the bootstrap frame, and keyframe 0 is then fixed wherever it landed:
        // every later pose inherits the offset. A Sim(3)-aligned ATE cannot see
        // it, and neither can the noise-free synthetic test (zero residual =>
        // zero gradient) — but the unaligned tape-measure test is measuring
        // exactly that offset. Skipping BA for the first two insertions costs
        // very little; upstream's own bootstrap BA (`run_initial_ba`) pins the
        // older keyframe for the same reason.
        // The wheel interval this keyframe closes, queued even when BA does not run: a factor
        // is keyed by `Frame::idx` and acts in every later solve whose window holds both ends.
        // Vision's word for this keyframe, before BA (or the per-frame fusion) leans it on the
        // gyro and the wheels: PnP's pose when this frame was tracked, else the keyframe's own
        // (a bootstrap keyframe, which no fusion touched).
        if let Some(kf) = self.map.keyframes().last() {
            //
            // The inertial ladder above may have moved the keyframe since (an alignment rotates
            // the world), so the fusion's offset is carried onto the keyframe as it stands:
            // `vision * published^-1 * kf`. Bit for bit the keyframe's own pose when the fusion
            // did not move it.
            let kf_pose = kf.frame.pose_world_to_cam;
            let vision = match self.vision_pose {
                Some((idx, vision, published))
                    if idx == kf.frame.idx && pose_differs(&vision, &published) =>
                {
                    vision.compose(&published.inverse()).compose(&kf_pose)
                }
                _ => kf_pose,
            };
            let tracked_in = TrackedIn {
                world_generation: self.world_generation,
                map_settled: self.map_settled || !self.config.enable_local_ba,
            };
            self.kf_tracked_poses
                .push_back((kf.frame.idx, tracked_in, vision));
            while self.kf_tracked_poses.len() > KF_TRACKED_POSES {
                self.kf_tracked_poses.pop_front();
            }
        }
        let new_wheel_factors = match self.wheel.as_mut() {
            Some(w) => w.on_keyframe(prev_edge, frame.idx, stamp_ns),
            None => Vec::new(),
        };
        self.check_motion(&new_wheel_factors, timestamp_sec);
        let wheel_factors: Vec<WheelOdomFactor> = self
            .wheel
            .as_ref()
            .map_or_else(Vec::new, |w| w.factors().to_vec());

        let ba_window_ready =
            self.config.enable_local_ba && self.map.keyframes().len() > BA_ACTIVE_KEYFRAMES;
        let suspect_in_window = self.vision_suspect_in_ba_window();
        self.note_vision_degraded(suspect_in_window, frame.idx);
        let vision_suspect = ba_window_ready && suspect_in_window;
        if vision_suspect {
            self.consistency.ba_skipped += 1;
        }
        let ran_ba = ba_window_ready && !vision_suspect;
        if ran_ba {
            // Every keyframe tracked from here on is tracked against a map a solve has held.
            self.map_settled = true;
            // kornia-slam's only BA entry points solve a `BaSnapshot` and hand back an update
            // for `apply_ba_update` to validate and write. `local_window` is the selection
            // kornia-slam's own local mapping captures: the solvers' active keyframes, their
            // live landmarks, and as fixed context the keyframes observing those landmarks plus
            // the active keyframes' IMU neighbours. Same solve as cloning the whole map (#143:
            // byte-identical trajectory, peak RSS 768 -> 504 MB), at a cost that no longer
            // grows with the map.
            let snapshot = self.map.ba_snapshot(&local_window(&self.map));
            let update = match inertial_ba {
                // 15 DOF per keyframe (pose, velocity, bias) instead of 6. Mirrors
                // kornia-slam's own local mapping, which branches on exactly this flag.
                Some((imu_t_bc, gravity_world)) => run_local_inertial_ba(
                    snapshot,
                    &self.camera,
                    Some(imu_t_bc),
                    gravity_world,
                    &wheel_factors,
                ),
                // A map built without the inertial path holds no IMU factor, so the 15-DOF solve
                // is the visual one plus the wheel edges; velocity and bias ride along under the
                // solver's damping floor and nothing reads them. With the inertial path on but
                // not yet applied, the map DOES hold IMU factors, at an unaligned gravity this
                // solve would trust: the wheels wait for the initialization there.
                None if self.inertial.is_none() && !wheel_factors.is_empty() => {
                    run_local_inertial_ba(
                        snapshot,
                        &self.camera,
                        None,
                        Vec3F64::ZERO,
                        &wheel_factors,
                    )
                }
                None => run_local_ba(snapshot, &self.camera),
            };
            // Refused wholesale (a non-finite estimate or IMU factor, or a world frame that moved
            // since the snapshot, which cannot happen on this synchronous path): the map keeps
            // its pre-BA state, exactly what a solver failure has always left. Counted, because
            // that state is indistinguishable from a BA that ran; throttled, because a cause
            // that persists refuses every keyframe's BA until the map drops.
            // A bias outside the plausible range is refused with the WHOLE update: the poses and
            // velocities were co-estimated with it, and written alone it would sit in the map
            // where the random-walk edges hold every later keyframe to it. See `BiasLimit`.
            let implausible = self.inertial.as_ref().and_then(|inert| {
                let limit = inert.cfg.bias_limit;
                update
                    .keyframe_biases()
                    .find(|b| !limit.admits(b))
                    .map(|b| (b, limit))
            });
            if let Some((bias, limit)) = implausible {
                self.ba_updates_refused += 1;
                if let Some(inert) = self.inertial.as_mut() {
                    inert.stats.bias_rejected += 1;
                }
                self.bias_refusals_in_a_row += 1;
                if self.bias_refusals_in_a_row >= BIAS_REFUSALS_BEFORE_DROP {
                    warning!(
                        "keyframe: {} local BA results in a row wanted an implausible IMU bias; \
                         dropping the map (keeps the plausible bias, re-initializes) frame={}",
                        self.bias_refusals_in_a_row,
                        frame.idx
                    );
                    self.bias_refusals_in_a_row = 0;
                    self.pending_map_reset = true;
                }
                if should_log_throttled(
                    self.last_ba_refusal_log,
                    timestamp_sec,
                    true,
                    LOSS_LOG_RETRY_SEC,
                ) {
                    self.last_ba_refusal_log = Some((timestamp_sec, true));
                    warning!(
                        "keyframe: local BA result refused, its IMU bias is outside the plausible \
                         range; map left as it was: frame={} gyro_bias={} accel_bias={} \
                         gyro_limit={} accel_limit={} refused_total={}",
                        frame.idx,
                        bias.gyro.length(),
                        bias.accel.length(),
                        limit.gyro_rad_s,
                        limit.accel_m_s2,
                        self.ba_updates_refused
                    );
                }
            } else if let Err(e) = {
                // A plausible bias ends the run, written or refused for another reason.
                self.bias_refusals_in_a_row = 0;
                self.map.apply_ba_update(update)
            } {
                self.ba_updates_refused += 1;
                if should_log_throttled(
                    self.last_ba_refusal_log,
                    timestamp_sec,
                    true,
                    LOSS_LOG_RETRY_SEC,
                ) {
                    self.last_ba_refusal_log = Some((timestamp_sec, true));
                    warning!(
                        "keyframe: local BA result refused, map left as it was: frame={} \
                         refused_total={} error={}",
                        frame.idx,
                        self.ba_updates_refused,
                        e.to_string()
                    );
                }
            }
        }

        // The keyframe just inserted is the newest: `Frame::idx` only grows and `insert_keyframe`
        // appends. O(1) here, where `Map::get_keyframe(frame.idx)` is a linear scan
        // over every keyframe since the last reset.
        if let Some(newest_kf) = self.map.keyframes().last() {
            debug_assert_eq!(newest_kf.frame.idx, frame.idx);
            // Pose sync-back: BA may have moved the newest keyframe.
            if ran_ba {
                self.state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
            }
            // After BA, not before: the inertial solve writes velocity and bias onto the last
            // three keyframes, and the NEXT edge must be linearized at that refreshed bias.
            // Without this the next edge is linearized at the bootstrap bias forever, which
            // drives the inertial BA's 0.02 repropagation threshold to fire on every factor of
            // every solve — a purely numerical cost that also makes the first-order bias
            // correction the dominant residual, which the optimizer cannot tell apart from
            // signal. Where no alignment or VI-BA wrote the keyframe, this reads back what
            // `seeded_keyframe` put on it.
            if let Some(inert) = self.inertial.as_mut() {
                inert.bias = newest_kf.imu_bias;
                self.state.velocity_world = newest_kf.velocity_world;
            }
        }

        cull_landmarks(&mut self.map);

        debug!(
            "keyframe inserted: frame={} grown={} fused={} neighbors={} map_points={} keyframes={}",
            frame.idx,
            total_grown,
            n_fused,
            neighbor_kf_indices.len(),
            self.map.num_active_map_points(),
            self.map.keyframes().len()
        );
        true
    }

    // ── Motion consistency ───────────────────────────────────────────────────

    /// Checks each wheel factor just integrated against the gyro and vision over the same
    /// keyframe interval ([`crate::consistency`]): counts the verdict, remembers an interval
    /// vision disagreed on, and drops the wheel factor of one the wheels disagreed on.
    ///
    /// Every interval the wheels MOVED in is logged with all three sensors' numbers, whatever
    /// the verdict (an on-robot drive is read off these lines); a notable verdict while parked is
    /// throttled. Each line also carries the interval's out-of-plane sample and path length
    /// ([`crate::consistency::OutOfPlane`]), the data the factor's `z_sigma`, `z_slope` and
    /// `tilt_sigma` are fitted from.
    fn check_motion(&mut self, factors: &[(WheelOdomFactor, f64)], timestamp_sec: f64) {
        for (f, path_m) in factors {
            let (wheels, gyro, vision) = self.interval_sensors(f);
            let Some(vision) = vision else {
                // No comparable vision, so no verdict. The gyro alone still screens the wheels:
                // a factor that contradicts it is slip or a head that moved on its joint (the
                // gyro does not slip), and BA should not see it. Keeping it was the old answer
                // for every unarbitrated interval.
                if gyro
                    .is_some_and(|g| crate::consistency::rotations_disagree(&wheels.rotation, &g))
                    && let Some(w) = self.wheel.as_mut()
                    && w.drop_factor(f.from_idx, f.to_idx)
                {
                    self.consistency.wheels_dropped_on_gyro += 1;
                }
                continue;
            };
            let motion = IntervalMotion {
                wheels,
                vision,
                gyro,
            };
            let verdict = motion.verdict();
            let translation = motion.translation_disagrees();
            self.consistency.count(verdict, translation);
            match verdict {
                MotionVerdict::VisionDisagrees => {
                    self.vision_suspect.push_back((f.from_idx, f.to_idx));
                    while self.vision_suspect.len() > KF_TRACKED_POSES {
                        self.vision_suspect.pop_front();
                    }
                }
                MotionVerdict::WheelsDisagree => {
                    if let Some(w) = self.wheel.as_mut() {
                        w.drop_factor(f.from_idx, f.to_idx);
                    }
                }
                MotionVerdict::Consistent => {
                    if let Some(inert) = self.inertial.as_ref() {
                        self.consistent_gyro_bias = Some(inert.bias.gyro);
                    }
                }
                _ => {}
            }
            let notable = translation
                || !matches!(
                    verdict,
                    MotionVerdict::Consistent | MotionVerdict::Inconclusive
                );
            let moving = motion.wheels_moved();
            let oop = crate::consistency::OutOfPlane::of(f, &motion.vision);
            if moving {
                self.last_motion = motion_check(
                    &motion,
                    verdict,
                    translation,
                    self.last_motion.checks + 1,
                    &oop,
                    *path_m,
                );
            }
            let speak = moving
                || (notable
                    && should_log_throttled(
                        self.last_consistency_log,
                        timestamp_sec,
                        true,
                        LOSS_LOG_RETRY_SEC,
                    ));
            if !speak {
                continue;
            }
            if !moving {
                self.last_consistency_log = Some((timestamp_sec, true));
            }
            let deg = |r: &Mat3F64| crate::consistency::angle(r).to_degrees();
            let action = match verdict {
                MotionVerdict::Consistent if translation => {
                    "consistent rotation, TRANSLATION disagrees (counted only: slip and frozen \
                     vision look alike)"
                }
                MotionVerdict::Consistent => "consistent",
                MotionVerdict::Inconclusive => "inconclusive",
                MotionVerdict::VisionDisagrees => {
                    "VISION disagrees, tracking DEGRADED, local BA held off while it is in the \
                     window"
                }
                MotionVerdict::WheelsDisagree => {
                    "WHEELS disagree (slip, or the head moved on its joint), the interval's \
                     factor dropped"
                }
                MotionVerdict::GyroDisagrees => "GYRO disagrees (bias or extrinsic), counted only",
            };
            let c = self.consistency;
            let line = format!(
                "motion check: {action}: kf {}->{} wheels_deg={:.2} gyro_deg={:.2} \
                 vision_deg={:.2} wheels_mm={:.1} vision_mm={:.1} oop_z_mm={:.2} \
                 oop_roll_deg={:.3} oop_pitch_deg={:.3} path_m={:.4} (checked={} vision={} \
                 wheels={} gyro={} translation={})",
                f.from_idx,
                f.to_idx,
                deg(&motion.wheels.rotation),
                motion.gyro.as_ref().map_or(f64::NAN, deg),
                deg(&motion.vision.rotation),
                motion.wheels.translation.length() * 1e3,
                motion.vision.translation.length() * 1e3,
                oop.z_m * 1e3,
                oop.roll_rad.to_degrees(),
                oop.pitch_rad.to_degrees(),
                path_m,
                c.checked,
                c.vision_disagrees,
                c.wheels_disagree,
                c.gyro_disagrees,
                c.translation_disagrees
            );
            if notable {
                warning!("{}", line.as_str());
            } else {
                info!("{}", line.as_str());
            }
        }
    }

    /// Edge-logs the degraded state: a vision-suspect interval entering or leaving the local BA
    /// window. The published pose stays vision's while degraded (so it may be frozen); what
    /// changes is that no solve bends the map or the bias to agree with it.
    fn note_vision_degraded(&mut self, suspect_in_window: bool, frame_idx: usize) {
        if suspect_in_window == self.vision_degraded {
            return;
        }
        self.vision_degraded = suspect_in_window;
        if suspect_in_window {
            self.consistency.degraded_entries += 1;
            warning!(
                "tracking DEGRADED: vision disagrees with the wheels and the gyro (frozen or \
                 self-tracking view); local BA held off, the published pose is vision's and may \
                 be frozen frame={} times_degraded={}",
                frame_idx,
                self.consistency.degraded_entries
            );
        } else {
            info!(
                "tracking RECOVERED: no vision-suspect interval left in the local BA window \
                 frame={}",
                frame_idx
            );
        }
    }

    /// Whether tracking is DEGRADED: an interval vision disagreed with the wheels and the gyro
    /// on is still in the local BA window ([`crate::consistency`]).
    pub fn vision_degraded(&self) -> bool {
        self.vision_degraded
    }

    /// Why the last frame was or was not measured; see [`FrameReport`].
    pub fn frame_report(&self) -> FrameReport {
        self.frame_report
    }

    /// Maps dropped by the size bound, a sustained loss or refused biases, lifetime. A requested
    /// [`reset`](Self::reset) is not counted.
    pub fn map_drops(&self) -> u64 {
        self.map_drops
    }

    /// Maps discarded by any reset, lifetime: bumps on EVERY new map (requested reset, loss, size
    /// bound), and on nothing else. See the field.
    pub fn map_generation(&self) -> u64 {
        self.map_generation
    }

    /// Whether a map was anchored since construction or the last requested reset.
    pub fn anchored_since_reset(&self) -> bool {
        self.anchored_since_reset
    }

    /// The last keyframe interval the wheels moved in; `checks == 0` before the first.
    pub fn last_motion_check(&self) -> MotionCheck {
        self.last_motion
    }

    /// The last frame's left keypoints, with `FEATURE_*` classes: stereo depth, and, on a tracked
    /// frame, PnP inlier or outlier among the map matches.
    pub fn frame_features(&self) -> &[FeaturePoint] {
        &self.features
    }

    /// The inertial initialization's progress, and the current `(gyro, accel)` bias.
    pub fn inertial_diag(&self) -> (InertialStage, [f64; 3], [f64; 3]) {
        let Some(inert) = self.inertial.as_ref() else {
            return (InertialStage::Off, [0.0; 3], [0.0; 3]);
        };
        let s = &inert.stats;
        let stage = if !self.state.imu_initialized {
            InertialStage::Uninitialized
        } else if s.second_refinement_done {
            InertialStage::Refine2
        } else if s.first_refinement_done {
            InertialStage::Refine1
        } else if inert.applied_stage == Some(InitStage::Static) {
            InertialStage::Static
        } else {
            InertialStage::Initial
        };
        let (g, a) = (inert.bias.gyro, inert.bias.accel);
        (stage, [g.x, g.y, g.z], [a.x, a.y, a.z])
    }

    /// Records the frame's failure cause; the first one wins, since later ones (a re-bootstrap
    /// that also failed) follow from it.
    fn note_failure(&mut self, reason: FailureReason) {
        if self.frame_report.failure == FailureReason::None {
            self.frame_report.failure = reason;
        }
    }

    /// Marks the feature classes of a tracked frame: every map match an outlier, then the ones
    /// PnP's final gate kept inliers. `(map point, keypoint)` pairs.
    fn classify_matches(&mut self, all: &[(usize, usize)], kept: &[(usize, usize)]) {
        for &(_, kp) in all {
            if let Some(f) = self.features.get_mut(kp) {
                f.class |= FEATURE_OUTLIER;
            }
        }
        for &(_, kp) in kept {
            if let Some(f) = self.features.get_mut(kp) {
                f.class = (f.class & !FEATURE_OUTLIER) | FEATURE_INLIER;
            }
        }
    }

    /// The interval a wheel factor spans, as `(wheels, gyro, vision)` saw it: the wheels through
    /// the mount at each end, the gyro (the interval's IMU edge, at the bias of the last interval
    /// all three agreed on, else the current one; see `consistent_gyro_bias`; `None` without
    /// one) and vision (the keyframes' tracked poses). Vision is `None` when either tracked pose
    /// is gone or from another world (a map drop, an inertial alignment or pivot, a re-anchor),
    /// or the interval straddles the map's first local BA (see `map_settled`): its two poses
    /// would then be in two frames, and their difference a frame change, not motion.
    fn interval_sensors(&self, f: &WheelOdomFactor) -> (Pose3d, Option<Mat3F64>, Option<Pose3d>) {
        let tracked = |idx: usize| {
            self.kf_tracked_poses
                .iter()
                .find(|(i, t, _)| *i == idx && t.world_generation == self.world_generation)
                .map(|(_, t, p)| (*t, *p))
        };
        let vision = match (tracked(f.from_idx), tracked(f.to_idx)) {
            (Some((from_in, p0)), Some((to_in, p1))) if from_in == to_in => {
                Some(p0.compose(&p1.inverse()))
            }
            _ => None,
        };
        let gyro = self.inertial.as_ref().and_then(|inert| {
            let r_bc = inert.cfg.imu_t_bc.rotation;
            self.map
                .imu_factors()
                .iter()
                .find(|e| e.prev_kf_idx == f.from_idx && e.curr_kf_idx == f.to_idx)
                .map(|e| {
                    let p = &e.preintegrated;
                    let bias = self.consistent_gyro_bias.unwrap_or(inert.bias.gyro);
                    let db = bias - p.bias.gyro;
                    let r_b = p.delta_rotation
                        * kornia_algebra::SO3F64::exp(p.d_rotation_d_bias_gyro * db).matrix();
                    r_bc.transpose() * r_b * r_bc
                })
        });
        let wheels = f
            .t_cb_from
            .compose(&f.measurement.delta_pose())
            .compose(&f.t_cb_to.inverse());
        (wheels, gyro, vision)
    }

    /// Whether an interval vision disagreed on ends at a keyframe local BA would free. Prunes
    /// the ones that have left the window for good.
    fn vision_suspect_in_ba_window(&mut self) -> bool {
        let kfs = self.map.keyframes();
        let Some(oldest_free) = kfs
            .len()
            .checked_sub(BA_ACTIVE_KEYFRAMES)
            .and_then(|i| kfs.get(i))
            .map(|kf| kf.frame.idx)
        else {
            return false;
        };
        self.vision_suspect.retain(|&(_, to)| to >= oldest_free);
        !self.vision_suspect.is_empty()
    }

    /// The motion-consistency counters; see [`crate::consistency`].
    pub fn consistency_stats(&self) -> ConsistencyStats {
        self.consistency
    }

    // ── Inertial ─────────────────────────────────────────────────────────────

    /// Everything the inertial path does on a keyframe insertion, in the order kornia-slam's
    /// removed `SlamPipeline` did it: build the edge, prune, then run the initialization ladder.
    ///
    /// Returns `Some((imu_t_bc, gravity_world))` when local BA should take the inertial path,
    /// and `None` — including for every call with the inertial path off — when it should not.
    ///
    /// `enable_inertial_ba` gates BOTH ways the inertial path can move a published pose: the
    /// 15-DOF local BA selected by the return value here, and the initialization commit
    /// (`apply_initialization` scales and rotates the WHOLE map and bumps the world
    /// generation). With it off, the ladder still runs
    /// `try_initialize` and counts the result, but never applies it: see
    /// [`Self::accept_inertial_init`].
    ///
    /// The state is `take`n for the duration because the initializer wants `&mut Map` and
    /// `&mut` the inertial bias and gravity at the same time, which it cannot have while a
    /// field of `self` is borrowed. The move is a handful of scalars plus the ring's `VecDeque`
    /// handle; nothing is copied.
    fn on_keyframe_inertial(
        &mut self,
        prev_edge: Option<(usize, u64)>,
        curr_kf_idx: usize,
        stamp_ns: u64,
    ) -> Option<(Pose3d, Vec3F64)> {
        let mut inert = self.inertial.take()?;
        // `epoch_ns` is set by `process_frame` before this can run; the fallback only keeps the
        // arithmetic sane if that ever stops being true.
        let epoch = self.epoch_ns.unwrap_or(stamp_ns);

        if let Some((prev_kf_idx, prev_stamp_ns)) = prev_edge {
            self.add_imu_edge(
                &mut inert,
                prev_kf_idx,
                prev_stamp_ns,
                curr_kf_idx,
                stamp_ns,
                epoch,
            );
        }
        // Unconditional — refusal included. `t0` for the next interval is exactly this stamp,
        // so nothing earlier can ever be asked for again, and this (not the capacity bound) is
        // what holds the buffer at one keyframe interval in steady state. Without it, a run of
        // refusals would grow the ring until the capacity bound started evicting, at which
        // point the refusals become self-sustaining.
        inert.buffer.prune_before(stamp_ns);

        self.run_inertial_init(&mut inert, ts_sec(stamp_ns, epoch));

        let handles = (self.state.imu_initialized && inert.cfg.enable_inertial_ba)
            .then_some((inert.cfg.imu_t_bc, inert.gravity_world));
        self.inertial = Some(inert);
        handles
    }

    /// Preintegrates `[prev_stamp_ns, stamp_ns]` and adds it to the map, or refuses and says why.
    ///
    /// A refusal costs one visual-only keyframe pair. Every refusal here is a case where
    /// `PreintegratedImu::from_measurements` would have returned a well-formed delta anyway:
    /// see the gates in [`crate::imu`]. That asymmetry — a missing factor is recoverable, a
    /// wrong one poisons a bias shared across the whole window — is why the gates are strict.
    fn add_imu_edge(
        &mut self,
        inert: &mut InertialState,
        prev_kf_idx: usize,
        prev_stamp_ns: u64,
        curr_kf_idx: usize,
        stamp_ns: u64,
        epoch: u64,
    ) {
        // Converted once, straight out of the ring, into the one owned copy the `ImuFactor`
        // keeps. Seconds relative to the tracker's frame epoch — the SAME conversion
        // `process_frame` applies to frame stamps. `from_measurements` filters the samples by
        // `[t0, t1]`, so a second epoch here would drop every one of them and return the
        // zero-delta case.
        let raw: Vec<_> = match inert
            .buffer
            .window(prev_stamp_ns, stamp_ns, &inert.cfg.gates)
        {
            Ok(samples) => samples.map(|s| s.to_measurement(epoch)).collect(),
            Err(error) => {
                match error {
                    ImuWindowError::DroppedSamples { .. } => inert.stats.refused_dropped += 1,
                    ImuWindowError::Gap { .. } => inert.stats.refused_gap += 1,
                    ImuWindowError::TooFewSamples { .. } => inert.stats.refused_too_few += 1,
                    ImuWindowError::NoSamples { .. } => inert.stats.refused_no_samples += 1,
                    ImuWindowError::NonPositiveInterval { .. } => {
                        inert.stats.refused_bad_interval += 1
                    }
                }
                let span_ms = (stamp_ns.saturating_sub(prev_stamp_ns)) as f64 * 1e-6;
                warning!(
                    "inertial: IMU edge refused; this keyframe pair stays visual-only: \
                     prev_kf={} kf={} span_ms={} error={}",
                    prev_kf_idx,
                    curr_kf_idx,
                    span_ms,
                    error.to_string()
                );
                return;
            }
        };

        let t0 = ts_sec(prev_stamp_ns, epoch);
        let t1 = ts_sec(stamp_ns, epoch);
        let preint = PreintegratedImu::from_measurements(inert.bias, inert.cfg.calib, &raw, t0, t1);

        // Negated on purpose: a NaN `dt` must be refused too, which `dt <= 0.0` would let through.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let non_positive_dt = !(preint.dt > 0.0);
        if non_positive_dt {
            // Should be unreachable behind the gates above. Checked anyway because
            // `Map::apply_insertion` validates the factor's `[t0, t1]` interval but not the
            // integrated `dt`, and `vi_ba_schur` answers a singular covariance with a
            // 1e6-diagonal information matrix and a printed warning — a hugely-weighted
            // zero-motion constraint, not an error.
            inert.stats.refused_zero_dt += 1;
            warning!(
                "inertial: preintegration returned dt = 0 despite passing coverage; edge dropped: \
                 prev_kf={} kf={} samples={}",
                prev_kf_idx,
                curr_kf_idx,
                raw.len()
            );
            return;
        }
        // Behind the ingest gate (`ImuBufferStats::non_finite`) this should be unreachable too.
        // Checked because the map admits a non-finite factor and `Map::apply_ba_update` then
        // refuses every BA write-back whose window holds both its endpoints, visual included.
        if !preintegration_is_finite(&preint) {
            inert.stats.refused_non_finite += 1;
            warning!(
                "inertial: preintegration is not finite despite passing coverage; edge dropped: \
                 prev_kf={} kf={} samples={}",
                prev_kf_idx,
                curr_kf_idx,
                raw.len()
            );
            return;
        }

        // Both indices are `Frame::idx`, NOT positions in `Map::keyframes` — `ImuFactor` is
        // keyed by frame index everywhere it is read. `apply_insertion` refuses an endpoint
        // the map does not hold, a self-edge, a duplicate edge and a non-increasing interval.
        //
        // `raw` is handed over as well, and kept for the life of the factor: it is what
        // the inertial BA's repropagation re-integrates once the bias has drifted past 0.02
        // from this edge's linearization point. Passing an empty Vec compiles and leaves the
        // edge permanently stuck at a stale linearization.
        let factor = ImuFactor {
            prev_kf_idx,
            curr_kf_idx,
            preintegrated: preint,
            raw_samples: raw,
            t0,
            t1,
        };
        if let Err(e) = self.map.apply_insertion(MapInsertion {
            imu_factors: vec![factor],
            ..MapInsertion::default()
        }) {
            inert.stats.refused_by_map += 1;
            warning!(
                "inertial: the map refused the IMU edge; this keyframe pair stays visual-only: \
                 prev_kf={} kf={} error={}",
                prev_kf_idx,
                curr_kf_idx,
                e.to_string()
            );
            return;
        }
        inert.stats.factors_added += 1;
    }

    /// The staged visual-inertial initialization of Campos et al., ORB-SLAM3, IEEE T-RO 2021:
    /// an initial solve once the window is ready, then a first refinement at 5 s and a second
    /// at 15 s.
    ///
    /// All three stages are needed. kornia-slam's own EuRoC V101 numbers put the gravity error
    /// at 23 degrees after the initial solve, 1.13 after the first refinement and 0.78 after the
    /// second — and gravity error is not a diagnostic here, it is the axis the whole map gets
    /// rotated onto.
    ///
    /// With `enable_inertial_ba` off, `state.imu_initialized` never becomes true (only
    /// `apply_initialization` sets it), so only the initial-solve branch is reachable; it runs
    /// once to a dry acceptance and then stops.
    fn run_inertial_init(&mut self, inert: &mut InertialState, now_sec: f64) {
        let Some(start_idx) = inert.start_kf_idx else {
            return;
        };

        if !self.state.imu_initialized {
            // Checked before the throttle and before `ready`, so that a latched dry run costs
            // nothing per keyframe — not even the keyframe/factor scan `ready` does.
            if inert.dry_run_accepted {
                return;
            }
            // Before the motion solve and unthrottled: it is a pass over one window's samples,
            // not an LM solve. A refusal falls through, so a robot that never stops still
            // initializes the way it always has.
            if self.try_static_init(inert, now_sec) {
                return;
            }
            // Throttled because `try_initialize` runs a 200-iteration LM over a window that
            // never shrinks, so retrying on every keyframe is a cost that grows with the
            // session for a solve that just failed its own gates.
            if inert
                .last_init_attempt_sec
                .is_some_and(|t| now_sec - t < inert.cfg.init_retry.get::<second>())
            {
                return;
            }
            let ready = inert.initializer.ready(&self.map, Some(start_idx));
            // Gates the raw-sample scan inside the line, not just the print. Measured on a
            // Jetson Orin with an OAK-D at 640x400, parked: per keyframe this was 1,920 lines and
            // ~900 kB in 90 s, i.e. ~78 MB/day, each one re-scanning a window of 9,353 samples
            // and growing.
            if should_log_throttled(
                inert.last_window_log,
                now_sec,
                ready,
                inert.cfg.init_retry.get::<second>(),
            ) {
                inert.last_window_log = Some((now_sec, ready));
                self.log_init_window(inert, start_idx, ready);
            }
            if !ready {
                return;
            }
            inert.last_init_attempt_sec = Some(now_sec);
            inert.stats.init_attempts += 1;

            // prior_a = 1e5, the STEREO value: this tracker cannot bootstrap monocularly (see
            // `bootstrap_stereo`), so the 1e10 accelerometer-bias prior the mono path uses to
            // stop scale and bias trading off against each other does not apply. Scale is
            // pinned to 1 by the stereo baseline and the factor zeroes its column.
            let result = inert.initializer.try_initialize(
                &self.map,
                Some(inert.cfg.imu_t_bc),
                inert.bias,
                start_idx,
                1e2,
                1e5,
                false,
            );
            match result {
                Some(init) => {
                    self.accept_inertial_init(inert, init, start_idx, now_sec, InitStage::Initial)
                }
                None => info!(
                    "inertial: initial solve rejected (solve failed, or |accel bias| over 1 m/s^2): start_kf={}",
                    start_idx
                ),
            }
            return;
        }

        if inert.static_hold.is_some() && !self.release_static_hold(inert, now_sec) {
            return;
        }
        let Some(window_start) = inert.window_start_sec else {
            return;
        };
        let init_elapsed = now_sec - window_start;
        // Past 50 s the window is far too large for a single joint solve to be worth attempting;
        // this mirrors the removed `SlamPipeline`'s own ceiling.
        if init_elapsed >= 50.0 {
            return;
        }
        let (prior_g, stage) = if !inert.stats.first_refinement_done && init_elapsed > 5.0 {
            (1.0, InitStage::Refine1)
        } else if inert.stats.first_refinement_done
            && !inert.stats.second_refinement_done
            && init_elapsed > 15.0
        {
            (0.0, InitStage::Refine2)
        } else {
            return;
        };

        inert.stats.init_attempts += 1;
        // `already_initialized = true` changes the seeding, not just the priors: velocities come
        // from the keyframes instead of finite differences, and Rwg seeds at the fixed
        // gI -> (0, +G, 0) rotation rather than identity.
        let result = inert.initializer.try_initialize(
            &self.map,
            Some(inert.cfg.imu_t_bc),
            inert.bias,
            start_idx,
            prior_g,
            1e5,
            true,
        );
        self.finish_refinement(inert, result, start_idx, now_sec, stage);
    }

    /// The stationary initialization: finds a still window ending at the newest keyframe and
    /// initializes from it through [`Self::accept_inertial_init`], the same gate as the motion
    /// solve (applied with `enable_inertial_ba`, a dry run without).
    ///
    /// Returns whether it settled the question for this window: applied, or accepted dry. `false`
    /// (no `static_init` config, no still window, or a refusal by kornia-slam or the map) leaves
    /// the motion solve to run.
    ///
    /// Applied, it writes zero velocity from the still window on and finite-difference velocities
    /// to the initialization window's earlier keyframes ([`Self::pre_window_velocities`]), and
    /// holds the refinements until the robot moves ([`InertialState::static_hold`]). A map
    /// refusal is not retried for `init_retry` ([`InertialState::static_map_refused_at`]).
    fn try_static_init(&mut self, inert: &mut InertialState, now_sec: f64) -> bool {
        let Some(cfg) = inert.cfg.static_init else {
            return false;
        };
        let (Some(start_idx), Some(newest)) = (inert.start_kf_idx, self.state.last_keyframe_idx)
        else {
            return false;
        };
        if inert
            .static_map_refused_at
            .is_some_and(|t| now_sec - t < inert.cfg.init_retry.get::<second>())
        {
            return false;
        }
        let found = self
            .still_window(newest, start_idx, &cfg)
            .map_err(StaticRefusal::NotStill)
            .and_then(|w| {
                inert
                    .initializer
                    .initialize_stationary(
                        &self.map,
                        inert.cfg.imu_t_bc,
                        inert.bias,
                        w.first_kf,
                        &w.stats,
                        cfg.gravity_tolerance,
                    )
                    .map(|init| (w, init))
                    .map_err(StaticRefusal::Init)
            });
        let (window, mut init) = match found {
            Ok(found) => found,
            Err(reason) => {
                inert.stats.static_refused += 1;
                if should_log_throttled(
                    inert.last_static_log,
                    now_sec,
                    false,
                    inert.cfg.init_retry.get::<second>(),
                ) {
                    inert.last_static_log = Some((now_sec, false));
                    info!(
                        "inertial: no still window for the stationary initialization: kf={} reason={}",
                        newest,
                        reason.to_string()
                    );
                }
                return false;
            }
        };
        inert.stats.static_accepted += 1;
        info!(
            "inertial: still window found: first_kf={} kf={} span_s={} samples={} accel_rms={} \
             gyro_rms={} gyro_mean={}",
            window.first_kf,
            newest,
            window.span_s,
            window.stats.samples,
            window.stats.accel_rms_dev,
            window.stats.gyro_rms_dev,
            window.stats.mean_gyro.length()
        );
        // Keyframes of the initialization window before the still one get the velocity their
        // visual track shows, not the zero `initialize_stationary` writes only from the still
        // window on: their IMU edges stay in VI-BA, and an edge spanning real motion between two
        // keyframes that both claim rest is a residual the size of that speed.
        let before = self.pre_window_velocities(start_idx, window.first_kf);
        let apply_from = if before.is_empty() {
            window.first_kf
        } else {
            start_idx
        };
        init.velocities_world.splice(0..0, before);
        #[cfg(test)]
        if inert.refuse_static_alignment {
            init.velocities_world.clear();
        }
        self.accept_inertial_init(inert, init, apply_from, now_sec, InitStage::Static);
        if self.state.imu_initialized {
            inert.static_hold = Some(newest);
            inert.stats.first_refinement_done = false;
            inert.stats.second_refinement_done = false;
        } else if !inert.dry_run_accepted {
            inert.static_map_refused_at = Some(now_sec);
        }
        self.state.imu_initialized || inert.dry_run_accepted
    }

    /// Velocities, in the current world, for the keyframes in `[start_idx, first_still)` in map
    /// order, by finite differences of their camera centres over the keyframe stamps the IMU
    /// edges carry (central where both neighbours are known, one-sided at the ends; the still
    /// window's first keyframe is the right neighbour of the last). A keyframe with no stamped
    /// neighbour gets zero. Empty when the still window starts at the initialization window.
    ///
    /// The camera centre's velocity stands in for the IMU's, as in [`Self::carried_velocity`].
    fn pre_window_velocities(&self, start_idx: usize, first_still: usize) -> Vec<Vec3F64> {
        if first_still <= start_idx {
            return Vec::new();
        }
        let mut stamps: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for f in self.map.imu_factors() {
            stamps.insert(f.prev_kf_idx, f.t0);
            stamps.insert(f.curr_kf_idx, f.t1);
        }
        let kfs: Vec<(usize, Vec3F64)> = self
            .map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx && kf.frame.idx <= first_still)
            .map(|kf| {
                (
                    kf.frame.idx,
                    kf.frame.pose_world_to_cam.inverse().translation,
                )
            })
            .collect();
        let at = |i: usize| {
            kfs.get(i)
                .and_then(|(idx, c)| Some((*stamps.get(idx)?, *c)))
        };
        kfs.iter()
            .enumerate()
            .take_while(|(_, (idx, _))| *idx < first_still)
            .map(|(i, _)| {
                let prev = i.checked_sub(1).and_then(at).or_else(|| at(i));
                let next = at(i + 1).or_else(|| at(i));
                match (prev, next) {
                    (Some((t0, c0)), Some((t1, c1))) if t1 - t0 > 1e-3 => (c1 - c0) / (t1 - t0),
                    _ => Vec3F64::ZERO,
                }
            })
            .collect()
    }

    /// Whether the refinements held by a stationary initialization may run: the camera has moved
    /// [`STATIC_REFINE_MOTION_M`] or turned [`STATIC_REFINE_MOTION_RAD`] from the still window's
    /// newest keyframe. On release the refinement window is re-armed at the newest keyframe's
    /// predecessor (the edge into it spans the start of the motion), so the 5 s / 15 s schedule
    /// counts from when the robot started moving and the solve does not span the whole stop.
    ///
    /// Returns `true` only on the keyframe that releases it; that keyframe is the window's
    /// second, so the refinement branch it falls into has nothing scheduled yet anyway.
    fn release_static_hold(&self, inert: &mut InertialState, now_sec: f64) -> bool {
        let Some(hold) = inert.static_hold else {
            return true;
        };
        let Some(newest) = self.map.keyframes().last() else {
            return false;
        };
        if let Some(held) = self.map.get_keyframe(hold) {
            let (p0, p1) = (held.frame.pose_world_to_cam, newest.frame.pose_world_to_cam);
            let moved = (p1.inverse().translation - p0.inverse().translation).length();
            let r = p1.rotation * p0.rotation.transpose();
            let cos = ((r.x_axis.x + r.y_axis.y + r.z_axis.z) - 1.0) * 0.5;
            let turned = cos.clamp(-1.0, 1.0).acos();
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN pose is not motion
            if !(moved >= STATIC_REFINE_MOTION_M || turned >= STATIC_REFINE_MOTION_RAD) {
                return false;
            }
        }
        // Held keyframe gone (a reset clears the hold, so this is defensive): release as well.
        let newest_idx = newest.frame.idx;
        let (start, t0) = self
            .map
            .imu_factors()
            .iter()
            .rev()
            .find(|f| f.curr_kf_idx == newest_idx)
            .map_or((newest_idx, now_sec), |f| (f.prev_kf_idx, f.t0));
        inert.static_hold = None;
        inert.start_kf_idx = Some(start);
        inert.window_start_sec = Some(t0);
        info!(
            "inertial: motion after the stationary initialization; refinements scheduled from \
             here: held_kf={} start_kf={} kf={}",
            hold, start, newest_idx
        );
        true
    }

    /// The newest run of contiguous IMU edges ending at keyframe `newest` that spans at least
    /// `cfg.window`, if every stillness check passes over it.
    ///
    /// Contiguous by construction: each edge is found by the keyframe the previous one started
    /// at, so a refused edge (a gap, dropped samples) ends the run, and so does the window start
    /// (a re-bootstrap has no edge into its anchor). The samples are each edge's own retained
    /// raw samples, exactly what the factors integrated.
    fn still_window(
        &self,
        newest: usize,
        start_idx: usize,
        cfg: &StaticInitConfig,
    ) -> Result<StillWindow, NotStill> {
        let want_s = cfg.window.get::<second>();
        let factors = self.map.imu_factors();
        let mut edges: Vec<&ImuFactor> = Vec::new();
        let mut span_s = 0.0;
        let mut kf = newest;
        while span_s < want_s {
            let Some(edge) = factors
                .iter()
                .rev()
                .find(|f| f.curr_kf_idx == kf && f.prev_kf_idx >= start_idx)
            else {
                break;
            };
            span_s += edge.t1 - edge.t0;
            kf = edge.prev_kf_idx;
            edges.push(edge);
        }
        edges.reverse();
        let too_short = NotStill::TooShort { span_s, want_s };
        let first = edges.first().ok_or(too_short)?;
        // Adjacent edges share their boundary sample; count it once.
        let mut last_t = f64::NEG_INFINITY;
        let samples = edges.iter().flat_map(|e| e.raw_samples.iter()).filter(|m| {
            let fresh = m.timestamp > last_t;
            last_t = last_t.max(m.timestamp);
            fresh
        });
        let stats = ImuWindowStats::from_samples(samples).ok_or(too_short)?;
        check_imu(span_s, &stats, cfg)?;

        if let Some(w) = self.wheel.as_ref() {
            let epoch = self.epoch_ns.unwrap_or(0);
            let ns = |t: f64| epoch + (t * 1e9).round() as u64;
            let last = edges.last().expect("first exists");
            let delta = w.delta(ns(first.t0), ns(last.t1), w.cfg.max_gap_ns()).ok();
            check_wheels(delta, cfg)?;
        }

        let poses: Vec<Pose3d> = std::iter::once(first.prev_kf_idx)
            .chain(edges.iter().map(|e| e.curr_kf_idx))
            .filter_map(|idx| self.map.get_keyframe(idx))
            .map(|kf| kf.frame.pose_world_to_cam)
            .collect();
        check_camera(&poses, cfg)?;

        Ok(StillWindow {
            first_kf: first.prev_kf_idx,
            span_s,
            stats,
        })
    }

    /// The anchor velocity for a carried alignment, measured at the bootstrap `stamp_ns`: the
    /// camera's velocity over the last [`CARRY_VELOCITY_WINDOW_NS`] of wheel odometry, in the
    /// world through `pose_world_to_cam`. `None` without a wheel feed or without readings that
    /// cover the interval.
    ///
    /// The camera's velocity stands in for the IMU's: they differ by the angular rate times the
    /// IMU-to-camera lever arm (centimetres on a stereo head), well inside what one VI-BA solve
    /// corrects. What it must not be is the stale velocity from before the drop.
    fn carried_velocity(&mut self, stamp_ns: u64, pose_world_to_cam: Pose3d) -> Option<Vec3F64> {
        let w = self.wheel.as_mut()?;
        let t0 = stamp_ns.checked_sub(CARRY_VELOCITY_WINDOW_NS)?;
        // `M` takes world-to-camera at t0 to t1: `T_c1w = M T_c0w`, so `M^-1` takes camera-1
        // coordinates to camera-0 and its translation is camera 1's centre in camera 0.
        let c0_c1 = w.camera_motion(t0, stamp_ns)?.inverse();
        let r_wc1 = pose_world_to_cam.inverse().rotation;
        let r_wc0 = r_wc1 * c0_c1.rotation.transpose();
        let v = r_wc0 * c0_c1.translation / (CARRY_VELOCITY_WINDOW_NS as f64 * 1e-9);
        (v.x.is_finite() && v.y.is_finite() && v.z.is_finite()).then_some(v)
    }

    /// Gives up a carried alignment whose anchor velocity could not be measured: the inertial
    /// path initializes again (still window or motion solve), which rotates the world and bumps
    /// the generation, as every loss did before the carry existed.
    fn drop_carried_alignment(&mut self, frame_idx: usize) {
        let Some(inert) = self.inertial.as_mut() else {
            return;
        };
        inert.carried_alignment = false;
        inert.stats.carry_dropped += 1;
        warning!(
            "inertial: carried alignment dropped, no wheel velocity at the re-bootstrap; the \
             inertial path initializes again: frame={} wheel_feed={}",
            frame_idx,
            self.wheel.is_some()
        );
    }

    /// Adopts an alignment a map drop carried: the new map's anchor keyframe was seeded in a
    /// world that is still gravity-aligned, at `carried`, the velocity measured at the bootstrap,
    /// so the tracker is initialized from its first keyframe with no rotation and no extra
    /// world-generation bump.
    ///
    /// The refinements are marked spent: each one rotates the world, and a carried world is
    /// already the product of the initialization (and refinements) that aligned it.
    fn adopt_carried_alignment(&mut self, carried: Option<Vec3F64>, timestamp_sec: f64) {
        let Some(inert) = self.inertial.as_mut() else {
            return;
        };
        let Some(velocity) = carried else {
            return;
        };
        inert.carried_alignment = false;
        inert.static_hold = None;
        self.state.imu_init_timestamp_sec = Some(timestamp_sec);
        inert.stats.first_refinement_done = true;
        inert.stats.second_refinement_done = true;
        info!(
            "inertial: alignment carried into the new map; gravity and bias kept, no \
             re-initialization: speed_mps={} gyro_bias={}",
            velocity.length(),
            inert.bias.gyro.length()
        );
    }

    /// Applies a refinement's outcome and marks the stage done WHATEVER it was — rejected by the
    /// solve, or accepted and then refused by the map. The stage is scheduled once, as in the
    /// ORB-SLAM3 reference implementation, and a failed refinement that retried every keyframe
    /// would be the unthrottled solve the initial-solve path is careful to avoid. A failure
    /// leaves the tracker on the previous initialization.
    fn finish_refinement(
        &mut self,
        inert: &mut InertialState,
        result: Option<ImuInitResult>,
        start_idx: usize,
        now_sec: f64,
        stage: InitStage,
    ) {
        match result {
            Some(init) => self.accept_inertial_init(inert, init, start_idx, now_sec, stage),
            None => info!("inertial: refinement rejected: stage={}", stage.as_str()),
        }
        if stage == InitStage::Refine1 {
            inert.stats.first_refinement_done = true;
        } else {
            inert.stats.second_refinement_done = true;
        }
    }

    /// What the initialization window looks like, at every `ready` evaluation and on BOTH of its
    /// answers. Observation only; nothing here gates anything.
    ///
    /// New measurement, not a restatement of the gate: no deployed gate (kornia-slam 9d3a927
    /// `imu_init.rs:93-140` — >=10 kfs, >=1 s dt, >=0.05 m of VISUAL displacement, which VO drift
    /// satisfies on a parked robot) can see whether the IMU moved, and `ready` answers a bare
    /// bool its caller drops. `info!` because a production log filter typically drops debug,
    /// and this line is the only record of why the initializer is still waiting.
    ///
    /// Throttled by the caller on [`InertialState::last_window_log`]. Each call is one line AND
    /// one pass over the window's raw samples, so the caller gates both; do not add further
    /// per-sample cost here.
    fn log_init_window(&self, inert: &InertialState, start_idx: usize, ready: bool) {
        // Scoped exactly as `ImuInitializer::ready` scopes its own — keyframes by `frame.idx`,
        // factors by `curr_kf_idx` — so the line explains THAT verdict and not a differently
        // scoped one.
        let kfs: Vec<&Keyframe> = self
            .map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        let factors = || {
            self.map
                .imu_factors()
                .iter()
                .filter(|f| f.curr_kf_idx >= start_idx)
        };
        let imu_dt_s: f64 = factors().map(|f| f.preintegrated.dt).sum();
        let disp_m = match (kfs.first(), kfs.last()) {
            (Some(first), Some(last)) => (last.frame.pose_world_to_cam.inverse().translation
                - first.frame.pose_world_to_cam.inverse().translation)
                .length(),
            _ => 0.0,
        };
        let exc = self.init_window_excitation(start_idx);
        // Which of `ready`'s two time thresholds applied: 1 s stereo, 2 s mono.
        let stereo = kfs.first().is_some_and(|kf| kf.frame.is_stereo());
        info!(
            "inertial: init window: ready={} start_kf={} kfs={} imu_dt_s={} disp_m={} \
             rot_exc_rad={} accel_var={} imu_samples={} stereo={} init_retry_sec={}",
            ready,
            start_idx,
            kfs.len(),
            imu_dt_s,
            disp_m,
            exc.rot_rad,
            exc.accel_var,
            exc.samples,
            stereo,
            inert.cfg.init_retry.get::<second>()
        );
    }

    /// Excitation over exactly the factors `ImuInitializer::ready` counts for its summed-dt gate:
    /// `curr_kf_idx >= start_idx`. Split out of the log line so the scoping is testable — a
    /// window that swept in pre-bootstrap factors would read excited for motion that predates it.
    fn init_window_excitation(&self, start_idx: usize) -> WindowExcitation {
        window_excitation(
            self.map
                .imu_factors()
                .iter()
                .filter(|f| f.curr_kf_idx >= start_idx)
                .map(|f| f.raw_samples.as_slice()),
        )
    }

    /// The one place an accepted initialization is allowed to touch the map, and the gate on it.
    ///
    /// `enable_inertial_ba` off means the result is counted and logged but NOT applied. This is
    /// the whole content of the flag's "cannot move a published pose" promise, and it is not
    /// redundant with the local-BA selection in `on_keyframe_inertial`: `apply_initialization`
    /// scales and rotates every keyframe and map point (`Map::apply_inertial_alignment`), then
    /// the world generation goes up, which `task.rs` publishes as `VioStatus::reset_epoch` and
    /// which every consumer answers by wiping its trail and re-deriving its anchor. Without this
    /// gate, a yaw-unverified extrinsic with the flag off would produce that wipe at the initial
    /// solve (~2-5 s in) and at both refinements (>5 s, >15 s) — three frame snaps per run, on
    /// a guess.
    ///
    /// What the dry run keeps: `init_attempts`, `init_accepted` (so `init N/M accepted,
    /// initialized=false` in the stop line reads as "the gates pass, nothing was applied"),
    /// and a `warn!` carrying the scale, gravity and bias the solve produced — the numbers a
    /// caller compares against the static gravity read to decide whether the yaw candidate
    /// is the right one. With the wrong candidate the gyro/visual inconsistency lands in the
    /// bias estimate, so a large `gyro_bias` here is the symptom to look for.
    fn accept_inertial_init(
        &mut self,
        inert: &mut InertialState,
        init: ImuInitResult,
        start_idx: usize,
        now_sec: f64,
        stage: InitStage,
    ) {
        // An implausible bias is a failed solve whatever its gates said (with a wrong extrinsic
        // the gyro/visual inconsistency lands exactly there): refused before anything is
        // applied or counted as accepted, so the map, the world and the bias stay as they were.
        if !inert.cfg.bias_limit.admits(&init.bias) {
            inert.stats.bias_rejected += 1;
            warning!(
                "inertial: initialization refused, its IMU bias is outside the plausible range; \
                 map left unchanged: stage={} gyro_bias={} accel_bias={} gyro_limit={} \
                 accel_limit={}",
                stage.as_str(),
                init.bias.gyro.length(),
                init.bias.accel.length(),
                inert.cfg.bias_limit.gyro_rad_s,
                inert.cfg.bias_limit.accel_m_s2
            );
            return;
        }
        if inert.cfg.enable_inertial_ba {
            self.apply_inertial_init(inert, init, start_idx, now_sec, stage);
            return;
        }
        inert.stats.init_accepted += 1;
        inert.dry_run_accepted = true;
        warning!(
            "inertial: initialization ACCEPTED but NOT applied (enable_inertial_ba is off); \
             the map, the world generation and the bias are untouched: stage={} scale={} \
             gravity_x={} gravity_y={} gravity_z={} gyro_bias={} accel_bias={}",
            stage.as_str(),
            init.scale,
            init.gravity_world.x,
            init.gravity_world.y,
            init.gravity_world.z,
            init.bias.gyro.length(),
            init.bias.accel.length()
        );
    }

    /// Moves every held covariance into the world an inertial alignment just produced: the
    /// alignment is one rigid motion of the world (`T_new_old`, read off `reference`, a keyframe's
    /// world-to-camera pose before it, against the same keyframe now), and a world left
    /// perturbation crosses a change of world by `Ad(T_new_old)`. Its scale (one, to the
    /// stereo baseline's accuracy) is not applied.
    fn realign_covariances(&mut self, reference: Option<(usize, Pose3d)>) {
        let Some((idx, old_cw)) = reference else {
            return;
        };
        let Some(new_cw) = self
            .map
            .get_keyframe(idx)
            .map(|kf| kf.frame.pose_world_to_cam)
        else {
            return;
        };
        let ad = covariance::pose_adjoint(&new_cw.inverse().compose(&old_cw));
        for c in self.kf_cov.values_mut() {
            *c = covariance::congruence(&ad, c);
        }
        if let Some(o) = self.coast_origin.as_mut() {
            // Both world covariances; the motion since the measured origin is the base's own.
            o.cov = covariance::congruence(&ad, &o.cov);
            o.accumulated = covariance::congruence(&ad, &o.accumulated);
        }
    }

    /// Commits an accepted initialization to the map and bumps the world generation.
    ///
    /// Only reachable through [`Self::accept_inertial_init`] with `enable_inertial_ba` on.
    fn apply_inertial_init(
        &mut self,
        inert: &mut InertialState,
        init: ImuInitResult,
        start_idx: usize,
        now_sec: f64,
        stage: InitStage,
    ) {
        let (scale, gravity, bias) = (init.scale, init.gravity_world, init.bias);
        // The keyframe tracking resumes from, and its centre before the alignment moves it.
        let pivot = self
            .map_carried
            .then(|| {
                self.map
                    .keyframes()
                    .iter()
                    .rfind(|kf| kf.frame.idx >= start_idx)
                    .map(|kf| {
                        (
                            kf.frame.idx,
                            kf.frame.pose_world_to_cam.inverse().translation,
                        )
                    })
            })
            .flatten();
        // A keyframe's pose before the alignment moves the world, to read off afterwards what the
        // alignment did to it, so the covariances follow (`realign_covariances`).
        let reference = self
            .map
            .keyframes()
            .last()
            .map(|kf| (kf.frame.idx, kf.frame.pose_world_to_cam));
        let applied = inert.initializer.apply_initialization(
            &mut self.map,
            &mut inert.bias,
            &mut inert.gravity_world,
            init,
            start_idx,
        );
        match applied {
            Ok(Some(mut aligned)) => {
                inert.applied_stage = Some(stage);
                if let Some((idx, centre)) = pivot {
                    aligned.pose_world_to_cam =
                        self.repivot_map(idx, centre, aligned.pose_world_to_cam);
                }
                self.realign_covariances(reference);
                self.state.adopt_inertial_initialization(aligned)
            }
            // `apply_inertial_alignment` refuses an empty velocity list, and a non-empty one
            // names a window keyframe, so `Ok` with no keyframe to resume from should be
            // unreachable. If it happens the map HAS been scaled and rotated (and the bias and
            // gravity taken), so the generation must still move — but with no aligned pose or
            // velocity to resume from, the tracker does not claim to be initialized.
            // kornia-slam's own `adopt_inertial_initialization(None)` would mark it initialized
            // anyway; a tracker that switched to inertial BA without a velocity is worse than
            // one that retries.
            Ok(None) => {
                self.world_generation += 1;
                inert.stats.init_apply_refused += 1;
                error!(
                    "inertial: initialization moved the map but named no keyframe to resume \
                     from; staying visual-only: stage={} start_kf={} world_generation={}",
                    stage.as_str(),
                    start_idx,
                    self.world_generation
                );
                return;
            }
            // Refused before any write: map, bias and gravity are as they were, so nothing
            // moved and the generation stays. A refused initial solve leaves the tracker
            // visual-only until the next accepted one (`init_retry`); a refused refinement leaves
            // it on the previous initialization, and the stage is not retried.
            Err(e) => {
                inert.stats.init_apply_refused += 1;
                let outcome = if self.state.imu_initialized {
                    "staying on the previous initialization, stage not retried"
                } else {
                    "staying visual-only until the next accepted solve"
                };
                warning!(
                    "inertial: the map refused the initialization, map left unchanged; {}: \
                     stage={} scale={} error={}",
                    outcome,
                    stage.as_str(),
                    scale,
                    e.to_string()
                );
                return;
            }
        }
        // `apply_initialization` scales and rotates the whole map (`Map::apply_inertial_alignment`):
        // every keyframe pose and every map point moves, so that gravity lands on world +y. Poses
        // published before this instant and after it are in DIFFERENT frames, and the only
        // thing that says so on the wire is this counter — the same contract a requested reset
        // and a loss re-bootstrap already use. Consumers that map tracker poses into another
        // frame re-derive their anchor when it changes.
        self.world_generation += 1;
        self.state.imu_init_timestamp_sec = Some(now_sec);
        inert.stats.init_accepted += 1;
        warning!(
            "inertial: initialization applied; the map has been rotated onto gravity: stage={} \
             scale={} gravity_x={} gravity_y={} gravity_z={} gyro_bias={} accel_bias={} \
             world_generation={}",
            stage.as_str(),
            scale,
            gravity.x,
            gravity.y,
            gravity.z,
            bias.gyro.length(),
            bias.accel.length(),
            self.world_generation
        );
    }

    /// Translates the whole map so keyframe `idx`'s centre is back at `centre`, and returns
    /// `pose` (that keyframe's aligned pose) moved the same way.
    ///
    /// `apply_initialization` scales and rotates about the world ORIGIN. A map that restarted at
    /// identity sits on it and loses nothing, but a carried map sits wherever the robot drove:
    /// at `c` metres out, a gravity re-estimate of 1 deg moves the pose ~`0.017 c` and a scale
    /// of `s` moves it `(s - 1) c`, which grows with every drop. Gravity still lands on +y, since
    /// a translation does not rotate. On a refusal the map stays aligned about the origin.
    fn repivot_map(&mut self, idx: usize, centre: Vec3F64, pose: Pose3d) -> Pose3d {
        let Some(kf) = self.map.get_keyframe(idx) else {
            return pose;
        };
        let shift = centre - kf.frame.pose_world_to_cam.inverse().translation;
        // `X_w' = X_w + shift`, so `X_c = R (X_w' - shift) + t`.
        let moved = |p: &Pose3d| Pose3d::new(p.rotation, p.translation - p.rotation * shift);
        let (indices, before): (Vec<usize>, Vec<Pose3d>) = self
            .map
            .keyframes()
            .iter()
            .map(|kf| (kf.frame.idx, kf.frame.pose_world_to_cam))
            .unzip();
        let after: Vec<Pose3d> = before.iter().map(moved).collect();
        match self
            .map
            .apply_pose_graph_correction(&indices, &before, &after)
        {
            Ok(_) => moved(&pose),
            Err(e) => {
                warning!(
                    "inertial: could not re-centre the carried map after its alignment, the pose                      jumps by {} m: error={}",
                    shift.length(),
                    e.to_string()
                );
                pose
            }
        }
    }

    /// Inserts each seed as a new landmark, returning how many the map accepted.
    ///
    /// One at a time rather than as one `apply_insertion` batch: a batch is all-or-nothing, and
    /// one refused seed must cost one point, not the whole keyframe's worth. Every seed here
    /// references a free feature of a keyframe the map holds at a finite position, so a refusal
    /// is an invariant break; each is counted in [`Tracker::map_writes_refused`] and the
    /// total is logged. The map computes each new landmark's descriptor and scale geometry as it
    /// links it, from the reference keyframe it now holds.
    fn insert_landmarks(
        &mut self,
        seeds: impl IntoIterator<Item = LandmarkSeed>,
        site: &'static str,
    ) -> usize {
        let (mut inserted, mut refused) = (0usize, 0u64);
        for seed in seeds {
            match self.map.insert_landmark(seed) {
                Ok(_) => inserted += 1,
                Err(e) => {
                    refused += 1;
                    debug!(
                        "map refused a landmark: site={} kf={} feature={} error={}",
                        site,
                        seed.reference.keyframe_idx,
                        seed.reference.feature_idx,
                        e.to_string()
                    );
                }
            }
        }
        if refused > 0 {
            self.map_writes_refused += refused;
            warning!(
                "map refused landmark insertions: site={} refused={} inserted={}",
                site,
                refused,
                inserted
            );
        }
        inserted
    }

    /// Back-projects keyframe `curr_kf_idx`'s unassociated *close* (`z <= close_depth_threshold`)
    /// stereo keypoints into new metric map points. Far points are left to multi-view
    /// triangulation in the growth pass.
    fn add_close_stereo_points(&mut self, curr_kf_idx: usize, close_depth_threshold: f64) -> usize {
        let seeds: Vec<LandmarkSeed> = {
            let Some(curr_kf) = self.map.get_keyframe(curr_kf_idx) else {
                return 0;
            };
            let cam_points = unproject_stereo(&curr_kf.frame, &self.camera);
            if cam_points.is_empty() {
                return 0;
            }
            let pose_inv = curr_kf.frame.pose_world_to_cam.inverse();
            cam_points
                .iter()
                .filter(|(desc_idx, p_cam)| {
                    p_cam.z <= close_depth_threshold && curr_kf.map_point(*desc_idx).is_none()
                })
                .map(|(desc_idx, p_cam)| LandmarkSeed {
                    position: pose_inv.transform_point(p_cam),
                    color: keypoint_color(&curr_kf.frame, *desc_idx),
                    reference: ObservationKey {
                        keyframe_idx: curr_kf_idx,
                        feature_idx: *desc_idx,
                    },
                })
                .collect()
        };
        self.insert_landmarks(seeds, "stereo_densification")
    }

    /// Triangulates new map points from the unassociated features shared by
    /// `prev_kf_idx` and `curr_kf_idx`, gated by the pose-derived epipolar geometry
    /// (the triangulation search of Campos et al., ORB-SLAM3, IEEE T-RO 2021).
    fn grow_map_points_from_keyframe_pair(
        &mut self,
        prev_kf_idx: usize,
        curr_kf_idx: usize,
        match_config: OrbMatchConfig,
        triangulation_config: &TriangulationConfig,
    ) -> usize {
        // Read-only phase; the shared borrow of `self.map` ends with this block. Each entry is
        // `(position, color, prev_idx, curr_idx)`.
        let points: Vec<(Vec3F64, [u8; 3], usize, usize)> = {
            let (Some(prev_kf), Some(curr_kf)) = (
                self.map.get_keyframe(prev_kf_idx),
                self.map.get_keyframe(curr_kf_idx),
            ) else {
                return 0;
            };

            // Matching the full descriptor arrays and filtering afterwards
            // discards almost everything once the keyframes are mature: the best
            // matches always land on already-tracked features.
            let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
                .filter(|&i| prev_kf.map_point(i).is_none())
                .collect();
            let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
                .filter(|&i| curr_kf.map_point(i).is_none())
                .collect();
            if prev_unassoc.is_empty() || curr_unassoc.is_empty() {
                return 0;
            }

            let rel = Pose3d::between(
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
            );
            if rel.translation.length() <= 1e-8 {
                // No baseline: epipolar geometry degenerates and triangulation
                // would reject everything anyway.
                return 0;
            }
            let t = rel.translation;
            let t_skew = Mat3F64::from_cols(
                Vec3F64::new(0.0, t.z, -t.y),
                Vec3F64::new(-t.z, 0.0, t.x),
                Vec3F64::new(t.y, -t.x, 0.0),
            );
            let camera = &self.camera;
            // K^-1 built by hand: no distortion terms, valid only because the
            // camera is rectified (checked in `validate_camera`).
            let k_inv = Mat3F64::from_cols(
                Vec3F64::new(1.0 / camera.fx, 0.0, 0.0),
                Vec3F64::new(0.0, 1.0 / camera.fy, 0.0),
                Vec3F64::new(-camera.cx / camera.fx, -camera.cy / camera.fy, 1.0),
            );
            let f_mat = k_inv.transpose() * (t_skew * rel.rotation) * k_inv;

            // Near the epipole every keypoint is close to every epipolar line,
            // so the chi-square gate is uninformative there: wrong matches
            // survive and triangulate to depth-garbage that still reprojects
            // well in both views.
            let prev_center_world = prev_kf.frame.pose_world_to_cam.inverse().translation;
            let epipole_cam = curr_kf
                .frame
                .pose_world_to_cam
                .transform_point(&prev_center_world);
            let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
                Vec2F64::new(
                    camera.fx * epipole_cam.x / epipole_cam.z + camera.cx,
                    camera.fy * epipole_cam.y / epipole_cam.z + camera.cy,
                )
            });

            let prev_orients: Vec<f32> = prev_unassoc
                .iter()
                .map(|&i| prev_kf.frame.features.orientations[i])
                .collect();
            let prev_descs: Vec<[u8; 32]> = prev_unassoc
                .iter()
                .map(|&i| prev_kf.frame.features.descriptors[i])
                .collect();
            let curr_orients: Vec<f32> = curr_unassoc
                .iter()
                .map(|&i| curr_kf.frame.features.orientations[i])
                .collect();
            let curr_descs: Vec<[u8; 32]> = curr_unassoc
                .iter()
                .map(|&i| curr_kf.frame.features.descriptors[i])
                .collect();

            let sub_matches = match_orb_descriptors(
                &prev_orients,
                &prev_descs,
                &curr_orients,
                &curr_descs,
                match_config,
            );

            let mut pair_indices: Vec<(usize, usize)> = Vec::new();
            let mut matched_prev: Vec<Vec2F64> = Vec::new();
            let mut matched_curr: Vec<Vec2F64> = Vec::new();
            // `match_orb_descriptors` is ONE-DIRECTIONAL: it loops over side 1
            // (prev), takes the best j on side 2 (curr) and ratio-tests on the
            // prev side only. There is no cross-check and no uniqueness on the
            // curr side, so several `prev_idx` can name the same `curr_idx` —
            // and on a repeated pattern (tiled floor, brick, window mullions)
            // they all lie on the shared epipolar line by construction, so the
            // chi-square gate below keeps every one of them. The `curr_kf
            // .map_point(curr_idx).is_some()` guard further down runs *before*
            // the write phase, so it cannot see duplicates created in the same
            // batch: two distinct 3D points at different depths would be built
            // from one measurement. kornia-slam's map now refuses the second
            // link to an occupied feature (`FeatureOccupied`), so the loser
            // would no longer desync the two association directions — but it
            // would still be a wasted triangulation and a counted refusal, so
            // claim each `curr_idx` once here.
            let mut claimed_curr: HashSet<usize> = HashSet::new();
            for (prev_sub, curr_sub) in sub_matches {
                let (Some(&prev_idx), Some(&curr_idx)) =
                    (prev_unassoc.get(prev_sub), curr_unassoc.get(curr_sub))
                else {
                    continue;
                };
                if !claimed_curr.insert(curr_idx) {
                    continue;
                }
                let (Some(pu), Some(qu)) = (
                    prev_kf.frame.undistorted_xy(prev_idx, camera),
                    curr_kf.frame.undistorted_xy(curr_idx, camera),
                ) else {
                    continue;
                };
                let p = Vec2F64::new(pu[0] as f64, pu[1] as f64);
                let q = Vec2F64::new(qu[0] as f64, qu[1] as f64);

                let octave = curr_kf
                    .frame
                    .features
                    .octaves
                    .get(curr_idx)
                    .copied()
                    .unwrap_or(0);

                if let Some(e) = epipole_px {
                    let dx = q.x - e.x;
                    let dy = q.y - e.y;
                    if dx * dx + dy * dy < 100.0 * ORB_SCALE_FACTOR.powi(octave as i32) {
                        continue;
                    }
                }

                let l = f_mat * Vec3F64::new(p.x, p.y, 1.0);
                let line_norm_sq = l.x * l.x + l.y * l.y;
                if line_norm_sq <= 1e-12 {
                    continue;
                }
                let d = l.x * q.x + l.y * q.y + l.z;
                let sigma_sq = ORB_SCALE_FACTOR.powi(2 * octave as i32);
                if d * d > EPIPOLAR_CHI2 * sigma_sq * line_norm_sq {
                    continue;
                }

                pair_indices.push((prev_idx, curr_idx));
                matched_prev.push(p);
                matched_curr.push(q);
            }
            if pair_indices.len() < MIN_GROWTH_MATCHES {
                return 0;
            }

            let triangulated = match triangulate_matched_points(
                &matched_prev,
                &matched_curr,
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
                camera,
                triangulation_config,
            ) {
                Ok(pts) => pts,
                Err(e) => {
                    debug!("grow: triangulation failed: {}", e.to_string());
                    return 0;
                }
            };

            let mut points = Vec::new();
            for tp in &triangulated {
                let Some(&(prev_idx, curr_idx)) = pair_indices.get(tp.pair_index) else {
                    continue;
                };
                if curr_kf.map_point(curr_idx).is_some() {
                    continue;
                }
                let color = keypoint_color(&curr_kf.frame, curr_idx);
                points.push((tp.position, color, prev_idx, curr_idx));
            }
            points
        };

        // Write phase. `curr_kf` is each new point's reference (first) observer, as ORB-SLAM3
        // references a new point to the newer keyframe; the neighbour is linked as the second
        // observer, without which every new point has a single observation, which biases the
        // scale/normal geometry and makes culling over-aggressive.
        let (mut added, mut refused) = (0usize, 0u64);
        for (position, color, prev_idx, curr_idx) in points {
            let seed = LandmarkSeed {
                position,
                color,
                reference: ObservationKey {
                    keyframe_idx: curr_kf_idx,
                    feature_idx: curr_idx,
                },
            };
            let mp_idx = match self.map.insert_landmark(seed) {
                Ok(mp_idx) => mp_idx,
                Err(e) => {
                    refused += 1;
                    debug!(
                        "grow: map refused a landmark: kf={} feature={} error={}",
                        curr_kf_idx,
                        curr_idx,
                        e.to_string()
                    );
                    continue;
                }
            };
            added += 1;
            // A refused second link leaves a valid single-observation point, the state the
            // point is in anyway until the neighbour is linked.
            if let Err(e) = self.map.link_observation(prev_kf_idx, prev_idx, mp_idx) {
                refused += 1;
                debug!(
                    "grow: map refused the neighbour observation: kf={} feature={} map_point={} error={}",
                    prev_kf_idx,
                    prev_idx,
                    mp_idx,
                    e.to_string()
                );
            }
        }
        if refused > 0 {
            self.map_writes_refused += refused;
            warning!(
                "grow: map refused writes: prev_kf={} kf={} refused={} added={}",
                prev_kf_idx,
                curr_kf_idx,
                refused,
                added
            );
        }

        added
    }

    /// Forward-only subset of the neighbour fusion of Campos et al., ORB-SLAM3,
    /// IEEE T-RO 2021: project each map
    /// point observed by the current keyframe into every neighbour that does not
    /// yet observe it, and register the observation when it lands on a matching
    /// unassociated keypoint. No duplicate merging, no second-hop expansion.
    ///
    /// Cost is `O(neighbors x curr_map_points x nb_keypoints)` with no spatial
    /// index — profile this first if the live loop misses frame rate.
    fn fuse_into_neighbors(&mut self, curr_kf_idx: usize, neighbor_kf_indices: &[usize]) -> usize {
        let curr_mp_indices: Vec<usize> = match self.map.get_keyframe(curr_kf_idx) {
            Some(kf) => kf.map_point_by_desc_idx.iter().flatten().copied().collect(),
            None => return 0,
        };
        if curr_mp_indices.is_empty() {
            return 0;
        }

        let r2 = FUSE_SEARCH_RADIUS_PX * FUSE_SEARCH_RADIUS_PX;
        let mut n_fused = 0usize;
        let mut refused = 0u64;

        for &nb_kf_idx in neighbor_kf_indices {
            if nb_kf_idx == curr_kf_idx {
                continue;
            }

            // (kp_idx_in_nb_kf, mp_idx, hamming), collected under a shared
            // borrow and resolved below so one keypoint can't be double-claimed.
            let mut proposals: Vec<(usize, usize, u32)> = Vec::new();
            {
                let Some(nb_kf) = self.map.get_keyframe(nb_kf_idx) else {
                    continue;
                };

                for &mp_idx in &curr_mp_indices {
                    let mp = match self.map.map_points().get(mp_idx) {
                        Some(mp) if !mp.culled => mp,
                        _ => continue,
                    };
                    if mp.is_observed_by(nb_kf_idx) {
                        continue;
                    }

                    let p_cam = nb_kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 0.0 {
                        continue;
                    }
                    let Ok(pixel) =
                        self.camera
                            .project_to_image(&p_cam, 0.0, nb_kf.frame.image_size)
                    else {
                        continue;
                    };
                    let u = pixel.x as f32;
                    let v = pixel.y as f32;

                    let mut best_dist = u32::MAX;
                    let mut best_kp = usize::MAX;
                    for kp_idx in 0..nb_kf.frame.features.keypoints_xy.len() {
                        if nb_kf.map_point(kp_idx).is_some() {
                            continue;
                        }
                        let Some(kp) = nb_kf.frame.undistorted_xy(kp_idx, &self.camera) else {
                            continue;
                        };
                        let dx = kp[0] - u;
                        let dy = kp[1] - v;
                        if dx * dx + dy * dy > r2 {
                            continue;
                        }
                        let dist = hamming_distance(
                            &mp.descriptor,
                            &nb_kf.frame.features.descriptors[kp_idx],
                        );
                        if dist < best_dist {
                            best_dist = dist;
                            best_kp = kp_idx;
                        }
                    }

                    if best_dist <= FUSE_MAX_HAMMING && best_kp != usize::MAX {
                        proposals.push((best_kp, mp_idx, best_dist));
                    }
                }
            }

            proposals.sort_by_key(|&(_, _, dist)| dist);
            let mut taken_kp: HashSet<usize> = HashSet::new();
            for (kp_idx, mp_idx, _) in proposals {
                if taken_kp.contains(&kp_idx) {
                    continue;
                }
                let already = self
                    .map
                    .get_keyframe(nb_kf_idx)
                    .and_then(|kf| kf.map_point(kp_idx))
                    .is_some();
                if already {
                    continue;
                }
                // Links both directions at once: the neighbour's feature slot and the point's
                // observation record, which the "already observes" test above reads.
                if let Err(e) = self.map.link_observation(nb_kf_idx, kp_idx, mp_idx) {
                    refused += 1;
                    debug!(
                        "fuse: map refused a link: kf={} feature={} map_point={} error={}",
                        nb_kf_idx,
                        kp_idx,
                        mp_idx,
                        e.to_string()
                    );
                    continue;
                }
                taken_kp.insert(kp_idx);
                n_fused += 1;
            }
        }
        if refused > 0 {
            self.map_writes_refused += refused;
            warning!(
                "fuse: map refused links: kf={} refused={} fused={}",
                curr_kf_idx,
                refused,
                n_fused
            );
        }

        n_fused
    }
}

// ── Free helpers ─────────────────────────────────────────────────────────────

/// A static label for a map-projection rejection, for the structured log (the upstream enum
/// implements neither `Display` nor `Serialize`).
fn reject_reason_str(reason: MapProjectionRejectReason) -> &'static str {
    match reason {
        MapProjectionRejectReason::LowProjectionMatches => "low_projection_matches",
        MapProjectionRejectReason::PnpFailed => "pnp_failed",
        MapProjectionRejectReason::LowPnpInliers => "low_pnp_inliers",
        MapProjectionRejectReason::LowReferenceMatches => "low_reference_matches",
        MapProjectionRejectReason::LowReferenceCorrespondences => "low_reference_correspondences",
    }
}

/// The diagnostic label of a map-projection rejection.
fn failure_reason(reason: MapProjectionRejectReason) -> FailureReason {
    match reason {
        MapProjectionRejectReason::LowProjectionMatches => FailureReason::LowProjectionMatches,
        MapProjectionRejectReason::PnpFailed => FailureReason::PnpFailed,
        MapProjectionRejectReason::LowPnpInliers => FailureReason::LowPnpInliers,
        MapProjectionRejectReason::LowReferenceMatches => FailureReason::LowReferenceMatches,
        MapProjectionRejectReason::LowReferenceCorrespondences => {
            FailureReason::LowReferenceCorrespondences
        }
    }
}

/// One moving interval's motion check, as the status topic carries it.
fn motion_check(
    motion: &IntervalMotion,
    verdict: MotionVerdict,
    translation_disagrees: bool,
    checks: u64,
    oop: &crate::consistency::OutOfPlane,
    path_m: f64,
) -> MotionCheck {
    let deg = |r: &Mat3F64| crate::consistency::angle(r).to_degrees() as f32;
    MotionCheck {
        checks,
        verdict: match verdict {
            MotionVerdict::Consistent => VerdictLabel::Consistent,
            MotionVerdict::VisionDisagrees => VerdictLabel::VisionDisagrees,
            MotionVerdict::WheelsDisagree => VerdictLabel::WheelsDisagree,
            MotionVerdict::GyroDisagrees => VerdictLabel::GyroDisagrees,
            MotionVerdict::Inconclusive => VerdictLabel::Inconclusive,
        },
        translation_disagrees,
        wheels_deg: deg(&motion.wheels.rotation),
        gyro_deg: motion.gyro.as_ref().map_or(f32::NAN, deg),
        vision_deg: deg(&motion.vision.rotation),
        wheels_mm: (motion.wheels.translation.length() * 1e3) as f32,
        vision_mm: (motion.vision.translation.length() * 1e3) as f32,
        oop_z_mm: (oop.z_m * 1e3) as f32,
        oop_roll_deg: oop.roll_rad.to_degrees() as f32,
        oop_pitch_deg: oop.pitch_rad.to_degrees() as f32,
        path_m: path_m as f32,
    }
}

fn validate_camera(camera: &PinholeCamera, baseline_m: f64) -> Result<(), VioError> {
    if camera.k1 != 0.0 || camera.k2 != 0.0 || camera.p1 != 0.0 || camera.p2 != 0.0 {
        return Err(VioError::DistortedCamera {
            k1: camera.k1,
            k2: camera.k2,
            p1: camera.p1,
            p2: camera.p2,
        });
    }
    if !baseline_m.is_finite() || baseline_m <= 0.0 {
        return Err(VioError::InvalidBaseline {
            baseline: Length::new::<meter>(baseline_m),
        });
    }
    Ok(())
}

fn validate_frame_arrays(frame: &Frame) -> Result<(), VioError> {
    let n = frame.features.keypoints_xy.len();
    let stereo_ok = (frame.u_right.is_empty() && frame.depth.is_empty())
        || (frame.u_right.len() == n && frame.depth.len() == n);
    if !stereo_ok
        || frame.keypoint_colors.len() != n
        || frame.features.descriptors.len() != n
        || frame.features.octaves.len() != n
        || frame.features.orientations.len() != n
    {
        return Err(VioError::FrameArraysNotParallel {
            n_keypoints: n,
            n_u_right: frame.u_right.len(),
            n_depth: frame.depth.len(),
            n_colors: frame.keypoint_colors.len(),
            n_descriptors: frame.features.descriptors.len(),
            n_octaves: frame.features.octaves.len(),
            n_orientations: frame.features.orientations.len(),
        });
    }

    // Lengths are not enough: `process_frame` is public and is the seam the
    // tests and offline replays drive, so a caller that did its own stereo
    // can hand us `+inf`. See `VioError::NonFiniteFrameValue` for why that one
    // in particular is unrecoverable rather than merely wrong.
    let non_finite =
        |field: &'static str, index: usize, value: f64| VioError::NonFiniteFrameValue {
            field,
            index,
            value,
        };
    for (i, kp) in frame.features.keypoints_xy.iter().enumerate() {
        if !kp[0].is_finite() || !kp[1].is_finite() {
            let bad = if kp[0].is_finite() { kp[1] } else { kp[0] };
            return Err(non_finite("keypoints_xy", i, bad as f64));
        }
    }
    for (i, &u) in frame.u_right.iter().enumerate() {
        if !u.is_finite() {
            return Err(non_finite("u_right", i, u as f64));
        }
    }
    for (i, &d) in frame.depth.iter().enumerate() {
        if !d.is_finite() {
            return Err(non_finite("depth", i, d as f64));
        }
    }
    Ok(())
}

fn keypoint_color(frame: &Frame, desc_idx: usize) -> [u8; 3] {
    frame
        .keypoint_colors
        .get(desc_idx)
        .copied()
        .unwrap_or([128; 3])
}

/// Level `level`'s dimension, from the level-0 dimension.
///
/// Byte-for-byte the formula `OrbDetector` uses (`pyramid_size_at_level` in
/// kornia-imgproc's ORB extractor): f64 arithmetic and `round_ties_even`. It is
/// duplicated rather than approximated because the two pyramids MUST come out
/// the same size — `kornia_slam::stereo` takes a keypoint detected by
/// `OrbDetector` at octave `o`, scales its coordinate by `1.2^-o`, and indexes
/// *our* level `o` with it. A resolution whose scaled dimension lands on `.5`
/// resolves differently under `f32::round` (half away from zero) than under
/// `f64::round_ties_even`, and the two pyramids would silently disagree by a
/// pixel at that octave. 640x400 and 752x480 happen not to hit it; that is luck,
/// not a property, so do not "simplify" this back to `as f32` + `.round()`.
fn pyramid_dim_at_level(base: usize, level: usize) -> usize {
    // `as f32 as f64` is NOT redundant. `OrbDetector::downscale` is an f32 field, which
    // `TrackerConfig::new` sets to `ORB_SCALE_FACTOR as f32`, and `pyramid_size_at_level`
    // then widens it back with `(downscale as f64)`. That round trip gives
    // 1.2000000476837158, not the f64 1.2 that `ORB_SCALE_FACTOR` is — a ~4e-8 relative
    // difference, which is irrelevant everywhere except within one ULP of a `.5` rounding
    // boundary, where it is the whole answer. Reproduce the round trip rather than the
    // intent.
    let inv_scale = (ORB_SCALE_FACTOR as f32 as f64).powi(-(level as i32));
    ((base as f64 * inv_scale).round_ties_even() as usize).max(1)
}

/// Builds the octave pyramid `kornia_slam::stereo` reads: level `o` is the
/// **full-resolution** image resampled once by `1 / 1.2^o`.
///
/// This is not `OrbDetector`'s internal pyramid, and the difference is *pixel
/// content*, not geometry. Both size every level from level 0 (see
/// [`pyramid_dim_at_level`]) — `OrbDetector` does so deliberately, its own doc
/// noting that chaining `prev / scale` accumulates rounding drift — so the
/// `u * 1.2^-o` coordinate mapping is equally valid against either. What
/// differs is that `OrbDetector` chains its bilinear reductions level-to-level
/// (each level resampled from the one above), while this resamples every level
/// straight from full resolution: our upper levels are therefore more aliased
/// than the ones the descriptors were extracted from, which degrades the SAD
/// parabola fit at octave >= 1 rather than shifting any coordinate.
///
/// Two consequences worth knowing before changing this. It is not free — 7
/// full-image resizes per eye per frame, plus a `clone()` for level 0,
/// rebuilding a pyramid `OrbDetector` just built and discarded. And chaining
/// the reductions here (matching `OrbDetector` exactly, which is strictly more
/// faithful) is a behaviour change to stereo depth, so it needs the
/// `tests/stereo_depth.rs` numbers re-measured, not just a green test run.
fn build_u8_pyramid(img: &Image<u8, 1>) -> Result<Vec<Image<u8, 1>>, VioError> {
    let mut pyramid = Vec::with_capacity(ORB_N_LEVELS);
    pyramid.push(img.clone());
    for level in 1..ORB_N_LEVELS {
        let width = pyramid_dim_at_level(img.width(), level);
        let height = pyramid_dim_at_level(img.height(), level);
        let mut dst = Image::from_size_val(ImageSize { width, height }, 0u8)?;
        resize_fast_mono(img, &mut dst, InterpolationMode::Bilinear)?;
        pyramid.push(dst);
    }
    Ok(pyramid)
}

/// Samples one grey value per keypoint from the left image, truncating then
/// clamping the coordinates (the upstream kornia-slam example does exactly this).
fn sample_keypoint_colors(gray: &Image<u8, 1>, keypoints_xy: &[[f32; 2]]) -> Vec<[u8; 3]> {
    let (width, height) = (gray.width(), gray.height());
    if width == 0 || height == 0 {
        return vec![[128; 3]; keypoints_xy.len()];
    }
    let data = gray.as_slice();
    keypoints_xy
        .iter()
        .map(|kp| {
            // Negative f32 saturates to 0 on an `as usize` cast.
            let x = (kp[0] as usize).min(width - 1);
            let y = (kp[1] as usize).min(height - 1);
            let g = data.get(y * width + x).copied().unwrap_or(128);
            [g, g, g]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::f64::Frequency;
    use kornia_algebra::SO3F64;

    fn rectified_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 200.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_rejects_a_distorted_camera() {
        let camera = PinholeCamera {
            k1: -0.12,
            ..rectified_camera()
        };
        let err = validate_camera(&camera, 0.075).unwrap_err();
        assert!(matches!(err, VioError::DistortedCamera { .. }), "{err:?}");
    }

    #[test]
    fn test_rejects_a_non_positive_baseline() {
        let err = validate_camera(&rectified_camera(), 0.0).unwrap_err();
        assert!(matches!(err, VioError::InvalidBaseline { .. }), "{err:?}");
    }

    #[test]
    fn test_close_depth_is_thirty_five_baselines() {
        let cfg = TrackerConfig::new(Length::new::<meter>(0.075));
        let close = cfg
            .stereo_close_depth
            .expect("densification enabled")
            .get::<meter>();
        assert!((close - 2.625).abs() < 1e-12, "got {close}");
    }

    #[test]
    fn test_orb_pyramid_geometry_matches_kornia_slam_scale_tables() {
        let img = Image::from_size_val(
            ImageSize {
                width: 640,
                height: 400,
            },
            7u8,
        )
        .unwrap();
        let pyramid = build_u8_pyramid(&img).unwrap();
        assert_eq!(pyramid.len(), ORB_N_LEVELS);
        assert_eq!(pyramid[0].width(), 640);
        for (level, img) in pyramid.iter().enumerate() {
            let expect = (640.0 / 1.2f64.powi(level as i32)).round() as usize;
            assert_eq!(img.width(), expect.max(1), "level {level}");
        }
    }

    /// Our pyramid must be dimension-identical to `OrbDetector`'s at every
    /// level, for every resolution we ship.
    ///
    /// `kornia_slam::stereo` scales a keypoint that `OrbDetector` found at
    /// octave `o` by `1.2^-o` and indexes OUR level `o` with it. If the two
    /// pyramids disagree by even one pixel at some level, that indexing is
    /// wrong for every keypoint at that octave and the failure is a quiet
    /// disparity bias, not a panic. Checked against the exact upstream formula
    /// (f64 + `round_ties_even`), not against `1.2^-o` rounded some other way —
    /// that is the difference this test exists to catch.
    #[test]
    fn test_pyramid_levels_match_the_orb_extractor_sizes_exactly() {
        // Verbatim from kornia-imgproc `features/orb/extractor.rs`:
        // `pyramid_size_at_level`.
        fn kornia_size(base: usize, downscale: f32, level: usize) -> usize {
            let inv_scale = (downscale as f64).powi(-(level as i32));
            (base as f64 * inv_scale).round_ties_even() as usize
        }

        // 640x400 = the OAK stereo mode; 752x480 = EuRoC; the rest are odd
        // sizes chosen to exercise the rounding boundary.
        for (w0, h0) in [
            (640usize, 400usize),
            (752, 480),
            (1280, 720),
            (641, 401),
            (255, 173),
        ] {
            let img = Image::from_size_val(
                ImageSize {
                    width: w0,
                    height: h0,
                },
                7u8,
            )
            .unwrap();
            let pyr = build_u8_pyramid(&img).unwrap();
            assert_eq!(pyr.len(), ORB_N_LEVELS);
            for (level, img_l) in pyr.iter().enumerate() {
                assert_eq!(
                    (img_l.width(), img_l.height()),
                    (
                        kornia_size(w0, ORB_SCALE_FACTOR as f32, level).max(1),
                        kornia_size(h0, ORB_SCALE_FACTOR as f32, level).max(1)
                    ),
                    "{w0}x{h0} level {level}: our pyramid and OrbDetector's disagree, so \
                     kornia_slam::stereo's `u * 1.2^-o` mapping indexes the wrong pixels"
                );
            }
        }
    }

    #[test]
    fn test_colors_are_clamped_not_wrapped() {
        let img = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            9u8,
        )
        .unwrap();
        // Out of range on both sides; must clamp rather than index out of bounds.
        let colors = sample_keypoint_colors(&img, &[[-5.0, -5.0], [99.0, 99.0], [1.0, 1.0]]);
        assert_eq!(colors, vec![[9; 3]; 3]);
    }

    // ── Inertial ─────────────────────────────────────────────────────────────
    //
    // These drive `on_keyframe_inertial` directly with a hand-built map rather than through
    // synthetic imagery: what is under test is the ingest contract — which intervals are
    // refused, and what bounds reach the map's `ImuFactor` — and none of that depends on how
    // the keyframes were produced. The map DOES validate that both endpoints name keyframes it
    // holds (`Map::apply_insertion`), so every test inserts featureless keyframes at its
    // endpoints first — including the refusal tests, where a missing endpoint would have the map
    // refuse the factor whatever the gate under test did. `bare_keyframe` is enough for that.

    /// A real epoch, not 0: the whole reason samples are buffered in `u64` nanoseconds is that
    /// this value in `f64` seconds leaves ~400 ns of resolution against a 5 ms sample period.
    const EPOCH_NS: u64 = 1_757_000_000_000_000_000;
    const IMU_PERIOD_NS: u64 = 5_000_000;

    fn inertial_config() -> InertialConfig {
        // 90 degrees about z. Asymmetric under transpose, so a convention slip in anything
        // that consumes it would change the answer rather than cancel.
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        InertialConfig::new(
            Pose3d::from_rt(rotation, Vec3F64::new(0.01, -0.02, 0.03)),
            // EuRoC ADIS16448 densities. Fine for a test, wrong for any other IMU — which is
            // why the real ones have to come from config with no default.
            kornia_sensors::imu::ImuCalib {
                gyro_noise: 1.6968e-4,
                accel_noise: 2.0e-3,
                gyro_bias_noise: 1.9393e-5,
                accel_bias_noise: 3.0e-3,
            },
            Frequency::new::<hertz>(200.0),
        )
        .expect("a proper rotation and positive densities")
    }

    fn tracker_with_inertial() -> Tracker {
        // `InertialConfig::new` defaults the flag to true (the RON reader is what defaults it
        // to false), so this is the fully-armed tracker.
        tracker_with_inertial_ba(true)
    }

    fn tracker_with_inertial_ba(enable_inertial_ba: bool) -> Tracker {
        let mut config = TrackerConfig::new(Length::new::<meter>(0.075));
        let mut inertial = inertial_config();
        inertial.enable_inertial_ba = enable_inertial_ba;
        config.inertial = Some(inertial);
        let mut tracker = Tracker::new(rectified_camera(), config);
        // Normally set by the first frame; set here because these tests never feed one.
        tracker.epoch_ns = Some(EPOCH_NS);
        tracker
    }

    /// The landmark snapshot window: the newest `cap` SLOTS, culled ones skipped rather than
    /// backfilled from older slots, indices exactly as the map holds them.
    #[test]
    fn test_newest_live_map_points_windows_the_tail_and_skips_culled() {
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        // kornia-slam links every landmark to a feature of a keyframe it holds, so the ten
        // landmarks hang off ten features of one keyframe, in slot order.
        tracker
            .map
            .insert_keyframe(keyframe_with_features(0, 10))
            .expect("a fresh keyframe id");
        for i in 0..10 {
            let slot = tracker
                .map
                .insert_landmark(LandmarkSeed {
                    position: Vec3F64::new(i as f64, -(i as f64), 0.5 * i as f64),
                    color: [0; 3],
                    reference: ObservationKey {
                        keyframe_idx: 0,
                        feature_idx: i,
                    },
                })
                .expect("a free feature and a finite position");
            assert_eq!(slot, i, "landmark slots are appended in order");
        }
        assert!(tracker.map.remove_landmark(2).expect("a known landmark"));
        assert!(tracker.map.remove_landmark(7).expect("a known landmark"));

        let got: Vec<_> = tracker.newest_live_map_points(5).collect();
        let want: Vec<_> = [5usize, 6, 8, 9]
            .into_iter()
            .map(|i| (i, [i as f64, -(i as f64), 0.5 * i as f64]))
            .collect();
        assert_eq!(got, want);

        // A cap past the map's size is the whole live map.
        assert_eq!(tracker.newest_live_map_points(100).count(), 8);
        assert_eq!(tracker.newest_live_map_points(0).count(), 0);
    }

    /// Nothing in kornia-slam removes a keyframe, so an unbounded map is the shipped behaviour
    /// and `max_keyframes` is the only thing standing between clean tracking and the silent
    /// stall measured on a Jetson Orin with an OAK-D at 640x400 (757 MB RSS, zero poses after
    /// 2.5 h).
    #[test]
    fn test_the_map_bound_drops_the_map_and_bumps_the_generation() {
        let mut tracker = tracker_with_inertial();
        tracker.config.max_keyframes = Some(4);
        for i in 0..6 {
            tracker
                .map
                .insert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)))
                .expect("a fresh keyframe id");
        }
        let gen_before = tracker.world_generation();
        let maps_before = tracker.map_generation();

        // Not armed: the bound must never fire on its own, or every consumer sees spurious
        // reset epochs.
        tracker.apply_pending_map_reset();
        assert_eq!(tracker.map.keyframes().len(), 6, "unarmed reset fired");
        assert_eq!(tracker.world_generation(), gen_before);
        assert_eq!(tracker.map_generation(), maps_before);

        tracker.pending_map_reset = true;
        tracker.apply_pending_map_reset();
        assert!(tracker.map.keyframes().is_empty(), "map survived the bound");
        assert_eq!(
            tracker.world_generation(),
            gen_before + 1,
            "a dropped map MUST bump the generation, or a consumer composes two worlds"
        );
        assert_eq!(
            tracker.map_generation(),
            maps_before + 1,
            "a dropped map is a new map"
        );
        // One-shot: the flag is consumed, so the next frame does not reset again.
        assert!(!tracker.pending_map_reset);
        // A requested reset is a new map too, though it is not a drop.
        let drops = tracker.map_drops();
        tracker.reset();
        assert_eq!(tracker.map_generation(), maps_before + 2);
        assert_eq!(
            tracker.map_drops(),
            drops,
            "a requested reset is not a drop"
        );
    }

    /// The arming predicate itself, which nothing else reaches: `tracking_step` is private and
    /// no test drives a frame through it, so every conjunct here is a way to disable the bound
    /// or fire it every frame with a green suite.
    #[test]
    fn test_the_bound_arms_only_on_an_insertion_that_reaches_the_cap() {
        let cases = [
            // (cap, keyframes, inserted, should_arm)
            (Some(4), 4, true, true),
            (Some(4), 5, true, true),
            (Some(4), 3, true, false),
            // Not an insertion: the map did not grow, so nothing can have crossed.
            (Some(4), 9, false, false),
            // Opt-in. `None` is every graph without the key, and must never arm.
            (None, 50, true, false),
        ];
        for (cap, kfs, inserted, want) in cases {
            let mut tracker = tracker_with_inertial();
            tracker.config.max_keyframes = cap;
            for i in 0..kfs {
                tracker
                    .map
                    .insert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)))
                    .expect("a fresh keyframe id");
            }
            tracker.arm_map_bound(inserted);
            assert_eq!(
                tracker.pending_map_reset, want,
                "cap={cap:?} keyframes={kfs} inserted={inserted}"
            );
        }
    }

    /// The TRACKING-LOSS reset must keep the bias too — it is the drop a live graph takes most.
    ///
    /// A test that only exercised the `max_keyframes` bound could not see a loss path that ran
    /// `reset()` and re-estimated the bias from zero on nearly every real drop, so this drives
    /// the loss path's own entry point.
    #[test]
    fn test_a_tracking_loss_map_drop_keeps_the_imu_bias() {
        let measured = ImuBias {
            gyro: Vec3F64::new(0.0026, -0.0011, 0.0009),
            accel: Vec3F64::new(0.01, -0.02, 0.03),
        };
        let mut tracker = tracker_with_inertial();
        tracker.config.reset_map_on_loss = true;
        tracker.inertial.as_mut().unwrap().bias = measured;

        // The loss path the live graph takes, reached the way `tracking_step` reaches it.
        tracker.reset_keeping_calibration();
        assert_eq!(
            tracker.inertial.as_ref().unwrap().bias.gyro,
            measured.gyro,
            "a tracking-loss drop re-estimated the bias from zero"
        );

        // And the requested reset remains the one path that DOES start it over.
        tracker.inertial.as_mut().unwrap().bias = measured;
        let initial = tracker.inertial.as_ref().unwrap().cfg.initial_bias;
        tracker.reset();
        assert_eq!(
            tracker.inertial.as_ref().unwrap().bias.gyro,
            initial.gyro,
            "a requested reset must still start the bias over"
        );
    }

    /// A gyro prior must rotate the camera IN PLACE, not swing it about the world origin.
    ///
    /// `Pose3d` is world-to-camera, so the centre is `-R^T t`. Writing `rotation` alone moves
    /// the predicted centre by ~theta * |t|, which is metres once the robot is any distance
    /// out, and worst exactly while turning. The rotation residual
    /// that validated the prior compares rotations only, so it could not see this.
    #[test]
    fn test_the_gyro_prior_rotates_about_the_camera_centre_not_the_world_origin() {
        use kornia_3d::pose::Pose3d;
        // Camera 5 m from the map origin, looking along +x.
        let centre = Vec3F64::new(5.0, 0.0, 0.0);
        let before = Pose3d::from_rt(Mat3F64::IDENTITY, -(Mat3F64::IDENTITY * centre));
        assert!(
            (before.inverse().translation - centre).length() < 1e-9,
            "fixture wrong: centre must round-trip"
        );

        // A 0.03 rad yaw, the size one 66 ms frame at 0.45 rad/s produces.
        let th: f64 = 0.03;
        let (c, sn) = (th.cos(), th.sin());
        let d = Mat3F64::from_cols_array(&[c, sn, 0.0, -sn, c, 0.0, 0.0, 0.0, 1.0]);

        // THE PRODUCTION FUNCTION, not a copy of its body beside it: re-deriving the two lines
        // here would pass whether or not `tracking_step` actually calls them.
        let candidate = rotate_about_centre(before, d.transpose() * before.rotation);

        let moved = (candidate.inverse().translation - centre).length();
        assert!(
            moved < 1e-9,
            "the gyro prior teleported the camera centre by {moved:.4} m"
        );

        // And the naive version really does move it — otherwise this test proves nothing.
        let mut naive = before;
        naive.rotation = d.transpose() * before.rotation;
        let naive_moved = (naive.inverse().translation - centre).length();
        assert!(
            naive_moved > 0.1,
            "writing rotation alone should move the centre ~theta*|t| = 0.15 m; got {naive_moved:.4}"
        );
    }

    /// A map-size reset keeps the IMU bias; a requested reset does not.
    ///
    /// Bias is a property of the sensor. Dropping it when the MAP hit a size cap makes the next
    /// initial solve land wide and the first refinement that corrects it rotate the whole world
    /// — measured as gravity_y 8.94 then 9.81, one large world rotation per map cycle.
    #[test]
    fn test_a_map_bound_reset_keeps_the_imu_bias_and_a_requested_reset_does_not() {
        let measured = ImuBias {
            gyro: Vec3F64::new(0.0026, -0.0011, 0.0009),
            accel: Vec3F64::new(0.01, -0.02, 0.03),
        };

        let mut tracker = tracker_with_inertial();
        tracker.inertial.as_mut().unwrap().bias = measured;
        tracker.reset_keeping_calibration();
        let kept = tracker.inertial.as_ref().unwrap().bias;
        assert_eq!(
            kept.gyro, measured.gyro,
            "map-bound reset lost the gyro bias"
        );
        assert_eq!(
            kept.accel, measured.accel,
            "map-bound reset lost the accel bias"
        );

        let mut tracker = tracker_with_inertial();
        tracker.inertial.as_mut().unwrap().bias = measured;
        let initial = tracker.inertial.as_ref().unwrap().cfg.initial_bias;
        tracker.reset();
        assert_eq!(
            tracker.inertial.as_ref().unwrap().bias.gyro,
            initial.gyro,
            "a requested reset must start the bias over"
        );

        // Neither carries `gravity_world`: the bootstrap after a reset defines a new world.
        // MOVED OFF THE DEFAULT FIRST — asserting it equals the default without ever changing
        // it passes whether or not the reset touches it, which is no test at all.
        let mut tracker = tracker_with_inertial();
        tracker.inertial.as_mut().unwrap().gravity_world = Vec3F64::new(0.2, 8.9, -4.0);
        tracker.reset_keeping_calibration();
        assert!(
            (tracker.inertial.as_ref().unwrap().gravity_world.z + GRAVITY_MAGNITUDE).abs() < 1e-9,
            "gravity_world must not survive a reset — it is world-frame"
        );
    }

    /// A requested reset while the bound is armed must not drop the FRESH map on the next
    /// frame and bump the generation a second time for no visible reason.
    #[test]
    fn test_a_reset_disarms_the_pending_bound() {
        let mut tracker = tracker_with_inertial();
        tracker.config.max_keyframes = Some(2);
        for i in 0..3 {
            tracker
                .map
                .insert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)))
                .expect("a fresh keyframe id");
        }
        tracker.arm_map_bound(true);
        assert!(tracker.pending_map_reset);

        tracker.reset();
        assert!(!tracker.pending_map_reset, "reset left the bound armed");
        let gen_after_reset = tracker.world_generation();
        tracker.apply_pending_map_reset();
        assert_eq!(
            tracker.world_generation(),
            gen_after_reset,
            "a disarmed bound must not bump the generation again"
        );
    }

    /// A featureless keyframe at `idx` with its camera centre at `centre`. `apply_insertion`,
    /// `ready` and `apply_initialization` read `frame.idx`, `pose_world_to_cam` and (`ready`)
    /// `is_stereo()`, so this is enough to make them do their real work.
    fn bare_keyframe(idx: usize, centre: Vec3F64) -> Keyframe {
        Keyframe::from_frame(Frame {
            idx,
            features: kornia_imgproc::features::OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::from_rt(Mat3F64::IDENTITY, -centre),
            image_size: ImageSize {
                width: 640,
                height: 400,
            },
            keypoint_colors: Vec::new(),
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        })
    }

    /// A keyframe at the origin with `n` distinct features, for tests that need landmarks: the
    /// map links a landmark only to a feature that has both a descriptor and a keypoint.
    fn keyframe_with_features(idx: usize, n: usize) -> Keyframe {
        Keyframe::from_frame(Frame {
            idx,
            features: kornia_imgproc::features::OrbFeatures {
                keypoints_xy: (0..n).map(|i| [10.0 * i as f32, 10.0]).collect(),
                orientations: vec![0.0; n],
                descriptors: (0..n).map(|i| [i as u8; 32]).collect(),
                octaves: vec![0; n],
            },
            pose_world_to_cam: Pose3d::IDENTITY,
            image_size: ImageSize {
                width: 640,
                height: 400,
            },
            keypoint_colors: vec![[0; 3]; n],
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        })
    }

    /// Featureless keyframes at `indices`, for the IMU-edge endpoints the map requires.
    fn insert_bare_keyframes(tracker: &mut Tracker, indices: &[usize]) {
        for &i in indices {
            tracker
                .map
                .insert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)))
                .expect("a fresh keyframe id");
        }
    }

    /// What `try_initialize` hands back when its gates pass. Gravity deliberately OFF world
    /// +y and scale deliberately not 1, so that `apply_initialization` — if it runs — cannot
    /// be a no-op on the map: `apply_inertial_alignment` gets a non-identity rotation and a
    /// non-unit scale.
    fn accepted_init() -> ImuInitResult {
        ImuInitResult {
            scale: 1.7,
            gravity_world: Vec3F64::new(0.3, 9.7, -0.8),
            velocities_world: vec![Vec3F64::new(0.1, 0.0, 0.0)],
            bias: ImuBias {
                gyro: Vec3F64::new(0.01, -0.02, 0.03),
                accel: Vec3F64::new(0.1, 0.2, -0.1),
            },
        }
    }

    fn pose_bits(p: &Pose3d) -> ([u64; 9], [u64; 3]) {
        (
            p.rotation.to_cols_array().map(f64::to_bits),
            p.translation.to_array().map(f64::to_bits),
        )
    }

    fn imu_samples(from_ns: u64, n: usize) -> Vec<RawImuSample> {
        (0..n)
            .map(|i| RawImuSample {
                stamp_ns: from_ns + i as u64 * IMU_PERIOD_NS,
                gyro: [0.0, 0.0, 0.05],
                accel: [0.0, 9.81, 0.0],
            })
            .collect()
    }

    #[test]
    fn test_the_inertial_path_is_off_unless_an_extrinsic_is_configured() {
        assert!(
            TrackerConfig::new(Length::new::<meter>(0.075))
                .inertial
                .is_none(),
            "the default config must never carry a guessed extrinsic"
        );
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        assert!(tracker.inertial_stats().is_none());

        // Pushing samples at a tracker that cannot read them is a wiring mistake, and it looks
        // exactly like a working inertial front end from the outside. So it is an error.
        let err = tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 10), 0, false)
            .unwrap_err();
        assert!(matches!(err, VioError::InertialDisabled), "{err:?}");
    }

    #[test]
    fn test_a_disabled_tracker_adds_no_factor_and_keeps_local_ba_visual() {
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        tracker.epoch_ns = Some(EPOCH_NS);
        let handles = tracker.on_keyframe_inertial(
            Some((0, EPOCH_NS + 1_000_000_000)),
            1,
            EPOCH_NS + 1_500_000_000,
        );
        // `None` is what selects `run_local_ba` over `run_local_inertial_ba` at the call site.
        assert!(handles.is_none());
        assert!(tracker.map().imu_factors().is_empty());
    }

    #[test]
    fn test_already_aligned_samples_are_refused_rather_than_rotated_twice() {
        let mut tracker = tracker_with_inertial();
        let err = tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 10), 0, true)
            .unwrap_err();
        assert!(matches!(err, VioError::InertialSamplesAligned), "{err:?}");
        // And nothing was buffered, so a later interval cannot half-succeed on them.
        assert_eq!(tracker.inertial_stats().unwrap().buffer.accepted, 0);
    }

    /// The public edge takes the unit-typed payload sample; its `tov` and rad/s, m/s^2 values
    /// must land in the ring unchanged.
    #[test]
    fn test_push_imu_accepts_payload_samples() {
        use cu_stereo_payloads::ImuPayload;

        let mut tracker = tracker_with_inertial();
        let payload = (0..3u64).map(|i| ImuSample {
            tov: CuTime::from_nanos(EPOCH_NS + i * IMU_PERIOD_NS),
            imu: ImuPayload::from_raw([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], 25.0),
        });
        assert_eq!(tracker.push_imu(payload, 0, false).expect("inertial on"), 3);
        let raw = RawImuSample::from(ImuSample {
            tov: CuTime::from_nanos(EPOCH_NS),
            imu: ImuPayload::from_raw([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], 25.0),
        });
        assert_eq!(raw.stamp_ns, EPOCH_NS);
        assert!((raw.accel[2] - f64::from(9.81f32)).abs() < 1e-12, "{raw:?}");
        assert!((raw.gyro[0] - f64::from(0.1f32)).abs() < 1e-12, "{raw:?}");
    }

    #[test]
    fn test_an_imu_edge_spans_exactly_the_two_keyframe_stamps() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        // 1.0 s through 1.995 s, so the interval is fully covered at both ends.
        tracker
            .push_raw_imu(imu_samples(t0_ns, 200), 0, false)
            .expect("inertial path is on");
        insert_bare_keyframes(&mut tracker, &[7, 11]);

        let handles = tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        // Not initialized yet (no window armed, so the ladder does not run), so BA stays visual.
        assert!(handles.is_none());

        let factors = tracker.map().imu_factors();
        assert_eq!(factors.len(), 1);
        let f = &factors[0];
        // Both indices are `Frame::idx`, not positions in the keyframe vector.
        assert_eq!((f.prev_kf_idx, f.curr_kf_idx), (7, 11));
        // Seconds since the TRACKER's epoch, the same rebasing `process_frame` applies to
        // frame stamps — a second epoch here would make `from_measurements` filter every
        // sample away and return a zero delta with no error.
        assert!((f.t0 - 1.0).abs() < 1e-12, "t0 = {}", f.t0);
        assert!((f.t1 - 1.6).abs() < 1e-12, "t1 = {}", f.t1);
        // The integration covers the whole interval, not just the span of the samples.
        assert!(
            (f.preintegrated.dt - 0.6).abs() < 1e-9,
            "dt = {}",
            f.preintegrated.dt
        );
        // Samples at 1.000, 1.005 ... 1.600 inclusive.
        assert_eq!(f.raw_samples.len(), 121);
        assert!(
            f.raw_samples.first().unwrap().timestamp >= f.t0
                && f.raw_samples.last().unwrap().timestamp <= f.t1
        );
        // Retained on the factor, which is what the inertial BA's 0.02 repropagation
        // re-integrates once the bias has drifted past its linearization point. An empty Vec
        // would compile.
        assert!(!f.raw_samples.is_empty());

        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.factors_added, 1);
        // What the map is pinning, as the stop line reports it: the one factor and its 121
        // retained raw samples.
        assert_eq!((stats.retained_factors, stats.retained_samples), (1, 121));
        // Pruned to the keyframe stamp: 1.600 s through 1.995 s inclusive.
        assert_eq!(stats.buffer.buffered, 80);
    }

    /// The excitation statistics must be scoped to the window `ImuInitializer::ready` is scoped
    /// to — factors with `curr_kf_idx >= start_idx`. A pre-window factor swept in would credit
    /// the window with motion that happened before it opened, and the whole point of measuring
    /// this is to decide later whether the window itself was excited.
    ///
    /// The per-statistic maths is pinned in `imu.rs`; what this pins is the filter.
    #[test]
    fn test_init_window_excitation_covers_exactly_the_factors_ready_counts() {
        let mut tracker = tracker_with_inertial();
        let stamp = |i: u64| EPOCH_NS + 1_000_000_000 + i * 500_000_000;
        // 3.0 s of 200 Hz samples from the first stamp: covers both intervals with margin.
        tracker
            .push_raw_imu(imu_samples(stamp(0), 600), 0, false)
            .unwrap();
        // Two edges: 3 -> 4 lands before a window opened at 5, and 7 -> 8 lands inside it.
        insert_bare_keyframes(&mut tracker, &[3, 4, 7, 8]);
        tracker.on_keyframe_inertial(Some((3, stamp(0))), 4, stamp(1));
        tracker.on_keyframe_inertial(Some((7, stamp(1))), 8, stamp(2));
        assert_eq!(tracker.map().imu_factors().len(), 2);

        // 0.5 s of the fixture's constant 0.05 rad/s about z, over 101 samples.
        let inside = tracker.init_window_excitation(5);
        assert_eq!(inside.samples, 101);
        assert!((inside.rot_rad - 0.05 * 0.5).abs() < 1e-9, "{inside:?}");
        // Both factors in scope: the angle doubles. This is what an unscoped (or inverted)
        // filter would return for the call above, so the two numbers must differ.
        let both = tracker.init_window_excitation(0);
        // 202, not 201: abutting factors each retain the sample on their shared boundary, so a
        // boundary sample is counted twice. Harmless for both statistics at 200 Hz, stated here
        // because the count is otherwise off by one against the sample stream.
        assert_eq!(both.samples, 202);
        assert!((both.rot_rad - 2.0 * 0.05 * 0.5).abs() < 1e-9, "{both:?}");
        // The fixture IMU is parked: one constant 9.81 magnitude, so the translational
        // statistic is exactly zero whichever scoping applies.
        assert_eq!((inside.accel_var, both.accel_var), (0.0, 0.0));
    }

    #[test]
    fn test_an_interval_with_reported_drops_is_refused_not_shortened() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        // Two batches at an unbroken 200 Hz cadence, so there is NO gap to detect: the only
        // evidence that samples went missing is the source's cumulative counter. Without the
        // check, `from_measurements` returns a full-0.6 s delta and nothing anywhere says the
        // stream was holed.
        tracker
            .push_raw_imu(imu_samples(t0_ns, 100), 0, false)
            .unwrap();
        tracker
            .push_raw_imu(imu_samples(t0_ns + 100 * IMU_PERIOD_NS, 100), 12, false)
            .unwrap();
        insert_bare_keyframes(&mut tracker, &[7, 11]);

        let handles = tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        assert!(handles.is_none());
        assert!(
            tracker.map().imu_factors().is_empty(),
            "a holed interval must produce NO factor, not a short one"
        );
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.refused_dropped, 1);
        // The gate refused it, not the map.
        assert_eq!(stats.refused_by_map, 0);
        assert_eq!(stats.factors_added, 0);
        // Still pruned: the refusal must not let the buffer grow without bound.
        assert_eq!(stats.buffer.buffered, 80);
    }

    #[test]
    fn test_an_under_covered_interval_is_refused() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        // Every other sample: 100 Hz. Each gap is 10 ms, inside the 15 ms limit, so only the
        // coverage count can catch it.
        let thinned: Vec<RawImuSample> = imu_samples(t0_ns, 200).into_iter().step_by(2).collect();
        tracker.push_raw_imu(thinned, 0, false).unwrap();
        insert_bare_keyframes(&mut tracker, &[7, 11]);

        tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        assert!(tracker.map().imu_factors().is_empty());
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.refused_too_few, 1);
        assert_eq!(stats.refused_by_map, 0);
    }

    /// An edge naming a keyframe the map does not hold used to be pushed anyway and vanish from
    /// the solve later; kornia-slam's map now refuses it. The refusal must be counted, not
    /// reported as an added factor.
    #[test]
    fn test_an_edge_to_a_keyframe_the_map_lacks_is_refused_and_counted() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        tracker
            .push_raw_imu(imu_samples(t0_ns, 200), 0, false)
            .unwrap();
        // Only the newer endpoint exists.
        insert_bare_keyframes(&mut tracker, &[11]);

        tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        assert!(tracker.map().imu_factors().is_empty());
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.factors_added, 0);
        assert_eq!(stats.refused_by_map, 1);
        assert_eq!(
            stats.refused_total(),
            1,
            "the total must see the new counter"
        );
    }

    /// An initialization the map refuses (here: no keyframe velocities) must leave everything as
    /// it was — map, bias, gravity, generation — and the tracker visual-only, with the refusal
    /// counted rather than reported as an acceptance.
    #[test]
    fn test_a_refused_initialization_leaves_the_tracker_visual_only() {
        let mut tracker = tracker_with_inertial_ba(true);
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, Vec3F64::new(0.5, -0.25, 2.0)))
            .expect("a fresh keyframe id");
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let pose_before = pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam);
        let refused = ImuInitResult {
            velocities_world: Vec::new(),
            ..accepted_init()
        };

        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, refused, 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);

        assert_eq!(
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam),
            pose_before
        );
        assert_eq!(tracker.world_generation(), 0);
        assert!(!tracker.state.imu_initialized);
        assert!(tracker.state.imu_init_timestamp_sec.is_none());
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(
            (inert.bias.gyro, inert.bias.accel),
            (inert.cfg.initial_bias.gyro, inert.cfg.initial_bias.accel)
        );
        assert_eq!(
            inert.gravity_world,
            Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE)
        );
        assert_eq!(inert.stats.init_accepted, 0);
        assert_eq!(inert.stats.init_apply_refused, 1);
        // And local BA stays on the visual path.
        let handles = tracker.on_keyframe_inertial(None, 4, EPOCH_NS + 3_000_000_000);
        assert!(handles.is_none());
    }

    #[test]
    fn test_the_very_first_keyframe_has_no_edge_and_that_is_not_a_refusal() {
        let mut tracker = tracker_with_inertial();
        tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 200), 0, false)
            .unwrap();
        // `prev_edge` is `None` on the bootstrap keyframe: there is no previous keyframe to
        // span from.
        tracker.on_keyframe_inertial(None, 0, EPOCH_NS + 500_000_000);
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.factors_added, 0);
        assert_eq!(stats.refused_total(), 0);
    }

    /// The flag's promise, as the config comments state it: with `enable_inertial_ba` off the
    /// inertial path cannot move a published pose. An accepted initialization is the other
    /// way it could (`apply_initialization` rotates and scales the whole map and bumps the
    /// world generation, which goes on the wire as `reset_epoch`), so the gate has to hold
    /// regardless of what the solve returns.
    #[test]
    fn test_with_inertial_ba_off_an_accepted_initialization_is_counted_but_never_applied() {
        let mut tracker = tracker_with_inertial_ba(false);
        let centre = Vec3F64::new(0.5, -0.25, 2.0);
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, centre))
            .expect("a fresh keyframe id");
        tracker.state.pose_world_to_cam = tracker.map.keyframes()[0].frame.pose_world_to_cam;
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let pose_before = pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam);
        let state_pose_before = pose_bits(&tracker.state.pose_world_to_cam);

        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, accepted_init(), 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);

        // Byte-identical, not approximately equal: a rotation by the ~5 degrees between the
        // synthetic gravity and +y would pass a loose tolerance.
        assert_eq!(
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam),
            pose_before,
            "the keyframe pose moved"
        );
        assert_eq!(
            pose_bits(&tracker.state.pose_world_to_cam),
            state_pose_before
        );
        assert_eq!(
            tracker.world_generation(),
            0,
            "reset_epoch would have gone on the wire"
        );
        assert!(!tracker.state.imu_initialized);
        assert!(tracker.state.imu_init_timestamp_sec.is_none());
        let inert = tracker.inertial.as_ref().unwrap();
        // Neither the linearization bias nor the gravity handed to inertial BA moved either.
        assert_eq!(
            (inert.bias.gyro, inert.bias.accel),
            (inert.cfg.initial_bias.gyro, inert.cfg.initial_bias.accel)
        );
        assert_eq!(
            inert.gravity_world,
            Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE)
        );
        // ...but the acceptance is observable, which is the point of running dry at all.
        assert_eq!(inert.stats.init_accepted, 1);
        assert!(inert.dry_run_accepted);
        // And local BA stays on the visual path.
        let handles = tracker.on_keyframe_inertial(None, 4, EPOCH_NS + 3_000_000_000);
        assert!(handles.is_none());
    }

    /// Positive control for the test above: the same call with the flag ON does move
    /// everything the negative test asserts is still. Without this, that test would also pass
    /// against an `apply_initialization` that happened to be a no-op on a one-keyframe map.
    #[test]
    fn test_with_inertial_ba_on_an_accepted_initialization_rotates_the_map() {
        let mut tracker = tracker_with_inertial_ba(true);
        let centre = Vec3F64::new(0.5, -0.25, 2.0);
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, centre))
            .expect("a fresh keyframe id");
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let pose_before = pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam);
        // A motion model from before the alignment, which moves the frame it was measured in.
        tracker.state.velocity = Some(Pose3d::IDENTITY);

        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, accepted_init(), 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);

        // Adoption resumes from the aligned keyframe and drops the stale motion model.
        assert_eq!(
            pose_bits(&tracker.state.pose_world_to_cam),
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam)
        );
        assert!(tracker.state.velocity.is_none());
        assert_ne!(
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam),
            pose_before,
            "the synthetic init must be one that moves a pose, or the dry-run test is vacuous"
        );
        assert_eq!(tracker.world_generation(), 1);
        assert!(tracker.state.imu_initialized);
        assert_eq!(tracker.state.imu_init_timestamp_sec, Some(2.5));
        let inert = tracker.inertial.as_ref().unwrap();
        let want = accepted_init().bias;
        assert_eq!((inert.bias.gyro, inert.bias.accel), (want.gyro, want.accel));
        assert_eq!(
            inert.gravity_world,
            Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)
        );
        assert_eq!(inert.stats.init_accepted, 1);
        assert!(!inert.dry_run_accepted);
    }

    /// A dry acceptance ends the initial-solve loop. Otherwise, with `state.imu_initialized` never
    /// set, the ladder would re-solve every `init_retry` for the rest of the session over
    /// a window that never shrinks. The window here is built to make `ready` TRUE, so the
    /// early return is what is being measured — a false `ready` would hide a missing latch.
    #[test]
    fn test_a_dry_acceptance_stops_the_initializer_from_re_solving() {
        let mut tracker = tracker_with_inertial_ba(false);
        // Ten keyframes 0.25 s apart, 0.03 m apart: 2.25 s of IMU time over nine edges and
        // 0.27 m of displacement, past every `ready` gate (10 kfs / 1 s stereo, 2 s mono /
        // 0.05 m). The frames carry no stereo data, so the mono 2 s applies.
        const N: usize = 10;
        const STEP_NS: u64 = 250_000_000;
        let stamp = |i: usize| EPOCH_NS + 1_000_000_000 + i as u64 * STEP_NS;
        for i in 0..N {
            tracker
                .map
                .insert_keyframe(bare_keyframe(i, Vec3F64::new(0.03 * i as f64, 0.0, 0.0)))
                .expect("a fresh keyframe id");
        }
        // Samples covering the whole span at 200 Hz, plus a margin past the last stamp.
        let n_samples = (N as u64 * STEP_NS / IMU_PERIOD_NS) as usize + 10;
        tracker
            .push_raw_imu(imu_samples(stamp(0), n_samples), 0, false)
            .unwrap();
        tracker.arm_inertial_window(0, 1.0, stamp(0));
        // Simulate the acceptance that already happened for this window.
        tracker.inertial.as_mut().unwrap().dry_run_accepted = true;

        for i in 1..N {
            let handles = tracker.on_keyframe_inertial(Some((i - 1, stamp(i - 1))), i, stamp(i));
            assert!(handles.is_none());
        }

        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.factors_added, N as u64 - 1);
        assert!(
            inert.initializer.ready(&tracker.map, Some(0)),
            "the window must be ready, or the latch is not what stopped the solve"
        );
        assert_eq!(
            inert.stats.init_attempts, 0,
            "the ladder re-solved after a dry acceptance"
        );
        // The latch belongs to the window: re-arming (a re-bootstrap) clears it.
        tracker.arm_inertial_window(0, 1.0, stamp(0));
        assert!(!tracker.inertial.as_ref().unwrap().dry_run_accepted);
    }

    #[test]
    fn test_a_reset_drops_the_initialization_window_and_keeps_the_samples() {
        let mut tracker = tracker_with_inertial();
        tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 200), 0, false)
            .unwrap();
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        tracker.inertial.as_mut().unwrap().last_window_log = Some((0.0, false));
        assert!(tracker.inertial.as_ref().unwrap().start_kf_idx.is_some());

        tracker.reset();
        let inert = tracker.inertial.as_ref().unwrap();
        // `start_kf_idx` named a keyframe in the dropped map. `ImuInitializer::ready` would
        // keep counting against it and `try_initialize` would silently skip every factor whose
        // endpoints it could not find — a solve over an empty problem, reported as a rejection.
        assert!(inert.start_kf_idx.is_none());
        assert!(inert.last_window_log.is_none());
        // The samples are the sensor's, not the map's: the next anchor prunes them, and
        // clearing them here cost every new map its first IMU edge.
        assert_eq!(inert.buffer.stats().buffered, 200);
        assert!(tracker.last_keyframe_stamp_ns.is_none());
    }

    #[test]
    fn test_pose_convention_is_world_to_cam() {
        // The rotation MUST be non-identity. With `R = I` the correct inverse
        // translation `-R^T t` and the classic wrong one `-t` are numerically
        // identical, so an identity-rotation version of this test passes
        // against exactly the bug it is named after.
        //
        // 90 deg yaw about y, column-major: a camera whose centre is at world
        // (2, 0, 3), so both the rotation and the translation are load-bearing.
        let r_wc = Mat3F64::from_cols(
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
        );
        let centre = Vec3F64::new(2.0, 0.0, 3.0);
        let world_to_cam = Pose3d::new(r_wc, -(r_wc * centre));
        let tracked = TrackedPose {
            stamp: CuTime::from_nanos(0),
            pose_world_to_cam: world_to_cam,
            keyframe: false,
        };
        let back = tracked.cam_in_world();
        assert!(
            (back.translation - centre).length() < 1e-12,
            "cam_in_world gave {:?}, want {centre:?}",
            back.translation
        );
        // Negative control: negating the translation without applying R^T is
        // the bug, and it must give a different answer here — otherwise the
        // assertion above is satisfied for the wrong reason.
        assert!(
            (-world_to_cam.translation - centre).length() > 1.0,
            "pick a pose where -t and -R^T t actually differ"
        );
        // And the rotation half round-trips: R_cw * R_wc = I.
        let round_trip = back.rotation * world_to_cam.rotation;
        assert!(
            (round_trip - Mat3F64::IDENTITY)
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-12)
        );
    }

    /// On a parked robot the excitation gate holds `ready=false` forever, which is CORRECT, so
    /// "never ready" is the normal state and the throttle has to survive it without ever
    /// swallowing the transition out of it.
    #[test]
    fn test_the_init_window_line_is_throttled_but_never_loses_the_transition() {
        const RETRY: f64 = 5.0;
        let due = |last: Option<(f64, bool)>, now_sec: f64, ready: bool| {
            should_log_throttled(last, now_sec, ready, RETRY)
        };

        // First sighting always speaks.
        assert!(due(None, 100.0, false));

        // Same verdict inside the interval stays quiet; at the interval it speaks again.
        assert!(!due(Some((100.0, false)), 100.5, false));
        assert!(!due(Some((100.0, false)), 104.9, false));
        assert!(due(Some((100.0, false)), 105.0, false));

        // A FLIPPED verdict speaks immediately, however recently we spoke. This is the case a
        // plain time throttle would swallow, and it is the only line anyone actually waits for.
        assert!(due(Some((100.0, false)), 100.01, true));
        assert!(due(Some((100.0, true)), 100.01, false));
    }

    /// The truth table above is the whole of `should_log_throttled`, so the loss caller needs no
    /// second copy of it — only that its interval is a usable one.
    #[test]
    fn test_the_loss_throttle_interval_is_positive() {
        const { assert!(LOSS_LOG_RETRY_SEC > 0.0) };
    }

    /// A requested reset drops the map, so a loss logged against the OLD map must not silence the
    /// first loss of the new one — and the suppressed tally must not carry across, or it would
    /// report losses of a map that no longer exists.
    #[test]
    fn test_a_reset_clears_the_loss_throttle_and_its_tally() {
        let mut tracker = tracker_with_inertial();
        tracker.last_loss_log = Some((10.0, true));
        tracker.losses_suppressed = 17;

        tracker.reset();

        assert!(tracker.last_loss_log.is_none());
        assert_eq!(tracker.losses_suppressed, 0);
    }

    /// A re-bootstrap re-arms the window WITHOUT an `InertialState::reset` — a tracking loss with
    /// `reset_map_on_loss` off, the default, takes exactly that path. The throttle state has to go
    /// with the window, or the first description of the new one is silenced by its predecessor.
    #[test]
    fn test_re_arming_the_window_clears_the_line_throttle() {
        let mut tracker = tracker_with_inertial();
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        tracker.inertial.as_mut().unwrap().last_window_log = Some((10.0, false));

        tracker.arm_inertial_window(7, 1.0, EPOCH_NS);
        assert!(tracker.inertial.as_ref().unwrap().last_window_log.is_none());
    }

    /// Every field of the locally owned `SystemState`, set, then reset: what kornia-slam's own
    /// `reset` keeps (the pose the re-bootstrap re-anchors at, the frame clock) and what it
    /// clears. A struct literal, so a new field cannot be added without choosing its side.
    #[test]
    fn test_system_state_reset_keeps_the_pose_and_clears_everything_else() {
        let pose = Pose3d::from_rt(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 3.0));
        let mut state = SystemState {
            pose_world_to_cam: pose,
            velocity: Some(Pose3d::IDENTITY),
            velocity_world: Vec3F64::new(0.4, 0.0, 0.0),
            last_frame_timestamp_sec: 4.5,
            imu_initialized: true,
            imu_init_timestamp_sec: Some(1.5),
            current_keyframe_idx: Some(5),
            last_keyframe_idx: Some(6),
            lost_since_sec: Some(3.0),
            mode: SystemMode::Tracking,
        };
        state.reset();

        assert_eq!(pose_bits(&state.pose_world_to_cam), pose_bits(&pose));
        assert_eq!(state.last_frame_timestamp_sec, 4.5);

        assert_eq!(state.mode, SystemMode::Bootstrap);
        assert!(state.velocity.is_none());
        assert_eq!(state.velocity_world, Vec3F64::ZERO);
        assert!(
            !state.imu_initialized,
            "the re-anchored map needs a new initialization"
        );
        assert!(state.imu_init_timestamp_sec.is_none());
        assert!(state.current_keyframe_idx.is_none());
        assert!(state.last_keyframe_idx.is_none());
        assert!(state.lost_since_sec.is_none());
    }

    /// A refinement the map refuses is not a de-initialization: the tracker stays on the
    /// previous initialization (inertial BA keeps running on it), the world does not move, and
    /// the stage is marked done so it is not re-solved every keyframe.
    #[test]
    fn test_a_refused_refinement_keeps_the_previous_initialization_and_is_not_retried() {
        let mut tracker = tracker_with_inertial_ba(true);
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, Vec3F64::new(0.5, -0.25, 2.0)))
            .expect("a fresh keyframe id");
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, accepted_init(), 3, 2.5, InitStage::Initial);
        assert!(tracker.state.imu_initialized);
        let generation = tracker.world_generation();
        let bias = inert.bias;

        let refused = ImuInitResult {
            velocities_world: Vec::new(),
            ..accepted_init()
        };
        tracker.finish_refinement(&mut inert, Some(refused), 3, 6.0, InitStage::Refine1);
        tracker.inertial = Some(inert);

        assert!(tracker.state.imu_initialized);
        assert_eq!(tracker.world_generation(), generation);
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!((inert.bias.gyro, inert.bias.accel), (bias.gyro, bias.accel));
        assert!(
            inert.stats.first_refinement_done,
            "a refused stage is not retried"
        );
        assert!(!inert.stats.second_refinement_done);
        assert_eq!(
            (inert.stats.init_accepted, inert.stats.init_apply_refused),
            (1, 1)
        );
        // Inertial BA stays selected.
        let handles = tracker.on_keyframe_inertial(None, 4, EPOCH_NS + 7_000_000_000);
        assert!(handles.is_some());
    }

    /// One NaN accelerometer reading must not reach the map. A factor carrying it is admitted by
    /// `Map::apply_insertion`, and `Map::apply_ba_update` then refuses every BA write-back whose
    /// capture holds both its endpoints — visual BA included, with inertial BA off.
    #[test]
    fn test_a_non_finite_imu_sample_gives_no_factor_and_leaves_visual_ba_writable() {
        let mut tracker = tracker_with_inertial_ba(false);
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        let mut samples = imu_samples(t0_ns, 200);
        samples[40].accel[1] = f64::NAN;
        tracker.push_raw_imu(samples, 0, false).unwrap();
        insert_bare_keyframes(&mut tracker, &[7, 11]);

        tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        assert!(tracker.map().imu_factors().is_empty());
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.buffer.non_finite, 1);
        assert_eq!(
            stats.refused_dropped, 1,
            "the reading is a hole in its interval"
        );
        assert_eq!(stats.factors_added, 0);

        // A whole-map capture: `local_window` holds only the newest keyframes and their IMU
        // neighbours, so a NaN factor on 7->11 would drop out of it and pass unseen.
        let everything = kornia_slam::mapping::map::BaWindow {
            keyframe_slots: (0..tracker.map.keyframes().len()).collect(),
            landmark_ids: (0..tracker.map.map_points().len()).collect(),
        };
        let update = run_local_ba(tracker.map.ba_snapshot(&everything), &tracker.camera);
        tracker
            .map
            .apply_ba_update(update)
            .expect("visual BA must stay writable");
    }

    /// Duplicate tracked claims are reachable (`estimate_pose`'s reference-keyframe fallback
    /// matches one-directionally), and are resolved first-claim-wins before the links, as
    /// upstream's `keyframe_insertion` resolves them — not handed to the map as refusals.
    #[test]
    fn test_duplicate_tracked_claims_link_the_first_and_are_not_refusals() {
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        tracker
            .map
            .insert_keyframe(keyframe_with_features(0, 4))
            .expect("a fresh keyframe id");
        for i in 0..3 {
            tracker
                .map
                .insert_landmark(LandmarkSeed {
                    position: Vec3F64::new(0.1 * i as f64, 0.0, 2.0),
                    color: [0; 3],
                    reference: ObservationKey {
                        keyframe_idx: 0,
                        feature_idx: i,
                    },
                })
                .expect("a free feature and a finite position");
        }
        tracker.state.current_keyframe_idx = Some(0);
        tracker.state.last_keyframe_idx = Some(0);
        tracker.state.mode = SystemMode::Tracking;

        // (landmark, feature): landmarks 0 and 1 both claim feature 2, and landmark 0 also
        // claims feature 1.
        let matches = [(0, 2), (1, 2), (0, 1), (2, 3)];
        let frame = keyframe_with_features(10, 4).frame;
        assert!(tracker.try_insert_keyframe(&frame, 0.6, EPOCH_NS, 100, &matches));

        assert_eq!(tracker.map_writes_refused(), 0);
        let kf = tracker.map.get_keyframe(10).expect("inserted");
        assert_eq!(kf.map_point(2), Some(0), "the first claim wins");
        assert_eq!(kf.map_point(1), None, "landmark 0 already holds a feature");
        assert_eq!(kf.map_point(3), Some(2));
    }

    // ── End to end ───────────────────────────────────────────────────────────
    //
    // Through `process_stereo`, on the example's scene: the only tests that reach keyframe
    // insertion, the map's link/insert contracts and the snapshot -> solve -> `apply_ba_update`
    // path together.

    /// Metres the scene camera moves along +x per frame: an eighth of the 0.1 m baseline, so
    /// each band shifts `disparity / 8` whole pixels and the rendering needs no interpolation.
    const SCENE_STEP_M: f64 = 0.1 / 8.0;

    /// The example's calibration (`examples/stereo_vio.rs`): 320x240, fx = fy = 300.
    fn scene_tracker(inertial: Option<InertialConfig>) -> Tracker {
        let mut config = TrackerConfig::new(Length::new::<meter>(0.1));
        config.inertial = inertial;
        let camera = PinholeCamera {
            fx: 300.0,
            fy: 300.0,
            cx: 160.0,
            cy: 120.0,
            ..rectified_camera()
        };
        let mut tracker = Tracker::new(camera, config);
        tracker.epoch_ns = Some(EPOCH_NS);
        tracker
    }

    /// Frame `seq` of the example's scene: a 2x2-cell texture in three depth bands (16, 24 and
    /// 32 px disparity), the camera `seq * SCENE_STEP_M` along +x. The same scene as
    /// `trackable_frame` in the task tests.
    fn scene_pair(seq: u32) -> (Image<u8, 1>, Image<u8, 1>, CuTime) {
        const BANDS: [u32; 3] = [16, 24, 32];
        const W: u32 = 320;
        const H: u32 = 240;
        let texture = |x: u32, y: u32| {
            let h = (x / 2).wrapping_mul(73_856_093) ^ (y / 2).wrapping_mul(19_349_663);
            (h.wrapping_mul(2_654_435_761) >> 24) as u8
        };
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for y in 0..H {
            let d = BANDS[((y / (H / 3)) as usize).min(2)];
            let shift = seq * (d / 8);
            for x in 0..W {
                left.push(texture(x + shift, y));
                right.push(texture(x + d + shift, y));
            }
        }
        let size = ImageSize {
            width: W as usize,
            height: H as usize,
        };
        (
            Image::new(size, left).expect("sized"),
            Image::new(size, right).expect("sized"),
            CuTime::from_nanos(EPOCH_NS + u64::from(seq) * 66_666_667),
        )
    }

    fn feed(tracker: &mut Tracker, seq: u32) -> TrackStatus {
        let (left, right, stamp) = scene_pair(seq);
        tracker
            .process_stereo(&left, &right, stamp)
            .expect("a valid pair")
    }

    #[test]
    fn test_the_synthetic_scene_tracks_through_keyframe_insertion_and_local_ba() {
        let mut tracker = scene_tracker(None);
        let mut bootstrap_landmarks = 0;
        let mut worst_m = 0.0f64;
        for seq in 0..40 {
            let TrackStatus::Tracked(pose) = feed(&mut tracker, seq) else {
                panic!("frame {seq} was not tracked");
            };
            if seq == 0 {
                bootstrap_landmarks = tracker.map().map_points().len();
                assert!(bootstrap_landmarks > 0);
            }
            let truth = Vec3F64::new(SCENE_STEP_M * f64::from(seq), 0.0, 0.0);
            worst_m = worst_m.max((pose.cam_in_world().translation - truth).length());
        }

        // Every write this path made was accepted, BA included.
        assert_eq!(tracker.map_writes_refused(), 0);
        assert_eq!(tracker.ba_updates_refused(), 0);
        let kfs = tracker.map().keyframes();
        assert!(
            kfs.len() > BA_ACTIVE_KEYFRAMES,
            "{} keyframes: local BA never ran",
            kfs.len()
        );
        // Tracked matches were linked into the newest keyframe: it observes landmarks the
        // bootstrap keyframe created.
        let newest = kfs.last().unwrap();
        assert!(
            newest
                .map_point_by_desc_idx
                .iter()
                .flatten()
                .any(|&mp| mp < bootstrap_landmarks),
            "the newest keyframe holds no bootstrap landmark"
        );
        // Metric and unaligned: the baseline fixes the scale and the bootstrap fixes the frame.
        assert!(worst_m < 0.005, "worst camera-centre error {worst_m} m");
    }

    /// The bias a map drop keeps, and the velocity once initialized, must survive the keyframes
    /// inserted after it: `Keyframe::from_frame` zeroes both, and the sync-back after each
    /// insertion copies the newest keyframe's bias into the next edge's linearization point.
    #[test]
    fn test_a_kept_bias_survives_the_keyframes_inserted_after_a_map_drop() {
        let measured = ImuBias {
            gyro: Vec3F64::new(0.0026, -0.0011, 0.0009),
            accel: Vec3F64::new(0.01, -0.02, 0.03),
        };
        let mut tracker = scene_tracker(Some(inertial_config()));
        tracker.inertial.as_mut().unwrap().bias = measured;
        tracker.reset_keeping_calibration();

        // The bootstrap keyframe, then the one the keyframe policy forces at an 8-frame gap.
        for seq in 0..10 {
            feed(&mut tracker, seq);
        }
        let kfs = tracker.map().keyframes();
        assert!(kfs.len() >= 2, "{} keyframes", kfs.len());
        for kf in kfs {
            assert_eq!(
                (kf.imu_bias.gyro, kf.imu_bias.accel),
                (measured.gyro, measured.accel),
                "keyframe {} was not seeded with the kept bias",
                kf.frame.idx
            );
        }
        let bias = tracker.inertial.as_ref().unwrap().bias;
        assert_eq!((bias.gyro, bias.accel), (measured.gyro, measured.accel));

        // Once initialized, the current velocity seeds the next keyframe too.
        let velocity = Vec3F64::new(0.2, 0.0, -0.1);
        tracker.state.imu_initialized = true;
        tracker.state.velocity_world = velocity;
        let kf = tracker.seeded_keyframe(tracker.map().keyframes()[0].frame.clone());
        assert_eq!(kf.velocity_world, velocity);
        assert_eq!(kf.imu_bias.gyro, measured.gyro);
    }

    /// When a loss's re-bootstrap into the KEPT map cannot anchor on the loss frame itself, the
    /// next frame that does anchor is the same dead-reckoned coast and must be `Untracked` too.
    /// A dropped map needs no deferral: its fresh world starts at identity.
    #[test]
    fn test_a_deferred_loss_re_bootstrap_is_not_published_as_measured() {
        for reset_map_on_loss in [false, true] {
            let mut tracker = scene_tracker(None);
            tracker.config.reset_map_on_loss = reset_map_on_loss;
            assert!(matches!(feed(&mut tracker, 0), TrackStatus::Tracked(_)));

            // A blank pair: tracking fails and, with one keyframe (no established map), the
            // loss path runs at once; its re-bootstrap on this featureless frame fails too.
            let blank = Image::from_size_val(
                ImageSize {
                    width: 320,
                    height: 240,
                },
                0u8,
            )
            .unwrap();
            let stamp = CuTime::from_nanos(EPOCH_NS + 66_666_667);
            let status = tracker.process_stereo(&blank, &blank, stamp).unwrap();
            assert!(matches!(status, TrackStatus::Untracked));
            assert_eq!(tracker.mode(), SystemMode::Bootstrap);

            // The camera has not moved (an occlusion), so the re-anchor is where it should be;
            // what is under test is only whether it is reported as a measurement.
            let (left, right, _) = scene_pair(0);
            let stamp = CuTime::from_nanos(EPOCH_NS + 2 * 66_666_667);
            let status = tracker.process_stereo(&left, &right, stamp).unwrap();
            assert_eq!(tracker.mode(), SystemMode::Tracking);
            assert_eq!(
                matches!(status, TrackStatus::Untracked),
                !reset_map_on_loss,
                "reset_map_on_loss={reset_map_on_loss}: {status:?}"
            );

            // The next frame tracks against the re-anchored map and is measured again.
            let (left, right, _) = scene_pair(1);
            let stamp = CuTime::from_nanos(EPOCH_NS + 3 * 66_666_667);
            let status = tracker.process_stereo(&left, &right, stamp).unwrap();
            assert!(matches!(status, TrackStatus::Tracked(_)), "{status:?}");
        }
    }

    // ── Wheel odometry ───────────────────────────────────────────────────────

    const FRAME_NS: u64 = 66_666_667;

    /// The scene camera looks along base +x (OpenCV x right = base -y, y down = base -z), so
    /// the scene's camera +x motion is the base driving to its right, along odom -y.
    fn scene_wheel_config() -> WheelOdomConfig {
        let r_bc = crate::wheel::mat3_from_rows(&[0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
        WheelOdomConfig::new(Pose3d::new(r_bc, Vec3F64::new(0.1, 0.0, 0.9)), 0.01, 0.01)
            .expect("valid")
    }

    fn scene_tracker_with(f: impl FnOnce(&mut TrackerConfig)) -> Tracker {
        let mut config = TrackerConfig::new(Length::new::<meter>(0.1));
        f(&mut config);
        let camera = PinholeCamera {
            fx: 300.0,
            fy: 300.0,
            cx: 160.0,
            cy: 120.0,
            ..rectified_camera()
        };
        let mut tracker = Tracker::new(camera, config);
        tracker.epoch_ns = Some(EPOCH_NS);
        tracker
    }

    /// Wheel readings that agree exactly with the scene, one per frame stamp for `seqs`.
    fn push_scene_wheels(tracker: &mut Tracker, seqs: std::ops::Range<u32>) {
        let n = tracker
            .push_wheel(seqs.clone().map(|seq| WheelSample {
                stamp_ns: EPOCH_NS + u64::from(seq) * FRAME_NS,
                x: 0.0,
                y: -SCENE_STEP_M * f64::from(seq),
                theta: 0.0,
            }))
            .expect("the wheel path is on");
        assert_eq!(n, seqs.len());
    }

    fn blank_frame(tracker: &mut Tracker, seq: u32) -> TrackStatus {
        let blank = Image::from_size_val(
            ImageSize {
                width: 320,
                height: 240,
            },
            0u8,
        )
        .unwrap();
        let stamp = CuTime::from_nanos(EPOCH_NS + u64::from(seq) * FRAME_NS);
        tracker.process_stereo(&blank, &blank, stamp).unwrap()
    }

    /// The camera-centre error of a published pose against the scene's truth at `seq`.
    fn centre_error(pose: &TrackedPose, seq: u32) -> f64 {
        let truth = Vec3F64::new(SCENE_STEP_M * f64::from(seq), 0.0, 0.0);
        (pose.cam_in_world().translation - truth).length()
    }

    /// Off means off: wheel odometry configured but never fed is the stereo tracker bit for bit
    /// (the prior falls back to constant velocity, no factor reaches BA), and without the
    /// config the tracker refuses readings rather than ignoring them.
    #[test]
    fn test_wheel_odom_without_readings_is_bit_identical_to_off() {
        let mut off = scene_tracker(None);
        let mut idle = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        for seq in 0..30 {
            let (a, b) = (feed(&mut off, seq), feed(&mut idle, seq));
            let (TrackStatus::Tracked(a), TrackStatus::Tracked(b)) = (a, b) else {
                panic!("frame {seq}: both must track");
            };
            assert_eq!(
                pose_bits(&a.pose_world_to_cam),
                pose_bits(&b.pose_world_to_cam)
            );
        }
        assert!(matches!(
            off.push_wheel(std::iter::empty()),
            Err(VioError::WheelOdomDisabled)
        ));
        assert_eq!(off.wheel_stats(), None);
        // An unplaced frame: no payload without wheels, an honest "not localized" with them.
        assert!(matches!(blank_frame(&mut off, 30), TrackStatus::Untracked));
        assert!(matches!(
            blank_frame(&mut idle, 30),
            TrackStatus::NotLocalized(_)
        ));
    }

    /// With readings, every keyframe pair gets a factor and local BA takes the wheel path on a
    /// visual-only map, with nothing refused and the trajectory as accurate as without.
    #[test]
    fn test_wheel_factors_reach_local_ba_on_a_visual_only_map() {
        let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        push_scene_wheels(&mut tracker, 0..41);
        let mut worst_m = 0.0f64;
        for seq in 0..40 {
            let TrackStatus::Tracked(pose) = feed(&mut tracker, seq) else {
                panic!("frame {seq} was not tracked");
            };
            worst_m = worst_m.max(centre_error(&pose, seq));
        }
        let w = tracker.wheel_stats().expect("on");
        assert!(w.edges_added >= 3, "{w:?}");
        assert_eq!(w.edges_refused + w.edges_expired, 0, "{w:?}");
        assert!(w.prior_frames >= 39, "{w:?}");
        assert!(tracker.map().keyframes().len() > BA_ACTIVE_KEYFRAMES);
        assert_eq!(tracker.ba_updates_refused(), 0);
        assert!(worst_m < 0.005, "worst camera-centre error {worst_m} m");
    }

    /// The factors must actually reach the solve, not just be counted: wheels that over-read
    /// by 30 % pull the BA-refined keyframes off the vision-only answer, which a tracking prior
    /// alone (PnP converges onto vision from any nearby seed) cannot do.
    ///
    /// At the default floors (1 mm / 1 mrad) the pull at 30 % is ~6.5 mm. Dropping the heading
    /// floor to the encoders' quantisation (0.2 mrad) cut it to 1.2 mm on this turn-free scene,
    /// next to 0.9 mm with the factors withheld: a robot that lowers the floors should check its
    /// solve still converges onto the wheels before trusting them.
    #[test]
    fn test_biased_wheels_pull_the_ba_keyframes() {
        let keyframe_error = |scale: f64| {
            let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
            tracker
                .push_wheel((0..41u32).map(|seq| WheelSample {
                    stamp_ns: EPOCH_NS + u64::from(seq) * FRAME_NS,
                    x: 0.0,
                    y: -scale * SCENE_STEP_M * f64::from(seq),
                    theta: 0.0,
                }))
                .unwrap();
            for seq in 0..40 {
                feed(&mut tracker, seq);
            }
            // Keyframe `idx` is frame `idx`, so its truth is `idx` steps along +x.
            tracker
                .map()
                .keyframes()
                .iter()
                .map(|kf| {
                    let truth = Vec3F64::new(SCENE_STEP_M * kf.frame.idx as f64, 0.0, 0.0);
                    (kf.frame.pose_world_to_cam.inverse().translation - truth).length()
                })
                .fold(0.0f64, f64::max)
        };
        let (exact, biased) = (keyframe_error(1.0), keyframe_error(1.3));
        assert!(exact < 0.002, "exact wheels: {exact} m");
        assert!(
            biased > 0.002,
            "biased wheels moved no keyframe ({biased} m): BA never saw them"
        );
    }

    /// Vision goes dark for six frames while the robot keeps driving. The published pose must
    /// keep moving with the wheels (coasting) and come back onto the truth, not freeze and not
    /// jump, whichever way the loss path re-anchors.
    #[test]
    fn test_coasting_keeps_a_continuous_pose_over_a_vision_gap() {
        for reset_map_on_loss in [false, true] {
            let mut tracker = scene_tracker_with(|c| {
                c.wheel_odom = Some(scene_wheel_config());
                c.reset_map_on_loss = reset_map_on_loss;
            });
            push_scene_wheels(&mut tracker, 0..31);
            for seq in 0..12 {
                assert!(matches!(feed(&mut tracker, seq), TrackStatus::Tracked(_)));
            }
            for seq in 12..18 {
                let TrackStatus::Coasting(pose) = blank_frame(&mut tracker, seq) else {
                    panic!("frame {seq}: a dark frame with wheels must coast");
                };
                let err = centre_error(&pose, seq);
                assert!(err < 0.01, "reset={reset_map_on_loss} frame {seq}: {err} m");
            }
            let mut tracked_again = false;
            for seq in 18..30 {
                let pose = match feed(&mut tracker, seq) {
                    TrackStatus::Tracked(p) => {
                        tracked_again = true;
                        p
                    }
                    // The frame that re-anchors is itself a coast.
                    TrackStatus::Coasting(p) => p,
                    other => panic!("frame {seq}: {other:?}"),
                };
                let err = centre_error(&pose, seq);
                assert!(err < 0.01, "reset={reset_map_on_loss} frame {seq}: {err} m");
            }
            assert!(tracked_again, "vision never took over again");
        }
    }

    /// A `max_keyframes` drop must not move the world: the next map is seeded at the pose
    /// propagated on the wheels, so every published pose stays on the truth while the epoch
    /// still bumps for consumers that key on it.
    #[test]
    fn test_a_map_drop_keeps_the_world_continuous() {
        let mut tracker = scene_tracker_with(|c| {
            c.wheel_odom = Some(scene_wheel_config());
            c.max_keyframes = Some(3);
        });
        push_scene_wheels(&mut tracker, 0..41);
        let mut coasted = 0;
        for seq in 0..40 {
            let pose = match feed(&mut tracker, seq) {
                TrackStatus::Tracked(p) => p,
                TrackStatus::Coasting(p) => {
                    coasted += 1;
                    p
                }
                other => panic!("frame {seq}: {other:?}"),
            };
            let err = centre_error(&pose, seq);
            assert!(err < 0.01, "frame {seq}: {err} m from the truth");
        }
        assert!(
            tracker.world_generation() >= 1,
            "the bound never dropped the map"
        );
        assert_eq!(
            coasted,
            tracker.world_generation(),
            "each drop's re-anchor frame, and only it, is a coast"
        );

        // Without wheels the same drop restarts the world at identity: the regression this
        // guards against.
        let mut plain = scene_tracker_with(|c| c.max_keyframes = Some(3));
        let mut jumped = false;
        for seq in 0..40 {
            if let TrackStatus::Tracked(p) = feed(&mut plain, seq) {
                jumped |= centre_error(&p, seq) > 0.05;
            }
        }
        assert!(
            jumped,
            "the drop no longer restarts the plain tracker; this test lost its control"
        );
    }

    /// A requested reset is still a new world at identity, wheels or not.
    #[test]
    fn test_a_requested_reset_does_not_carry_the_pose() {
        let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        push_scene_wheels(&mut tracker, 0..21);
        for seq in 0..10 {
            feed(&mut tracker, seq);
        }
        tracker.reset();
        let TrackStatus::Tracked(pose) = feed(&mut tracker, 10) else {
            panic!("a fresh bootstrap is a measurement");
        };
        assert_eq!(
            pose_bits(&pose.pose_world_to_cam),
            pose_bits(&Pose3d::IDENTITY)
        );
    }

    /// Readings released only `lag` frames after their stamp, as they arrive over the head/body
    /// link: the coast must still follow the truth. Holding the newest reading lost the stretch
    /// `[newest, frame]` from every interval (12.5 mm a frame here), and with a lag over one
    /// period integrated nothing.
    #[test]
    fn test_coasting_on_lagging_readings_keeps_up_with_the_truth() {
        for lag in [1u32, 2] {
            let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
            let mut released = 0u32;
            let mut release = |tracker: &mut Tracker, seq: u32| {
                let upto = (seq + 1).saturating_sub(lag);
                if upto > released {
                    push_scene_wheels(tracker, released..upto);
                    released = upto;
                }
            };
            for seq in 0..12 {
                release(&mut tracker, seq);
                assert!(matches!(feed(&mut tracker, seq), TrackStatus::Tracked(_)));
            }
            for seq in 12..18 {
                release(&mut tracker, seq);
                let TrackStatus::Coasting(pose) = blank_frame(&mut tracker, seq) else {
                    panic!("lag {lag} frame {seq}: a dark frame with wheels must coast");
                };
                let err = centre_error(&pose, seq);
                assert!(err < 0.01, "lag {lag} frame {seq}: {err} m");
            }
        }
    }

    /// The position block's trace of a covariance, m^2.
    fn position_trace(c: &Cov6) -> f64 {
        c[0] + c[7] + c[14]
    }

    /// Driving through `max_keyframes` drops, every published pose carries a covariance, it ends
    /// the drive larger than it started, and a drop never resets it: the first pose of each new
    /// map (the coast it was anchored on) is at least as uncertain as the last measured pose of
    /// the map before. A zero there would read as a robot that just became perfectly localised.
    #[test]
    fn world_sigma_grows_along_a_drive_and_is_carried_across_a_drop() {
        let mut tracker = scene_tracker_with(|c| {
            c.wheel_odom = Some(scene_wheel_config());
            c.max_keyframes = Some(3);
        });
        push_scene_wheels(&mut tracker, 0..41);
        let mut last_measured: Option<f64> = None;
        let mut first: Option<f64> = None;
        let mut drops = 0;
        let mut generation = tracker.map_generation();
        for seq in 0..40 {
            let status = feed(&mut tracker, seq);
            let cov = tracker
                .frame_covariance()
                .unwrap_or_else(|| panic!("frame {seq}: {status:?} carries no covariance"));
            assert!(covariance::is_valid(&cov), "frame {seq}");
            let tr = position_trace(&cov);
            first.get_or_insert(tr);
            if tracker.map_generation() != generation {
                generation = tracker.map_generation();
                drops += 1;
                let before = last_measured.expect("a measured pose before the drop");
                assert!(
                    tr > before,
                    "frame {seq}: the new map starts at trace {tr:e}, the old one ended at {before:e}"
                );
            }
            if matches!(status, TrackStatus::Tracked(_)) {
                last_measured = Some(tr);
            }
        }
        assert!(drops >= 2, "only {drops} drops");
        let (first, end) = (first.expect("a pose"), last_measured.expect("a pose"));
        assert!(
            end > first,
            "trace {first:e} at the start, {end:e} at the end"
        );
    }

    /// Without wheels a drop restarts the world at identity, so its covariance restarts at zero
    /// too: the old world's covariance is in the old world's axes, and carried into each new one
    /// it stacked up without bound across drops.
    ///
    /// Red before the fix: `coast_origin` was kept on every drop, carried or not.
    #[test]
    fn a_world_restarted_at_identity_starts_its_covariance_at_zero() {
        let mut tracker = scene_tracker_with(|c| c.max_keyframes = Some(3));
        let mut generation = tracker.map_generation();
        let mut drops = 0;
        let mut grew = false;
        for seq in 0..40 {
            let status = feed(&mut tracker, seq);
            let Some(cov) = tracker.frame_covariance() else {
                continue;
            };
            let tr = position_trace(&cov);
            grew |= tr > 0.0;
            if tracker.map_generation() != generation {
                generation = tracker.map_generation();
                drops += 1;
                assert_eq!(
                    tr, 0.0,
                    "frame {seq}: {status:?} starts the new world at {tr:e}"
                );
            }
        }
        assert!(drops >= 2, "only {drops} drops");
        assert!(
            grew,
            "the covariance never grew; this test lost its control"
        );
    }

    /// Blind on the wheels, the covariance grows every frame and faster than it did over the
    /// same number of tracked frames before: the wheels' propagated covariance per metre, against
    /// vision re-observing the map.
    #[test]
    fn coast_sigma_grows_faster_than_tracked() {
        let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        push_scene_wheels(&mut tracker, 0..31);
        let mut trace = Vec::new();
        for seq in 0..12 {
            assert!(matches!(feed(&mut tracker, seq), TrackStatus::Tracked(_)));
            trace.push(position_trace(
                &tracker.frame_covariance().expect("tracked"),
            ));
        }
        for seq in 12..18 {
            assert!(matches!(
                blank_frame(&mut tracker, seq),
                TrackStatus::Coasting(_)
            ));
            let tr = position_trace(&tracker.frame_covariance().expect("coasting"));
            assert!(tr > *trace.last().expect("a frame"), "frame {seq}: {tr:e}");
            trace.push(tr);
        }
        let tracked = trace[11] - trace[5];
        let coasted = trace[17] - trace[11];
        assert!(
            coasted > 2.0 * tracked.max(0.0),
            "six coasted frames grew the trace {coasted:e}, six tracked ones {tracked:e}"
        );
    }

    /// One base step of the coast tests: a turn with a lever arm, so the mount term is not zero.
    fn coast_chunk() -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, 0.3)).matrix(),
            Vec3F64::new(0.2, 0.05, 0.0),
        )
    }

    fn assert_cov_close(a: &Cov6, b: &Cov6, what: &str) {
        let scale = a.iter().chain(b).fold(0.0f64, |m, v| m.max(v.abs()));
        for k in 0..36 {
            assert!(
                (a[k] - b[k]).abs() <= 1e-9 * scale,
                "{what}: entry {k}: {:e} vs {:e}",
                a[k],
                b[k]
            );
        }
    }

    /// The mount error is systematic, so a coast's mount term is that of the TOTAL motion since
    /// the measured origin: five 1 s chunks of a turn give exactly what one interval of the same
    /// motion gives, and a turn driven back cancels it. Summed per chunk (as if each chunk had
    /// its own independent mount error) it came out 4x too small here, and would never cancel.
    ///
    /// Red before the fix: every chunk's factor carried its own mount term into the re-base.
    #[test]
    fn a_coast_s_mount_term_is_the_total_motion_s_not_the_chunks() {
        let extrinsic = [0.03, 0.035];
        let zero = [0.0; 36];
        let t_wb0 = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, 0.7)).matrix(),
            Vec3F64::new(1.0, -2.0, 0.3),
        );
        let (step, n) = (coast_chunk(), 5u64);
        let mut chunked = CoastOrigin::measured(0, zero);
        let (mut t_wb, mut total) = (t_wb0, CoastOrigin::measured(0, zero).since_measured);
        for k in 1..=n {
            t_wb = t_wb.compose(&step);
            total = total.compose(&step);
            chunked = chunked.advanced(k * COAST_REBASE_NS, &zero, &step, &t_wb, extrinsic);
        }
        let single = CoastOrigin::measured(0, zero).advanced(
            n * COAST_REBASE_NS,
            &zero,
            &total,
            &t_wb,
            extrinsic,
        );
        assert!(
            position_trace(&single.cov) > 1e-4,
            "no mount term: the test lost its premise"
        );
        assert_cov_close(&chunked.cov, &single.cov, "chunked vs one interval");
        // The control: one independent mount term per chunk, each at its own end pose.
        let mut per_chunk = zero;
        let mut t = t_wb0;
        for _ in 0..n {
            t = t.compose(&step);
            per_chunk = covariance::add(
                &per_chunk,
                &covariance::congruence(
                    &covariance::pose_adjoint(&t),
                    &crate::fusion::extrinsic_covariance(&step, extrinsic),
                ),
            );
        }
        assert!(
            (position_trace(&per_chunk) - position_trace(&single.cov)).abs()
                > 0.1 * position_trace(&single.cov),
            "the per-chunk sum matches the total: the test cannot tell them apart"
        );
        // Driven back to where it was measured, the mount error cancels.
        let back = chunked.advanced(
            (n + 1) * COAST_REBASE_NS,
            &zero,
            &total.inverse(),
            &t_wb0,
            extrinsic,
        );
        assert!(
            position_trace(&back.cov) < 1e-20,
            "{:e}",
            position_trace(&back.cov)
        );
    }

    /// The wheels' noise is independent per chunk and keeps accumulating across re-bases.
    #[test]
    fn a_coast_s_wheel_noise_accumulates_per_chunk() {
        let mut noise = [0.0; 36];
        for k in 0..6 {
            noise[k * 7] = 1e-4;
        }
        let id = CoastOrigin::measured(0, [0.0; 36]).since_measured;
        let mut o = CoastOrigin::measured(0, [0.0; 36]);
        for k in 1..=4 {
            o = o.advanced(k * COAST_REBASE_NS, &noise, &coast_chunk(), &id, [0.0, 0.0]);
        }
        assert_cov_close(&o.cov, &noise.map(|v| 4.0 * v), "four chunks of noise");
        assert_eq!(o.cov, o.accumulated);
    }

    /// An interval the wheels did not measure grows the covariance at the worst-case rate for
    /// its length, on top of everything so far (the mount term of the motion before it
    /// included), and the next chunk restarts from there.
    #[test]
    fn an_unmeasured_coast_interval_grows_at_the_worst_case_rate() {
        let t_wb = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));
        let o = CoastOrigin::measured(0, [0.0; 36]).advanced(
            COAST_REBASE_NS,
            &[0.0; 36],
            &coast_chunk(),
            &t_wb,
            [0.03, 0.035],
        );
        let blind = o.unmeasured(3 * COAST_REBASE_NS);
        assert_eq!(blind.stamp_ns, 3 * COAST_REBASE_NS);
        assert_eq!(blind.cov, blind.accumulated);
        for k in 0..6 {
            let rate = if k < 3 {
                COAST_UNMEASURED_M_PER_S
            } else {
                COAST_UNMEASURED_RAD_PER_S
            };
            let grown = blind.cov[k * 7] - o.cov[k * 7];
            assert!(
                (grown - (2.0 * rate).powi(2)).abs() < 1e-12,
                "axis {k}: {grown}"
            );
        }
    }

    /// Readings with a gap longer than `max_gap` under a coast: the interval from the coast
    /// origin cannot be integrated while the frame-to-frame prior still can, and the coast's
    /// covariance must keep growing (and the frames be counted) rather than fall back to the
    /// origin's while the pose keeps moving.
    ///
    /// Red before the fix: `coast_covariance` returned the origin's covariance on a failed
    /// interval, so the trace dropped back to the last measured frame's.
    #[test]
    fn a_coast_across_a_reading_gap_never_loses_its_growth() {
        let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        push_scene_wheels(&mut tracker, 0..15);
        // 15..23 never arrive: 0.6 s between readings 14 and 23, past the 0.5 s `max_gap`.
        push_scene_wheels(&mut tracker, 23..41);
        for seq in 0..12 {
            assert!(matches!(feed(&mut tracker, seq), TrackStatus::Tracked(_)));
        }
        let mut last = position_trace(&tracker.frame_covariance().expect("tracked"));
        let mut coasted_after_gap = 0;
        for seq in 12..36 {
            let status = blank_frame(&mut tracker, seq);
            let Some(cov) = tracker.frame_covariance() else {
                continue;
            };
            if !matches!(status, TrackStatus::Coasting(_)) {
                continue;
            }
            let tr = position_trace(&cov);
            assert!(tr >= last, "frame {seq}: trace {tr:e} fell below {last:e}");
            last = tr;
            coasted_after_gap += usize::from(seq > 23);
        }
        assert!(
            coasted_after_gap > 0,
            "nothing coasted past the gap: the test lost its premise"
        );
        assert!(
            tracker.coast_unmeasured() > 0,
            "no unmeasured interval was counted"
        );
    }

    /// A parked robot inserts a keyframe every eight frames; chained one onto the next, each
    /// would add its own relative covariance and the parked robot's would grow for as long as it
    /// stood. Tracked against the landmarks the scene was mapped from, it chains from that
    /// keyframe instead ([`Tracker::chain_base`]) and stays flat.
    #[test]
    fn a_parked_robot_s_covariance_stays_flat_across_its_keyframes() {
        let mut tracker = scene_tracker(None);
        let (left, right, _) = scene_pair(0);
        let mut trace = Vec::new();
        let mut keyframes = 0;
        for k in 0..60u64 {
            let stamp = CuTime::from_nanos(EPOCH_NS + k * FRAME_NS);
            let status = tracker.process_stereo(&left, &right, stamp).expect("valid");
            let TrackStatus::Tracked(p) = status else {
                panic!("frame {k}: {status:?}");
            };
            keyframes += usize::from(p.keyframe);
            trace.push(position_trace(
                &tracker.frame_covariance().expect("tracked"),
            ));
        }
        assert!(
            keyframes >= 5,
            "only {keyframes} keyframes: the test lost its premise"
        );
        let (early, late) = (trace[10], trace[59]);
        assert!(
            late < 1.5 * early,
            "parked: trace {early:e} at frame 10, {late:e} at frame 59 ({keyframes} keyframes)"
        );
    }

    /// Wheel stamps on another clock (+10 s) leave every frame outside the readings: no factor,
    /// no prior, and a counted, logged condition instead of a silent pure-vision run.
    #[test]
    fn test_readings_on_another_clock_are_flagged() {
        let mut tracker = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        tracker
            .push_wheel((0..41u32).map(|seq| WheelSample {
                stamp_ns: EPOCH_NS + 10_000_000_000 + u64::from(seq) * FRAME_NS,
                x: 0.0,
                y: -SCENE_STEP_M * f64::from(seq),
                theta: 0.0,
            }))
            .unwrap();
        for seq in 0..40 {
            feed(&mut tracker, seq);
        }
        let w = tracker.wheel_stats().expect("on");
        assert_eq!((w.edges_added, w.prior_frames), (0, 0), "{w:?}");
        assert_eq!(w.frames_uncovered, 40, "{w:?}");
        // The same readings on the frames' clock are not flagged.
        let mut ok = scene_tracker_with(|c| c.wheel_odom = Some(scene_wheel_config()));
        push_scene_wheels(&mut ok, 0..41);
        for seq in 0..40 {
            feed(&mut ok, seq);
        }
        assert_eq!(ok.wheel_stats().unwrap().frames_uncovered, 0);
    }

    /// A carried map's inertial alignment must not move the robot: it rotates and scales about
    /// the world origin, so at 20 m out the synthetic init (scale 1.7, gravity ~5 deg off +y)
    /// would throw the pose ~14 m.
    #[test]
    fn test_a_carried_maps_inertial_alignment_keeps_the_pose_in_place() {
        let mut tracker = tracker_with_inertial_ba(true);
        tracker.wheel = Some(WheelState::new(scene_wheel_config()));
        tracker.world_anchored = true;
        tracker.reset_keeping_calibration();
        assert!(tracker.map_carried);
        let centre = Vec3F64::new(20.0, 0.3, -4.0);
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, centre))
            .expect("a fresh keyframe id");
        tracker
            .map
            .insert_keyframe(bare_keyframe(4, centre + Vec3F64::new(0.1, 0.0, 0.0)))
            .expect("a fresh keyframe id");
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let rotation_before = tracker.map.keyframes()[1].frame.pose_world_to_cam.rotation;
        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, accepted_init(), 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);

        assert!(tracker.state.imu_initialized);
        let kf = tracker.map.get_keyframe(4).unwrap().frame.pose_world_to_cam;
        let moved = (kf.inverse().translation - (centre + Vec3F64::new(0.1, 0.0, 0.0))).length();
        assert!(moved < 0.01, "the resumed keyframe moved {moved} m");
        assert_eq!(
            pose_bits(&tracker.state.pose_world_to_cam),
            pose_bits(&kf),
            "tracking resumes from the re-centred keyframe"
        );
        assert_ne!(kf.rotation, rotation_before, "gravity was still aligned");
        // The rest of the map moved with it, rigidly: the 0.1 m between the two scaled by 1.7.
        let other = tracker.map.get_keyframe(3).unwrap().frame.pose_world_to_cam;
        let gap = (other.inverse().translation - kf.inverse().translation).length();
        assert!((gap - 0.17).abs() < 1e-9, "{gap}");
    }

    // ── Stationary initialization ────────────────────────────────────────────

    /// The true gyro bias the parked rig's samples carry.
    const PARKED_GYRO_BIAS: [f64; 3] = [0.003, -0.002, 0.001];

    /// Gravity in the scene's visual world (the bootstrap camera frame): the camera pitched
    /// 22.45 deg down, as on XLeRobot, so world +y is NOT down.
    fn parked_gravity_vis() -> Vec3F64 {
        let pitch = 22.45f64.to_radians();
        Vec3F64::new(0.0, pitch.cos(), pitch.sin()) * GRAVITY_MAGNITUDE
    }

    fn static_inertial_config() -> InertialConfig {
        let mut c = inertial_config();
        c.static_init = Some(StaticInitConfig::from_noise(&c.calib, c.gates.nominal_rate));
        c
    }

    /// A parked rig's IMU samples over `(from_ns, to_ns]` (one sample past `to_ns`, so a window
    /// ending on a frame stamp has no trailing gap), on the 5 ms grid: the tilted IMU of
    /// `inertial_config` at rest in the scene's world, with a gyro bias and small deterministic
    /// noise, plus `extra(k)` on the accelerometer.
    fn parked_imu(
        tracker: &Tracker,
        from_ns: Option<u64>,
        to_ns: u64,
        extra: impl Fn(u64) -> Vec3F64,
    ) -> Vec<RawImuSample> {
        let r_bc = tracker.inertial.as_ref().unwrap().cfg.imu_t_bc.rotation;
        // At rest the camera is the bootstrap one (identity), so R_wb = R_cb and the specific
        // force in the body is R_bc * (-g).
        let f = r_bc * (-parked_gravity_vis());
        let bg = Vec3F64::from_array(PARKED_GYRO_BIAS);
        let first = from_ns.map_or(0, |t| (t - EPOCH_NS) / IMU_PERIOD_NS + 2);
        let last = (to_ns - EPOCH_NS) / IMU_PERIOD_NS + 1;
        (first..=last)
            .map(|k| {
                let j = k as f64;
                let jitter = Vec3F64::new((1.7 * j).sin(), (2.9 * j).sin(), (4.3 * j).sin());
                let a = f + jitter * 0.02 + extra(k);
                let w = bg + jitter * 0.002;
                RawImuSample {
                    stamp_ns: EPOCH_NS + k * IMU_PERIOD_NS,
                    gyro: [w.x, w.y, w.z],
                    accel: [a.x, a.y, a.z],
                }
            })
            .collect()
    }

    /// Feeds frame `seq` of a parked robot: the scene's first pair at `seq`'s stamp, after the
    /// IMU samples up to it and, with wheels, a reading at it that has the base `wheel_x(seq)`
    /// metres along.
    fn feed_parked(
        tracker: &mut Tracker,
        seq: u32,
        accel_extra: &dyn Fn(u64) -> Vec3F64,
        wheel_x: &dyn Fn(u32) -> f64,
    ) -> TrackStatus {
        feed_parked_frame(tracker, seq, accel_extra, wheel_x, false)
    }

    /// [`feed_parked`], with `blank` replacing the pair by two black images: no keypoints, so a
    /// bootstrap on it fails (the dark room).
    fn feed_parked_frame(
        tracker: &mut Tracker,
        seq: u32,
        accel_extra: &dyn Fn(u64) -> Vec3F64,
        wheel_x: &dyn Fn(u32) -> f64,
        blank: bool,
    ) -> TrackStatus {
        let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
        let from = seq
            .checked_sub(1)
            .map(|p| EPOCH_NS + u64::from(p) * FRAME_NS);
        let samples = parked_imu(tracker, from, stamp, accel_extra);
        tracker.push_raw_imu(samples, 0, false).unwrap();
        if tracker.wheel.is_some() {
            tracker
                .push_wheel([WheelSample {
                    stamp_ns: stamp,
                    x: wheel_x(seq),
                    y: 0.0,
                    theta: 0.0,
                }])
                .unwrap();
        }
        let (mut left, mut right, _) = scene_pair(0);
        if blank {
            let size = left.size();
            left = Image::new(size, vec![0u8; size.width * size.height]).expect("sized");
            right = Image::new(size, vec![0u8; size.width * size.height]).expect("sized");
        }
        tracker
            .process_stereo(&left, &right, CuTime::from_nanos(stamp))
            .expect("a valid pair")
    }

    fn no_extra(_: u64) -> Vec3F64 {
        Vec3F64::ZERO
    }

    fn parked_wheels(_: u32) -> f64 {
        0.0
    }

    /// The angle between the specific force the IMU measures at rest, mapped into the world
    /// through the tracker's current camera pose, and world UP (-y): zero once the world is
    /// gravity-aligned, ~22 deg in the scene's raw world.
    fn up_error_deg(tracker: &Tracker) -> f64 {
        let r_bc = tracker.inertial.as_ref().unwrap().cfg.imu_t_bc.rotation;
        let f_body = r_bc * (-parked_gravity_vis());
        let r_wb = tracker.state.pose_world_to_cam.inverse().rotation * r_bc.transpose();
        let up = (r_wb * f_body).normalize();
        up.dot(Vec3F64::new(0.0, -1.0, 0.0))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    /// THE behaviour: a robot that never moves initializes the inertial path from a still
    /// window, the world is rotated onto the IMU's gravity (from 22 deg off to under half a
    /// degree), consumers see it as a reset, the gyro bias is the true one, and tracking and
    /// inertial BA carry on with nothing refused. The first 1.5 s vibrate, so the still window
    /// starts after the map's anchor.
    ///
    /// Mutations (each turned this red): making `try_static_init` return `false` before looking
    /// (the pre-change behaviour: parked never initializes); dropping the still-window horizon
    /// from the wheel-reading prune (the window's start is evicted, `WheelsSilent` forever).
    #[test]
    fn test_a_parked_robot_initializes_from_a_still_window() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let mut generation_at_init = None;
        let settling = |k: u64| {
            let a = if k < 300 {
                0.5 * (k as f64 * 1.3).sin()
            } else {
                0.0
            };
            Vec3F64::new(a, 0.0, 0.0)
        };
        for seq in 0..60 {
            let status = feed_parked(&mut tracker, seq, &settling, &parked_wheels);
            assert!(
                matches!(status, TrackStatus::Tracked(_)),
                "frame {seq}: {status:?}"
            );
            if seq == 0 {
                assert!(
                    up_error_deg(&tracker) > 20.0,
                    "the scene world starts unaligned"
                );
            }
            if tracker.state.imu_initialized && generation_at_init.is_none() {
                generation_at_init = Some(tracker.world_generation());
            }
        }
        assert!(
            tracker.state.imu_initialized,
            "a parked robot never initialized"
        );
        assert_eq!(
            generation_at_init,
            Some(1),
            "the alignment is a world reset"
        );
        assert_eq!(tracker.world_generation(), 1, "and only one");
        let up = up_error_deg(&tracker);
        assert!(up < 0.5, "world up is {up} deg off the IMU's gravity");

        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.static_accepted, 1);
        assert_eq!(inert.stats.init_attempts, 0, "the motion solve never ran");
        assert_eq!(
            inert.gravity_world,
            Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)
        );
        let bg_err = (inert.bias.gyro - Vec3F64::from_array(PARKED_GYRO_BIAS)).length();
        assert!(bg_err < 5e-4, "gyro bias off by {bg_err} rad/s");
        assert!(
            tracker.state.velocity_world.length() < 0.05,
            "{:?}",
            tracker.state.velocity_world
        );
        assert!(
            inert.static_hold.is_some(),
            "the refinements wait for motion"
        );
        assert!(!inert.stats.first_refinement_done && !inert.stats.second_refinement_done);
        assert!(
            inert.stats.static_refused > 0,
            "the vibrating stretch was not refused"
        );
        assert_eq!(tracker.ba_updates_refused(), 0);
        assert_eq!(tracker.map_writes_refused(), 0);
    }

    /// The vetoes, end to end: wheels that report creeping, or an IMU that reads vibration,
    /// keep a camera-still robot from initializing; the same run with both still (the test
    /// above) initializes.
    ///
    /// Mutation: dropping the wheel check from `still_window` turned the first case red.
    #[test]
    fn test_moving_wheels_or_a_vibrating_imu_veto_the_still_window() {
        let creeping = |seq: u32| 0.004 * f64::from(seq);
        let vibrating = |k: u64| Vec3F64::new(0.5 * (k as f64 * 1.3).sin(), 0.0, 0.0);
        for (case, accel, wheels) in [
            (
                "wheels creeping",
                &no_extra as &dyn Fn(u64) -> Vec3F64,
                &creeping as &dyn Fn(u32) -> f64,
            ),
            ("imu vibrating", &vibrating, &parked_wheels),
        ] {
            let mut tracker = scene_tracker_with(|c| {
                c.inertial = Some(static_inertial_config());
                c.wheel_odom = Some(scene_wheel_config());
            });
            for seq in 0..45 {
                feed_parked(&mut tracker, seq, accel, wheels);
            }
            let inert = tracker.inertial.as_ref().unwrap();
            assert!(!tracker.state.imu_initialized, "{case}: initialized");
            assert_eq!(inert.stats.static_accepted, 0, "{case}");
            assert!(inert.stats.static_refused > 0, "{case}: never looked");
            assert_eq!(tracker.world_generation(), 0, "{case}");
        }
    }

    /// Static init must not block the motion solve: a window that is not still falls through to
    /// it on the same keyframe.
    ///
    /// Mutation: returning `true` from `try_static_init` on a refusal turned this red.
    #[test]
    fn test_a_refused_still_window_falls_through_to_the_motion_solve() {
        let mut config = TrackerConfig::new(Length::new::<meter>(0.075));
        config.inertial = Some(static_inertial_config());
        let mut tracker = Tracker::new(rectified_camera(), config);
        tracker.epoch_ns = Some(EPOCH_NS);
        // The `ready` window of the dry-acceptance test: 10 keyframes, 0.27 m of motion, and
        // samples turning at 0.05 rad/s, which the still window refuses as a turn.
        const N: usize = 10;
        const STEP_NS: u64 = 250_000_000;
        let stamp = |i: usize| EPOCH_NS + 1_000_000_000 + i as u64 * STEP_NS;
        for i in 0..N {
            tracker
                .map
                .insert_keyframe(bare_keyframe(i, Vec3F64::new(0.03 * i as f64, 0.0, 0.0)))
                .expect("a fresh keyframe id");
        }
        let n_samples = (N as u64 * STEP_NS / IMU_PERIOD_NS) as usize + 10;
        tracker
            .push_raw_imu(imu_samples(stamp(0), n_samples), 0, false)
            .unwrap();
        tracker.arm_inertial_window(0, 1.0, stamp(0));
        for i in 1..N {
            tracker.state.last_keyframe_idx = Some(i);
            tracker.on_keyframe_inertial(Some((i - 1, stamp(i - 1))), i, stamp(i));
        }
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.static_accepted, 0);
        assert!(inert.stats.static_refused >= 1);
        assert!(
            inert.stats.init_attempts >= 1,
            "the motion solve never ran behind a refused still window"
        );
    }

    /// A `max_keyframes` drop with wheels keeps the world, so it keeps the alignment: the next
    /// map is inertial from its first keyframe, with no second still window, no rotation and no
    /// generation bump beyond the drop's own. Without wheels the world restarts at identity and
    /// the still window runs again.
    ///
    /// Mutation: not setting `carried_alignment` in `reset_inner` turned the wheel case red
    /// (a second static init, generation 3).
    #[test]
    fn test_a_map_drop_carries_the_alignment_or_re_runs_the_still_window() {
        for wheels in [true, false] {
            let mut tracker = scene_tracker_with(|c| {
                c.inertial = Some(static_inertial_config());
                c.max_keyframes = Some(8);
                if wheels {
                    c.wheel_odom = Some(scene_wheel_config());
                }
            });
            let mut initialized_after_drop_at = None;
            for seq in 0..95 {
                feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
                if tracker.world_generation() >= 2
                    && tracker.state.imu_initialized
                    && initialized_after_drop_at.is_none()
                {
                    initialized_after_drop_at = Some((seq, tracker.map().keyframes().len()));
                }
            }
            let inert = tracker.inertial.as_ref().unwrap();
            let (_, kfs) = initialized_after_drop_at.expect("initialized again after the drop");
            assert_eq!(
                inert.gravity_world,
                Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)
            );
            let up = up_error_deg(&tracker);
            assert!(up < 0.5, "wheels={wheels}: up {up} deg");
            if wheels {
                assert_eq!(kfs, 1, "inertial from the new map's anchor keyframe");
                assert_eq!(inert.stats.static_accepted, 1, "no second still window");
                assert_eq!(tracker.world_generation(), 2, "init + drop, nothing else");
                assert!(inert.stats.first_refinement_done && inert.stats.second_refinement_done);
            } else {
                assert_eq!(inert.stats.static_accepted, 2, "the still window re-ran");
                assert_eq!(tracker.world_generation(), 3, "init + drop + re-init");
            }
            assert_eq!(tracker.ba_updates_refused(), 0, "wheels={wheels}");
        }
    }
    /// How far the IMU runs ahead of the frame being processed on the robot: the task drains
    /// the IMU queue right before the solve, by which time the samples of the ~60 ms the frame
    /// spent in capture, transfer and queueing are already in. 2026-10-09 loop test: every drop
    /// refused its first edge with an "IMU gap of 52-64 ms".
    const IMU_LEAD_NS: u64 = 60_000_000;

    /// [`feed_parked`] with the IMU [`IMU_LEAD_NS`] ahead of the frame, as on the robot.
    /// `pushed_to` is the stamp the previous call pushed samples up to.
    fn feed_parked_imu_ahead(
        tracker: &mut Tracker,
        seq: u32,
        gyro_extra: &dyn Fn(u64) -> Vec3F64,
        pushed_to: &mut Option<u64>,
    ) -> TrackStatus {
        let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
        let to = stamp + IMU_LEAD_NS;
        let mut samples = parked_imu(tracker, *pushed_to, to, no_extra);
        for s in &mut samples {
            let w = gyro_extra((s.stamp_ns - EPOCH_NS) / IMU_PERIOD_NS);
            s.gyro = [s.gyro[0] + w.x, s.gyro[1] + w.y, s.gyro[2] + w.z];
        }
        *pushed_to = Some(to);
        tracker.push_raw_imu(samples, 0, false).unwrap();
        tracker
            .push_wheel([WheelSample {
                stamp_ns: stamp,
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            }])
            .unwrap();
        let (left, right, _) = scene_pair(0);
        tracker
            .process_stereo(&left, &right, CuTime::from_nanos(stamp))
            .expect("a valid pair")
    }

    /// BUG 3 (2026-10-09): a map drop must not lose the IMU samples already buffered past the
    /// frame the next map anchors on. They were cleared with the dropped map, so the first edge
    /// of every new map started ~60 ms (the frame's latency) before the oldest sample left and
    /// was refused as a gap — a new map whose first keyframe pair had no IMU edge, i.e. a local
    /// BA window whose bias nothing tied to the carried one.
    ///
    /// Mutation: restoring `self.buffer.clear()` in `InertialState::reset` turned this red
    /// (one gap refusal per drop).
    #[test]
    fn test_a_map_drop_keeps_the_imu_samples_past_the_new_anchor() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
            c.max_keyframes = Some(8);
        });
        let mut pushed_to = None;
        for seq in 0..95 {
            feed_parked_imu_ahead(&mut tracker, seq, &no_extra, &mut pushed_to);
        }
        assert!(
            tracker.world_generation() >= 2,
            "init + at least one drop, got generation {}",
            tracker.world_generation()
        );
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.refused_gap, 0, "{stats:?}");
        assert_eq!(stats.refused_total(), 0, "{stats:?}");
        assert!(tracker.state.imu_initialized);
    }

    /// The 2026-10-09 loop test's diverged gyro bias: the commanded turn rate.
    fn implausible_bias(sane: ImuBias) -> ImuBias {
        ImuBias {
            gyro: Vec3F64::new(0.0, 0.0, 0.2845),
            accel: sane.accel,
        }
    }

    /// Writes `bias` onto every keyframe of the map, as a solver result would.
    fn write_map_bias(tracker: &mut Tracker, bias: ImuBias) {
        use kornia_slam::mapping::map::{InertialAlignment, KeyframeVelocity};
        let keyframe_velocities = tracker
            .map
            .keyframes()
            .iter()
            .map(|kf| KeyframeVelocity {
                keyframe_idx: kf.frame.idx,
                velocity_world: Vec3F64::ZERO,
            })
            .collect();
        tracker
            .map
            .apply_inertial_alignment(InertialAlignment {
                scale: 1.0,
                rotation: kornia_algebra::SO3F64::IDENTITY,
                keyframe_velocities,
                bias,
            })
            .expect("a valid alignment");
    }

    /// BUG 2 guard: a local BA update carrying a gyro bias above `BiasLimit` is refused WHOLE
    /// (map left as it was), the newest keyframe's plausible bias is what the next edge is
    /// linearized at, and the stop line counts it.
    ///
    /// Mutation: dropping the `implausible` check before `apply_ba_update` turned this red (BA
    /// wrote the window, `bias_rejected` 0).
    #[test]
    fn test_a_local_ba_update_with_an_implausible_bias_is_refused() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let mut seq = park_until_static_init(&mut tracker);
        while tracker.map().keyframes().len() < 5 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            seq += 1;
        }
        let sane = tracker.inertial.as_ref().unwrap().bias;
        let bad = implausible_bias(sane);
        write_map_bias(&mut tracker, bad);
        let poses: Vec<Pose3d> = tracker
            .map()
            .keyframes()
            .iter()
            .map(|kf| kf.frame.pose_world_to_cam)
            .collect();
        let (n, refused) = (
            tracker.map().keyframes().len(),
            tracker.ba_updates_refused(),
        );
        while tracker.map().keyframes().len() == n {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            seq += 1;
        }
        assert_eq!(tracker.ba_updates_refused(), refused + 1);
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.bias_rejected, 1);
        assert_eq!(inert.bias.gyro, sane.gyro, "the next edge's bias moved");
        for (kf, pose) in tracker.map().keyframes().iter().zip(&poses) {
            assert_eq!(
                kf.imu_bias.gyro, bad.gyro,
                "kf {} was written",
                kf.frame.idx
            );
            assert_eq!(kf.frame.pose_world_to_cam.translation, pose.translation);
        }
    }

    /// BUG 2 guard: an initialization whose bias is above `BiasLimit` is refused before
    /// anything is applied or counted as accepted — map, world generation and bias untouched.
    ///
    /// Mutation: dropping the `admits` check at the top of `accept_inertial_init` turned this
    /// red (the map rotated, generation 1, bias adopted).
    #[test]
    fn test_an_initialization_with_an_implausible_bias_is_refused() {
        let mut tracker = tracker_with_inertial_ba(true);
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, Vec3F64::new(0.5, -0.25, 2.0)))
            .expect("a fresh keyframe id");
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let pose_before = pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam);
        let mut init = accepted_init();
        init.bias.gyro = Vec3F64::new(0.0, 0.0, 0.2845);
        let mut inert = tracker.inertial.take().unwrap();
        let bias_before = inert.bias;
        tracker.accept_inertial_init(&mut inert, init, 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);
        assert_eq!(
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam),
            pose_before
        );
        assert_eq!(tracker.world_generation(), 0);
        assert!(!tracker.state.imu_initialized);
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.bias.gyro, bias_before.gyro);
        assert_eq!(inert.stats.init_accepted, 0);
        assert_eq!(inert.stats.bias_rejected, 1);
    }

    /// BUG 2 guard: a map drop never carries an implausible bias — not as the calibration and
    /// not with the alignment. The next map starts from `initial_bias` and initializes again;
    /// a plausible bias still carries (the drop-keeps-the-bias tests).
    ///
    /// Mutation: carrying the bias regardless (`inert.reset(keep_calibration)`) turned this red.
    #[test]
    fn test_a_map_drop_never_carries_an_implausible_bias() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let seq = park_until_static_init(&mut tracker);
        feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
        let sane = tracker.inertial.as_ref().unwrap().bias;
        tracker.inertial.as_mut().unwrap().bias = implausible_bias(sane);
        tracker.pending_map_reset = true;
        tracker.apply_pending_map_reset();
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.bias.gyro, inert.cfg.initial_bias.gyro);
        assert_eq!(inert.bias.accel, inert.cfg.initial_bias.accel);
        assert!(!inert.carried_alignment, "the alignment rode along");
        assert_eq!(inert.stats.bias_rejected, 1);
    }

    /// Runs a parked robot with wheels until the stationary initialization lands, returning the
    /// next frame sequence number.
    fn park_until_static_init(tracker: &mut Tracker) -> u32 {
        for seq in 0..80 {
            feed_parked(tracker, seq, &no_extra, &parked_wheels);
            if tracker.state.imu_initialized {
                return seq + 1;
            }
        }
        panic!("a parked robot never initialized");
    }

    /// Finding 3: a parked robot past both refinement times (5 s, 15 s) keeps the one static
    /// alignment. The refinements wait for motion rather than re-solving the parked window,
    /// which on stationary data separates nothing the still window did not, rotates the world
    /// again (a second wipe for every consumer) and spends each once-only stage.
    ///
    /// Mutation: dropping the `static_hold` gate from the refinement branch turned this red
    /// (both stages spent while parked).
    #[test]
    fn test_a_parked_robot_past_the_refinement_times_keeps_one_alignment() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let from = park_until_static_init(&mut tracker);
        // 18 s more at 15 fps.
        for seq in from..from + 270 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
        }
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(
            tracker.world_generation(),
            1,
            "one alignment, no refinement wipe"
        );
        assert!(
            !inert.stats.first_refinement_done && !inert.stats.second_refinement_done,
            "a refinement was spent on the parked window"
        );
        assert_eq!(inert.stats.init_attempts, 0, "no solve ran while parked");
        assert!(inert.static_hold.is_some());
        assert!(up_error_deg(&tracker) < 0.5);
    }

    /// The hold's release: the camera moving 0.2 m from the still window's newest keyframe
    /// re-arms the refinement window at the moving keyframe's predecessor, stamped at its edge's
    /// start, so the 5 s / 15 s schedule counts from the motion. Below the threshold nothing
    /// changes.
    ///
    /// Mutation: re-arming at the held keyframe instead of the edge into the newest turned this
    /// red.
    #[test]
    fn test_motion_releases_the_static_hold_and_restarts_the_refinement_clock() {
        let mut tracker = tracker_with_inertial_ba(true);
        tracker.epoch_ns = Some(EPOCH_NS);
        for (i, x) in [(0, 0.0), (1, 0.0), (2, 0.05)] {
            tracker
                .map
                .insert_keyframe(bare_keyframe(i, Vec3F64::new(x, 0.0, 0.0)))
                .expect("a fresh keyframe id");
        }
        let edge = |prev: usize, curr: usize, t0: f64, t1: f64| ImuFactor {
            prev_kf_idx: prev,
            curr_kf_idx: curr,
            preintegrated: PreintegratedImu::new(ImuBias::default(), inertial_config().calib),
            raw_samples: Vec::new(),
            t0,
            t1,
        };
        let mut inert = tracker.inertial.take().unwrap();
        inert.static_hold = Some(1);
        inert.start_kf_idx = Some(0);
        inert.window_start_sec = Some(0.0);

        // Keyframe 2: 5 cm from the held one. Held.
        tracker
            .map
            .apply_insertion(MapInsertion {
                imu_factors: vec![edge(1, 2, 1.0, 2.0)],
                ..Default::default()
            })
            .unwrap();
        assert!(!tracker.release_static_hold(&mut inert, 2.0));
        assert_eq!(inert.static_hold, Some(1));
        assert_eq!(inert.start_kf_idx, Some(0));

        // Keyframe 3: 0.4 m away. Released, re-armed at keyframe 2 from t = 2.0.
        tracker
            .map
            .insert_keyframe(bare_keyframe(3, Vec3F64::new(0.4, 0.0, 0.0)))
            .expect("a fresh keyframe id");
        tracker
            .map
            .apply_insertion(MapInsertion {
                imu_factors: vec![edge(2, 3, 2.0, 3.0)],
                ..Default::default()
            })
            .unwrap();
        assert!(tracker.release_static_hold(&mut inert, 3.0));
        assert_eq!(inert.static_hold, None);
        assert_eq!(inert.start_kf_idx, Some(2));
        assert_eq!(inert.window_start_sec, Some(2.0));
    }

    /// Finding 2: a carried alignment does not carry a velocity from before the drop. A robot
    /// that drove into a `max_keyframes` drop at 0.5 m/s, then sat in the dark (the bootstrap
    /// failing for 2 s) and parked, resumes on the new map at the velocity the wheels measure at
    /// the bootstrap, ~0, with the alignment and no extra generation bump.
    ///
    /// Mutation: seeding the anchor from the pre-drop `velocity_world` (the original carry)
    /// turned this red at 0.5 m/s.
    #[test]
    fn test_a_carried_alignment_takes_the_bootstrap_velocity_not_the_stale_one() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let from = park_until_static_init(&mut tracker);
        for seq in from..from + 5 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
        }
        // The drop arrives with the tracker believing it is moving.
        tracker.state.velocity_world = Vec3F64::new(0.5, 0.0, 0.0);
        tracker.reset_keeping_calibration();
        let generation = tracker.world_generation();
        let from = from + 5;
        for seq in from..from + 30 {
            feed_parked_frame(&mut tracker, seq, &no_extra, &parked_wheels, true);
            assert!(
                !tracker.state.imu_initialized,
                "no map, nothing initialized"
            );
        }
        let mut resumed = None;
        for seq in from + 30..from + 40 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            if tracker.state.imu_initialized {
                resumed = Some(tracker.state.velocity_world);
                break;
            }
        }
        let v = resumed.expect("the alignment carried into the new map");
        assert!(v.length() < 0.01, "anchor velocity {v:?}");
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(
            inert.stats.static_accepted, 1,
            "carried, not re-initialized"
        );
        assert_eq!(inert.stats.carry_dropped, 0);
        assert_eq!(
            tracker.world_generation(),
            generation,
            "no bump beyond the drop's"
        );
        let anchor = tracker.map.keyframes().first().unwrap();
        assert!(
            anchor.velocity_world.length() < 0.01,
            "{:?}",
            anchor.velocity_world
        );
    }

    /// Finding 2, the fallback: with no wheel velocity at the re-bootstrap (here, wheel readings
    /// stop at the drop) the alignment is dropped rather than resumed on a guess, and the still
    /// window initializes again.
    ///
    /// Mutation: adopting the alignment with a zero velocity when none is measured turned this
    /// red.
    #[test]
    fn test_a_carried_alignment_without_a_wheel_velocity_re_initializes() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let from = park_until_static_init(&mut tracker);
        tracker.reset_keeping_calibration();
        let generation = tracker.world_generation();
        // Frames resume 30 s later, the wheel ring holding nothing near them.
        let from = from + 450;
        let mut stamp_wheels = false;
        for seq in from..from + 60 {
            let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
            let prev = EPOCH_NS + u64::from(seq - 1) * FRAME_NS;
            let samples = parked_imu(&tracker, Some(prev), stamp, no_extra);
            tracker.push_raw_imu(samples, 0, false).unwrap();
            if stamp_wheels {
                tracker
                    .push_wheel([WheelSample {
                        stamp_ns: stamp,
                        x: 0.0,
                        y: 0.0,
                        theta: 0.0,
                    }])
                    .unwrap();
            }
            let (left, right, _) = scene_pair(0);
            tracker
                .process_stereo(&left, &right, CuTime::from_nanos(stamp))
                .expect("a valid pair");
            // Readings return after the bootstrap, so the still window can vouch for the base.
            stamp_wheels = true;
        }
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.carry_dropped, 1);
        assert_eq!(inert.stats.static_accepted, 2, "the still window ran again");
        assert!(tracker.state.imu_initialized);
        assert_eq!(
            tracker.world_generation(),
            generation + 1,
            "re-init is a reset"
        );
    }

    /// A keyframe for the bare-map tests that counts as stereo.
    fn stereo_bare_keyframe(idx: usize, centre: Vec3F64) -> Keyframe {
        let mut frame = bare_keyframe(idx, centre).frame;
        frame.features = kornia_imgproc::features::OrbFeatures {
            keypoints_xy: vec![[10.0, 10.0]],
            orientations: vec![0.0],
            descriptors: vec![[0u8; 32]],
            octaves: vec![0],
        };
        frame.keypoint_colors = vec![[0; 3]];
        frame.u_right = vec![5.0];
        frame.depth = vec![2.0];
        Keyframe::from_frame(frame)
    }

    /// Finding 4: a robot that drove 0.1 m/s, stopped, and initialized from the still window
    /// gives the keyframes it drove through their visual velocity, not the zero the still window
    /// writes, so the IMU edges spanning that motion do not enter VI-BA claiming rest at both
    /// ends. The still keyframes stay at zero, and every keyframe takes the new bias.
    ///
    /// Mutation: applying from the still window's first keyframe only (the original behaviour)
    /// turned this red.
    #[test]
    fn test_keyframes_before_the_still_window_get_their_visual_velocity() {
        let mut config = TrackerConfig::new(Length::new::<meter>(0.075));
        config.inertial = Some(static_inertial_config());
        let mut tracker = Tracker::new(rectified_camera(), config);
        tracker.epoch_ns = Some(EPOCH_NS);
        const STEP_NS: u64 = 250_000_000;
        let stamp = |i: usize| EPOCH_NS + 1_000_000_000 + i as u64 * STEP_NS;
        // 0..=3 drive 2.5 cm per keyframe (0.1 m/s), 3..=8 sit still.
        let x = |i: usize| 0.025 * i.min(3) as f64;
        for i in 0..9 {
            tracker
                .map
                .insert_keyframe(stereo_bare_keyframe(i, Vec3F64::new(x(i), 0.0, 0.0)))
                .unwrap_or_else(|e| panic!("kf {i}: {e}"));
        }
        let n = (9 * STEP_NS / IMU_PERIOD_NS) as usize + 10;
        let samples: Vec<RawImuSample> = (0..n)
            .map(|k| {
                let j = k as f64;
                RawImuSample {
                    stamp_ns: stamp(0) + k as u64 * IMU_PERIOD_NS,
                    gyro: [0.002, -0.001, 0.0015],
                    accel: [0.01 * (1.7 * j).sin(), 9.81, 0.01 * (2.9 * j).sin()],
                }
            })
            .collect();
        tracker.push_raw_imu(samples, 0, false).unwrap();
        tracker.arm_inertial_window(0, 1.0, stamp(0));
        for i in 1..9 {
            tracker.state.last_keyframe_idx = Some(i);
            tracker.on_keyframe_inertial(Some((i - 1, stamp(i - 1))), i, stamp(i));
        }
        assert!(
            tracker.state.imu_initialized,
            "the still window initialized"
        );
        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.static_accepted, 1);
        assert_eq!(inert.stats.init_attempts, 0, "the motion solve never ran");
        let bias = inert.bias;
        for kf in tracker.map.keyframes() {
            let speed = kf.velocity_world.length();
            match kf.frame.idx {
                0..=2 => assert!(
                    (speed - 0.1).abs() < 1e-6,
                    "kf {} drove at 0.1 m/s, has {speed}",
                    kf.frame.idx
                ),
                3 => {}
                _ => assert!(speed < 1e-9, "still kf {} has {speed}", kf.frame.idx),
            }
            assert_eq!(kf.imu_bias.gyro, bias.gyro, "kf {}", kf.frame.idx);
        }
    }

    /// Finding 6: a stationary result the map refuses is not retried on every keyframe. The
    /// still window is searched again only after `init_retry` (5 s), as the motion solve is.
    ///
    /// Mutation: dropping the `static_map_refused_at` throttle turned this red (one refusal per
    /// keyframe).
    #[test]
    fn test_a_map_refused_static_init_is_throttled() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        tracker.inertial.as_mut().unwrap().refuse_static_alignment = true;
        let mut first_refusal = None;
        for seq in 0..120 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            let inert = tracker.inertial.as_ref().unwrap();
            if inert.stats.init_apply_refused > 0 && first_refusal.is_none() {
                first_refusal = Some((seq, tracker.map.keyframes().len()));
            }
        }
        let (seq0, kfs0) = first_refusal.expect("the map refused the still window");
        let inert = tracker.inertial.as_ref().unwrap();
        let elapsed_s = f64::from(120 - seq0) * FRAME_NS as f64 * 1e-9;
        let kfs_since = tracker.map.keyframes().len() - kfs0;
        assert!(kfs_since > 3, "too few keyframes to tell: {kfs_since}");
        let allowed = 1 + (elapsed_s / 5.0).floor() as u64;
        assert!(
            inert.stats.init_apply_refused <= allowed,
            "{} map refusals in {elapsed_s:.1} s over {kfs_since} keyframes",
            inert.stats.init_apply_refused
        );
        assert!(!tracker.state.imu_initialized);
        assert_eq!(tracker.world_generation(), 0);
    }

    // ── Motion consistency ───────────────────────────────────────────────────

    /// The scene's base yaw axis in the camera: `scene_wheel_config` mounts the camera with
    /// base +z along camera -y.
    fn base_up_in_cam() -> Vec3F64 {
        Vec3F64::new(0.0, -1.0, 0.0)
    }

    /// A frame of FROZEN vision (the scene's first pair, as a head pointed at the robot's own
    /// body sees) after the IMU samples up to it, the gyro turning the base at `gyro_rad_s`, and
    /// a wheel reading at heading `wheel_theta(seq)`.
    fn feed_frozen_turn(
        tracker: &mut Tracker,
        seq: u32,
        gyro_rad_s: f64,
        wheel_theta: &dyn Fn(u32) -> f64,
    ) {
        let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
        let from = seq
            .checked_sub(1)
            .map(|p| EPOCH_NS + u64::from(p) * FRAME_NS);
        let r_bc = tracker.inertial.as_ref().unwrap().cfg.imu_t_bc.rotation;
        let turn = r_bc * (base_up_in_cam() * gyro_rad_s);
        let mut samples = parked_imu(tracker, from, stamp, no_extra);
        for s in &mut samples {
            s.gyro = [s.gyro[0] + turn.x, s.gyro[1] + turn.y, s.gyro[2] + turn.z];
        }
        tracker.push_raw_imu(samples, 0, false).unwrap();
        tracker
            .push_wheel([WheelSample {
                stamp_ns: stamp,
                x: 0.0,
                y: 0.0,
                theta: wheel_theta(seq),
            }])
            .unwrap();
        let (left, right, _) = scene_pair(0);
        tracker
            .process_stereo(&left, &right, CuTime::from_nanos(stamp))
            .expect("a valid pair");
    }

    /// THE LOOP TEST, synthetic: the base turns at 0.3 rad/s, the wheels and the gyro both say
    /// so, and vision is frozen. The check calls vision the odd one out and holds local BA off
    /// while such an interval is in its window: no solve reconciles the three by bending the
    /// keyframes or the bias (on the robot it put the turn rate into the gyro bias), every
    /// keyframe of the frozen turn keeps the pose vision tracked, and the bias the next edge is
    /// linearized at is the parked one.
    ///
    /// Mutation: running local BA regardless (`ran_ba = ba_window_ready`) turned this red (the
    /// solves moved the turn's keyframes off their tracked poses).
    #[test]
    fn test_frozen_vision_in_a_turn_holds_local_ba_off() {
        const RATE: f64 = 0.3;
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let start = park_until_static_init(&mut tracker);
        let refused_before = tracker.ba_updates_refused();
        let t = |seq: u32| f64::from(seq - start) * FRAME_NS as f64 * 1e-9;
        for seq in start..start + 60 {
            feed_frozen_turn(&mut tracker, seq, RATE, &|s| RATE * t(s));
        }
        let c = tracker.consistency_stats();
        assert!(c.vision_disagrees >= 2, "{c:?}");
        assert_eq!(c.wheels_disagree, 0, "{c:?}");
        assert!(c.ba_skipped >= 2, "{c:?}");
        assert_eq!(
            tracker.ba_updates_refused(),
            refused_before,
            "a solve over frozen vision ran and proposed a bias"
        );
        let inert = tracker.inertial.as_ref().unwrap();
        let bg_err = (inert.bias.gyro - Vec3F64::from_array(PARKED_GYRO_BIAS)).length();
        assert!(bg_err < 5e-3, "gyro bias moved by {bg_err} rad/s");
        // Past the first interval (which straddles the start of the turn), nothing moved them.
        let mut checked = 0;
        for kf in tracker.map().keyframes() {
            if kf.frame.idx <= start as usize + 8 {
                continue;
            }
            let (_, _, tracked) = tracker
                .kf_tracked_poses
                .iter()
                .find(|(i, _, _)| *i == kf.frame.idx)
                .expect("a recent keyframe");
            assert_eq!(
                pose_bits(&kf.frame.pose_world_to_cam),
                pose_bits(tracked),
                "kf {} was moved by BA",
                kf.frame.idx
            );
            checked += 1;
        }
        assert!(checked >= 3, "{checked} keyframes in the frozen turn");
    }

    /// Wheels that report a 0.3 rad/s turn the gyro and vision do not see: slip (or a base
    /// lifted off the floor). Every such interval's factor is dropped before a solve sees it,
    /// and BA runs on (it is the wheels that are wrong).
    ///
    /// Mutation: not dropping the factor (`check_motion` returning nothing) turned this red.
    #[test]
    fn test_slipping_wheels_lose_their_factor() {
        const RATE: f64 = 0.3;
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let start = park_until_static_init(&mut tracker);
        let t = |seq: u32| f64::from(seq - start) * FRAME_NS as f64 * 1e-9;
        for seq in start..start + 60 {
            feed_frozen_turn(&mut tracker, seq, 0.0, &|s| RATE * t(s));
        }
        let c = tracker.consistency_stats();
        assert!(c.wheels_disagree >= 2, "{c:?}");
        assert_eq!((c.vision_disagrees, c.ba_skipped), (0, 0), "{c:?}");
        let turning = tracker
            .wheel
            .as_ref()
            .unwrap()
            .factors()
            .iter()
            .filter(|f| f.measurement.delta[2].abs() > 0.05)
            .count();
        assert_eq!(turning, 0, "a slipping interval's factor reached BA");
    }

    /// Parked, all three agree: every checked interval is consistent and nothing is dropped.
    #[test]
    fn test_a_parked_robot_is_consistent() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let start = park_until_static_init(&mut tracker);
        for seq in start..start + 40 {
            feed_frozen_turn(&mut tracker, seq, 0.0, &|_| 0.0);
        }
        let c = tracker.consistency_stats();
        assert!(c.checked >= 3, "{c:?}");
        assert_eq!(c.consistent, c.checked, "{c:?}");
        assert_eq!(c.ba_skipped, 0);
    }

    /// One frame of a STILL robot (parked IMU) whose camera sees the scene from `view` (camera in
    /// the scene's world, identity = the parked view), with a wheel reading `(x, theta)` at the
    /// frame, or none (the stream silent).
    fn feed_still(tracker: &mut Tracker, seq: u32, view: &Pose3d, wheel: Option<(f64, f64)>) {
        let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
        let from = seq
            .checked_sub(1)
            .map(|p| EPOCH_NS + u64::from(p) * FRAME_NS);
        let samples = parked_imu(tracker, from, stamp, no_extra);
        tracker.push_raw_imu(samples, 0, false).unwrap();
        if let Some((x, theta)) = wheel {
            tracker
                .push_wheel([WheelSample {
                    stamp_ns: stamp,
                    x,
                    y: 0.0,
                    theta,
                }])
                .unwrap();
        }
        let (left, right) = render_pair(view);
        tracker
            .process_stereo(&left, &right, CuTime::from_nanos(stamp))
            .expect("a valid pair");
    }

    /// A tracker on the parked scene with the wheels on and, when given, the parked interval.
    fn parked_gate_tracker(interval_s: Option<f64>) -> Tracker {
        scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            let mut w = scene_wheel_config();
            w.parked_keyframe_interval = interval_s.map(Time::new::<cu29::units::si::time::second>);
            c.wheel_odom = Some(w);
        })
    }

    /// 2026-10-11, parked robot: kornia-slam's policy inserted a keyframe every 8 frames, so the
    /// map hit `max_keyframes` and dropped every ~75 s. With the interval set, a parked robot
    /// inserts about one keyframe per interval once initialized; until then, and with it unset,
    /// exactly what it did before.
    #[test]
    fn test_a_parked_robot_withholds_keyframes_for_the_interval() {
        let frames = 150; // 10 s at 15 Hz after the initialization
        let count = |interval_s: Option<f64>| {
            let mut t = parked_gate_tracker(interval_s);
            let at_init = park_still_until_initialized(&mut t);
            for seq in at_init.0..at_init.0 + frames {
                feed_still(&mut t, seq, &Pose3d::IDENTITY, Some((0.0, 0.0)));
            }
            let skipped = t.wheel_stats().unwrap().parked_keyframes_skipped;
            (at_init, t.map.keyframes().len() - at_init.1, skipped)
        };
        let (off_init, off, off_skipped) = count(None);
        let (on_init, on, on_skipped) = count(Some(4.0));
        assert_eq!(on_init, off_init, "the still window waits for no gate");
        assert_eq!(off_skipped, 0);
        assert!(off >= 10, "the policy's forced keyframes: {off}");
        // One per 4 s over 10 s: never none, at most 3.
        assert!((1..=3).contains(&on), "parked keyframes {on} (off: {off})");
        assert!(on_skipped > 0);
    }

    /// The gate withholds only while ALL still signals hold: wheels driving a few millimetres a
    /// frame, or the camera turning with the wheels at rest (a head pan), insert as before.
    #[test]
    fn test_parked_gate_yields_to_wheel_motion_and_to_a_camera_turn() {
        let base = {
            let mut t = parked_gate_tracker(None);
            let (s0, k0) = park_still_until_initialized(&mut t);
            for seq in s0..s0 + 60 {
                feed_still(&mut t, seq, &Pose3d::IDENTITY, Some((0.0, 0.0)));
            }
            t.map.keyframes().len() - k0
        };
        // Wheels creep 2 mm a frame (the camera does not see it; the gate must still yield).
        let mut wheels = parked_gate_tracker(Some(100.0));
        let (s0, k0) = park_still_until_initialized(&mut wheels);
        for seq in s0..s0 + 60 {
            let x = 0.002 * f64::from(seq - s0);
            feed_still(&mut wheels, seq, &Pose3d::IDENTITY, Some((x, 0.0)));
        }
        assert_eq!(wheels.wheel_stats().unwrap().parked_keyframes_skipped, 0);
        assert_eq!(wheels.map.keyframes().len() - k0, base);
        // Parked 20 frames, then the camera turns 0.1 deg a frame with the wheels at rest: from
        // 0.5 deg on it inserts.
        let mut pan = parked_gate_tracker(Some(100.0));
        let (s0, k0) = park_still_until_initialized(&mut pan);
        for seq in s0..s0 + 60 {
            let deg = 0.1 * f64::from((seq - s0).saturating_sub(20));
            feed_still(&mut pan, seq, &yaw_view(deg), Some((0.0, 0.0)));
        }
        let skipped = pan.wheel_stats().unwrap().parked_keyframes_skipped;
        let added = pan.map.keyframes().len() - k0;
        assert!(skipped > 0, "the first 20 frames are parked");
        assert!(added >= 3, "the pan must insert: {added} (still: {base})");
    }

    /// A reject run that expires suspends the fusion until the next keyframe and relies on the
    /// policy forcing one within `max_frames_between`. On a still robot the parked gate withheld
    /// it, leaving the fusion off for the rest of the interval.
    ///
    /// Red before the fix: no keyframe in the frames after the suspension.
    #[test]
    fn test_parked_gate_never_withholds_the_keyframe_a_suspended_fusion_waits_for() {
        let mut t = parked_gate_tracker(Some(100.0));
        let (s0, _) = park_still_until_initialized(&mut t);
        let mut seq = s0;
        for _ in 0..20 {
            feed_still(&mut t, seq, &Pose3d::IDENTITY, Some((0.0, 0.0)));
            seq += 1;
        }
        assert!(t.wheel_stats().unwrap().parked_keyframes_skipped > 0);
        // What the reject-run expiry in `tracking_step` sets.
        let suspended = t.state.last_keyframe_idx;
        t.fusion_suspended_on = suspended;
        let (k0, budget) = (
            t.map.keyframes().len(),
            t.config.keyframe_policy.max_frames_between,
        );
        for _ in 0..budget {
            feed_still(&mut t, seq, &Pose3d::IDENTITY, Some((0.0, 0.0)));
            seq += 1;
        }
        assert!(
            t.map.keyframes().len() > k0,
            "no keyframe within {budget} frames of the suspension"
        );
        assert_ne!(t.state.last_keyframe_idx, suspended);
    }

    /// The robot's config: a 10 s parked interval at the real 200 Hz with the IMU drained 300 ms
    /// ahead of the frame (a slow solve). The interval's keyframe edge reaches back the full 10 s
    /// and must still be integrated, not refused as a leading gap.
    ///
    /// Red before the fix: `refused_gap` 1, the ring holding 10.24 s and the edge needing 10.3 s.
    #[test]
    fn test_a_full_parked_interval_keeps_its_imu_edge_with_the_imu_ahead() {
        const LEAD_NS: u64 = 300_000_000;
        let mut t = parked_gate_tracker(Some(10.0));
        let (s0, k0) = park_still_until_initialized(&mut t);
        let refused = |t: &Tracker| t.inertial.as_ref().unwrap().stats.refused_gap;
        let refused0 = refused(&t);
        let mut imu_to = EPOCH_NS + u64::from(s0 - 1) * FRAME_NS;
        let frames = (10.6e9 / FRAME_NS as f64) as u32;
        for seq in s0..s0 + frames {
            let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
            let samples = parked_imu(&t, Some(imu_to), stamp + LEAD_NS, no_extra);
            imu_to = stamp + LEAD_NS;
            t.push_raw_imu(samples, 0, false).unwrap();
            t.push_wheel([WheelSample {
                stamp_ns: stamp,
                x: 0.0,
                y: 0.0,
                theta: 0.0,
            }])
            .unwrap();
            let (left, right) = render_pair(&Pose3d::IDENTITY);
            t.process_stereo(&left, &right, CuTime::from_nanos(stamp))
                .expect("a valid pair");
        }
        assert!(t.wheel_stats().unwrap().parked_keyframes_skipped > 0);
        assert!(t.map.keyframes().len() > k0, "the interval's keyframe");
        assert_eq!(refused(&t), refused0, "the parked edge was refused");
    }

    /// Feeds still frames until the inertial path initializes; the next seq and the keyframes.
    fn park_still_until_initialized(t: &mut Tracker) -> (u32, usize) {
        for seq in 0..80 {
            feed_still(t, seq, &Pose3d::IDENTITY, Some((0.0, 0.0)));
            if t.state.imu_initialized {
                return (seq + 1, t.map.keyframes().len());
            }
        }
        panic!("a parked robot never initialized");
    }

    fn yaw_view(deg: f64) -> Pose3d {
        Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, deg.to_radians(), 0.0)).matrix(),
            Vec3F64::ZERO,
        )
    }

    /// 2026-10-10, black box on the PARKED robot: every `max_keyframes` drop (~90 s) was followed
    /// by `vision_degraded` at keyframe 4 of the new map, cleared at keyframe 7, with fusion never
    /// rejecting and no motion check recorded — a false WARN and a false black-box capture per
    /// drop. A young map's PnP is noisy (its first solve's edges started 0.7-1.3 deg off the
    /// gyro), and an interval with the wheels and the gyro at ~0 and vision ~2 deg read as
    /// "vision disagrees".
    ///
    /// Here vision steps 2.5 deg twice with the robot still, once in the young map (before its
    /// first local BA) and once after it settled. Neither is a frozen view: nothing turned. And
    /// the interval across the new map's first local BA, which re-pivots it, is not checked.
    ///
    /// Red before the fix (degraded). Mutations: dropping the motion-evidence gate in
    /// `IntervalMotion::verdict` turned it red (degraded); dropping the first-BA gate in
    /// `interval_sensors` turned the "interval across the first BA was checked" assertion red.
    #[test]
    fn test_a_parked_robot_is_never_degraded_across_a_map_drop() {
        for fusion in [false, true] {
            let mut tracker = scene_tracker_with(|c| {
                c.inertial = Some(static_inertial_config());
                c.wheel_odom = Some(scene_wheel_config());
                c.per_frame_fusion = fusion.then(FusionConfig::default);
                c.max_keyframes = Some(8);
            });
            let (mut view_deg, mut young_step, mut settled_step) = (0.0, false, false);
            let mut straddle_seen = false;
            for seq in 0..200 {
                let drops = tracker.map_drops();
                let kfs = tracker.map().keyframes().len();
                if drops == 1 && !young_step && kfs == 2 {
                    view_deg += 2.5;
                    young_step = true;
                }
                if drops == 1 && young_step && !settled_step && kfs == 7 {
                    view_deg += 2.5;
                    settled_step = true;
                }
                let checked = tracker.consistency_stats().checked;
                feed_still(&mut tracker, seq, &yaw_view(view_deg), Some((0.0, 0.0)));
                assert!(
                    !tracker.vision_degraded(),
                    "fusion={fusion} seq={seq}: a parked robot read as DEGRADED {:?}",
                    tracker.consistency_stats()
                );
                // The new map's first local BA runs as its 4th keyframe is inserted, after that
                // keyframe's pose was recorded; the 5th is tracked against the map it moved. The
                // interval between them spans that pivot.
                let straddles = kfs == BA_ACTIVE_KEYFRAMES + 1
                    && tracker.map().keyframes().len() == BA_ACTIVE_KEYFRAMES + 2;
                if drops >= 1 && tracker.map_drops() == drops && straddles {
                    assert_eq!(
                        tracker.consistency_stats().checked,
                        checked,
                        "fusion={fusion} seq={seq}: the interval across the first BA was checked"
                    );
                    straddle_seen = true;
                }
            }
            assert!(
                young_step && settled_step,
                "fusion={fusion}: both steps fed"
            );
            assert!(tracker.map_drops() >= 2, "fusion={fusion}: across drops");
            assert!(straddle_seen, "fusion={fusion}: a first BA after a drop");
            let c = tracker.consistency_stats();
            assert_eq!(
                (c.vision_disagrees, c.degraded_entries, c.ba_skipped),
                (0, 0, 0),
                "fusion={fusion}: {c:?}"
            );
            assert!(c.checked >= 4, "fusion={fusion}: still checked: {c:?}");
        }
    }

    /// The other half of that capture: the one motion check recorded spanned a deploy restart, the
    /// body's odometry starting over at the origin while the robot sat at 5.78 deg. An interval
    /// spanning an odometry restart, or a silent stretch past `max_gap`, is not integrated, so it
    /// is never checked (here it would read as slipping wheels and drop a factor) nor handed to
    /// BA. The intervals on either side are checked as usual.
    ///
    /// Red before the fix (the restart interval counted as `wheels_disagree`).
    #[test]
    fn test_an_odometry_restart_inside_an_interval_is_not_checked() {
        const AT: f64 = 0.1;
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let still = Pose3d::IDENTITY;
        let mut seq = 0;
        while !tracker.state.imu_initialized {
            feed_still(&mut tracker, seq, &still, Some((0.3, AT)));
            seq += 1;
            assert!(seq < 80, "never initialized");
        }
        for _ in 0..30 {
            feed_still(&mut tracker, seq, &still, Some((0.3, AT)));
            seq += 1;
        }
        let before = tracker.consistency_stats();
        assert!(before.checked >= 2, "{before:?}");
        // The body restarts: same clock, the odometry back at the origin, the robot unmoved.
        for _ in 0..30 {
            feed_still(&mut tracker, seq, &still, Some((0.0, 0.0)));
            seq += 1;
        }
        let c = tracker.consistency_stats();
        assert_eq!(c.wheels_disagree, 0, "a restart read as slip: {c:?}");
        let w = tracker.wheel_stats().unwrap();
        assert_eq!(w.edges_discontinuous, 1, "{w:?}");
        // The stream goes silent for 1 s (past the 0.5 s max_gap), then resumes.
        for _ in 0..15 {
            feed_still(&mut tracker, seq, &still, None);
            seq += 1;
        }
        let refused = tracker.wheel_stats().unwrap().edges_refused;
        for _ in 0..30 {
            feed_still(&mut tracker, seq, &still, Some((0.0, 0.0)));
            seq += 1;
        }
        let w = tracker.wheel_stats().unwrap();
        assert!(w.edges_refused > refused, "the gap interval refused: {w:?}");
        let c = tracker.consistency_stats();
        assert_eq!(c.wheels_disagree, 0, "a restart read as slip: {c:?}");
        assert_eq!(c.consistent, c.checked, "{c:?}");
        assert!(
            c.checked >= before.checked + 4,
            "checked on both sides: {c:?}"
        );
    }

    /// A gyro bias local BA absorbed UNDER the limit must not flip the check: the gyro is
    /// integrated at the bias of the last interval all three agreed on, not the current one. A
    /// slow frozen turn (0.07 rad/s) with 0.045 rad/s of it already in the bias reads, at the
    /// current bias, as a gyro that half-agrees with frozen vision.
    ///
    /// Mutation: integrating at the current bias (`consistent_gyro_bias` ignored) turned this
    /// red (the verdict flipped away from vision).
    #[test]
    fn test_an_absorbed_gyro_bias_does_not_side_the_gyro_with_frozen_vision() {
        const RATE: f64 = 0.07;
        const ABSORBED: f64 = 0.045;
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let mut seq = park_until_static_init(&mut tracker);
        for _ in 0..40 {
            feed_frozen_turn(&mut tracker, seq, 0.0, &|_| 0.0);
            seq += 1;
        }
        assert!(
            tracker.consistent_gyro_bias.is_some(),
            "parked intervals agree"
        );
        // Part of the turn rate already in the bias, map and next edge alike, as a solve that
        // ran before the check could act leaves it.
        let r_bc = tracker.inertial.as_ref().unwrap().cfg.imu_t_bc.rotation;
        let mut absorbed = tracker.inertial.as_ref().unwrap().bias;
        absorbed.gyro += r_bc * (base_up_in_cam() * ABSORBED);
        assert!(
            tracker
                .inertial
                .as_ref()
                .unwrap()
                .cfg
                .bias_limit
                .admits(&absorbed),
            "under the limit, so BiasLimit cannot catch it"
        );
        write_map_bias(&mut tracker, absorbed);
        tracker.inertial.as_mut().unwrap().bias = absorbed;
        let before = tracker.consistency_stats();
        let start = seq;
        let t = |s: u32| f64::from(s - start) * FRAME_NS as f64 * 1e-9;
        for _ in 0..60 {
            feed_frozen_turn(&mut tracker, seq, RATE, &|s| RATE * t(s));
            seq += 1;
        }
        let c = tracker.consistency_stats();
        assert!(c.vision_disagrees >= before.vision_disagrees + 2, "{c:?}");
        assert_eq!(c.wheels_disagree, before.wheels_disagree, "{c:?}");
        assert!(tracker.vision_degraded(), "frozen vision is flagged");
        assert!(c.degraded_entries >= 1, "{c:?}");
    }

    /// `BiasLimit` refuses; a RUN of refusals drops the map, so BA does not stay off until
    /// `max_keyframes` with an implausible bias in the window. The next map holds only plausible
    /// biases.
    ///
    /// Mutation: never arming the drop (`BIAS_REFUSALS_BEFORE_DROP` unreachable) turned this red
    /// (the bad bias still in the map after the run).
    #[test]
    fn test_a_run_of_implausible_bias_refusals_drops_the_map() {
        let mut tracker = scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
        });
        let mut seq = park_until_static_init(&mut tracker);
        while tracker.map().keyframes().len() < 5 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            seq += 1;
        }
        let sane = tracker.inertial.as_ref().unwrap().bias;
        write_map_bias(&mut tracker, implausible_bias(sane));
        let refused = tracker.ba_updates_refused();
        let mut fed = 0;
        while tracker.ba_updates_refused() < refused + u64::from(BIAS_REFUSALS_BEFORE_DROP) {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            seq += 1;
            fed += 1;
            assert!(fed < 400, "never refused {BIAS_REFUSALS_BEFORE_DROP} times");
        }
        // The drop is deferred to the next frame, like the `max_keyframes` one.
        for _ in 0..3 {
            feed_parked(&mut tracker, seq, &no_extra, &parked_wheels);
            seq += 1;
        }
        let limit = tracker.inertial.as_ref().unwrap().cfg.bias_limit;
        let kfs = tracker.map().keyframes();
        assert!(
            kfs.len() < 5 + BIAS_REFUSALS_BEFORE_DROP as usize,
            "{} keyframes: no drop",
            kfs.len()
        );
        for kf in kfs {
            assert!(
                limit.admits(&kf.imu_bias),
                "kf {} still holds {:?}",
                kf.frame.idx,
                kf.imu_bias
            );
        }
        assert!(limit.admits(&tracker.inertial.as_ref().unwrap().bias));
    }

    // ── Per-frame fusion ─────────────────────────────────────────────────────

    /// A camera ANYWHERE in the scene of [`scene_pair`], ray-cast: the three bands are planes at
    /// `Z = fx * B / disparity` in the first camera's frame, each carrying its texture where the
    /// first camera saw it. Pure translation along x reproduces `scene_pair` exactly (pinned by a
    /// test), so the moving tests below see the scene every other test tracks.
    fn render_pair(cam_in_world: &Pose3d) -> (Image<u8, 1>, Image<u8, 1>) {
        const BANDS: [u32; 3] = [16, 24, 32];
        const W: usize = 320;
        const H: usize = 240;
        const ROWS: f64 = (H / 3) as f64;
        let (f, cx, cy, baseline) = (300.0f64, 160.0f64, 120.0f64, 0.1f64);
        let texture = |x: u32, y: u32| {
            let h = (x / 2).wrapping_mul(73_856_093) ^ (y / 2).wrapping_mul(19_349_663);
            (h.wrapping_mul(2_654_435_761) >> 24) as u8
        };
        let eye = |pose: Pose3d| {
            let mut out = vec![0u8; W * H];
            // Nearest plane first.
            let planes: Vec<(usize, f64)> = (0..3)
                .rev()
                .map(|i| (i, f * baseline / f64::from(BANDS[i])))
                .collect();
            for y in 0..H {
                for x in 0..W {
                    let d_c = Vec3F64::new((x as f64 - cx) / f, (y as f64 - cy) / f, 1.0);
                    let d_w = pose.rotation * d_c;
                    let c = pose.translation;
                    for &(band, z) in &planes {
                        if d_w.z <= 0.0 {
                            break;
                        }
                        let t = (z - c.z) / d_w.z;
                        if t <= 0.0 {
                            continue;
                        }
                        let p = c + d_w * t;
                        let v = f * p.y / z + cy + 1e-6;
                        if v < 0.0 || ((v / ROWS) as usize).min(2) != band || v >= H as f64 {
                            continue;
                        }
                        let u = f * p.x / z + cx + 1e-6;
                        out[y * W + x] = texture(u.floor() as i64 as u32, v.floor() as u32);
                        break;
                    }
                }
            }
            Image::new(
                ImageSize {
                    width: W,
                    height: H,
                },
                out,
            )
            .expect("sized")
        };
        let right = cam_in_world.compose(&Pose3d::new(
            Mat3F64::IDENTITY,
            Vec3F64::new(baseline, 0.0, 0.0),
        ));
        (eye(*cam_in_world), eye(right))
    }

    #[test]
    fn test_the_ray_cast_scene_is_the_translating_scene() {
        for seq in [0u32, 3, 8] {
            let pose = Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(SCENE_STEP_M * f64::from(seq), 0.0, 0.0),
            );
            let (l, r) = render_pair(&pose);
            let (sl, sr, _) = scene_pair(seq);
            assert_eq!(l.as_slice(), sl.as_slice(), "left {seq}");
            assert_eq!(r.as_slice(), sr.as_slice(), "right {seq}");
        }
    }

    /// The parked, inertial, wheeled tracker of the consistency tests, with per-frame fusion
    /// on or off.
    fn fusion_tracker(fusion: bool) -> Tracker {
        scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom = Some(scene_wheel_config());
            c.per_frame_fusion = fusion.then(FusionConfig::default);
        })
    }

    /// The IMU of a rig whose camera follows `cam_at(t)` (seconds since `EPOCH_NS`, world = the
    /// first camera), over `(from_ns, to_ns]` on the 5 ms grid: kinematics by central
    /// differences, gravity [`parked_gravity_vis`], the parked gyro bias and jitter.
    fn moving_imu(
        tracker: &Tracker,
        cam_at: &dyn Fn(f64) -> Pose3d,
        from_ns: Option<u64>,
        to_ns: u64,
    ) -> Vec<RawImuSample> {
        let t_cb = tracker.inertial.as_ref().unwrap().cfg.imu_t_bc.inverse();
        let body = |t: f64| cam_at(t).compose(&t_cb);
        let g = parked_gravity_vis();
        let bg = Vec3F64::from_array(PARKED_GYRO_BIAS);
        let h = 1e-3;
        let first = from_ns.map_or(0, |t| (t - EPOCH_NS) / IMU_PERIOD_NS + 2);
        let last = (to_ns - EPOCH_NS) / IMU_PERIOD_NS + 1;
        (first..=last)
            .map(|k| {
                let t = (k * IMU_PERIOD_NS) as f64 * 1e-9;
                let (b0, b, b1) = (body(t - h), body(t), body(t + h));
                let a_w = (b1.translation - b.translation * 2.0 + b0.translation) * (1.0 / (h * h));
                let r_bw = b.rotation.transpose();
                let jf = k as f64;
                let jitter = Vec3F64::new((1.7 * jf).sin(), (2.9 * jf).sin(), (4.3 * jf).sin());
                let f_b = r_bw * (a_w - g) + jitter * 0.02;
                let w_b = SO3F64::from_matrix(&(b0.rotation.transpose() * b1.rotation)).log()
                    * (1.0 / (2.0 * h))
                    + bg
                    + jitter * 0.002;
                RawImuSample {
                    stamp_ns: EPOCH_NS + k * IMU_PERIOD_NS,
                    gyro: [w_b.x, w_b.y, w_b.z],
                    accel: [f_b.x, f_b.y, f_b.z],
                }
            })
            .collect()
    }

    /// One frame of a rig following `cam_at`, the wheels reading heading `wheel_theta`, the
    /// pair rendered at `view` (the truth unless a test corrupts it).
    fn feed_moving(
        tracker: &mut Tracker,
        seq: u32,
        cam_at: &dyn Fn(f64) -> Pose3d,
        wheel_theta: f64,
        view: &Pose3d,
    ) -> TrackStatus {
        let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
        let from = seq
            .checked_sub(1)
            .map(|p| EPOCH_NS + u64::from(p) * FRAME_NS);
        let samples = moving_imu(tracker, cam_at, from, stamp);
        tracker.push_raw_imu(samples, 0, false).unwrap();
        tracker
            .push_wheel([WheelSample {
                stamp_ns: stamp,
                x: 0.0,
                y: 0.0,
                theta: wheel_theta,
            }])
            .unwrap();
        let (left, right) = render_pair(view);
        tracker
            .process_stereo(&left, &right, CuTime::from_nanos(stamp))
            .expect("a valid pair")
    }

    /// [`feed_moving`] with the wheels reading `[x, y, theta]`.
    fn feed_moving_xy(
        tracker: &mut Tracker,
        seq: u32,
        cam_at: &dyn Fn(f64) -> Pose3d,
        wheel: [f64; 3],
        view: &Pose3d,
    ) -> TrackStatus {
        let stamp = EPOCH_NS + u64::from(seq) * FRAME_NS;
        let from = seq
            .checked_sub(1)
            .map(|p| EPOCH_NS + u64::from(p) * FRAME_NS);
        let samples = moving_imu(tracker, cam_at, from, stamp);
        tracker.push_raw_imu(samples, 0, false).unwrap();
        tracker
            .push_wheel([WheelSample {
                stamp_ns: stamp,
                x: wheel[0],
                y: wheel[1],
                theta: wheel[2],
            }])
            .unwrap();
        let (left, right) = render_pair(view);
        tracker
            .process_stereo(&left, &right, CuTime::from_nanos(stamp))
            .expect("a valid pair")
    }

    fn published(status: &TrackStatus) -> Pose3d {
        match status {
            TrackStatus::Tracked(p) | TrackStatus::Coasting(p) | TrackStatus::NotLocalized(p) => {
                p.pose_world_to_cam
            }
            TrackStatus::Untracked => panic!("nothing published"),
        }
    }

    /// (a) THE BEHAVIOUR: parked, the gyro and the wheels say nothing moved, and one frame's
    /// features say the camera jumped 25 mm (the scene as seen two steps along; a moving object
    /// filling the view looks the same to PnP). With per-frame fusion the gate rejects that
    /// frame's vision and the published pose stays within a few millimetres; without it, the
    /// pose jumps by the full 25 mm. The frames after it are accepted again.
    ///
    /// Mutations (each turned this red): fusion off (the `fusion = false` arm is the control);
    /// the gate never rejecting (`chi2 > gate` replaced by `false` in `fusion::fuse`).
    #[test]
    fn test_a_corrupted_frame_while_parked_does_not_move_the_pose() {
        for fusion in [false, true] {
            let mut tracker = fusion_tracker(fusion);
            let start = park_until_static_init(&mut tracker);
            let still = |_t: f64| Pose3d::IDENTITY;
            let mut seq = start;
            let mut before = Pose3d::IDENTITY;
            for _ in 0..6 {
                before = published(&feed_moving(
                    &mut tracker,
                    seq,
                    &still,
                    0.0,
                    &Pose3d::IDENTITY,
                ));
                seq += 1;
            }
            let jumped = Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(2.0 * SCENE_STEP_M, 0.0, 0.0),
            );
            let status = feed_moving(&mut tracker, seq, &still, 0.0, &jumped);
            seq += 1;
            let moved_mm =
                (published(&status).inverse().translation - before.inverse().translation).length()
                    * 1e3;
            let report = tracker.frame_report();
            if fusion {
                assert!(moved_mm < 3.0, "fused pose moved {moved_mm} mm");
                // The prediction is a coast, not a measurement.
                assert!(matches!(status, TrackStatus::Coasting(_)), "{status:?}");
                assert_eq!(report.failure, FailureReason::VisionRejected);
                assert!(report.vision_rejected && report.fused, "{report:?}");
                assert_eq!(tracker.fusion_stats().rejects, 1);
                assert!(tracker.fusion_stats().last_chi2 > 100.0);
            } else {
                assert!(
                    moved_mm > 20.0,
                    "control: vision alone moved only {moved_mm} mm"
                );
            }
            // Back to the true view: accepted again (with fusion) and in place.
            for _ in 0..4 {
                let status = feed_moving(&mut tracker, seq, &still, 0.0, &Pose3d::IDENTITY);
                seq += 1;
                assert!(matches!(status, TrackStatus::Tracked(_)), "{status:?}");
            }
            if fusion {
                assert!(!tracker.frame_report().vision_rejected);
                let back_mm = (tracker.last_pose().inverse().translation
                    - before.inverse().translation)
                    .length()
                    * 1e3;
                assert!(back_mm < 3.0, "{back_mm} mm after the bad frame");
            }
        }
    }

    /// A view that stays displaced (the robot pushed or carried, the prediction wrong) is
    /// outvoted for [`MAX_CONSECUTIVE_REJECTS`] frames, each publishing the prediction in place
    /// as a coast. The next frame's vision is accepted, and it STAYS accepted: the fusion waits
    /// for a keyframe instead of predicting every later frame from the same stale one. The
    /// tracker moves to vision at once; the published pose closes on it at
    /// [`PUBLISH_CLOSE_M_PER_S`] (the run's end held, not snapped).
    ///
    /// Mutations: not suspending the fusion on expiry (`fusion_suspended_on` left unset) turned
    /// this red (rejected again on the frame after the accepted one, a 25 mm flicker per run);
    /// the hold never set turned it red (the published pose jumped the 25 mm in one frame).
    #[test]
    fn test_a_run_of_rejections_ends_with_vision_accepted_for_good() {
        let mut tracker = fusion_tracker(true);
        let start = park_until_static_init(&mut tracker);
        // No keyframe from here on: the default policy would make the accepted frame one at once
        // (8 frames since the last), which re-anchors the prediction by itself and hides whether
        // the fusion waits. A map whose insertion is refused, or a laxer policy, does not.
        tracker.config.keyframe_policy.max_frames_between = usize::MAX;
        tracker.config.keyframe_policy.ref_ratio = 0.0;
        let still = |_t: f64| Pose3d::IDENTITY;
        let before = published(&feed_moving(
            &mut tracker,
            start,
            &still,
            0.0,
            &Pose3d::IDENTITY,
        ));
        let jumped = Pose3d::new(
            Mat3F64::IDENTITY,
            Vec3F64::new(2.0 * SCENE_STEP_M, 0.0, 0.0),
        );
        let moved_mm = |status: &TrackStatus| {
            (published(status).inverse().translation - before.inverse().translation).length() * 1e3
        };
        for (i, seq) in (start + 1..)
            .take(MAX_CONSECUTIVE_REJECTS as usize)
            .enumerate()
        {
            let status = feed_moving(&mut tracker, seq, &still, 0.0, &jumped);
            assert!(tracker.frame_report().vision_rejected, "frame {i}");
            assert!(
                matches!(status, TrackStatus::Coasting(_)),
                "frame {i}: {status:?}"
            );
            assert!(
                moved_mm(&status) < 3.0,
                "frame {i}: {} mm",
                moved_mm(&status)
            );
        }
        assert_eq!(tracker.fusion_stats().reject_runs_expired, 0);
        let first = start + 1 + MAX_CONSECUTIVE_REJECTS;
        let mut last_mm = 0.0;
        let frame_s = FRAME_NS as f64 * 1e-9;
        for seq in first..first + 20 {
            let status = feed_moving(&mut tracker, seq, &still, 0.0, &jumped);
            assert!(
                matches!(status, TrackStatus::Tracked(_)),
                "frame {seq}: {status:?}"
            );
            assert!(!tracker.frame_report().vision_rejected, "frame {seq}");
            // The tracker is on vision's pose (25 mm along) at once, not the old one...
            let internal_mm =
                (tracker.last_pose().inverse().translation - before.inverse().translation).length()
                    * 1e3;
            assert!(
                internal_mm > 15.0,
                "frame {seq}: back at the old pose, {internal_mm} mm from it"
            );
            // ...and the published pose closes on it at the hold's rate, not in one jump.
            let mm = moved_mm(&status);
            assert!(
                mm - last_mm <= PUBLISH_CLOSE_M_PER_S * frame_s * 1e3 + 0.1,
                "frame {seq}: published stepped {} mm",
                mm - last_mm
            );
            last_mm = mm;
        }
        assert!(last_mm > 20.0, "never closed on vision: {last_mm} mm");
        let f = tracker.fusion_stats();
        assert_eq!(f.reject_runs_expired, 1, "{f:?}");
        assert_eq!(f.rejects, u64::from(MAX_CONSECUTIVE_REJECTS), "{f:?}");
        // ...and the expiry reaches the status topic.
        let diag = crate::diag::frame_diagnostics(
            &tracker,
            &TrackStatus::Untracked,
            crate::diag::StreamCounters::default(),
        );
        assert_eq!(diag.fusion.reject_runs_expired, 1);
    }

    /// A straight drive at 0.15 m/s, gravity settled by the still window so the IMU's position
    /// rows are in: vision, the IMU and the wheels agree, nothing is rejected, and the published
    /// motion follows the truth. The keyframes' velocities are whatever the tracker gave them,
    /// which is what a stale one on the robot would test.
    #[test]
    fn test_a_straight_drive_with_good_vision_is_followed_not_rejected() {
        const SPEED: f64 = 0.15;
        let mut tracker = fusion_tracker(true);
        let start = park_until_static_init(&mut tracker);
        assert!(imu_position_trusted(tracker.inertial_diag().0));
        let t_start = f64::from(start) * FRAME_NS as f64 * 1e-9;
        let dist = move |t: f64| SPEED * (t - t_start).max(0.0);
        let t_cb = tracker.wheel.as_ref().unwrap().cfg.t_cb;
        let cam_at = move |t: f64| {
            let fwd = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(dist(t), 0.0, 0.0));
            t_cb.compose(&fwd).compose(&t_cb.inverse())
        };
        let t_of = |seq: u32| f64::from(seq) * FRAME_NS as f64 * 1e-9;
        let first = published(&feed_moving_xy(
            &mut tracker,
            start,
            &cam_at,
            [0.0, 0.0, 0.0],
            &Pose3d::IDENTITY,
        ));
        let mut worst_mm = 0.0f64;
        for seq in start + 1..start + 17 {
            let truth = cam_at(t_of(seq)).inverse();
            let status = feed_moving_xy(
                &mut tracker,
                seq,
                &cam_at,
                [dist(t_of(seq)), 0.0, 0.0],
                &cam_at(t_of(seq)),
            );
            assert!(
                matches!(status, TrackStatus::Tracked(_)),
                "frame {seq}: {status:?}"
            );
            assert!(
                !tracker.frame_report().vision_rejected,
                "frame {seq} rejected"
            );
            let rel = first.compose(&published(&status).inverse());
            let err = truth.compose(&rel);
            worst_mm = worst_mm.max(err.translation.length() * 1e3);
        }
        let f = tracker.fusion_stats();
        assert!(f.frames >= 16, "{f:?}");
        assert_eq!(f.rejects, 0, "{f:?}");
        assert!(
            worst_mm < 8.0,
            "worst translation error {worst_mm} mm over 160 mm ({f:?})"
        );
    }

    /// The keyframe motion check records PnP's pose, not the fused one: it is vision's word,
    /// compared against the gyro and the wheels, so it must not already lean on them.
    ///
    /// Mutation: recording `kf.frame.pose_world_to_cam` again turned this red.
    #[test]
    fn test_the_keyframe_motion_check_records_vision_not_the_fusion() {
        const RATE: f64 = 0.3;
        let mut tracker = fusion_tracker(true);
        let start = park_until_static_init(&mut tracker);
        let t_start = f64::from(start) * FRAME_NS as f64 * 1e-9;
        let theta = move |t: f64| RATE * (t - t_start).max(0.0);
        let t_cb = tracker.wheel.as_ref().unwrap().cfg.t_cb;
        let cam_at = move |t: f64| {
            let rz = Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.0, 0.0, theta(t))).matrix(),
                Vec3F64::ZERO,
            );
            t_cb.compose(&rz).compose(&t_cb.inverse())
        };
        let t_of = |seq: u32| f64::from(seq) * FRAME_NS as f64 * 1e-9;
        let mut checked = 0;
        for seq in start..start + 24 {
            let truth = cam_at(t_of(seq));
            let status = feed_moving(&mut tracker, seq, &cam_at, theta(t_of(seq)), &truth);
            let TrackStatus::Tracked(p) = status else {
                continue;
            };
            if !p.keyframe || !tracker.frame_report().fused {
                continue;
            }
            let (idx, vision, published) = tracker.vision_pose.expect("tracked by PnP");
            let &(kf_idx, _, recorded) = tracker.kf_tracked_poses.back().unwrap();
            assert_eq!(kf_idx, idx);
            // No alignment on these keyframes (the world was aligned while parked), so the
            // recorded pose is PnP's, up to the noise of `vision * published^-1 * kf` (the
            // keyframe as stored is not bit for bit the published pose): far below what the
            // fusion moved it by, which is what recording the fused pose would leave.
            let d = recorded.compose(&vision.inverse());
            let moved = published.compose(&vision.inverse());
            let (d_m, moved_m) = (d.translation.length(), moved.translation.length());
            assert!(
                moved_m > 1e-6,
                "frame {seq}: the fusion moved it {moved_m} m"
            );
            assert!(
                d_m < moved_m * 1e-2,
                "frame {seq}: {d_m} m off vision, fusion {moved_m}"
            );
            checked += 1;
        }
        assert!(checked >= 1, "no fused keyframe in the run");
    }

    #[test]
    fn test_imu_position_rows_wait_for_settled_gravity() {
        for (stage, want) in [
            (InertialStage::Off, false),
            (InertialStage::Uninitialized, false),
            (InertialStage::Initial, false),
            (InertialStage::Refine1, false),
            (InertialStage::Refine2, true),
            (InertialStage::Static, true),
        ] {
            assert_eq!(imu_position_trusted(stage), want, "{stage:?}");
        }
    }

    /// The wheels' rotation covariance is rotated into the camera at the interval's end.
    #[test]
    fn test_the_wheel_vote_uses_the_mount_at_the_interval_end() {
        let mut cov6 = vec![0.0; 36];
        cov6[3 * 6 + 3] = 1e-6; // roll
        cov6[4 * 6 + 4] = 4e-6; // pitch
        cov6[5 * 6 + 5] = 9e-6; // yaw
        let end = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.0, 0.0, std::f64::consts::FRAC_PI_2)).matrix(),
            Vec3F64::ZERO,
        );
        let c = wheel_rotation_cov_in_camera(&cov6, &end);
        // A quarter turn about z swaps the roll and pitch variances.
        assert!((c.x_axis.x - 4e-6).abs() < 1e-15, "{c:?}");
        assert!((c.y_axis.y - 1e-6).abs() < 1e-15, "{c:?}");
        assert!((c.z_axis.z - 9e-6).abs() < 1e-15, "{c:?}");
    }

    /// (b) A real turn with good vision: the base turns at 0.3 rad/s about its z for 16 frames
    /// (~17 deg, the camera swinging on its 0.1 m lever arm), and the gyro, the wheels and
    /// vision all see it. The gate rejects nothing, no wheel factor is dropped, and the published
    /// motion follows the truth.
    #[test]
    fn test_a_real_turn_with_good_vision_is_followed_not_rejected() {
        const RATE: f64 = 0.3;
        let mut tracker = fusion_tracker(true);
        let start = park_until_static_init(&mut tracker);
        let t_start = f64::from(start) * FRAME_NS as f64 * 1e-9;
        let theta = |t: f64| RATE * (t - t_start).max(0.0);
        let t_cb = tracker.wheel.as_ref().unwrap().cfg.t_cb;
        let cam_at = move |t: f64| {
            let rz = Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.0, 0.0, theta(t))).matrix(),
                Vec3F64::ZERO,
            );
            t_cb.compose(&rz).compose(&t_cb.inverse())
        };
        let t_of = |seq: u32| f64::from(seq) * FRAME_NS as f64 * 1e-9;
        let first = published(&feed_moving(
            &mut tracker,
            start,
            &cam_at,
            0.0,
            &Pose3d::IDENTITY,
        ));
        let mut worst_deg = 0.0f64;
        let mut worst_mm = 0.0f64;
        for seq in start + 1..start + 17 {
            let truth = cam_at(t_of(seq));
            let status = feed_moving(&mut tracker, seq, &cam_at, theta(t_of(seq)), &truth);
            assert!(
                matches!(status, TrackStatus::Tracked(_)),
                "frame {seq}: {status:?}"
            );
            assert!(
                !tracker.frame_report().vision_rejected,
                "frame {seq} rejected"
            );
            // Camera motion since the turn started, in the first frame's camera: the tracker's
            // world was rotated by the inertial alignment, the relative motion was not.
            let rel = first.compose(&published(&status).inverse());
            let err = truth.inverse().compose(&rel);
            worst_deg = worst_deg.max(crate::consistency::angle(&err.rotation).to_degrees());
            worst_mm = worst_mm.max(err.translation.length() * 1e3);
        }
        let f = tracker.fusion_stats();
        assert!(f.frames >= 16, "{f:?}");
        assert_eq!((f.rejects, f.wheel_dropped), (0, 0), "{f:?}");
        assert!(worst_deg < 0.3, "worst rotation error {worst_deg} deg");
        assert!(worst_mm < 5.0, "worst translation error {worst_mm} mm");
    }

    /// (c) Wheels that report a 0.3 rad/s turn the gyro and vision do not see (slip, or the head
    /// turning on its mast): the per-frame wheel factor is dropped, nothing is rejected, and the
    /// published pose does not turn.
    ///
    /// Mutation: never dropping the wheel factor (`disagree` forced false in `fuse_frame`) turned
    /// this red (the slipping wheels outvoted vision and turned the published pose).
    #[test]
    fn test_slipping_wheels_lose_their_per_frame_factor() {
        const RATE: f64 = 0.3;
        let mut tracker = fusion_tracker(true);
        let start = park_until_static_init(&mut tracker);
        let still = |_t: f64| Pose3d::IDENTITY;
        let before = published(&feed_moving(
            &mut tracker,
            start,
            &still,
            0.0,
            &Pose3d::IDENTITY,
        ));
        let mut worst_deg = 0.0f64;
        for seq in start + 1..start + 20 {
            let theta = RATE * f64::from(seq - start) * FRAME_NS as f64 * 1e-9;
            let status = feed_moving(&mut tracker, seq, &still, theta, &Pose3d::IDENTITY);
            let r = published(&status).rotation * before.rotation.transpose();
            worst_deg = worst_deg.max(crate::consistency::angle(&r).to_degrees());
        }
        let f = tracker.fusion_stats();
        assert!(
            worst_deg < 0.2,
            "the published pose turned {worst_deg} deg ({f:?})"
        );
        assert!(f.wheel_dropped >= 10, "{f:?}");
        assert_eq!(f.rejects, 0, "{f:?}");
    }

    /// The robot's head mount (app `head.ron`, `wheel_odom.cam_in_base_*`): the camera 1.17 m up,
    /// 0.106 m off the base's rotation axis, pitched 22 deg down; the robot's wheel noise.
    fn robot_cam_in_base() -> Pose3d {
        Pose3d::new(
            crate::wheel::mat3_from_rows(&[
                0.013874125,
                -0.379527910,
                0.925076253,
                -0.999332488,
                -0.036531869,
                0.0,
                0.033794765,
                -0.924458754,
                -0.379781418,
            ]),
            Vec3F64::new(-0.088381105, 0.058421493, 1.166114481),
        )
    }

    fn robot_fusion_tracker(fusion: Option<FusionConfig>) -> Tracker {
        scene_tracker_with(|c| {
            c.inertial = Some(static_inertial_config());
            c.wheel_odom =
                Some(WheelOdomConfig::new(robot_cam_in_base(), 0.05, 0.01).expect("valid"));
            c.per_frame_fusion = fusion;
        })
    }

    /// The robot's `per_frame_fusion` block (app `head.ron`): the mount's lever arm known only
    /// to 10 cm until the URDF is corrected.
    fn robot_fusion_config() -> FusionConfig {
        FusionConfig {
            extrinsic_sigma_m: 0.1,
            ..FusionConfig::default()
        }
    }

    /// The model errors of the real rig, which the configured mount and wheel model leave out.
    #[derive(Debug, Clone, Copy)]
    struct TurnErrors {
        /// The camera's true lever arm minus the configured one, base frame, metres.
        lever_err_m: Vec3F64,
        /// The head's yaw lag on the mast, ramped in over the turn's first 0.3 s, radians.
        play_rad: f64,
        /// The wheels' heading over-read, a fraction of the angle turned.
        wheel_scale: f64,
        /// The base origin's sideways scrub per radian turned, metres.
        scrub_m_per_rad: f64,
    }

    /// A well calibrated rig: a 1 cm lever-arm error, 2 deg of head play, a 2 % wheel heading
    /// error and 2.5 cm/rad of scrub.
    fn calibrated_turn_errors() -> TurnErrors {
        TurnErrors {
            lever_err_m: Vec3F64::new(0.007, -0.007, 0.0),
            play_rad: 0.0349,
            wheel_scale: 0.02,
            scrub_m_per_rad: 0.025,
        }
    }

    /// The robot as measured on the 2026-10-10 drive: the camera circles a turning axis at base
    /// `(+0.120, +0.040)` m from it (both 38 deg turns, within 1 mm of each other, from the
    /// published poses before and after each turn), where the URDF mount puts it at
    /// `(-0.088, +0.058)`: 21 cm off, ahead of the axle instead of behind it. Plus the head play
    /// and the wheel heading error of [`calibrated_turn_errors`].
    fn measured_turn_errors() -> TurnErrors {
        TurnErrors {
            lever_err_m: Vec3F64::new(0.1197 + 0.088381105, 0.0402 - 0.058421493, 0.0),
            scrub_m_per_rad: 0.0,
            ..calibrated_turn_errors()
        }
    }

    /// What a turn on the robot did to the per-frame fusion, frame by frame.
    #[derive(Debug, Default)]
    struct TurnRun {
        rejects: u64,
        expired: u64,
        /// Largest frame-to-frame change of the published camera centre's error against the
        /// truth, mm: a snap shows here, the turn's own motion does not.
        worst_step_mm: f64,
        worst_err_mm: f64,
        wheel_dropped: u64,
        imu_dropped: u64,
        chi2: Vec<(f64, u8, bool)>,
        steps_mm: Vec<f64>,
        /// The steps on the frames that END a rejection run (the previous frame rejected, this
        /// one not), mm.
        end_steps_mm: Vec<f64>,
    }

    /// A base turning in place at `rate` rad/s for `frames` frames, the robot's mount, and the
    /// model errors of the real rig: the camera's true lever arm `lever_err_m` off the
    /// configured one, the head lagging the turn by `play_rad` on its mast (ramped in over the
    /// first 0.3 s, pivoting on the camera), and the wheels over-reading the angle by
    /// `wheel_scale`. Vision sees the truth.
    fn run_robot_turn(
        rate: f64,
        frames: u32,
        e: TurnErrors,
        fusion: Option<FusionConfig>,
    ) -> (Tracker, TurnRun) {
        let TurnErrors {
            lever_err_m,
            play_rad,
            wheel_scale,
            scrub_m_per_rad,
        } = e;
        let mut tracker = robot_fusion_tracker(fusion);
        let start = park_until_static_init(&mut tracker);
        // The robot's scene keeps its inliers, so keyframes come at the policy's
        // `max_frames_between` (8 frames, 0.67 s at 12 Hz), not every 3 as this small scene's.
        tracker.config.keyframe_policy.min_frames_between = 8;
        let t_start = f64::from(start) * FRAME_NS as f64 * 1e-9;
        let theta = move |t: f64| rate * (t - t_start).max(0.0);
        let cfg_cam_in_base = robot_cam_in_base();
        let true_cam_in_base = Pose3d::new(
            cfg_cam_in_base.rotation,
            cfg_cam_in_base.translation + lever_err_m,
        );
        // The head pans on the mast 0.178 m behind the base origin (URDF `head_pan_joint`).
        let p_pan = Vec3F64::new(-0.178, 0.0, 0.0);
        let play = move |t: f64| {
            let s = ((t - t_start) / 0.3).clamp(0.0, 1.0);
            -play_rad * s * rate.signum()
        };
        // The camera in the base at `t`: yawed by the play about the pan axis.
        let cam_in_base_at = move |t: f64| {
            let pivot = Pose3d::new(Mat3F64::IDENTITY, p_pan);
            let yaw = Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.0, 0.0, play(t))).matrix(),
                Vec3F64::ZERO,
            );
            pivot
                .compose(&yaw)
                .compose(&pivot.inverse())
                .compose(&true_cam_in_base)
        };
        // World = the first camera: T_w<-c(t) = T_c0<-b0 * T_b0<-b(t) * T_b<-c(t), the base
        // turning about its origin while scrubbing sideways (castor drag, wheel slip) by
        // `scrub_m_per_rad` of turn, which the wheels do not see.
        let cam_at = move |t: f64| {
            let rz = Pose3d::new(
                SO3F64::exp(Vec3F64::new(0.0, 0.0, theta(t))).matrix(),
                Vec3F64::new(0.0, scrub_m_per_rad * theta(t).abs(), 0.0),
            );
            true_cam_in_base
                .inverse()
                .compose(&rz)
                .compose(&cam_in_base_at(t))
        };
        let t_of = |seq: u32| f64::from(seq) * FRAME_NS as f64 * 1e-9;
        let first = published(&feed_moving(
            &mut tracker,
            start,
            &cam_at,
            0.0,
            &cam_at(t_of(start)),
        ));
        let mut run = TurnRun::default();
        let mut prev_err: Option<Vec3F64> = None;
        let mut prev_rejected = false;
        for seq in start + 1..start + 1 + frames {
            let truth = cam_at(t_of(seq));
            let status = feed_moving(
                &mut tracker,
                seq,
                &cam_at,
                theta(t_of(seq)) * (1.0 + wheel_scale),
                &truth,
            );
            let f = tracker.fusion_stats();
            run.chi2.push((
                f.last_chi2,
                f.last_dof,
                tracker.frame_report().vision_rejected,
            ));
            // The camera centre since the turn started, in the first frame's camera.
            let rel = first.compose(&published(&status).inverse());
            let err = rel.translation - truth.translation;
            let rejected = tracker.frame_report().vision_rejected;
            if let Some(p) = prev_err {
                let step = (err - p).length() * 1e3;
                run.steps_mm.push((step * 10.0).round() / 10.0);
                run.worst_step_mm = run.worst_step_mm.max(step);
                if prev_rejected && !rejected {
                    run.end_steps_mm.push(step);
                }
            }
            prev_rejected = rejected;
            run.worst_err_mm = run.worst_err_mm.max(err.length() * 1e3);
            prev_err = Some(err);
        }
        let f = tracker.fusion_stats();
        run.rejects = f.rejects;
        run.expired = f.reject_runs_expired;
        run.wheel_dropped = f.wheel_dropped;
        run.imu_dropped = f.imu_dropped;
        (tracker, run)
    }

    /// (a) THE LIVE BUG (2026-10-10, robot on the floor, +-40 deg turns at 0.3 rad/s): vision
    /// agreed with the gyro within 0.6 deg, yet the gate rejected it for 15 frames in a row
    /// (chi2 179-259), the prediction coasted, and the published pose snapped 55-65 mm when the
    /// run expired. The wheel factor predicted the camera's swing through a mount whose lever arm
    /// is 21 cm off ([`measured_turn_errors`]) with a 1 mm translation sigma, so a few frames into
    /// any turn vision read as wrong. Here, with the robot's mount, its wheel noise and its
    /// configured `extrinsic_sigma_m`: vision is kept through the whole turn and the published
    /// camera never steps. A calibrated rig keeps it at the default sigma too.
    ///
    /// Mutations (each turned this red): the extrinsic term left out of `MotionFactor::wheel`,
    /// i.e. the model before it (14 rejects in the +0.3 rad/s turn, the published camera 69 mm
    /// off the truth, 14 mm steps); the robot's sigma at the default 0.03 (13 rejects, 18 mm
    /// steps).
    #[test]
    fn test_a_robot_turn_with_the_measured_mount_error_keeps_vision() {
        for rate in [0.3, -0.3] {
            // The control: vision alone, the scene's own frame-to-frame noise.
            let (_, alone) = run_robot_turn(rate, 24, measured_turn_errors(), None);
            let (_, measured) = run_robot_turn(
                rate,
                24,
                measured_turn_errors(),
                Some(robot_fusion_config()),
            );
            let (_, calibrated) = run_robot_turn(
                rate,
                24,
                calibrated_turn_errors(),
                Some(FusionConfig::default()),
            );
            println!(
                "rate {rate}\n vision alone {alone:?}\n measured {measured:?}\n calibrated {calibrated:?}"
            );
            for run in [&measured, &calibrated] {
                assert!(run.rejects <= 1, "rate {rate}: {run:?}");
                assert_eq!(run.expired, 0, "rate {rate}: {run:?}");
                assert!(
                    run.worst_step_mm < alone.worst_step_mm + 1.0,
                    "rate {rate}: steps {} mm against vision alone's {} mm: {run:?}",
                    run.worst_step_mm,
                    alone.worst_step_mm
                );
            }
        }
    }

    /// (b) Parked, the robot's mount and its loose `extrinsic_sigma_m` (10 cm): PnP outliers
    /// (the scene as seen 12.5 and 25 mm along, a moving object filling the view) are still
    /// rejected and the published pose does not move. The extrinsic term acts only through the
    /// motion, so a parked gate is exactly as tight as with no term at all.
    ///
    /// Mutation: the term applied whatever the motion (`I - Ad` replaced by `I` in
    /// `fusion::extrinsic_covariance`) turned this red (both outliers accepted).
    #[test]
    fn test_parked_outliers_are_still_rejected_with_the_robot_mount() {
        let mut tracker = robot_fusion_tracker(Some(robot_fusion_config()));
        let start = park_until_static_init(&mut tracker);
        let still = |_t: f64| Pose3d::IDENTITY;
        let mut seq = start;
        let mut before = Pose3d::IDENTITY;
        let mut good = |tracker: &mut Tracker, seq: &mut u32| {
            for _ in 0..6 {
                before = published(&feed_moving(tracker, *seq, &still, 0.0, &Pose3d::IDENTITY));
                *seq += 1;
            }
            before
        };
        for steps in [1.0, 2.0] {
            let before = good(&mut tracker, &mut seq);
            let jumped = Pose3d::new(
                Mat3F64::IDENTITY,
                Vec3F64::new(steps * SCENE_STEP_M, 0.0, 0.0),
            );
            let rejects = tracker.fusion_stats().rejects;
            let status = feed_moving(&mut tracker, seq, &still, 0.0, &jumped);
            seq += 1;
            let f = tracker.fusion_stats();
            let moved_mm =
                (published(&status).inverse().translation - before.inverse().translation).length()
                    * 1e3;
            println!("{steps} steps: chi2 {} moved {moved_mm} mm", f.last_chi2);
            assert_eq!(f.rejects, rejects + 1, "{steps} steps accepted: {f:?}");
            assert!(moved_mm < 3.0, "{steps} steps: moved {moved_mm} mm");
        }
    }

    /// (c) A rejection run that ends does not snap the published pose: the measured 21 cm mount
    /// error under a sigma far too small for it (the default 3 cm), so runs do start and expire,
    /// and each end is held and closed at [`PUBLISH_CLOSE_M_PER_S`] instead of jumping.
    ///
    /// Mutation: the hold never set (`publish_hold_target` left `None`) turned this red: a 46 mm
    /// step on the frame the run expired, the live run's 55-65 mm snap.
    #[test]
    fn test_the_end_of_a_rejection_run_does_not_snap_the_published_pose() {
        for rate in [0.3, -0.3] {
            let (_, run) = run_robot_turn(
                rate,
                30,
                measured_turn_errors(),
                Some(FusionConfig::default()),
            );
            println!("rate {rate}: {run:?}");
            assert!(run.expired >= 1, "rate {rate}: no run expired: {run:?}");
            assert!(!run.end_steps_mm.is_empty(), "rate {rate}: {run:?}");
            assert!(
                run.end_steps_mm.iter().all(|&s| s < 10.0),
                "rate {rate}: {run:?}"
            );
        }
    }

    /// What the fusion costs per frame, against the whole frame, on the parked-then-turning run
    /// (printed with `--nocapture`). The fusion is a few small dense solves over the kept
    /// matches; it must stay a small fraction of `process_stereo`.
    #[test]
    fn test_per_frame_fusion_cost() {
        let mut costs = Vec::new();
        let mut frames = Vec::new();
        for fusion in [false, true] {
            let mut tracker = fusion_tracker(fusion);
            let start = park_until_static_init(&mut tracker);
            let still = |_t: f64| Pose3d::IDENTITY;
            let mut frame_us = Vec::new();
            for seq in start..start + 40 {
                let t = std::time::Instant::now();
                feed_moving(&mut tracker, seq, &still, 0.0, &Pose3d::IDENTITY);
                frame_us.push(t.elapsed().as_micros() as u64);
                if fusion && tracker.frame_report().fused {
                    costs.push(u64::from(tracker.fusion_stats().last_cost_us));
                }
            }
            frame_us.sort_unstable();
            frames.push(frame_us[frame_us.len() / 2]);
        }
        costs.sort_unstable();
        assert!(costs.len() >= 30, "{} fused frames", costs.len());
        let (p50, p90) = (costs[costs.len() / 2], costs[costs.len() * 9 / 10]);
        println!(
            "per-frame fusion: cost p50={p50} us p90={p90} us; frame p50 off={} us on={} us",
            frames[0], frames[1]
        );
        assert!(
            p50 * 10 < frames[1],
            "fusion p50 {p50} us is over 10 % of the frame p50 {} us",
            frames[1]
        );
    }
}
