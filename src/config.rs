//! The `ComponentConfig` accessors this task needs.
//!
//! Kept local to the component: a component should not depend on an application's config
//! module.

use cu29::prelude::{ComponentConfig, CuResult};
use cu29::units::si::f64::Frequency;
use cu29::units::si::frequency::hertz;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::ImuCalib;

use crate::error::VioError;
use crate::fusion::FusionConfig;
use crate::imu::InertialConfig;
use crate::stationary::StaticInitConfig;
use crate::wheel::{WheelOdomConfig, mat3_from_rows};

/// What the tracker does with its map after a sustained tracking loss: the `on_tracking_loss`
/// key, spelled `"keep_map"` or `"reset_map"`.
///
/// `reset_map` is the setting for a long-running graph, and the reason is worth stating because
/// the obvious improvement is a trap. Relocalization does recover onto an existing map, so
/// keeping it looks strictly better, and it does stop the visible re-bootstrap thrash. But
/// kornia-slam never culls keyframes, so without a `max_keyframes` bound the drop is the ONLY
/// bound on map size: with it off, an 18 h run on a Jetson Orin with an OAK-D at 640x400 reached
/// 46,649 keyframes and 1 GB RSS, and per-frame cost spiralled until the tracker ran at ~0.1 fps
/// while the rest of the graph kept running, so the stall was SILENT. Visible thrash beats a
/// silent stall.
///
/// `keep_map`, the default, is the setting for an offline replay that must accumulate one map
/// across losses, so its output can be compared frame for frame with another run of the tracker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnTrackingLoss {
    /// Re-bootstrap into the existing map.
    #[default]
    KeepMap,
    /// Drop the map and start a new tracker world.
    ResetMap,
}

/// Reads the optional `on_tracking_loss` policy, defaulting to [`OnTrackingLoss::KeepMap`].
pub(crate) fn on_tracking_loss(config: Option<&ComponentConfig>) -> CuResult<OnTrackingLoss> {
    Ok(config
        .map(|c| c.get_value::<OnTrackingLoss>(ON_TRACKING_LOSS))
        .transpose()?
        .flatten()
        .unwrap_or_default())
}

/// Reads an optional POSITIVE integer key.
///
/// Zero is refused rather than accepted: as a `max_keyframes` it would drop the map on every
/// insertion, which reads as "VIO is broken", not as "the config said 0". A negative RON
/// literal fails the `u32` read here rather than wrapping to ~4.3e9 and silently disabling
/// whatever it bounds.
pub(crate) fn optional_positive_usize(
    config: Option<&ComponentConfig>,
    key: &'static str,
) -> CuResult<Option<usize>> {
    let raw = config.map(|c| c.get::<u32>(key)).transpose()?.flatten();
    match raw {
        Some(0) => Err(format!("`{key}` must be greater than 0").into()),
        other => Ok(other.map(|v| v as usize)),
    }
}

/// The `inertial` block of a `vio` component config.
///
/// One key, all-or-nothing, with `deny_unknown_fields` and no `serde(default)` on anything
/// physical: the extrinsic and the four noise densities have no defensible default, and a
/// silently-defaulted one is not a degraded answer but a plausible, wrong map. Omitting the
/// whole `inertial` key leaves the tracker on the stereo-only path it ships with.
///
/// ```ron
/// ( id: "vio",
///   type: "cu_kornia_vio::StereoVio",
///   config: {
///     "inertial": (
///       // R_BC, ROW-major, mapping camera axes into IMU/body axes: X_body = R_BC * X_cam.
///       // "cam" is the RECTIFIED left camera — the frame the tracker's own poses live in.
///       t_bc_rotation: [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
///       // Lever arm, metres, IMU origin relative to that camera. NOT optional and NOT
///       // zero-by-default: a rotation-only extrinsic is a silent lever-arm error, which
///       // injects a spurious specific force w x (w x e) that grows with angular rate.
///       t_bc_translation: [0.0, 0.0, 0.0],
///       gyro_noise: 0.0,        // rad/s/sqrt(Hz)
///       accel_noise: 0.0,       // m/s^2/sqrt(Hz)
///       gyro_bias_noise: 0.0,   // rad/s^2/sqrt(Hz)
///       accel_bias_noise: 0.0,  // m/s^3/sqrt(Hz)
///       rate_hz: 200.0,
///     ),
///   },
/// ),
/// ```
///
/// The values above are placeholders and every one of them is rejected: the densities must be
/// strictly positive, and an identity `t_bc_rotation` would only ever be correct on a device
/// where the IMU and the rectified camera share axes exactly. The device's own `T_BC` cannot be
/// relied on (the OAK-D's `imu_to_camera_extrinsics` query can fail on a given unit), so the key
/// must carry a measured value or be absent. There is deliberately no default; see
/// `t_bc_rotation`.
///
/// This is the WIRE form, and crate-private for that reason: its numbers are plain `f64` because
/// that is what a RON file holds. [`InertialRon::to_config`] is the one place they become the
/// unit-typed [`InertialConfig`] the tracker takes.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InertialRon {
    /// `R_BC`, ROW-major. See the type doc for the convention and why it is not column-major.
    ///
    /// NO DEFAULT. RTAB-Map's hardcoded first-generation OAK-D transform is the obvious
    /// candidate, and it fails the gravity cross-check: measured on a Jetson Orin with an OAK-D
    /// at 640x400, a static accelerometer read of `(-1.34, +9.56, +0.38)` mapped through it lands
    /// on camera -x, 82 deg from where an upright module puts gravity (camera -y), on a unit whose
    /// decoded picture is upright and landscape, so the module is not rolled. A published
    /// per-board value that fails the one free check is worse than no value: it looks
    /// calibrated. The rotation comes from a measurement of the unit in use or it is absent.
    pub t_bc_rotation: [f64; 9],
    /// `t_BC` in metres. No default either; a zero lever arm is a silent error under rotation.
    pub t_bc_translation: [f64; 3],
    /// Gyroscope white noise density, rad/s/sqrt(Hz).
    pub gyro_noise: f64,
    /// Accelerometer white noise density, m/s^2/sqrt(Hz).
    pub accel_noise: f64,
    /// Gyroscope bias random walk, rad/s^2/sqrt(Hz).
    pub gyro_bias_noise: f64,
    /// Accelerometer bias random walk, m/s^3/sqrt(Hz).
    pub accel_bias_noise: f64,
    /// Configured sample rate, Hz. The denominator of the interval coverage check.
    pub rate_hz: f64,
    /// Whether the inertial path may move a published pose. It gates two things, and both
    /// matter: the switch of local BA to the 15-DOF inertial solve once initialized, and the
    /// COMMIT of an accepted initialization — `apply_initialization` scales and rotates the
    /// whole map onto the estimated gravity and bumps the world generation, which goes on the
    /// wire as `reset_epoch` and makes every consumer wipe its trail and re-derive its anchor.
    /// With this off, the factors are still built, `try_initialize` still runs and its
    /// acceptance is still counted (`init_accepted` with `initialized=false` in the stop line)
    /// and logged with the gravity and bias it found, but nothing is applied: the map, the
    /// world generation and the bias stay where the visual path left them.
    ///
    /// Defaults to `false` here even though [`InertialConfig::new`] defaults it to `true`:
    /// reading a config to turn the inertial path ON is already the risky step, and the
    /// numbers above are what a user needs to decide whether the extrinsic is right
    /// BEFORE it is allowed to rotate the world. Gating only the local-BA switch would not be
    /// enough: a yaw-unverified extrinsic would still re-register the world at each of the three
    /// staged inertial-initialization refinements (Campos et al., ORB-SLAM3, IEEE T-RO 2021),
    /// three published-pose jumps in the first 15 s of every run.
    #[serde(default)]
    pub enable_inertial_ba: bool,
    /// Initialize while parked, from a still window; see [`StaticInitRon`]. Absent, the
    /// inertial path waits for the robot to move, as it always has.
    #[serde(default)]
    pub static_init: Option<StaticInitRon>,
    /// Largest gyro bias norm the tracker believes, rad/s; see [`crate::BiasLimit`]. Absent is
    /// [`crate::BiasLimit::DEFAULT_GYRO_RAD_S`].
    #[serde(default)]
    pub gyro_bias_limit: Option<f64>,
    /// Largest accelerometer bias norm, m/s^2. Absent is
    /// [`crate::BiasLimit::DEFAULT_ACCEL_M_S2`].
    #[serde(default)]
    pub accel_bias_limit: Option<f64>,
}

/// The `static_init` block inside `inertial`.
///
/// ```ron
/// static_init: (
///   enabled: true,
///   // Optional; each defaults to the value derived in `StaticInitConfig::from_noise`.
///   window_s: 1.2,            // contiguous IMU edges the still window must span
///   accel_rms_max: 0.186,     // m/s^2, accelerometer vector spread
///   gyro_rms_max: 0.0118,     // rad/s, gyroscope vector spread
///   gyro_mean_max: 0.02,      // rad/s, largest believable gyro bias
///   gravity_tolerance: 0.3,   // m/s^2, | |mean accel| - 9.81 |
/// ),
/// ```
///
/// Off unless `enabled`. The spread ceilings default to four times what the block's own noise
/// densities predict at `rate_hz`, so a config with honest densities needs nothing else. Like the
/// rest of `inertial`, it moves the world only with `enable_inertial_ba` on: with it off an
/// accepted still window is counted and logged, not applied.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StaticInitRon {
    /// Off unless set.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub window_s: Option<f64>,
    #[serde(default)]
    pub accel_rms_max: Option<f64>,
    #[serde(default)]
    pub gyro_rms_max: Option<f64>,
    #[serde(default)]
    pub gyro_mean_max: Option<f64>,
    #[serde(default)]
    pub gravity_tolerance: Option<f64>,
}

impl StaticInitRon {
    /// The config over `defaults`, or `None` when disabled. Validated either way.
    fn to_config(&self, defaults: StaticInitConfig) -> Result<Option<StaticInitConfig>, VioError> {
        let mut cfg = defaults;
        for (field, value, slot) in [
            (
                "static_init.accel_rms_max",
                self.accel_rms_max,
                &mut cfg.accel_rms_max,
            ),
            (
                "static_init.gyro_rms_max",
                self.gyro_rms_max,
                &mut cfg.gyro_rms_max,
            ),
            (
                "static_init.gyro_mean_max",
                self.gyro_mean_max,
                &mut cfg.gyro_mean_max,
            ),
            (
                "static_init.gravity_tolerance",
                self.gravity_tolerance,
                &mut cfg.gravity_tolerance,
            ),
        ] {
            if let Some(v) = value {
                *slot = v;
            }
            if !slot.is_finite() || *slot <= 0.0 {
                return Err(VioError::InertialParamInvalid {
                    field,
                    value: *slot,
                });
            }
        }
        if let Some(v) = self.window_s {
            if !v.is_finite() || v <= 0.0 {
                return Err(VioError::InertialParamInvalid {
                    field: "static_init.window_s",
                    value: v,
                });
            }
            cfg.window = cu29::units::si::f64::Time::new::<cu29::units::si::time::second>(v);
        }
        Ok(self.enabled.then_some(cfg))
    }
}

impl InertialRon {
    /// Converts to the tracker's config, validating the extrinsic and the densities.
    ///
    /// The rotation arrives ROW-major because that is how a human writes a matrix, and
    /// `Mat3F64` is glam-backed COLUMN-major. The transpose below is therefore mandatory —
    /// and it is an easy mistake to make, because a transposed rotation
    /// is still a valid rotation and nothing downstream can report it. It is spelled as ONE
    /// named operation rather than nine hand-placed indices, because the indices are where the
    /// mistake gets made; `test_rotation_is_read_row_major` pins it either way.
    ///
    /// # Errors
    ///
    /// [`VioError::InertialExtrinsicNotRigid`] or [`VioError::InertialParamInvalid`].
    pub fn to_config(&self) -> Result<InertialConfig, VioError> {
        // Row-major on the wire, glam is column-major: reading the rows as columns gives R^T.
        let rotation = Mat3F64::from_cols_array(&self.t_bc_rotation).transpose();
        let translation = Vec3F64::from_array(self.t_bc_translation);
        let calib = ImuCalib {
            gyro_noise: self.gyro_noise,
            accel_noise: self.accel_noise,
            gyro_bias_noise: self.gyro_bias_noise,
            accel_bias_noise: self.accel_bias_noise,
        };
        let mut cfg = InertialConfig::new(
            Pose3d::from_rt(rotation, translation),
            calib,
            Frequency::new::<hertz>(self.rate_hz),
        )?;
        cfg.enable_inertial_ba = self.enable_inertial_ba;
        for (field, value, slot) in [
            (
                "gyro_bias_limit",
                self.gyro_bias_limit,
                &mut cfg.bias_limit.gyro_rad_s,
            ),
            (
                "accel_bias_limit",
                self.accel_bias_limit,
                &mut cfg.bias_limit.accel_m_s2,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    return Err(VioError::InertialParamInvalid { field, value });
                }
                *slot = value;
            }
        }
        if let Some(block) = &self.static_init {
            cfg.static_init = block.to_config(StaticInitConfig::from_noise(
                &cfg.calib,
                cfg.gates.nominal_rate,
            ))?;
        }
        Ok(cfg)
    }
}

/// Reads the optional `inertial` block.
///
/// A malformed or partial block is a hard error, not a fallback to the stereo-only path: a
/// config that carries the key means to turn the inertial path on, and starting visual-only
/// because a field was misspelled is the silent failure this whole module is written against.
pub(crate) fn optional_inertial(config: Option<&ComponentConfig>) -> CuResult<Option<InertialRon>> {
    Ok(config
        .map(|c| c.get_value::<InertialRon>(INERTIAL))
        .transpose()?
        .flatten())
}

/// The `wheel_odom` block of a `vio` component config.
///
/// ```ron
/// "wheel_odom": (
///   enabled: true,
///   // Camera pose in the base frame (the URDF base -> rectified left camera), ROW-major.
///   // The fallback until a `CameraMount` arrives on the bus; a fixed mount needs nothing else.
///   cam_in_base_rotation: [0.0, 0.0, 1.0,  -1.0, 0.0, 0.0,  0.0, -1.0, 0.0],
///   cam_in_base_translation: [0.1, 0.0, 0.9],
///   translation_noise: 0.02,   // m per sqrt(m) of path
///   rotation_noise: 0.05,      // rad per sqrt(rad) turned
///   // Optional:
///   z_sigma: 0.01,             // m, planar stiffness on z
///   tilt_sigma: 0.01,          // rad, planar stiffness on roll and pitch
///   huber_chi2: 12.59,         // 6-DOF chi-square, 95 %; absent keeps the factor quadratic
///   max_gap_s: 0.5,            // longest hole between readings an interval may span
/// ),
/// ```
///
/// Absent, or with `enabled: false` (the default), the tracker behaves exactly as without the
/// block. Present, the mount and the two noise figures are required: there is no defensible
/// default for where a camera sits or how good a robot's wheels are, and a wrong one is not an
/// error but a confidently bent trajectory. The planar stiffness defaults to kornia-slam's
/// (1 cm, 0.01 rad), itself unmeasured.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WheelOdomRon {
    /// Off unless set, so a block can sit in a graph for an A/B without acting.
    #[serde(default)]
    pub enabled: bool,
    /// `R` of the camera in the base, ROW-major: `X_base = R X_cam + t`.
    pub cam_in_base_rotation: [f64; 9],
    /// `t` of the camera in the base, metres.
    pub cam_in_base_translation: [f64; 3],
    /// m per sqrt(m) of path.
    pub translation_noise: f64,
    /// rad per sqrt(rad) turned.
    pub rotation_noise: f64,
    #[serde(default)]
    pub z_sigma: Option<f64>,
    #[serde(default)]
    pub tilt_sigma: Option<f64>,
    #[serde(default)]
    pub huber_chi2: Option<f64>,
    #[serde(default)]
    pub max_gap_s: Option<f64>,
}

impl WheelOdomRon {
    /// The tracker's config, or `None` when the block is disabled. Validated either way, so a
    /// disabled block with a typo in it fails at startup rather than on the day it is enabled.
    pub fn to_config(&self) -> Result<Option<WheelOdomConfig>, VioError> {
        let cam_in_base = Pose3d::from_rt(
            mat3_from_rows(&self.cam_in_base_rotation),
            Vec3F64::from_array(self.cam_in_base_translation),
        );
        let mut cfg =
            WheelOdomConfig::new(cam_in_base, self.translation_noise, self.rotation_noise)?;
        if let Some(v) = self.z_sigma {
            cfg.z_sigma = v;
        }
        if let Some(v) = self.tilt_sigma {
            cfg.tilt_sigma = v;
        }
        cfg.huber_chi2 = self.huber_chi2;
        if let Some(v) = self.max_gap_s {
            // `from_secs_f64` would panic on a negative; `validate` refuses it instead.
            cfg.max_gap = cu29::units::si::f64::Time::new::<cu29::units::si::time::second>(v);
        }
        cfg.validate()?;
        Ok(self.enabled.then_some(cfg))
    }
}

/// Reads the optional `wheel_odom` block; malformed is a hard error, like `inertial`.
pub(crate) fn optional_wheel_odom(
    config: Option<&ComponentConfig>,
) -> CuResult<Option<WheelOdomRon>> {
    Ok(config
        .map(|c| c.get_value::<WheelOdomRon>(WHEEL_ODOM))
        .transpose()?
        .flatten())
}

/// The `per_frame_fusion` block of a `vio` component config.
///
/// ```ron
/// "per_frame_fusion": (
///   enabled: true,
///   // Optional: the vision-vs-prediction gate as a 6-DOF chi-square value, scaled to the
///   // frame's actual degrees of freedom. Default 22.458, the 99.9 % quantile.
///   chi2_gate: 22.458,
///   // Optional: the camera-to-base mount's uncertainty, metres and radians. Acts only through
///   // the motion (zero on a parked robot): a turn moves the wheel factor's camera prediction
///   // by `|dtheta| * sigma`. Defaults 0.03 and 0.035.
///   extrinsic_sigma_m: 0.03,
///   extrinsic_sigma_rad: 0.035,
/// ),
/// ```
///
/// Absent, or `enabled: false` (the default), the published pose is PnP's, as it always was.
/// Enabled, each tracked frame's pose is optimized against its map matches, the IMU from the last
/// keyframe and the wheels from the last keyframe together, and a frame whose vision disagrees
/// with the motion sensors past the gate publishes their prediction instead; see
/// [`crate::fusion`]. It acts only through the `inertial` and `wheel_odom` blocks: with neither
/// there is nothing to fuse and the tracker is unchanged.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FusionRon {
    /// Off unless set.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub chi2_gate: Option<f64>,
    #[serde(default)]
    pub extrinsic_sigma_m: Option<f64>,
    #[serde(default)]
    pub extrinsic_sigma_rad: Option<f64>,
}

impl FusionRon {
    /// The tracker's config, or `None` when disabled. Validated either way.
    pub fn to_config(&self) -> Result<Option<FusionConfig>, VioError> {
        let mut cfg = FusionConfig::default();
        if let Some(g) = self.chi2_gate {
            cfg.chi2_gate = g;
        }
        if let Some(v) = self.extrinsic_sigma_m {
            cfg.extrinsic_sigma_m = v;
        }
        if let Some(v) = self.extrinsic_sigma_rad {
            cfg.extrinsic_sigma_rad = v;
        }
        cfg.validate()?;
        Ok(self.enabled.then_some(cfg))
    }
}

/// Reads and validates the optional `per_frame_fusion` block; malformed is a hard error.
pub(crate) fn optional_fusion(config: Option<&ComponentConfig>) -> CuResult<Option<FusionConfig>> {
    let ron = config
        .map(|c| c.get_value::<FusionRon>(PER_FRAME_FUSION))
        .transpose()?
        .flatten();
    match ron {
        Some(r) => r.to_config().map_err(|e| e.to_string().into()),
        None => Ok(None),
    }
}

/// Reads an optional finite, strictly-positive float key.
pub(crate) fn optional_positive_f64(
    config: Option<&ComponentConfig>,
    key: &'static str,
) -> CuResult<Option<f64>> {
    let raw = config.map(|c| c.get::<f64>(key)).transpose()?.flatten();
    match raw {
        Some(v) if !v.is_finite() || v <= 0.0 => {
            Err(format!("`{key}` must be finite and greater than 0, got {v}").into())
        }
        other => Ok(other),
    }
}

/// The config keys `StereoVio::new` reads, each spelled ONCE.
///
/// The macro is what makes the list trustworthy: a second hand-typed copy of the strings could
/// only be kept honest by a test scraping the reads out of the source, which is fragile under
/// line wrapping and formatting. One declaration states it instead.
///
/// Both halves come from the same source text, so the list cannot lose an entry the reads
/// still use. That direction is the dangerous one: [`deny_unknown_keys`] REFUSES anything unlisted,
/// so a key read-but-unlisted gets a correct RON rejected at startup, with the real key named as
/// the unknown one.
macro_rules! config_keys {
    ($($name:ident => $lit:literal),+ $(,)?) => {
        $(#[doc = concat!("The `", $lit, "` config key.")]
        pub(crate) const $name: &str = $lit;)+
        /// Every key [`StereoVio`](crate::StereoVio) reads from its RON config, in declaration
        /// order. Public and read-only, so an application can assert the list its own configs
        /// are written against.
        pub const CONFIG_KEYS: &[&str] = &[$($lit),+];
    };
}

config_keys! {
    ON_TRACKING_LOSS => "on_tracking_loss",
    MAX_KEYFRAMES => "max_keyframes",
    ORB_KEYPOINTS => "orb_keypoints",
    SEARCH_RADIUS_PX => "search_radius_px",
    MAX_COVISIBLE_KEYFRAMES => "max_covisible_keyframes",
    PNP_LM_ITERATIONS => "pnp_lm_iterations",
    INERTIAL => "inertial",
    LANDMARK_BUFFERS => "landmark_buffers",
    WHEEL_ODOM => "wheel_odom",
    PER_FRAME_FUSION => "per_frame_fusion",
}

/// Refuses a config key `task` does not read.
///
/// `cu29` stores a component's config as `HashMap<String, Value>` and hands each task whatever
/// the RON carried: a key nobody reads is discarded in SILENCE — no parse error, no warning, no
/// log difference. A graph config can then carry keys under a comment asserting they are inert
/// while the deployed binary reads them, and only a measurement of the output settles which is
/// true. A typo behaves identically to a correct key whose dependency is too old, and BOTH
/// directions of the mistake read the same from outside: "the knob did nothing".
///
/// Refusing at construction makes both loud at startup instead of never. Behaviour-neutral for a
/// correct RON, which is what keeps it out of any A/B.
///
/// `task` and `known` are parameters rather than baked in so an application can apply the same
/// check, in the same `(config, task, keys)` shape, to its own tasks.
pub fn deny_unknown_keys(
    config: Option<&ComponentConfig>,
    task: &'static str,
    known: &[&str],
) -> CuResult<()> {
    let Some(config) = config else {
        return Ok(());
    };
    // Allocation-free on the success path: `Filter`'s lower size_hint is 0, so `collect` starts
    // at capacity 0 and never pushes, and `sort_unstable` sits behind the early return.
    let mut unknown: Vec<&str> = config
        .0
        .keys()
        .map(String::as_str)
        .filter(|k| !known.contains(k))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    Err(format!("unknown config key(s) {unknown:?} for {task}; it reads {known:?}").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::config::Value;

    #[test]
    fn test_on_tracking_loss_reads_the_snake_case_spelling_only() {
        let parse = |text: &str| {
            let mut cfg = ComponentConfig::new();
            cfg.set(ON_TRACKING_LOSS, text.to_string());
            on_tracking_loss(Some(&cfg))
        };
        assert_eq!(parse("reset_map").ok(), Some(OnTrackingLoss::ResetMap));
        assert_eq!(parse("keep_map").ok(), Some(OnTrackingLoss::KeepMap));
        assert!(
            parse("ResetMap").is_err(),
            "one canonical spelling, no aliases"
        );
        assert!(parse("reset").is_err());
        assert_eq!(on_tracking_loss(None).ok(), Some(OnTrackingLoss::KeepMap));
    }

    /// `optional_positive_f64` is the guard on every float geometry knob, and had no test.
    #[test]
    fn test_non_positive_or_non_finite_float_key_is_refused() {
        let mut cfg = ComponentConfig::new();
        cfg.set("r", 8.0f64);
        assert_eq!(optional_positive_f64(Some(&cfg), "r").unwrap(), Some(8.0));
        // Absent is fine — every one of these keys is optional.
        assert_eq!(optional_positive_f64(Some(&cfg), "absent").unwrap(), None);
        assert_eq!(optional_positive_f64(None, "r").unwrap(), None);

        for bad in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
            let mut c = ComponentConfig::new();
            c.set("r", bad);
            assert!(
                optional_positive_f64(Some(&c), "r").is_err(),
                "{bad} was accepted as a positive finite value"
            );
        }
    }

    fn ron(rotation: [f64; 9]) -> InertialRon {
        InertialRon {
            t_bc_rotation: rotation,
            t_bc_translation: [0.01, -0.02, 0.03],
            gyro_noise: 1.6e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9e-5,
            accel_bias_noise: 3.0e-3,
            rate_hz: 200.0,
            enable_inertial_ba: false,
            static_init: None,
            gyro_bias_limit: None,
            accel_bias_limit: None,
        }
    }

    /// The `inertial` block read the way a graph reads it, with `extra` appended to its fields.
    fn inertial_ron(extra: &str) -> CuResult<Option<InertialRon>> {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ "inertial": (
                 t_bc_rotation: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                 t_bc_translation: [0.0, 0.0, 0.0],
                 gyro_noise: 0.00012, accel_noise: 0.0019,
                 gyro_bias_noise: 0.000002, accel_bias_noise: 0.00004,
                 rate_hz: 200.0, enable_inertial_ba: true {extra} ) }} ) ],
                 cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None)?;
        let graph = cfg.get_graph(None)?;
        let id = graph.get_node_id_by_name("vio").expect("the node");
        let node = graph.get_node(id).expect("the node");
        optional_inertial(node.get_instance_config())
    }

    /// The bias limits: absent is the default, a value replaces its own norm only, and a
    /// non-positive or non-finite one is refused.
    ///
    /// Mutation: dropping the `*slot = value` write turned this red.
    #[test]
    fn test_the_bias_limit_keys() {
        let cfg = |extra: &str| inertial_ron(extra).unwrap().expect("present").to_config();
        assert_eq!(cfg("").unwrap().bias_limit, crate::BiasLimit::default());
        let tuned = cfg(", gyro_bias_limit: Some(0.03)").unwrap().bias_limit;
        assert_eq!(tuned.gyro_rad_s, 0.03);
        assert_eq!(tuned.accel_m_s2, crate::BiasLimit::DEFAULT_ACCEL_M_S2);
        let tuned = cfg(", accel_bias_limit: Some(0.8)").unwrap().bias_limit;
        assert_eq!(tuned.accel_m_s2, 0.8);
        assert_eq!(tuned.gyro_rad_s, crate::BiasLimit::DEFAULT_GYRO_RAD_S);
        for bad in ["gyro_bias_limit: Some(0.0)", "accel_bias_limit: Some(-1.0)"] {
            assert!(
                matches!(
                    cfg(&format!(", {bad}")),
                    Err(VioError::InertialParamInvalid { .. })
                ),
                "{bad} accepted"
            );
        }
    }

    /// `static_init`: absent and disabled are off; enabled takes the noise-derived defaults; an
    /// override replaces exactly its field; a typo or a non-positive value is refused.
    ///
    /// Mutation: ignoring `enabled` (always `Some`) turned this red.
    #[test]
    fn test_the_static_init_block() {
        let cfg = |extra: &str| inertial_ron(extra).unwrap().expect("present").to_config();

        assert!(cfg("").unwrap().static_init.is_none(), "absent is off");
        assert!(
            cfg(", static_init: (window_s: 2.0)")
                .unwrap()
                .static_init
                .is_none(),
            "enabled defaults to false"
        );

        let on = cfg(", static_init: (enabled: true)").unwrap();
        let derived = StaticInitConfig::from_noise(&on.calib, on.gates.nominal_rate);
        assert_eq!(on.static_init, Some(derived));

        let tuned = cfg(", static_init: (enabled: true, gyro_rms_max: 0.02, window_s: 1.5)")
            .unwrap()
            .static_init
            .unwrap();
        assert_eq!(tuned.gyro_rms_max, 0.02);
        assert_eq!(tuned.window.get::<cu29::units::si::time::second>(), 1.5);
        assert_eq!(tuned.accel_rms_max, derived.accel_rms_max);

        assert!(
            inertial_ron(", static_init: (enabled: true, gyro_rms_mx: 0.02)").is_err(),
            "unknown field accepted"
        );
        for bad in [
            "gravity_tolerance: 0.0",
            "window_s: -1.0",
            "accel_rms_max: -0.1",
        ] {
            assert!(
                matches!(
                    cfg(&format!(", static_init: (enabled: false, {bad})")),
                    Err(VioError::InertialParamInvalid { .. })
                ),
                "{bad} accepted"
            );
        }
    }

    /// The row-major -> column-major hand-off, checked on an asymmetric rotation so that a
    /// missing transpose changes the answer. A 90 degree yaw about y is symmetric under
    /// transpose in the wrong way to catch it; this is 90 degrees about z.
    #[test]
    fn test_rotation_is_read_row_major() {
        // Rows of R_BC: (0,-1,0), (1,0,0), (0,0,1).
        let cfg = ron([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .to_config()
            .expect("a proper rotation");
        // R * (1,0,0) must be the FIRST COLUMN, which the rows above make (0,1,0).
        let mapped = cfg.imu_t_bc.rotation * Vec3F64::new(1.0, 0.0, 0.0);
        assert!(
            (mapped - Vec3F64::new(0.0, 1.0, 0.0)).length() < 1e-12,
            "got {mapped:?}; a transposed read would give (0,-1,0)"
        );
        assert_eq!(cfg.imu_t_bc.translation.y, -0.02);
    }

    #[test]
    fn test_non_rotation_extrinsic_is_rejected() {
        // Scaled by 2: still orthogonal, no longer orthonormal.
        let err = ron([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
            .to_config()
            .unwrap_err();
        assert!(
            matches!(err, VioError::InertialExtrinsicNotRigid { .. }),
            "{err:?}"
        );
        // A reflection: det = -1, every row still unit length.
        let err = ron([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .to_config()
            .unwrap_err();
        assert!(
            matches!(err, VioError::InertialExtrinsicNotRigid { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn test_zero_noise_density_is_rejected() {
        let mut r = ron([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        r.gyro_noise = 0.0;
        let err = r.to_config().unwrap_err();
        match err {
            VioError::InertialParamInvalid { field, .. } => assert_eq!(field, "gyro_noise"),
            other => panic!("{other:?}"),
        }
    }

    /// `deny_unknown_keys` must REFUSE, not ignore. Without this nothing proves the function
    /// does anything — the key list alone is inert.
    #[test]
    fn test_typo_is_refused_rather_than_discarded() {
        // A TRANSPOSITION, not the singular `orb_keypoint`. The error echoes `known`, which
        // contains `orb_keypoints`, and the singular is a PREFIX of it — so `contains` would pass
        // on the echoed list alone, whether or not the message ever named the offending key. That
        // is the one property these assertions exist to prove.
        let err = deny_unknown_keys(Some(&cfg(&["orb_keypionts"])), "task", CONFIG_KEYS)
            .expect_err("an unknown key must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("orb_keypionts"),
            "must name the offender: {msg}"
        );
        assert!(msg.contains("orb_keypoints"), "must name the fix: {msg}");
        assert!(msg.contains("task"), "must name the task: {msg}");
    }

    /// The other half: every key the graphs actually ship is accepted. A checker that refuses
    /// everything would pass the test above.
    #[test]
    fn test_keys_the_graphs_ship_are_accepted() {
        deny_unknown_keys(Some(&cfg(CONFIG_KEYS)), "task", CONFIG_KEYS).unwrap();
        deny_unknown_keys(None, "task", CONFIG_KEYS).unwrap();
    }

    /// The `wheel_odom` block read the way a graph reads it: through a RON config file.
    fn wheel_ron(block: &str) -> CuResult<Option<WheelOdomRon>> {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ "wheel_odom": {block} }} ) ],
                 cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None)?;
        let graph = cfg.get_graph(None)?;
        let id = graph.get_node_id_by_name("vio").expect("the node");
        let node = graph.get_node(id).expect("the node");
        optional_wheel_odom(node.get_instance_config())
    }

    const WHEEL_BLOCK: &str = "(enabled: true, \
        cam_in_base_rotation: [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0], \
        cam_in_base_translation: [0.1, 0.0, 0.9], translation_noise: 0.02, rotation_noise: 0.05)";

    #[test]
    fn test_a_wheel_odom_block_is_read_row_major_and_inverted_once() {
        let ron = wheel_ron(WHEEL_BLOCK).unwrap().expect("present");
        let cfg = ron.to_config().unwrap().expect("enabled");
        // The camera's +z (optical axis) is the base's +x: t_cb maps base +x onto camera +z.
        let fwd = cfg.t_cb.rotation * Vec3F64::new(1.0, 0.0, 0.0);
        assert!(
            (fwd - Vec3F64::new(0.0, 0.0, 1.0)).length() < 1e-12,
            "{fwd:?}"
        );
        assert_eq!(
            cfg.z_sigma,
            kornia_slam::mapping::wheel_odom::WheelOdomMeasurement::DEFAULT_Z_SIGMA
        );
        assert_eq!(cfg.huber_chi2, None);
    }

    #[test]
    fn test_a_wheel_odom_block_is_off_unless_enabled() {
        let text = WHEEL_BLOCK.replace("enabled: true, ", "");
        let ron = wheel_ron(&text).unwrap().expect("present");
        assert!(
            ron.to_config().unwrap().is_none(),
            "enabled defaults to false"
        );
        assert!(optional_wheel_odom(None).unwrap().is_none());
    }

    #[test]
    fn test_a_malformed_wheel_odom_block_is_refused() {
        // An unknown field: a misspelt optional must not silently keep its default.
        let typo = WHEEL_BLOCK.replace("rotation_noise", "rotation_noise: 0.05, z_sigmaa");
        assert!(wheel_ron(&typo).is_err(), "unknown field accepted");
        // A required field missing.
        let partial = WHEEL_BLOCK.replace(", translation_noise: 0.02", "");
        assert!(wheel_ron(&partial).is_err(), "missing noise accepted");
        // A non-positive sigma, even while disabled.
        let zero = WHEEL_BLOCK
            .replace("enabled: true", "enabled: false")
            .replace(")", ", z_sigma: 0.0)");
        let ron = wheel_ron(&zero).unwrap().expect("parses");
        assert!(matches!(
            ron.to_config(),
            Err(VioError::WheelParamInvalid {
                field: "z_sigma",
                ..
            })
        ));
    }

    #[test]
    fn test_wheel_odom_is_a_known_key() {
        assert!(CONFIG_KEYS.contains(&"wheel_odom"));
        deny_unknown_keys(Some(&cfg(&["wheel_odom"])), "task", CONFIG_KEYS).unwrap();
        assert!(deny_unknown_keys(Some(&cfg(&["wheel_odometry"])), "task", CONFIG_KEYS).is_err());
    }

    /// The `per_frame_fusion` block read the way a graph reads it.
    fn fusion_ron(block: &str) -> CuResult<Option<FusionConfig>> {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ "per_frame_fusion": {block} }} ) ],
                 cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None)?;
        let graph = cfg.get_graph(None)?;
        let id = graph.get_node_id_by_name("vio").expect("the node");
        let node = graph.get_node(id).expect("the node");
        optional_fusion(node.get_instance_config())
    }

    /// Off unless enabled, the gate defaulted or read, and a bad block refused at startup, even
    /// while disabled.
    #[test]
    fn test_the_per_frame_fusion_block() {
        assert!(CONFIG_KEYS.contains(&"per_frame_fusion"));
        assert_eq!(optional_fusion(None).unwrap(), None);
        assert_eq!(
            fusion_ron("(chi2_gate: 30.0)").unwrap(),
            None,
            "enabled defaults to false"
        );
        assert_eq!(
            fusion_ron("(enabled: true)").unwrap(),
            Some(FusionConfig::default())
        );
        assert_eq!(
            fusion_ron("(enabled: true, chi2_gate: 30.0)")
                .unwrap()
                .map(|c| c.chi2_gate),
            Some(30.0)
        );
        assert!(fusion_ron("(enabled: false, chi2_gate: 0.0)").is_err());
        assert!(
            fusion_ron("(enabled: true, chi2: 30.0)").is_err(),
            "unknown field"
        );
        let c = fusion_ron("(enabled: true, extrinsic_sigma_m: 0.05, extrinsic_sigma_rad: 0.02)")
            .unwrap()
            .unwrap();
        assert_eq!((c.extrinsic_sigma_m, c.extrinsic_sigma_rad), (0.05, 0.02));
        assert!(fusion_ron("(enabled: true, extrinsic_sigma_m: -0.01)").is_err());
        assert!(fusion_ron("(enabled: true, extrinsic_sigma_rad: 0.0)").is_err());
    }

    fn cfg(keys: &[&str]) -> ComponentConfig {
        ComponentConfig(
            keys.iter()
                .map(|k| ((*k).to_string(), Value::from("1".to_string())))
                .collect(),
        )
    }
}
