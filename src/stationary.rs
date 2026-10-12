//! Stillness detection for the stationary inertial initialization.
//!
//! kornia-slam's `ImuInitializer::initialize_stationary` turns a still window into gravity, gyro
//! bias and zero velocity, and leaves deciding that the window WAS still to the caller. This is
//! that decision. Every check here is one way a moving robot could otherwise be read as parked,
//! and a wrong "parked" is not a degraded answer: it puts the robot's acceleration into gravity,
//! and the whole map is then rotated onto that.
//!
//! A window is still when ALL of these hold over it:
//!
//! * the IMU: the vector spread of the accelerometer and of the gyroscope are both under their
//!   thresholds (vibration, tilting in place, a head re-tilt), and the mean angular rate is under
//!   a plausible gyro bias (a slow constant turn, which has no spread);
//! * the wheels, when the wheel feed is on: their net motion over the window is under a floor,
//!   and readings cover the window at all;
//! * the camera: the keyframes in the window agree on where the camera is and which way it
//!   points, to within VO noise.
//!
//! kornia-slam adds two of its own: the map must be stereo (metric), and the mean specific force
//! must be within `gravity_tolerance` of standard gravity.

use cu29::units::si::f64::{Frequency, Time};
use cu29::units::si::frequency::hertz;
use cu29::units::si::time::second;
use kornia_3d::pose::Pose3d;
use kornia_sensors::imu::ImuCalib;
use kornia_slam::inertial::ImuWindowStats;

use crate::wheel::WheelDelta;

/// Multiple of the expected noise-only spread a still window may show.
///
/// A still window's spread estimate over ~240 samples sits within ~10 % of the true noise, so 4x
/// leaves room for noise the datasheet densities do not model (fans, the USB link, a slightly
/// worse unit) while staying an order of magnitude under what driving produces: measured parked
/// on XLeRobot, the accelerometer norm's standard deviation is 0.026 m/s^2, exactly the 0.0269 the
/// configured density predicts per axis at 200 Hz.
pub const NOISE_MULTIPLE: f64 = 4.0;

/// Default window length, seconds: long enough that a few hundred samples pin the means (gyro
/// bias to ~1e-4 rad/s), short enough to fit inside a stop.
pub const DEFAULT_WINDOW_S: f64 = 1.2;

/// Configuration of the stationary initialization. Build with [`StaticInitConfig::from_noise`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticInitConfig {
    /// Minimum span of contiguous IMU edges the window must cover.
    pub window: Time,
    /// Ceiling on the accelerometer's vector spread over the window, m/s^2.
    pub accel_rms_max: f64,
    /// Ceiling on the gyroscope's vector spread over the window, rad/s.
    pub gyro_rms_max: f64,
    /// Ceiling on the mean angular rate, rad/s: the largest gyro bias believed. Catches a slow
    /// constant turn, which the spread cannot see.
    pub gyro_mean_max: f64,
    /// Largest `| |mean accel - accel bias| - 9.81 |` accepted, m/s^2.
    pub gravity_tolerance: f64,
    /// Ceiling on the net wheel translation over the window, metres.
    pub wheel_translation_max: f64,
    /// Ceiling on the net wheel rotation over the window, radians.
    pub wheel_rotation_max: f64,
    /// Ceiling on how far any window keyframe's camera centre sits from the newest one's, metres.
    pub camera_translation_max: f64,
    /// Ceiling on any window keyframe's camera rotation relative to the newest one, radians.
    pub camera_rotation_max: f64,
}

impl StaticInitConfig {
    /// Thresholds derived from the IMU's own noise densities and sample rate.
    ///
    /// A white-noise density `s` sampled at `r` Hz gives a per-sample, per-axis standard deviation
    /// of `s * sqrt(r)`, and the vector spread [`ImuWindowStats`] reports is `sqrt(3)` times that.
    /// The ceilings are [`NOISE_MULTIPLE`] times it. On XLeRobot's BNO086 (0.0019 m/s^2/sqrt(Hz),
    /// 0.00012 rad/s/sqrt(Hz), 200 Hz) that is 0.186 m/s^2 and 0.0118 rad/s.
    ///
    /// The rest are not noise figures. `gyro_mean_max` 0.02 rad/s is ~6x the stationary bias
    /// measured on a BNO086 (0.0031 rad/s). `gravity_tolerance` 0.3 m/s^2 is 3 % of g: an
    /// accelerometer scale error is well inside it, a robot braking at 0.3 m/s^2 is not. The wheel
    /// floors (5 mm, 0.01 rad) are NET motion, so encoder jitter does not accumulate. The camera
    /// translation floor (1 cm) is ~50x the drift measured on a parked robot's keyframes (0.2 mm).
    ///
    /// The camera ROTATION floor (0.25 deg) is the one that bounds the gyro bias. Nothing else ties
    /// the gyro to the visual track: a rotation the wheels cannot see (a head pan, a slow base turn
    /// with the wheel feed off) at under `gyro_mean_max` has no spread and passes the IMU checks,
    /// and its rate would be written as gyro bias. Over the 1.2 s window 0.25 deg admits at most
    /// ~0.0036 rad/s of unseen rotation, about the BNO086's own bias (0.0031), where 1 deg admitted
    /// 0.0145, 5x it. Stereo keyframe attitude noise is hundredths of a degree, so a parked robot
    /// still clears it; a refusal only falls through to the motion solve.
    pub fn from_noise(calib: &ImuCalib, rate: Frequency) -> Self {
        let per_sample = |density: f64| density * rate.get::<hertz>().sqrt() * 3f64.sqrt();
        Self {
            window: Time::new::<second>(DEFAULT_WINDOW_S),
            accel_rms_max: NOISE_MULTIPLE * per_sample(calib.accel_noise),
            gyro_rms_max: NOISE_MULTIPLE * per_sample(calib.gyro_noise),
            gyro_mean_max: 0.02,
            gravity_tolerance: 0.3,
            wheel_translation_max: 0.005,
            wheel_rotation_max: 0.01,
            camera_translation_max: 0.01,
            camera_rotation_max: 0.25f64.to_radians(),
        }
    }
}

/// Why a window is not still.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NotStill {
    /// The contiguous IMU edges ending at the newest keyframe span less than the window.
    #[error("only {span_s:.2} s of contiguous IMU edges, need {want_s:.2} s")]
    TooShort {
        /// Span covered, seconds.
        span_s: f64,
        /// Span required, seconds.
        want_s: f64,
    },
    /// The accelerometer spread is over its ceiling.
    #[error("accelerometer spread {rms:.4} m/s^2 over {max:.4}")]
    AccelSpread {
        /// Measured spread.
        rms: f64,
        /// The ceiling.
        max: f64,
    },
    /// The gyroscope spread is over its ceiling.
    #[error("gyroscope spread {rms:.5} rad/s over {max:.5}")]
    GyroSpread {
        /// Measured spread.
        rms: f64,
        /// The ceiling.
        max: f64,
    },
    /// The mean angular rate is larger than any believable gyro bias.
    #[error("mean angular rate {rate:.5} rad/s over {max:.5}")]
    Turning {
        /// Measured mean rate.
        rate: f64,
        /// The ceiling.
        max: f64,
    },
    /// The wheels moved.
    #[error("wheels moved {translation_m:.4} m / {rotation_rad:.4} rad")]
    WheelsMoved {
        /// Net translation.
        translation_m: f64,
        /// Net rotation.
        rotation_rad: f64,
    },
    /// The wheel feed is on but no readings cover the window, so nothing vouches for the base.
    #[error("wheel feed on but no readings cover the window")]
    WheelsSilent,
    /// The window's keyframes disagree on the camera pose.
    #[error("camera moved {translation_m:.4} m / {rotation_rad:.4} rad across the window")]
    CameraMoved {
        /// Largest centre offset from the newest keyframe.
        translation_m: f64,
        /// Largest rotation from the newest keyframe.
        rotation_rad: f64,
    },
}

/// `value <= max`, false for a NaN: every ceiling here refuses what it cannot compare.
fn within(value: f64, max: f64) -> bool {
    value <= max
}

/// The IMU half: the window spans enough time, and neither sensor shows motion.
pub(crate) fn check_imu(
    span_s: f64,
    stats: &ImuWindowStats,
    cfg: &StaticInitConfig,
) -> Result<(), NotStill> {
    let want_s = cfg.window.get::<second>();
    if span_s < want_s {
        return Err(NotStill::TooShort { span_s, want_s });
    }
    check_imu_stats(stats, cfg)
}

/// The per-statistic part of [`check_imu`], without the span: accelerometer and gyroscope
/// spread and the mean rate. The parked keyframe run ([`crate::parked`]) applies it chunk by
/// chunk, where the span is whatever arrived since the last frame.
pub(crate) fn check_imu_stats(
    stats: &ImuWindowStats,
    cfg: &StaticInitConfig,
) -> Result<(), NotStill> {
    if !within(stats.accel_rms_dev, cfg.accel_rms_max) {
        return Err(NotStill::AccelSpread {
            rms: stats.accel_rms_dev,
            max: cfg.accel_rms_max,
        });
    }
    if !within(stats.gyro_rms_dev, cfg.gyro_rms_max) {
        return Err(NotStill::GyroSpread {
            rms: stats.gyro_rms_dev,
            max: cfg.gyro_rms_max,
        });
    }
    let rate = stats.mean_gyro.length();
    if !within(rate, cfg.gyro_mean_max) {
        return Err(NotStill::Turning {
            rate,
            max: cfg.gyro_mean_max,
        });
    }
    Ok(())
}

/// The wheel half: `None` when the wheel feed is on but could not integrate the window.
pub(crate) fn check_wheels(
    delta: Option<WheelDelta>,
    cfg: &StaticInitConfig,
) -> Result<(), NotStill> {
    let d = delta.ok_or(NotStill::WheelsSilent)?;
    let [dx, dy, dtheta] = d.delta;
    let (translation_m, rotation_rad) = (dx.hypot(dy), dtheta.abs());
    if !(within(translation_m, cfg.wheel_translation_max)
        && within(rotation_rad, cfg.wheel_rotation_max))
    {
        return Err(NotStill::WheelsMoved {
            translation_m,
            rotation_rad,
        });
    }
    Ok(())
}

/// The camera half: every window keyframe's world-to-camera pose against the newest one's.
pub(crate) fn check_camera(
    poses_world_to_cam: &[Pose3d],
    cfg: &StaticInitConfig,
) -> Result<(), NotStill> {
    let Some(newest) = poses_world_to_cam.last() else {
        return Ok(());
    };
    let (c_new, r_new) = (newest.inverse().translation, newest.rotation);
    let (mut translation_m, mut rotation_rad) = (0.0f64, 0.0f64);
    for p in poses_world_to_cam {
        translation_m = translation_m.max((p.inverse().translation - c_new).length());
        let r = p.rotation * r_new.transpose();
        let cos = ((r.x_axis.x + r.y_axis.y + r.z_axis.z) - 1.0) * 0.5;
        rotation_rad = rotation_rad.max(cos.clamp(-1.0, 1.0).acos());
    }
    if !(within(translation_m, cfg.camera_translation_max)
        && within(rotation_rad, cfg.camera_rotation_max))
    {
        return Err(NotStill::CameraMoved {
            translation_m,
            rotation_rad,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{SO3F64, Vec3F64};
    use kornia_sensors::imu::ImuMeasurement;

    const RATE: f64 = 200.0;

    /// XLeRobot's head.ron densities.
    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 0.00012,
            accel_noise: 0.0019,
            gyro_bias_noise: 0.000002,
            accel_bias_noise: 0.00004,
        }
    }

    fn cfg() -> StaticInitConfig {
        StaticInitConfig::from_noise(&calib(), Frequency::new::<hertz>(RATE))
    }

    /// Deterministic noise with unit variance per axis: a sum of incommensurate sines scaled by
    /// sqrt(2), so the window spread lands where the density says.
    fn unit_noise(k: usize, phase: f64) -> Vec3F64 {
        let t = k as f64;
        Vec3F64::new(
            (1.7 * t + phase).sin(),
            (2.9 * t + 2.0 * phase).sin(),
            (4.3 * t + 3.0 * phase).sin(),
        ) * std::f64::consts::SQRT_2
    }

    /// A window of `n` samples on a tilted IMU with a gyro bias, the datasheet noise, and
    /// `extra(k)` (accel, gyro) added on top.
    fn window(n: usize, extra: impl Fn(usize) -> (Vec3F64, Vec3F64)) -> ImuWindowStats {
        let r_wb = SO3F64::exp(Vec3F64::new(0.4, -0.2, 0.39)).matrix();
        let f = r_wb.transpose() * Vec3F64::new(0.0, -9.81, 0.0);
        let bg = Vec3F64::new(0.0031, -0.001, 0.002);
        let sa = calib().accel_noise * RATE.sqrt();
        let sg = calib().gyro_noise * RATE.sqrt();
        let samples: Vec<ImuMeasurement> = (0..n)
            .map(|k| {
                let (ea, eg) = extra(k);
                ImuMeasurement {
                    timestamp: k as f64 / RATE,
                    accel: f + unit_noise(k, 0.7) * sa + ea,
                    gyro: bg + unit_noise(k, 2.3) * sg + eg,
                }
            })
            .collect();
        ImuWindowStats::from_samples(&samples).unwrap()
    }

    fn still(_: usize) -> (Vec3F64, Vec3F64) {
        (Vec3F64::ZERO, Vec3F64::ZERO)
    }

    /// The derived thresholds are the documented numbers for the robot's IMU.
    ///
    /// Mutation: dropping the `sqrt(3)` vector factor turned this red.
    #[test]
    fn thresholds_derive_from_the_noise_densities() {
        let c = cfg();
        assert!(
            (c.accel_rms_max - 0.18615).abs() < 1e-4,
            "{}",
            c.accel_rms_max
        );
        assert!(
            (c.gyro_rms_max - 0.011757).abs() < 1e-5,
            "{}",
            c.gyro_rms_max
        );
        assert_eq!(c.window.get::<second>(), DEFAULT_WINDOW_S);
    }

    /// A parked window passes at datasheet noise; vibration, a tilt in place, a head re-tilt, a
    /// slow constant turn and a short window are each refused by the check meant for it.
    ///
    /// Mutations (each turned this red): dropping the mean-rate check (the slow turn passes);
    /// comparing the gyro spread against `accel_rms_max` (a head re-tilt passes); dropping the
    /// span check.
    #[test]
    fn imu_check_accepts_a_parked_window_and_refuses_each_kind_of_motion() {
        let c = cfg();
        let n = 240;
        let parked = window(n, still);
        assert!(parked.accel_rms_dev < 0.06, "{}", parked.accel_rms_dev);
        assert_eq!(check_imu(1.2, &parked, &c), Ok(()));

        // Motor vibration: 0.3 m/s^2 at ~25 Hz on one axis.
        let vib = window(n, |k| {
            let a = 0.3 * (2.0 * std::f64::consts::PI * 25.0 * k as f64 / RATE).sin();
            (Vec3F64::new(a, 0.0, 0.0), Vec3F64::ZERO)
        });
        assert!(matches!(
            check_imu(1.2, &vib, &c),
            Err(NotStill::AccelSpread { .. })
        ));

        // A head re-tilt: 0.1 rad/s for half the window, accelerometer untouched.
        let retilt = window(n, |k| {
            let w = if k < n / 2 { 0.1 } else { 0.0 };
            (Vec3F64::ZERO, Vec3F64::new(0.0, 0.0, w))
        });
        assert!(matches!(
            check_imu(1.2, &retilt, &c),
            Err(NotStill::GyroSpread { .. })
        ));

        // A slow constant turn: no spread at all, only a mean rate.
        let turn = window(n, |_| (Vec3F64::ZERO, Vec3F64::new(0.0, 0.03, 0.0)));
        assert!(turn.gyro_rms_dev < c.gyro_rms_max);
        assert!(matches!(
            check_imu(1.2, &turn, &c),
            Err(NotStill::Turning { .. })
        ));

        assert!(matches!(
            check_imu(0.9, &parked, &c),
            Err(NotStill::TooShort { .. })
        ));

        let mut nan = parked;
        nan.accel_rms_dev = f64::NAN;
        assert!(
            check_imu(1.2, &nan, &c).is_err(),
            "a NaN spread is not still"
        );
    }

    /// Wheels: parked readings with encoder jitter pass; a creep, a turn on the spot and a silent
    /// feed are refused.
    ///
    /// Mutation: checking only the translation (a turn on the spot passes) turned this red.
    #[test]
    fn wheel_check_refuses_motion_and_silence() {
        let c = cfg();
        let d = |dx: f64, dy: f64, dth: f64| {
            Some(WheelDelta {
                delta: [dx, dy, dth],
                path_m: 0.05,
                path_rad: 0.0,
                covariance: [[0.0; 3]; 3],
                speed_m_s: [0.0; 2],
                rate_rad_s: [0.0; 2],
            })
        };
        // Path length is jitter-inflated; the net motion is what counts.
        assert_eq!(check_wheels(d(0.0001, -0.0001, 0.0002), &c), Ok(()));
        assert!(matches!(
            check_wheels(d(0.02, 0.0, 0.0), &c),
            Err(NotStill::WheelsMoved { .. })
        ));
        assert!(matches!(
            check_wheels(d(0.0, 0.0, 0.05), &c),
            Err(NotStill::WheelsMoved { .. })
        ));
        assert_eq!(check_wheels(None, &c), Err(NotStill::WheelsSilent));
    }

    /// Camera: keyframes at VO noise pass; a drift of centimetres, a rotation of degrees, or a
    /// slow pan of one degree is refused, whichever keyframe carries it.
    ///
    /// Mutations (each turned this red): comparing only against the FIRST keyframe instead of
    /// every one (the middle keyframe's excursion passes); the original 1 deg rotation floor (the
    /// slow pan passes).
    #[test]
    fn camera_check_refuses_a_moving_track() {
        let c = cfg();
        let at = |x: f64, yaw: f64| {
            Pose3d::from_rt(
                SO3F64::exp(Vec3F64::new(0.0, yaw, 0.0)).matrix(),
                Vec3F64::new(x, 0.0, 0.0),
            )
            .inverse()
        };
        assert_eq!(
            check_camera(&[at(0.0002, 0.0), at(0.0, 0.0001), at(0.0001, 0.0)], &c),
            Ok(())
        );
        assert!(matches!(
            check_camera(&[at(0.0, 0.0), at(0.05, 0.0), at(0.0, 0.0)], &c),
            Err(NotStill::CameraMoved { .. })
        ));
        assert!(matches!(
            check_camera(&[at(0.0, 0.0), at(0.0, 0.1), at(0.0, 0.0)], &c),
            Err(NotStill::CameraMoved { .. })
        ));
        // A head pan at 0.0145 rad/s over 1.2 s: 1 deg, under `gyro_mean_max` and invisible to the
        // wheels, so the camera is the only check that sees it. Its rate must not become bias.
        let pan = 0.0145 * 1.2;
        assert!(pan < 1f64.to_radians());
        assert!(matches!(
            check_camera(&[at(0.0, 0.0), at(0.0, pan / 2.0), at(0.0, pan)], &c),
            Err(NotStill::CameraMoved { .. })
        ));
        assert_eq!(check_camera(&[], &c), Ok(()));
    }
}
