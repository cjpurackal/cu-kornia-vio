//! Per-frame fused tracking: the current frame's pose optimized against its map matches, the IMU
//! and the wheels together, with a Mahalanobis gate between vision and the motion sensors.
//!
//! # Why
//!
//! Without this the three sensors meet only in keyframe local BA (~1.5/s on the robot). The pose
//! published at every frame is PnP alone: the gyro and the wheels only SEED it. A frame whose
//! matches are few or wrong publishes whatever PnP made of them, even while the gyro and the
//! wheels both say the robot did not move.
//!
//! # What is solved
//!
//! ORB-SLAM3's `PoseInertialOptimizationLastKeyFrame` (Campos et al., IEEE T-RO 2021), cut to the
//! 6-DOF pose of the current frame. The last keyframe is held fixed and three kinds of factor act
//! on the frame:
//!
//! * **Visual reprojection** of each kept map match, sigma `1.2^octave` px, Huber kernel at the
//!   2-DOF 95 % quantile (ORB-SLAM3's `sqrt(5.991)`).
//! * **IMU preintegration** from the keyframe, at the keyframe's bias and velocity. Rotation
//!   always; position once the inertial path is initialized (before that there is no metric
//!   velocity or gravity to predict it with). The current frame's velocity is the only state
//!   the velocity residual would constrain, so it is marginalized exactly by leaving that
//!   residual out; the frame's bias is held at the keyframe's.
//! * **Wheel odometry** from the keyframe: kornia-slam's BA factor, `r = Log(dT^-1 T_wb_kf^-1
//!   T_wb)`, with the same measurement and noise model ([`crate::wheel`]) and the mount at each
//!   end.
//!
//! The weights are the inverse covariances already configured: the IMU noise densities through
//! the preintegration, the wheels' `translation_noise` / `rotation_noise` and planar stiffness.
//!
//! # What ORB-SLAM3 leaves out, and why it is added here
//!
//! ORB-SLAM3 trusts the keyframe's bias and velocity exactly. Over a keyframe interval (0.7 s on
//! the robot) at the configured densities that makes the gyro worth 0.1 mrad and the IMU position
//! 0.6 mm, so a bias error of 1 mrad/s, a 1 deg gravity error or the 2 deg residual of the measured
//! camera-IMU extrinsic would each be read as vision being wrong. The prediction's covariance is
//! therefore widened by the keyframe state's own uncertainty: [`KF_GYRO_BIAS_SIGMA`],
//! [`KF_ACCEL_BIAS_SIGMA`], [`KF_VELOCITY_SIGMA`] and [`GYRO_RELATIVE_SIGMA`]. They decide how far
//! the motion sensors can outvote vision, and none of them is measured on the robot.
//!
//! # The gate
//!
//! Every motion factor is a measurement of the current pose. At the vision-only pose their
//! stacked residual `r` has the innovation covariance `S = C + J Sigma_v J^T` (`C` the factors'
//! covariance, `Sigma_v` vision's from its Hessian): `chi2 = r^T S^-1 r` is the Mahalanobis
//! distance between what vision placed and what the motion sensors predicted, with as many degrees
//! of freedom as there are residual rows. Past the gate, vision is rejected for the frame and the
//! motion sensors' own optimum is published instead.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SO3F64, Vec3F64};
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement, PreintegratedImu};
use kornia_slam::mapping::wheel_odom::WheelOdomMeasurement;

use crate::error::VioError;

/// The default gate: the 6-DOF chi-square 99.9 % quantile. Scaled to the frame's actual degrees of
/// freedom by [`gate_for_dof`].
pub const DEFAULT_CHI2_GATE: f64 = 22.458;

/// One sigma of the keyframe's gyro bias, rad/s, once the inertial path is initialized. A
/// still-window or VI-BA estimate; ten times what the configured density predicts for a 1.2 s
/// still window (0.00012 / sqrt(1.2)), because nothing has measured how good the estimate is.
pub const KF_GYRO_BIAS_SIGMA: f64 = 1e-3;

/// The same before the inertial path is initialized: the bias is then the configured initial one,
/// and a stationary gyro bias of 3 mrad/s has been measured on a BNO086.
pub const UNINITIALIZED_GYRO_BIAS_SIGMA: f64 = 1e-2;

/// One sigma of the keyframe's accelerometer bias, m/s^2. Also absorbs a gravity-direction error:
/// 0.1 m/s^2 is ~0.6 deg of tilt.
pub const KF_ACCEL_BIAS_SIGMA: f64 = 0.1;

/// One sigma of the keyframe's velocity, m/s.
pub const KF_VELOCITY_SIGMA: f64 = 0.05;

/// One sigma of the gyro's rotation as a fraction of the angle turned: the extrinsic (a 2 deg fit
/// residual on the robot), the gyro scale and the camera-IMU time offset all grow with the turn.
pub const GYRO_RELATIVE_SIGMA: f64 = 0.05;

/// Huber threshold on a reprojection's normalized squared error: the 2-DOF 95 % quantile.
const HUBER_REPROJECTION_CHI2: f64 = 5.991;

/// Levenberg-Marquardt iterations per solve. Started from PnP's own optimum (or the prediction),
/// the solve is a few-millimetre correction and converges in two or three.
const MAX_ITERATIONS: usize = 8;

/// Central-difference step for the motion factors' Jacobians.
const JACOBIAN_EPS: f64 = 1e-6;

/// Per-frame fused tracking, the tracker side of the `per_frame_fusion` config block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionConfig {
    /// The gate on the vision-vs-prediction Mahalanobis distance, as a 6-DOF chi-square value;
    /// see [`gate_for_dof`].
    pub chi2_gate: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            chi2_gate: DEFAULT_CHI2_GATE,
        }
    }
}

impl FusionConfig {
    /// Refuses a gate that is not finite and positive.
    pub fn validate(&self) -> Result<(), VioError> {
        if !self.chi2_gate.is_finite() || self.chi2_gate <= 0.0 {
            return Err(VioError::FusionParamInvalid {
                field: "chi2_gate",
                value: self.chi2_gate,
            });
        }
        Ok(())
    }
}

/// The per-frame fusion's counters. Diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FusionStats {
    /// Frames the fusion ran on (a tracked frame with at least one motion factor), lifetime.
    pub frames: u64,
    /// Frames whose vision the gate rejected, lifetime.
    pub rejects: u64,
    /// Rejections that ran past [`MAX_CONSECUTIVE_REJECTS`] and were handed to the loss path.
    pub reject_runs_expired: u64,
    /// Frames whose wheel factor was left out because the wheels disagreed with the gyro (slip,
    /// or the head turned on its mast), lifetime.
    pub wheel_dropped: u64,
    /// Rejections in a row, now.
    pub consecutive_rejects: u32,
    /// The last frame's chi2; NaN when the fusion did not run on it.
    pub last_chi2: f64,
    /// Its degrees of freedom; 0 when the fusion did not run.
    pub last_dof: u8,
    /// The last fused frame's solve cost, microseconds (gate, and the fused or predicted solve).
    pub last_cost_us: u32,
}

/// Rejections in a row before the frame is handed to the loss path instead. A prediction that has
/// coasted this long (~1.2 s at the head's 12 Hz) is the likelier one to be wrong (wheels slipping
/// straight, a bias walking): vision that stays self-consistent against it gets the relocalization
/// and re-anchor machinery rather than being outvoted forever.
pub const MAX_CONSECUTIVE_REJECTS: u32 = 15;

/// 6x6 row-major.
pub(crate) type Mat6 = [[f64; 6]; 6];

/// One kept map match: the landmark, where the frame saw it, and its sigma.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VisualObs {
    pub point_world: Vec3F64,
    pub pixel: [f64; 2],
    pub inv_sigma: f64,
}

/// What a motion factor measures.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MotionTerm {
    /// IMU preintegration from the keyframe. `position` adds the position rows.
    Imu {
        imu_t_bc: Pose3d,
        kf_r_bw: Mat3F64,
        kf_p_wb: Vec3F64,
        kf_velocity: Vec3F64,
        gravity: Vec3F64,
        dt: f64,
        delta_rotation: Mat3F64,
        delta_position: Vec3F64,
        position: bool,
    },
    /// Wheel odometry from the keyframe.
    Wheel {
        delta_inv: Pose3d,
        kf_t_bw: Pose3d,
        t_cb_to: Pose3d,
    },
}

/// A motion factor on the current pose: its residual model and covariance (`dim` x `dim`).
#[derive(Debug, Clone)]
pub(crate) struct MotionFactor {
    pub term: MotionTerm,
    pub dim: usize,
    pub covariance: Vec<f64>,
    pub information: Vec<f64>,
}

impl MotionFactor {
    /// The IMU factor of a keyframe-to-frame preintegration, integrated at the keyframe's bias.
    ///
    /// `gyro_bias_sigma` is the keyframe bias's uncertainty ([`KF_GYRO_BIAS_SIGMA`] or
    /// [`UNINITIALIZED_GYRO_BIAS_SIGMA`]). `None` for a singular covariance.
    #[allow(clippy::too_many_arguments)]
    pub fn imu(
        pim: &PreintegratedImu,
        imu_t_bc: Pose3d,
        kf_pose_cw: &Pose3d,
        kf_velocity: Vec3F64,
        gravity: Vec3F64,
        position: bool,
        gyro_bias_sigma: f64,
    ) -> Option<Self> {
        let kf_t_bw = imu_t_bc.compose(kf_pose_cw);
        let dim = if position { 6 } else { 3 };
        // 9x9 [rot, vel, pos], symmetric, so its storage order does not matter here.
        let c9 = |r: usize, c: usize| pim.covariance[r * 9 + c];
        let idx = [0usize, 1, 2, 6, 7, 8];
        let mut cov = vec![0.0; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                cov[r * dim + c] = c9(idx[r], idx[c]);
            }
        }
        // The keyframe state's uncertainty, propagated through the bias Jacobians.
        let sg2 = gyro_bias_sigma * gyro_bias_sigma;
        let sa2 = KF_ACCEL_BIAS_SIGMA * KF_ACCEL_BIAS_SIGMA;
        let jrg = pim.d_rotation_d_bias_gyro;
        let jpg = pim.d_position_d_bias_gyro;
        let jpa = pim.d_position_d_bias_accel;
        let theta = SO3F64::from_matrix(&pim.delta_rotation).log().length();
        let rel = (GYRO_RELATIVE_SIGMA * theta).powi(2);
        let add = |cov: &mut Vec<f64>, r0: usize, c0: usize, m: Mat3F64| {
            let a = m.to_cols_array();
            for c in 0..3 {
                for r in 0..3 {
                    cov[(r0 + r) * dim + c0 + c] += a[c * 3 + r];
                }
            }
        };
        add(
            &mut cov,
            0,
            0,
            jrg * jrg.transpose() * sg2 + Mat3F64::IDENTITY * rel,
        );
        if position {
            let vel = (KF_VELOCITY_SIGMA * pim.dt).powi(2);
            add(&mut cov, 0, 3, jrg * jpg.transpose() * sg2);
            add(&mut cov, 3, 0, jpg * jrg.transpose() * sg2);
            add(
                &mut cov,
                3,
                3,
                jpg * jpg.transpose() * sg2 + jpa * jpa.transpose() * sa2 + Mat3F64::IDENTITY * vel,
            );
        }
        let information = invert(&cov, dim)?;
        Some(Self {
            term: MotionTerm::Imu {
                imu_t_bc,
                kf_r_bw: kf_t_bw.rotation,
                kf_p_wb: kf_t_bw.inverse().translation,
                kf_velocity,
                gravity,
                dt: pim.dt,
                delta_rotation: pim.delta_rotation,
                delta_position: pim.delta_position,
                position,
            },
            dim,
            covariance: cov,
            information,
        })
    }

    /// The wheel factor of a keyframe-to-frame interval: kornia-slam's BA factor with the
    /// keyframe end fixed. `None` for a measurement whose information is not positive definite.
    pub fn wheel(
        measurement: &WheelOdomMeasurement,
        t_cb_from: Pose3d,
        t_cb_to: Pose3d,
        kf_pose_cw: &Pose3d,
    ) -> Option<Self> {
        let info = measurement.information()?;
        let information: Vec<f64> = info.iter().flatten().copied().collect();
        let covariance = invert(&information, 6)?;
        Some(Self {
            term: MotionTerm::Wheel {
                delta_inv: measurement.delta_pose().inverse(),
                kf_t_bw: t_cb_from.inverse().compose(kf_pose_cw),
                t_cb_to,
            },
            dim: 6,
            covariance,
            information,
        })
    }

    /// The residual at the current world-to-camera pose; the first `dim` entries are used.
    pub fn residual(&self, pose_cw: &Pose3d) -> [f64; 6] {
        match self.term {
            MotionTerm::Imu {
                imu_t_bc,
                kf_r_bw,
                kf_p_wb,
                kf_velocity,
                gravity,
                dt,
                delta_rotation,
                delta_position,
                position,
            } => {
                // The VI-BA residual (`vi_ba_schur::imu_residual_and_jacobians`), rows R and p.
                let t_bw = imu_t_bc.compose(pose_cw);
                let r_wb = t_bw.rotation.transpose();
                let rot = SO3F64::from_matrix(&(delta_rotation.transpose() * kf_r_bw * r_wb)).log();
                let mut out = [rot.x, rot.y, rot.z, 0.0, 0.0, 0.0];
                if position {
                    let p_wb = t_bw.inverse().translation;
                    let dp = kf_r_bw
                        * (p_wb - kf_p_wb - kf_velocity * dt - gravity * (0.5 * dt * dt))
                        - delta_position;
                    out[3..].copy_from_slice(&[dp.x, dp.y, dp.z]);
                }
                out
            }
            MotionTerm::Wheel {
                delta_inv,
                kf_t_bw,
                t_cb_to,
            } => {
                let t_wb = pose_cw.inverse().compose(&t_cb_to);
                se3_log(&delta_inv.compose(&kf_t_bw).compose(&t_wb))
            }
        }
    }

    /// The residual and its central-difference Jacobian (`dim` x 6, row-major) under the left
    /// perturbation `Exp(xi) * T`.
    fn linearize(&self, pose_cw: &Pose3d) -> ([f64; 6], [[f64; 6]; 6]) {
        let r = self.residual(pose_cw);
        let mut j = [[0.0; 6]; 6];
        for k in 0..6 {
            let mut xi = [0.0; 6];
            xi[k] = JACOBIAN_EPS;
            let rp = self.residual(&se3_exp(&xi).compose(pose_cw));
            xi[k] = -JACOBIAN_EPS;
            let rm = self.residual(&se3_exp(&xi).compose(pose_cw));
            for row in 0..self.dim {
                j[row][k] = (rp[row] - rm[row]) / (2.0 * JACOBIAN_EPS);
            }
        }
        (r, j)
    }

    /// `r^T Omega r`.
    fn chi2(&self, pose_cw: &Pose3d) -> f64 {
        let r = self.residual(pose_cw);
        quadratic(&self.information, &r[..self.dim])
    }
}

/// What the gate decided for one frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusionOutcome {
    /// The pose to publish: fused, or the motion sensors' prediction when vision was rejected.
    pub pose: Pose3d,
    /// Whether vision was rejected.
    pub rejected: bool,
    /// The innovation's Mahalanobis distance, and its degrees of freedom.
    pub chi2: f64,
    pub dof: usize,
}

/// The gate and the solve for one frame. `None` when there is nothing to fuse (no motion factor)
/// or vision's own information is singular (too few matches to say anything about the pose).
///
/// `vision_pose` is PnP's optimum, the start of the fused solve; `prediction_seed` is where the
/// prediction solve starts when vision is rejected (the tracker's motion prior), which also fixes
/// any direction the motion factors leave unconstrained.
pub(crate) fn fuse(
    camera: &PinholeCamera,
    visual: &[VisualObs],
    factors: &[MotionFactor],
    vision_pose: &Pose3d,
    prediction_seed: &Pose3d,
    chi2_gate: f64,
) -> Option<FusionOutcome> {
    if factors.is_empty() {
        return None;
    }
    let (chi2, dof) = innovation_chi2(camera, visual, factors, vision_pose)?;
    if chi2 > gate_for_dof(chi2_gate, dof) {
        let pose = solve(camera, &[], factors, prediction_seed);
        return Some(FusionOutcome {
            pose,
            rejected: true,
            chi2,
            dof,
        });
    }
    Some(FusionOutcome {
        pose: solve(camera, visual, factors, vision_pose),
        rejected: false,
        chi2,
        dof,
    })
}

/// The 6-DOF chi-square value `chi2_6` moved to `dof` degrees of freedom at the same tail
/// probability, through the Wilson-Hilferty cube-root normal approximation (within 0.5 % of the
/// exact quantile at 3 to 15 DOF and 99-99.9 %).
pub fn gate_for_dof(chi2_6: f64, dof: usize) -> f64 {
    let wh = |k: f64| 2.0 / (9.0 * k);
    let z = ((chi2_6 / 6.0).cbrt() - (1.0 - wh(6.0))) / wh(6.0).sqrt();
    let m = dof as f64;
    m * (1.0 - wh(m) + z * wh(m).sqrt()).powi(3)
}

/// The Mahalanobis distance between the vision-only pose and the motion factors, and its DOF.
pub(crate) fn innovation_chi2(
    camera: &PinholeCamera,
    visual: &[VisualObs],
    factors: &[MotionFactor],
    vision_pose: &Pose3d,
) -> Option<(f64, usize)> {
    let (h_v, _, _) = visual_normal_equations(camera, visual, vision_pose);
    let h_flat: Vec<f64> = h_v.iter().flatten().copied().collect();
    let sigma_v = invert(&h_flat, 6)?;
    let m: usize = factors.iter().map(|f| f.dim).sum();
    let mut r = vec![0.0; m];
    let mut j = vec![[0.0; 6]; m];
    let mut s = vec![0.0; m * m];
    let mut row = 0;
    for f in factors {
        let (rf, jf) = f.linearize(vision_pose);
        for a in 0..f.dim {
            r[row + a] = rf[a];
            j[row + a] = jf[a];
            for b in 0..f.dim {
                s[(row + a) * m + row + b] = f.covariance[a * f.dim + b];
            }
        }
        row += f.dim;
    }
    // S += J Sigma_v J^T
    for a in 0..m {
        for b in 0..m {
            let mut acc = 0.0;
            for p in 0..6 {
                let mut jp = 0.0;
                for q in 0..6 {
                    jp += sigma_v[p * 6 + q] * j[b][q];
                }
                acc += j[a][p] * jp;
            }
            s[a * m + b] += acc;
        }
    }
    let mut x = r.clone();
    if !cholesky_solve(&mut s, m, &mut x) {
        return None;
    }
    let chi2: f64 = r.iter().zip(&x).map(|(a, b)| a * b).sum();
    chi2.is_finite().then_some((chi2, m))
}

/// Levenberg-Marquardt over the pose: robust reprojections plus quadratic motion factors.
pub(crate) fn solve(
    camera: &PinholeCamera,
    visual: &[VisualObs],
    factors: &[MotionFactor],
    start: &Pose3d,
) -> Pose3d {
    let mut pose = *start;
    let mut lambda = 1e-4;
    let mut cost = total_cost(camera, visual, factors, &pose);
    for _ in 0..MAX_ITERATIONS {
        let (mut h, mut b, _) = visual_normal_equations(camera, visual, &pose);
        for f in factors {
            let (r, j) = f.linearize(&pose);
            accumulate(&mut h, &mut b, &j, &f.information, &r, f.dim);
        }
        let max_diag = (0..6).map(|i| h[i][i]).fold(0.0f64, f64::max);
        if max_diag.is_nan() || max_diag <= 0.0 {
            break;
        }
        let mut improved = false;
        for _ in 0..6 {
            let mut a: Vec<f64> = h.iter().flatten().copied().collect();
            for i in 0..6 {
                a[i * 6 + i] += lambda * h[i][i].max(1e-9 * max_diag);
            }
            let mut step: Vec<f64> = b.iter().map(|v| -v).collect();
            if !cholesky_solve(&mut a, 6, &mut step) {
                lambda *= 10.0;
                continue;
            }
            let xi: [f64; 6] = std::array::from_fn(|i| step[i]);
            let candidate = se3_exp(&xi).compose(&pose);
            let c = total_cost(camera, visual, factors, &candidate);
            if c.is_finite() && c <= cost {
                pose = candidate;
                // Sub-micrometre and sub-microradian, or no relative progress: done.
                let converged = xi.iter().all(|v| v.abs() < 1e-7) || cost - c <= 1e-9 * cost;
                cost = c;
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                if converged {
                    return pose;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    pose
}

/// The robust visual cost plus the motion factors' `r^T Omega r`.
fn total_cost(
    camera: &PinholeCamera,
    visual: &[VisualObs],
    factors: &[MotionFactor],
    pose: &Pose3d,
) -> f64 {
    visual_cost(camera, visual, pose) + factors.iter().map(|f| f.chi2(pose)).sum::<f64>()
}

/// The robust reprojection cost alone: [`visual_normal_equations`]' third output, without the
/// Jacobians, for the LM step test.
fn visual_cost(camera: &PinholeCamera, visual: &[VisualObs], pose: &Pose3d) -> f64 {
    let k2 = HUBER_REPROJECTION_CHI2;
    visual
        .iter()
        .filter_map(|o| {
            let pc = pose.transform_point(&o.point_world);
            (pc.z > 1e-6).then(|| {
                let iz = 1.0 / pc.z;
                let eu = (camera.fx * pc.x * iz + camera.cx - o.pixel[0]) * o.inv_sigma;
                let ev = (camera.fy * pc.y * iz + camera.cy - o.pixel[1]) * o.inv_sigma;
                let s = eu * eu + ev * ev;
                if s <= k2 {
                    s
                } else {
                    2.0 * (k2 * s).sqrt() - k2
                }
            })
        })
        .sum()
}

/// `J^T W J`, `J^T W r` and the robust cost of the reprojections at `pose`.
fn visual_normal_equations(
    camera: &PinholeCamera,
    visual: &[VisualObs],
    pose: &Pose3d,
) -> (Mat6, [f64; 6], f64) {
    let mut h = [[0.0; 6]; 6];
    let mut b = [0.0; 6];
    let mut cost = 0.0;
    let k = HUBER_REPROJECTION_CHI2.sqrt();
    for o in visual {
        let pc = pose.transform_point(&o.point_world);
        if pc.z <= 1e-6 {
            continue;
        }
        let iz = 1.0 / pc.z;
        let u = camera.fx * pc.x * iz + camera.cx;
        let v = camera.fy * pc.y * iz + camera.cy;
        let e = [
            (u - o.pixel[0]) * o.inv_sigma,
            (v - o.pixel[1]) * o.inv_sigma,
        ];
        let s = e[0] * e[0] + e[1] * e[1];
        let (w, c) = if s <= k * k {
            (1.0, s)
        } else {
            let n = s.sqrt();
            (k / n, 2.0 * k * n - k * k)
        };
        cost += c;
        // d(pixel)/d(p_cam), scaled by 1/sigma.
        let fx = camera.fx * iz * o.inv_sigma;
        let fy = camera.fy * iz * o.inv_sigma;
        let dp = [[fx, 0.0, -fx * pc.x * iz], [0.0, fy, -fy * pc.y * iz]];
        // d(p_cam)/d(xi) = [I | -[p_cam]x] under the left perturbation.
        let neg_hat = [[0.0, pc.z, -pc.y], [-pc.z, 0.0, pc.x], [pc.y, -pc.x, 0.0]];
        let mut j = [[0.0; 6]; 2];
        for (row, d) in dp.iter().enumerate() {
            for c in 0..3 {
                j[row][c] = d[c];
                j[row][3 + c] = (0..3).map(|q| d[q] * neg_hat[q][c]).sum();
            }
        }
        for a in 0..6 {
            b[a] += w * (j[0][a] * e[0] + j[1][a] * e[1]);
            for c in 0..6 {
                h[a][c] += w * (j[0][a] * j[0][c] + j[1][a] * j[1][c]);
            }
        }
    }
    (h, b, cost)
}

/// `H += J^T Omega J`, `b += J^T Omega r` for a `dim`-row factor.
fn accumulate(
    h: &mut Mat6,
    b: &mut [f64; 6],
    j: &[[f64; 6]; 6],
    info: &[f64],
    r: &[f64; 6],
    dim: usize,
) {
    for p in 0..6 {
        for a in 0..dim {
            let mut wj = 0.0; // (Omega J)[a][p]
            let mut wr = 0.0;
            for c in 0..dim {
                wj += info[a * dim + c] * j[c][p];
                if p == 0 {
                    wr += info[a * dim + c] * r[c];
                }
            }
            if p == 0 {
                for (q, bq) in b.iter_mut().enumerate() {
                    *bq += j[a][q] * wr;
                }
            }
            for q in 0..6 {
                h[q][p] += j[a][q] * wj;
            }
        }
    }
}

fn quadratic(m: &[f64], v: &[f64]) -> f64 {
    let n = v.len();
    (0..n)
        .map(|a| (0..n).map(|b| v[a] * m[a * n + b] * v[b]).sum::<f64>())
        .sum()
}

/// In-place Cholesky solve of `a x = b` for symmetric positive definite `a` (`n` x `n`,
/// row-major); `b` becomes `x`. False when `a` is not positive definite.
fn cholesky_solve(a: &mut [f64], n: usize, b: &mut [f64]) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !d.is_finite() || d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    true
}

/// The inverse of a symmetric positive definite `n` x `n` matrix, row-major.
fn invert(m: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut out = vec![0.0; n * n];
    for c in 0..n {
        let mut a = m.to_vec();
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        if !cholesky_solve(&mut a, n, &mut e) {
            return None;
        }
        for r in 0..n {
            out[r * n + c] = e[r];
        }
    }
    Some(out)
}

/// `Exp([rho; omega])` with translation `J_l(omega) rho`, kornia-slam's tangent order.
pub(crate) fn se3_exp(xi: &[f64; 6]) -> Pose3d {
    let rho = Vec3F64::new(xi[0], xi[1], xi[2]);
    let omega = Vec3F64::new(xi[3], xi[4], xi[5]);
    Pose3d::new(
        SO3F64::exp(omega).matrix(),
        SO3F64::left_jacobian(omega) * rho,
    )
}

/// The inverse of [`se3_exp`].
fn se3_log(t: &Pose3d) -> [f64; 6] {
    let omega = SO3F64::from_matrix(&t.rotation).log();
    let rho = SO3F64::left_jacobian(omega).inverse() * t.translation;
    [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z]
}

/// The keyframe-to-frame preintegration, extended frame by frame instead of re-integrated.
///
/// A parked robot inserts keyframes seconds apart, so integrating the whole keyframe interval
/// again at every frame grows with the time since the keyframe (200 samples a second, each a 9x9
/// covariance step). The integration up to the newest sample is kept, and each frame integrates
/// only the samples since, then closes the tail to the frame stamp on a copy — exactly what
/// [`PreintegratedImu::from_measurements`] does over the whole interval, in the same order and
/// with the same float operations (pinned by a test). Restarted whenever the keyframe, the world
/// or the linearization bias changes.
#[derive(Debug, Clone)]
pub(crate) struct IncrementalPreintegration {
    key: (usize, u64, [u64; 6]),
    t0: f64,
    core: PreintegratedImu,
    /// The last integrated sample: its time and its reading (the tail is extrapolated on it).
    last: Option<ImuMeasurement>,
    end: f64,
}

impl IncrementalPreintegration {
    /// The preintegration over `[t0, t1]` from `samples` (any superset of the interval's, stamp
    /// sorted), reusing `cache` when it was built for the same keyframe, world and bias.
    #[allow(clippy::too_many_arguments)]
    pub fn extend(
        cache: &mut Option<Self>,
        kf_idx: usize,
        generation: u64,
        bias: ImuBias,
        calib: ImuCalib,
        samples: impl Iterator<Item = ImuMeasurement>,
        t0: f64,
        t1: f64,
    ) -> PreintegratedImu {
        let bits = [
            bias.gyro.x.to_bits(),
            bias.gyro.y.to_bits(),
            bias.gyro.z.to_bits(),
            bias.accel.x.to_bits(),
            bias.accel.y.to_bits(),
            bias.accel.z.to_bits(),
        ];
        let key = (kf_idx, generation, bits);
        let reuse = cache
            .as_ref()
            .is_some_and(|c| c.key == key && c.t0.to_bits() == t0.to_bits() && c.end <= t1);
        if !reuse {
            *cache = Some(Self {
                key,
                t0,
                core: PreintegratedImu::new(bias, calib),
                last: None,
                end: t0,
            });
        }
        let c = cache.as_mut().expect("set above");
        for m in samples {
            if m.timestamp > t1 {
                break;
            }
            if m.timestamp < t0 {
                continue;
            }
            let dt = m.timestamp - c.end;
            if dt > 0.0 {
                c.core.integrate(&m, dt);
                c.end = m.timestamp;
                c.last = Some(m);
            } else if c.last.is_none() {
                // A sample exactly at `t0`: not integrated, but the tail's reading if it is the
                // only one, as in `from_measurements`.
                c.last = Some(m);
            }
        }
        let mut out = c.core.clone();
        if c.end < t1
            && let Some(last) = c.last
        {
            out.integrate(&last, t1 - c.end);
        }
        out
    }
}

/// The Mahalanobis distance between the wheels' and the gyro's camera rotation over the same
/// interval (`R_c0c1` each, with their 3x3 camera-frame covariances), and whether it passes the
/// 3-DOF gate. The predictors' own vote, independent of vision: a wheel rotation the gyro does
/// not share is slip or the head turning on its mast, and either way the wheel factor would pull
/// the pose through a mount the interval broke.
pub(crate) fn wheels_agree_with_gyro(
    wheels: &Mat3F64,
    wheels_cov: &Mat3F64,
    gyro: &Mat3F64,
    gyro_cov: &Mat3F64,
    chi2_gate: f64,
) -> bool {
    let e = SO3F64::from_matrix(&(wheels.transpose() * *gyro)).log();
    let s = *wheels_cov + *gyro_cov;
    if s.determinant().abs() < 1e-300 {
        return true;
    }
    let d2 = e.dot(s.inverse() * e);
    d2 <= gate_for_dof(chi2_gate, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> PinholeCamera {
        PinholeCamera {
            fx: 300.0,
            fy: 300.0,
            cx: 160.0,
            cy: 120.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// A grid of landmarks 1-3 m ahead, observed exactly from `pose`.
    fn observations(pose: &Pose3d) -> Vec<VisualObs> {
        let cam = camera();
        let mut out = Vec::new();
        for i in 0..10 {
            for j in 0..8 {
                let pc = Vec3F64::new(
                    -0.8 + 0.18 * f64::from(i),
                    -0.5 + 0.14 * f64::from(j),
                    1.0 + 0.25 * f64::from((i + j) % 9),
                );
                let pw = pose.inverse().transform_point(&pc);
                out.push(VisualObs {
                    point_world: pw,
                    pixel: [cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy],
                    inv_sigma: 1.0,
                });
            }
        }
        out
    }

    fn wheel_factor_at(pose: &Pose3d, sigma_m: f64, sigma_rad: f64) -> MotionFactor {
        // Base == camera, keyframe at identity, a zero-length interval measured at `pose`.
        let t_cb = Pose3d::IDENTITY;
        let m = WheelOdomMeasurement {
            delta: [0.0; 3],
            covariance: [
                [sigma_m * sigma_m, 0.0, 0.0],
                [0.0, sigma_m * sigma_m, 0.0],
                [0.0, 0.0, sigma_rad * sigma_rad],
            ],
            z_sigma: sigma_m,
            tilt_sigma: sigma_rad,
        };
        // The keyframe placed so that the measurement's prediction is `pose`.
        MotionFactor::wheel(&m, t_cb, t_cb, pose).expect("positive definite")
    }

    fn centre(p: &Pose3d) -> Vec3F64 {
        p.inverse().translation
    }

    #[test]
    fn test_the_gate_scales_to_the_dof_it_is_given() {
        assert!((gate_for_dof(DEFAULT_CHI2_GATE, 6) - DEFAULT_CHI2_GATE).abs() < 1e-9);
        // Exact 99.9 % quantiles: 16.266 (3), 32.909 (12).
        assert!((gate_for_dof(DEFAULT_CHI2_GATE, 3) - 16.266).abs() < 0.15);
        assert!((gate_for_dof(DEFAULT_CHI2_GATE, 12) - 32.909).abs() < 0.15);
    }

    /// The analytic reprojection Jacobian against central differences of the residual.
    #[test]
    fn test_the_reprojection_jacobian_matches_central_differences() {
        let cam = camera();
        let truth = se3_exp(&[0.1, -0.05, 0.2, 0.05, -0.1, 0.02]);
        let obs = observations(&truth);
        let pose = se3_exp(&[0.003, 0.001, -0.002, 0.001, 0.002, -0.001]).compose(&truth);
        let (h, b, _) = visual_normal_equations(&cam, &obs, &pose);
        // b = J^T r and the cost gradient is 2 b (all inliers): compare with a numeric gradient.
        for k in 0..6 {
            let mut xi = [0.0; 6];
            xi[k] = 1e-7;
            let cp = visual_normal_equations(&cam, &obs, &se3_exp(&xi).compose(&pose)).2;
            xi[k] = -1e-7;
            let cm = visual_normal_equations(&cam, &obs, &se3_exp(&xi).compose(&pose)).2;
            let num = (cp - cm) / 2e-7;
            assert!(
                (num - 2.0 * b[k]).abs() < 1e-3 * (1.0 + num.abs()),
                "{k}: {num} vs {}",
                2.0 * b[k]
            );
        }
        assert!(h[0][0] > 0.0);
        // The step test's cost-only path is the same cost, Huber branch included.
        let far = se3_exp(&[0.05, 0.0, 0.0, 0.0, 0.02, 0.0]).compose(&truth);
        for p in [pose, far] {
            let (a, b) = (
                visual_cost(&cam, &obs, &p),
                visual_normal_equations(&cam, &obs, &p).2,
            );
            assert!((a - b).abs() <= 1e-9 * b.max(1.0), "{a} vs {b}");
        }
    }

    /// Agreement: vision and a wheel factor that both say `truth` fuse at `truth`, and the gate
    /// passes.
    #[test]
    fn test_agreeing_vision_and_wheels_fuse_at_the_truth() {
        let cam = camera();
        let truth = se3_exp(&[0.02, 0.0, 0.1, 0.0, 0.05, 0.0]);
        let obs = observations(&truth);
        let f = wheel_factor_at(&truth, 0.01, 0.01);
        let start = se3_exp(&[0.004, -0.002, 0.003, 0.002, 0.0, -0.001]).compose(&truth);
        let out = fuse(
            &cam,
            &obs,
            std::slice::from_ref(&f),
            &start,
            &start,
            DEFAULT_CHI2_GATE,
        )
        .expect("fused");
        assert!(!out.rejected, "chi2 {}", out.chi2);
        assert!((centre(&out.pose) - centre(&truth)).length() < 1e-6);
    }

    /// A vision pose 5 cm off a stiff wheel prediction is rejected, and the prediction published.
    #[test]
    fn test_vision_far_from_a_stiff_prediction_is_rejected() {
        let cam = camera();
        let truth = Pose3d::IDENTITY;
        let jumped = se3_exp(&[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let obs = observations(&jumped);
        let f = wheel_factor_at(&truth, 0.001, 0.001);
        let out = fuse(
            &cam,
            &obs,
            std::slice::from_ref(&f),
            &jumped,
            &jumped,
            DEFAULT_CHI2_GATE,
        )
        .expect("ran");
        assert!(out.rejected, "chi2 {}", out.chi2);
        assert!(centre(&out.pose).length() < 1e-6, "{:?}", centre(&out.pose));
    }

    /// Weighting: a loose prediction barely moves good vision; a stiff one pulls weak vision.
    #[test]
    fn test_the_weights_are_the_inverse_covariances() {
        let cam = camera();
        let truth = Pose3d::IDENTITY;
        let off = se3_exp(&[0.004, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let obs = observations(&off);
        let loose = wheel_factor_at(&truth, 0.05, 0.05);
        let out = fuse(&cam, &obs, std::slice::from_ref(&loose), &off, &off, 1e9).unwrap();
        let moved_loose = (centre(&out.pose) - centre(&off)).length();
        let stiff = wheel_factor_at(&truth, 0.0005, 0.0005);
        let out = fuse(&cam, &obs, std::slice::from_ref(&stiff), &off, &off, 1e9).unwrap();
        let moved_stiff = (centre(&out.pose) - centre(&off)).length();
        assert!(moved_loose < 2e-4, "{moved_loose}");
        assert!(moved_stiff > 1e-3, "{moved_stiff}");
    }

    #[test]
    fn test_wheels_that_turn_without_the_gyro_disagree() {
        let cov = Mat3F64::IDENTITY * 1e-6;
        let still = Mat3F64::IDENTITY;
        let turned = SO3F64::exp(Vec3F64::new(0.0, 0.02, 0.0)).matrix();
        assert!(wheels_agree_with_gyro(
            &still,
            &cov,
            &still,
            &cov,
            DEFAULT_CHI2_GATE
        ));
        assert!(!wheels_agree_with_gyro(
            &turned,
            &cov,
            &still,
            &cov,
            DEFAULT_CHI2_GATE
        ));
        // ...unless the two are loose enough to cover it.
        let loose = Mat3F64::IDENTITY * 1e-3;
        assert!(wheels_agree_with_gyro(
            &turned,
            &loose,
            &still,
            &loose,
            DEFAULT_CHI2_GATE
        ));
    }

    /// Frame by frame, the incremental preintegration is `from_measurements` over the whole
    /// interval, bit for bit; a new keyframe restarts it.
    #[test]
    fn test_the_incremental_preintegration_is_the_full_one() {
        let calib = ImuCalib {
            gyro_noise: 1.7e-4,
            accel_noise: 2e-3,
            gyro_bias_noise: 2e-5,
            accel_bias_noise: 3e-3,
        };
        let bias = ImuBias {
            gyro: Vec3F64::new(0.003, -0.002, 0.001),
            accel: Vec3F64::new(0.01, 0.0, -0.02),
        };
        let samples: Vec<ImuMeasurement> = (0..400)
            .map(|k| {
                let t = 0.0123 + 0.005 * f64::from(k);
                ImuMeasurement {
                    timestamp: t,
                    gyro: Vec3F64::new((3.0 * t).sin(), 0.2, (1.3 * t).cos()) * 0.3,
                    accel: Vec3F64::new(0.1 * t, 9.81, (2.0 * t).sin()),
                }
            })
            .collect();
        let bits = |p: &PreintegratedImu| {
            let mut v: Vec<u64> = p.covariance.iter().map(|x| x.to_bits()).collect();
            v.extend(p.delta_rotation.to_cols_array().iter().map(|x| x.to_bits()));
            v.extend(
                [p.delta_position, p.delta_velocity]
                    .iter()
                    .flat_map(|d| [d.x, d.y, d.z])
                    .map(f64::to_bits),
            );
            v.push(p.dt.to_bits());
            v
        };
        let mut cache = None;
        let t0 = 0.1;
        for frame in 1..20 {
            let t1 = t0 + 0.0667 * f64::from(frame);
            let full = PreintegratedImu::from_measurements(bias, calib, &samples, t0, t1);
            let inc = IncrementalPreintegration::extend(
                &mut cache,
                7,
                0,
                bias,
                calib,
                samples.iter().copied(),
                t0,
                t1,
            );
            assert_eq!(bits(&inc), bits(&full), "frame {frame}");
        }
        // A new keyframe: the old integration must not leak into the new interval.
        let t0 = 1.0;
        let t1 = 1.2;
        let full = PreintegratedImu::from_measurements(bias, calib, &samples, t0, t1);
        let inc = IncrementalPreintegration::extend(
            &mut cache,
            8,
            0,
            bias,
            calib,
            samples.iter().copied(),
            t0,
            t1,
        );
        assert_eq!(bits(&inc), bits(&full));
    }

    #[test]
    fn test_se3_log_inverts_exp() {
        let xi = [0.3, -0.2, 0.1, 0.4, -0.3, 0.2];
        let back = se3_log(&se3_exp(&xi));
        for k in 0..6 {
            assert!((back[k] - xi[k]).abs() < 1e-12);
        }
    }
}
