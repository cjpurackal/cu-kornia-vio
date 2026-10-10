//! The published pose's covariance, and the algebra that moves a 6x6 covariance between frames.
//!
//! Every covariance in this module is 6x6, row-major, over the tangent `[rho; omega]`
//! (translation first, kornia-slam's order). Which perturbation it describes is part of its
//! meaning, and three are in play:
//!
//! - **camera tangent**: `T_cw <- Exp(xi) T_cw`, the perturbation the per-frame fusion's
//!   Jacobians use ([`crate::fusion`]), so its Hessian inverts to a covariance here;
//! - **world left perturbation of `T_wc`**: `T_wc <- Exp(delta) T_wc`, what the tracker chains
//!   from keyframe to keyframe and what [`cu_stereo_payloads::VioPose::covariance`] carries. It
//!   composes with a change of world by the adjoint alone ([`congruence`] with [`adjoint`]), which
//!   is why it is the one that crosses a frame boundary;
//! - **ROS**: position error in world axes, then a small rotation about the world axes, which is
//!   what `nav_msgs/Odometry`'s `pose.covariance` means ([`ros_from_left`]).
//!
//! `T_cw <- Exp(xi) T_cw` is `T_wc <- T_wc Exp(-xi) = Exp(-Ad(T_wc) xi) T_wc`, so the camera
//! tangent maps to the world left perturbation by `Ad(T_wc)` (the sign does not reach a
//! covariance). A left perturbation moves the camera centre by `rho + omega x p`, not by `rho`:
//! far from the world origin the rotation uncertainty is most of the position uncertainty, so a
//! consumer that read `rho` as the position error would be wrong by metres once the robot is any
//! distance from where the world started.

use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};

/// A 6x6 covariance, row-major, over `[rho; omega]`.
pub type Cov6 = [f64; 36];

/// The variance a direction nothing measured is given: ROS's conventional "unknown" (1e6), not
/// zero, which every consumer reads as "perfectly known".
pub const UNKNOWN_VARIANCE: f64 = 1e6;

/// "No covariance": ROS's invalid marker, `-1` in element 0 and zeros elsewhere. A frame the
/// tracker has no covariance for carries this rather than a zero matrix.
pub const NO_COVARIANCE: Cov6 = {
    let mut c = [0.0; 36];
    c[0] = -1.0;
    c
};

/// Whether `c` is a covariance at all: not [`NO_COVARIANCE`] (a negative element 0) and finite.
pub fn is_valid(c: &Cov6) -> bool {
    c[0] >= 0.0 && c.iter().all(|v| v.is_finite())
}

/// `Ad(T)` of `T = (R, t)`, 6x6 row-major over `[rho; omega]`: `[[R, [t]x R], [0, R]]`. `r` is
/// row-major (`r[row][col]`).
pub fn adjoint(r: [[f64; 3]; 3], t: [f64; 3]) -> [f64; 36] {
    let tx = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
    let mut a = [0.0; 36];
    for i in 0..3 {
        for j in 0..3 {
            a[i * 6 + j] = r[i][j];
            a[(i + 3) * 6 + j + 3] = r[i][j];
            a[i * 6 + j + 3] = (0..3).map(|k| tx[i][k] * r[k][j]).sum();
        }
    }
    a
}

/// `A S A^T`, symmetrised: the product is symmetric only to rounding, and a consumer may check.
pub fn congruence(a: &[f64; 36], s: &Cov6) -> Cov6 {
    let mut as_ = [0.0; 36];
    for i in 0..6 {
        for j in 0..6 {
            as_[i * 6 + j] = (0..6).map(|k| a[i * 6 + k] * s[k * 6 + j]).sum();
        }
    }
    let mut out = [0.0; 36];
    for i in 0..6 {
        for j in i..6 {
            let v: f64 = (0..6).map(|k| as_[i * 6 + k] * a[j * 6 + k]).sum();
            out[i * 6 + j] = v;
            out[j * 6 + i] = v;
        }
    }
    out
}

/// `a + b`, element by element.
pub fn add(a: &Cov6, b: &Cov6) -> Cov6 {
    std::array::from_fn(|k| a[k] + b[k])
}

/// A world left perturbation of a pose whose position is `p` (world frame), in ROS's convention:
/// `G S G^T` with `G = [[I, -[p]x], [0, I]]`, since the left perturbation moves the position by
/// `rho + omega x p = rho - [p]x omega` and the rotation by `omega` about the world axes.
pub fn ros_from_left(s: &Cov6, p: [f64; 3]) -> Cov6 {
    let mut g = [0.0; 36];
    for k in 0..6 {
        g[k * 6 + k] = 1.0;
    }
    // -[p]x in the upper-right block.
    let neg_px = [[0.0, p[2], -p[1]], [-p[2], 0.0, p[0]], [p[1], -p[0], 0.0]];
    for i in 0..3 {
        for j in 0..3 {
            g[i * 6 + j + 3] = neg_px[i][j];
        }
    }
    congruence(&g, s)
}

/// The horizontal position sigma (metres, both horizontal axes together: the square root of the
/// position variance off the `up` axis) and the sigma of the rotation about `up` (radians), of a
/// covariance in ROS's convention ([`ros_from_left`]). `up` need not be normalised.
pub fn horizontal_sigmas(ros: &Cov6, up: [f64; 3]) -> (f64, f64) {
    let n = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
    let u = if n > 0.0 {
        [up[0] / n, up[1] / n, up[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    };
    let quad = |off: usize| -> f64 {
        (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| u[i] * ros[(off + i) * 6 + off + j] * u[j])
                    .sum::<f64>()
            })
            .sum()
    };
    let trace_p = ros[0] + ros[7] + ros[14];
    ((trace_p - quad(0)).max(0.0).sqrt(), quad(3).max(0.0).sqrt())
}

/// A row-major 3x3 of a kornia rotation matrix (column-major storage).
pub(crate) fn rows_of(m: &Mat3F64) -> [[f64; 3]; 3] {
    let c = m.to_cols_array();
    std::array::from_fn(|i| std::array::from_fn(|j| c[j * 3 + i]))
}

/// `Ad(T)` of a pose.
pub(crate) fn pose_adjoint(t: &Pose3d) -> [f64; 36] {
    let v: Vec3F64 = t.translation;
    adjoint(rows_of(&t.rotation), [v.x, v.y, v.z])
}

/// A camera-tangent covariance of `pose_cw` as the world left perturbation of its `T_wc`:
/// `Ad(T_wc) S Ad(T_wc)^T` (module doc).
pub(crate) fn world_from_camera_tangent(s: &Cov6, pose_cw: &Pose3d) -> Cov6 {
    congruence(&pose_adjoint(&pose_cw.inverse()), s)
}

/// A covariance as rows, the shape [`cu_stereo_payloads::VioPose::covariance`] carries.
pub fn to_rows(c: &Cov6) -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| c[i * 6 + j]))
}

/// The inverse of [`to_rows`].
pub fn from_rows(r: &[[f64; 6]; 6]) -> Cov6 {
    std::array::from_fn(|k| r[k / 6][k % 6])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::se3_exp;
    use kornia_algebra::SO3F64;

    /// A deterministic normal sampler (an LCG through Box-Muller): no `rand` in this crate.
    struct Normal(u64);

    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn sample(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// The covariance this module predicts for a camera-tangent covariance, in ROS's convention,
    /// against the sample covariance of the perturbed pose's position and world-axis rotation,
    /// with the camera FAR from the world origin (5, 3, 1) and turned, so that both the adjoint's
    /// `[t]x R` block and `G`'s `-[p]x` block carry most of the answer.
    #[test]
    fn left_world_covariance_matches_monte_carlo_far_from_origin() {
        let pose_wc = Pose3d::new(
            SO3F64::exp(Vec3F64::new(0.3, -0.6, 1.1)).matrix(),
            Vec3F64::new(5.0, 3.0, 1.0),
        );
        let pose_cw = pose_wc.inverse();
        let sigmas = [0.01, 0.02, 0.005, 0.004, 0.002, 0.008];
        let mut s_cw = [0.0; 36];
        for k in 0..6 {
            s_cw[k * 6 + k] = sigmas[k] * sigmas[k];
        }
        let predicted = ros_from_left(&world_from_camera_tangent(&s_cw, &pose_cw), [5.0, 3.0, 1.0]);

        let mut rng = Normal(7);
        let n = 40_000;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let xi: [f64; 6] = std::array::from_fn(|k| sigmas[k] * rng.sample());
            let wc = se3_exp(&xi).compose(&pose_cw).inverse();
            let dp = wc.translation - pose_wc.translation;
            let dw = SO3F64::from_matrix(&(wc.rotation * pose_wc.rotation.transpose())).log();
            samples.push([dp.x, dp.y, dp.z, dw.x, dw.y, dw.z]);
        }
        let mut empirical = [0.0; 36];
        for s in &samples {
            for i in 0..6 {
                for j in 0..6 {
                    empirical[i * 6 + j] += s[i] * s[j] / n as f64;
                }
            }
        }
        for i in 0..6 {
            for j in 0..6 {
                let (p, e) = (predicted[i * 6 + j], empirical[i * 6 + j]);
                let scale = (predicted[i * 6 + i] * predicted[j * 6 + j]).sqrt();
                assert!(
                    (p - e).abs() <= 0.05 * scale,
                    "[{i}][{j}]: predicted {p:e}, sampled {e:e} (scale {scale:e})"
                );
            }
        }
    }

    #[test]
    fn horizontal_sigmas_leave_out_the_up_axis() {
        let mut c = [0.0; 36];
        // x 3 cm, y 4 cm, z 1 m; yaw 0.1 rad, roll and pitch 1 rad.
        for (k, v) in [0.03f64, 0.04, 1.0, 1.0, 1.0, 0.1].iter().enumerate() {
            c[k * 6 + k] = v * v;
        }
        let (xy, yaw) = horizontal_sigmas(&c, [0.0, 0.0, 2.0]);
        assert!((xy - 0.05).abs() < 1e-12, "{xy}");
        assert!((yaw - 0.1).abs() < 1e-12, "{yaw}");
    }

    #[test]
    fn the_rows_round_trip_and_no_covariance_is_invalid() {
        let c: Cov6 = std::array::from_fn(|k| k as f64);
        assert_eq!(from_rows(&to_rows(&c)), c);
        assert_eq!(to_rows(&c)[2][3], 15.0);
        assert!(!is_valid(&NO_COVARIANCE));
        assert!(is_valid(&[0.0; 36]));
    }
}
