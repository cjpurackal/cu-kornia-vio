//! Pose-convention conversions.

use cu_spatial_payloads::Pose;
use kornia_3d::pose::Pose3d;

/// Converts the tracker's pose into the camera-in-world transform a consumer wants.
///
/// **The tracker's convention is world-to-camera; a transform wants camera-in-world.**
/// Publishing the un-inverted pose is the classic frustum-flies-backwards bug, and it does not
/// look like a bug: it looks like the camera driving the path in reverse.
pub fn cam_in_world(pose_world_to_cam: &Pose3d) -> Pose<f64> {
    let inv = pose_world_to_cam.inverse();
    // `Mat3F64` is column-major; `Pose::from_matrix` takes `mat[row][column]`.
    let cols = [
        inv.rotation.x_axis().to_array(),
        inv.rotation.y_axis().to_array(),
        inv.rotation.z_axis().to_array(),
    ];
    let t = inv.translation.to_array();
    let row = |r: usize| [cols[0][r], cols[1][r], cols[2][r], t[r]];
    Pose::from_matrix([row(0), row(1), row(2), [0.0, 0.0, 0.0, 1.0]])
}

/// The inverse direction: a [`Pose`] read as the `Pose3d` mapping its child frame into its
/// parent (`X_parent = R X_child + t`), with no inversion.
pub fn pose3d_from(pose: &Pose<f64>) -> Pose3d {
    let m = pose.to_matrix();
    Pose3d::new(
        crate::wheel::mat3_from_rows(&[
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ]),
        kornia_algebra::Vec3F64::new(m[0][3], m[1][3], m[2][3]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    /// A camera at (1, 2, 3) yawed +90 deg about world Z. The inverse must put the translation
    /// back at the camera centre and transpose the rotation, row by row.
    #[test]
    fn test_cam_in_world_inverts_and_keeps_row_major_order() {
        // world-to-camera rotation: yaw -90 deg about Z (columns listed).
        let r_wc = Mat3F64::from_cols(
            Vec3F64::new(0.0, -1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let centre = Vec3F64::new(1.0, 2.0, 3.0);
        let t_wc = -(r_wc * centre);
        let m = cam_in_world(&Pose3d::new(r_wc, t_wc)).to_matrix();
        let eps = 1e-12;
        let want = [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for (r, (got, want)) in m.iter().zip(want).enumerate() {
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() < eps, "[{r}][{c}]: {m:?}");
            }
        }
    }

    /// `pose3d_from` reads the matrix as it stands: it undoes `cam_in_world`'s inversion only
    /// when inverted again, and keeps the row-major layout.
    #[test]
    fn test_pose3d_from_round_trips_cam_in_world() {
        let r = Mat3F64::from_cols(
            Vec3F64::new(0.0, -1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let world_to_cam = Pose3d::new(r, Vec3F64::new(1.0, 2.0, 3.0));
        let back = pose3d_from(&cam_in_world(&world_to_cam)).inverse();
        assert!((back.translation - world_to_cam.translation).length() < 1e-12);
        let d = back.rotation - world_to_cam.rotation;
        assert!(d.to_cols_array().iter().all(|v| v.abs() < 1e-12));
    }
}
