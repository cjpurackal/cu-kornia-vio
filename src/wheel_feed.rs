//! `WheelFeed` and `MountFeed`: the inline sinks that move wheel odometry and camera mounts from
//! the graph onto a [`VioBus`](crate::VioBus), as [`ImuFeed`](crate::ImuFeed) does for the IMU.

use std::sync::Arc;

use cu_stereo_payloads::{CameraMount, WheelOdom};
use cu29::prelude::*;

use crate::pose::pose3d_from;
use crate::wheel::WheelSample;
use crate::wheel_channel::WheelQueue;

/// Resource bindings of [`WheelFeed`] and [`MountFeed`].
#[doc(hidden)]
#[allow(missing_docs)]
pub mod wheel_feed_resources {
    use super::WheelQueue;
    use cu29::resources;

    resources!({
        wheel => Shared<WheelQueue>,
    });
}

/// The capture time of a message, or `None` for an untimed one: a reading with no stamp cannot
/// be placed against a keyframe interval, so it is dropped rather than guessed.
fn stamp_ns(tov: Tov) -> Option<u64> {
    match tov {
        Tov::Time(t) => Some(t.as_nanos()),
        Tov::Range(r) => Some(r.end.as_nanos()),
        Tov::None => None,
    }
}

/// Pushes every [`WheelOdom`] reading it receives into the bus's [`WheelQueue`], stamped with
/// the message's `tov`.
///
/// ```ron
/// ( id: "wheel_feed", type: "cu_kornia_vio::WheelFeed", resources: { "wheel": "bus.wheel" } ),
/// ```
#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct WheelFeed {
    #[reflect(ignore)]
    wheel: Arc<WheelQueue>,
    /// Readings dropped for carrying no `tov`.
    untimed: u64,
}

// No state of its own: the queue is a bus resource.
impl Freezable for WheelFeed {}

impl CuSinkTask for WheelFeed {
    type Resources<'r> = wheel_feed_resources::Resources;
    type Input<'m> = input_msg!(WheelOdom);

    fn new(config: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        crate::config::deny_unknown_keys(config, "cu_kornia_vio::WheelFeed", &[])?;
        Ok(Self {
            wheel: resources.wheel,
            untimed: 0,
        })
    }

    fn process<'i>(&mut self, _ctx: &CuContext, input: &Self::Input<'i>) -> CuResult<()> {
        let Some(odom) = input.payload() else {
            return Ok(());
        };
        let Some(stamp_ns) = stamp_ns(input.tov) else {
            self.untimed += 1;
            // Once, at the first: a source that never stamps its tov leaves the wheels off for
            // good, and the count in `stop` is the only other trace.
            if self.untimed == 1 {
                warning!(
                    "wheel_feed: a wheel reading carries no tov and is dropped; the tracker \
                     gets no wheel odometry until the source stamps its messages"
                );
            }
            return Ok(());
        };
        self.wheel.push_odom(WheelSample {
            stamp_ns,
            x: odom.x,
            y: odom.y,
            theta: odom.theta,
        });
        Ok(())
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        if self.untimed > 0 {
            warning!(
                "wheel_feed: {} readings dropped for carrying no tov",
                self.untimed
            );
        }
        Ok(())
    }
}

/// Pushes every [`CameraMount`] it receives into the bus's [`WheelQueue`], stamped with the
/// message's `tov`. Only needed for a camera whose mount moves (a tilting head); a fixed mount
/// is the `wheel_odom` config's `cam_in_base_*`.
#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct MountFeed {
    #[reflect(ignore)]
    wheel: Arc<WheelQueue>,
    untimed: u64,
}

impl Freezable for MountFeed {}

impl CuSinkTask for MountFeed {
    type Resources<'r> = wheel_feed_resources::Resources;
    type Input<'m> = input_msg!(CameraMount);

    fn new(config: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        crate::config::deny_unknown_keys(config, "cu_kornia_vio::MountFeed", &[])?;
        Ok(Self {
            wheel: resources.wheel,
            untimed: 0,
        })
    }

    fn process<'i>(&mut self, _ctx: &CuContext, input: &Self::Input<'i>) -> CuResult<()> {
        let Some(mount) = input.payload() else {
            return Ok(());
        };
        let Some(stamp_ns) = stamp_ns(input.tov) else {
            self.untimed += 1;
            if self.untimed == 1 {
                warning!(
                    "mount_feed: a camera mount carries no tov and is dropped; the tracker \
                     keeps the configured mount until the source stamps its messages"
                );
            }
            return Ok(());
        };
        // Camera-in-base on the wire; the factor takes base-to-camera. Inverted once, here.
        self.wheel
            .push_mount(stamp_ns, pose3d_from(&mount.cam_in_base).inverse());
        Ok(())
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        if self.untimed > 0 {
            warning!(
                "mount_feed: {} mounts dropped for carrying no tov",
                self.untimed
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu_spatial_payloads::Pose;
    use cu29::clock::CuTime;

    fn queue() -> Arc<WheelQueue> {
        let q = Arc::new(WheelQueue::new());
        q.arm();
        q
    }

    fn res(q: &Arc<WheelQueue>) -> wheel_feed_resources::Resources {
        wheel_feed_resources::Resources {
            wheel: Arc::clone(q),
        }
    }

    #[test]
    fn test_wheel_feed_stamps_readings_with_the_message_tov_and_drops_untimed_ones() {
        let q = queue();
        let mut task = WheelFeed::new(None, res(&q)).unwrap();
        let ctx = CuContext::new_with_clock();
        let mut msg = CuMsg::new(Some(WheelOdom {
            x: 1.0,
            y: 2.0,
            theta: 0.3,
            covariance: [0.0; 9],
        }));
        msg.tov = Tov::Time(CuTime::from_nanos(42));
        task.process(&ctx, &msg).unwrap();
        msg.tov = Tov::None;
        task.process(&ctx, &msg).unwrap();
        let got = q.drain_with(|o, _, _| o.collect::<Vec<_>>()).unwrap();
        assert_eq!(
            got,
            [WheelSample {
                stamp_ns: 42,
                x: 1.0,
                y: 2.0,
                theta: 0.3
            }]
        );
        assert_eq!(task.untimed, 1);
    }

    #[test]
    fn test_mount_feed_inverts_camera_in_base_into_base_to_camera() {
        let q = queue();
        let mut task = MountFeed::new(None, res(&q)).unwrap();
        let ctx = CuContext::new_with_clock();
        let mut msg = CuMsg::new(Some(CameraMount {
            cam_in_base: Pose::from_matrix([
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.9],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        }));
        msg.tov = Tov::Time(CuTime::from_nanos(7));
        task.process(&ctx, &msg).unwrap();
        let got = q.drain_with(|_, m, _| m.collect::<Vec<_>>()).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 7);
        // X_cam = t_cb X_base: the base origin sits at -(0.1, 0, 0.9) in the camera.
        let origin = got[0].1.transform_point(&kornia_algebra::Vec3F64::ZERO);
        assert!((origin - kornia_algebra::Vec3F64::new(-0.1, 0.0, -0.9)).length() < 1e-12);
    }

    #[test]
    fn test_the_feeds_refuse_config_keys() {
        let mut cfg = ComponentConfig::new();
        cfg.set("rate", 4u32);
        assert!(WheelFeed::new(Some(&cfg), res(&queue())).is_err());
        assert!(MountFeed::new(Some(&cfg), res(&queue())).is_err());
    }
}
