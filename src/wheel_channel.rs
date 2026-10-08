//! The queue that gets wheel odometry and camera mounts into a BACKGROUNDED tracker.
//!
//! The same seam as [`crate::imu_channel`], for the same reason: `StereoVio` keeps exactly one
//! input edge so it can be backgrounded, and that edge refuses the input that arrives while a
//! solve is running, so nothing that must not be decimated may ride it. Inline sinks
//! ([`WheelFeed`](crate::WheelFeed), [`MountFeed`](crate::MountFeed)) push here and the tracker
//! drains on the frame it is given. Unarmed until a `StereoVio` with `wheel_odom` enabled exists,
//! so a graph that carries the edges without the feature pays one atomic load per message.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use kornia_3d::pose::Pose3d;

use crate::wheel::WheelSample;

/// Readings held between drains. At 100 Hz that is 10 s of a stalled consumer; the tracker's
/// own ring holds the interval it needs.
const CAPACITY: usize = 1024;
/// Mount samples held between drains.
const MOUNT_CAPACITY: usize = 64;

#[derive(Debug, Default)]
struct Pending {
    odom: VecDeque<WheelSample>,
    /// `(stamp_ns, t_cb)`: base-to-camera, already inverted from the payload's camera-in-base.
    mounts: VecDeque<(u64, Pose3d)>,
    evicted: u64,
}

/// A bounded queue of wheel readings and camera mounts, one per [`VioBus`](crate::VioBus).
#[derive(Debug, Default)]
pub struct WheelQueue {
    pending: Mutex<Pending>,
    armed: AtomicBool,
}

impl WheelQueue {
    /// An empty, unarmed queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a consumer; called by `StereoVio::new` when `wheel_odom` is enabled.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// Whether a consumer has armed the queue.
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// Queues one wheel reading. Call from an INLINE task. A poisoned lock drops the reading;
    /// the tracker refuses an interval with a hole in it rather than bridging it.
    pub fn push_odom(&self, sample: WheelSample) {
        if !self.is_armed() {
            return;
        }
        let Ok(mut p) = self.pending.lock() else {
            return;
        };
        p.odom.push_back(sample);
        while p.odom.len() > CAPACITY {
            p.odom.pop_front();
            p.evicted += 1;
        }
    }

    /// Queues the base-to-camera transform valid from `stamp_ns`.
    pub fn push_mount(&self, stamp_ns: u64, t_cb: Pose3d) {
        if !self.is_armed() {
            return;
        }
        let Ok(mut p) = self.pending.lock() else {
            return;
        };
        p.mounts.push_back((stamp_ns, t_cb));
        while p.mounts.len() > MOUNT_CAPACITY {
            p.mounts.pop_front();
        }
    }

    /// Runs `f` under the lock over everything queued, emptying it. `f` gets the readings, the
    /// mounts and the cumulative count of readings this queue evicted.
    pub(crate) fn drain_with<R>(
        &self,
        f: impl FnOnce(
            std::collections::vec_deque::Drain<'_, WheelSample>,
            std::collections::vec_deque::Drain<'_, (u64, Pose3d)>,
            u64,
        ) -> R,
    ) -> Option<R> {
        let Ok(mut p) = self.pending.lock() else {
            return None;
        };
        let p = &mut *p;
        let evicted = p.evicted;
        Some(f(p.odom.drain(..), p.mounts.drain(..), evicted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(t: u64) -> WheelSample {
        WheelSample {
            stamp_ns: t,
            x: t as f64,
            y: 0.0,
            theta: 0.0,
        }
    }

    #[test]
    fn test_an_unarmed_queue_does_not_buffer() {
        let q = WheelQueue::new();
        q.push_odom(s(1));
        q.push_mount(1, Pose3d::IDENTITY);
        assert_eq!(q.drain_with(|o, m, _| (o.count(), m.count())), Some((0, 0)));
    }

    #[test]
    fn test_push_then_drain_round_trips_and_overflow_is_counted() {
        let q = WheelQueue::new();
        q.arm();
        for t in 0..(CAPACITY as u64 + 5) {
            q.push_odom(s(t));
        }
        q.push_mount(7, Pose3d::IDENTITY);
        let (first, n, mounts, evicted) = q
            .drain_with(|o, m, e| {
                let v: Vec<_> = o.collect();
                (v[0].stamp_ns, v.len(), m.count(), e)
            })
            .unwrap();
        assert_eq!((first, n, mounts, evicted), (5, CAPACITY, 1, 5));
        assert_eq!(
            q.drain_with(|o, _, _| o.count()),
            Some(0),
            "drained means drained"
        );
    }
}
