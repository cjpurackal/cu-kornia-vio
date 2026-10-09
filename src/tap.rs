//! `VioTap`: what the tracker saw and concluded, per processed frame, for whoever is watching.
//!
//! A backgrounded `StereoVio` has ONE output (`CuAsyncTask` needs `Output = CuMsg<O>`), and that
//! output is the pose, which an untracked frame leaves empty. So diagnostics leave the same way the
//! IMU and the wheels arrive: through a [`VioBus`](crate::VioBus) resource. The tracker writes the
//! latest frame's [`VioDiagnostics`], its stereo pair (a refcount on the camera's buffers, never a
//! copy) and its classified keypoints; the inline [`VioTapSource`] emits them into the graph.
//!
//! Latest wins: a record the source has not taken when the next frame lands is overwritten and
//! counted. The source polls every copperlist (5 ms on the head graph) against a ~10 Hz tracker, so
//! that count is a stalled main loop, not a design rate.
//!
//! Inert until a source arms it: an unarmed tap is one atomic load per frame and retains no camera
//! buffer, so a graph that never declares a source pays nothing.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cu_stereo_payloads::{FeaturePoint, StereoPair, VioDiagnostics, VioFrame};
use cu29::pool::{CuHandle, CuHostMemoryPool, CuPool};
use cu29::prelude::*;

/// Keypoints one [`VioFrame::features`] buffer holds; a frame with more is truncated. Twice the
/// largest `orb_keypoints` any graph here has run (2026-09-19 sweep: 200-400).
pub const MAX_TAP_FEATURES: usize = 2048;

/// Feature buffers [`VioTapSource`] allocates up front. Each is held by the copperlist slots that
/// carry it and by a consumer while it draws; 8 copperlists plus a background encoder fit in 12.
const FEATURE_BUFFERS: usize = 12;

/// Pool id reported in copper's pool statistics.
const FEATURE_POOL_ID: &str = "cu_kornia_vio.tap_features";

/// The latest frame, as the tracker left it.
#[derive(Default)]
struct Slot {
    diag: Option<(Tov, VioDiagnostics)>,
    frame: Option<StereoPair>,
    features: Vec<FeaturePoint>,
}

/// The shared slot between the tracker and [`VioTapSource`]. Provided by a `VioBus` as
/// `<bus>.tap`.
#[derive(Default)]
pub struct VioTap {
    armed: AtomicBool,
    /// Whether the tracker should retain the stereo pair, not just the diagnostics.
    frames: AtomicBool,
    /// Records written; a reader compares against the last value it took.
    published: AtomicU64,
    /// Records overwritten before anyone took them.
    overwritten: AtomicU64,
    slot: Mutex<Slot>,
}

impl VioTap {
    /// An unarmed tap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a consumer. `frames` also retains each processed stereo pair (and so holds one
    /// camera buffer pair until the next frame replaces it).
    pub fn arm(&self, frames: bool) {
        self.frames.store(frames, Ordering::Relaxed);
        self.armed.store(true, Ordering::Release);
    }

    /// Whether a consumer is declared.
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// Records written so far.
    pub fn published(&self) -> u64 {
        self.published.load(Ordering::Acquire)
    }

    /// Records overwritten before a consumer took them.
    pub fn overwritten(&self) -> u64 {
        self.overwritten.load(Ordering::Relaxed)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Slot> {
        // A panic while holding the lock leaves plain data behind; keep serving it.
        self.slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes one processed frame. A no-op while unarmed. `features` is copied into a buffer the
    /// slot keeps, so a steady state allocates nothing.
    pub fn publish(
        &self,
        tov: Tov,
        diag: VioDiagnostics,
        frame: &StereoPair,
        features: &[FeaturePoint],
    ) {
        if !self.is_armed() {
            return;
        }
        let keep_frame = self.frames.load(Ordering::Relaxed);
        let mut slot = self.lock();
        if slot.diag.is_some() {
            self.overwritten.fetch_add(1, Ordering::Relaxed);
        }
        slot.diag = Some((tov, diag));
        slot.frame = keep_frame.then(|| frame.clone());
        slot.features.clear();
        slot.features.extend_from_slice(features);
        drop(slot);
        self.published.fetch_add(1, Ordering::Release);
    }

    /// Takes the latest record, if any: `f` gets the diagnostics with their `tov`, the pair (when
    /// retained) and the keypoints. The slot is left empty.
    pub fn take<R>(
        &self,
        f: impl FnOnce(Option<(Tov, VioDiagnostics)>, Option<StereoPair>, &[FeaturePoint]) -> R,
    ) -> R {
        let mut slot = self.lock();
        let diag = slot.diag.take();
        let frame = slot.frame.take();
        f(diag, frame, &slot.features)
    }
}

/// Resource bindings of [`VioTapSource`].
#[doc(hidden)]
#[allow(missing_docs)]
pub mod tap_resources {
    use super::VioTap;
    use cu29::resources;

    resources!({
        tap => Shared<VioTap>,
    });
}

/// The RON key that turns off frame retention: `"frames": false` emits the diagnostics alone.
pub const FRAMES_KEY: &str = "frames";

/// Emits the tracker's latest [`VioDiagnostics`] and [`VioFrame`] from the bus's [`VioTap`].
///
/// An inline source: it polls the tap every copperlist and emits only when the tracker wrote a
/// new record, so both outputs are empty on every other tick (no send on a bridge, no dispatch of
/// a background consumer on a stale input). Both carry the frame's capture `tov`.
#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct VioTapSource {
    #[reflect(ignore)]
    tap: Arc<VioTap>,
    #[reflect(ignore)]
    pool: Arc<CuHostMemoryPool<Vec<FeaturePoint>>>,
    seen: u64,
    emitted: u64,
    /// Frames emitted without keypoints because every feature buffer was checked out.
    pool_exhausted: u64,
}

impl Freezable for VioTapSource {}

/// Copies `points` into a pooled buffer, truncating at [`MAX_TAP_FEATURES`]; `None` when every
/// buffer is checked out.
fn pooled_features(
    pool: &CuHostMemoryPool<Vec<FeaturePoint>>,
    points: &[FeaturePoint],
) -> Option<CuHandle<Vec<FeaturePoint>>> {
    let handle = pool.acquire()?;
    handle.with_inner_mut(|inner| {
        let buf: &mut Vec<FeaturePoint> = inner.as_mut();
        buf.clear();
        buf.extend_from_slice(&points[..points.len().min(MAX_TAP_FEATURES)]);
    });
    Some(handle)
}

impl CuSrcTask for VioTapSource {
    type Output<'m> = output_msg!(VioDiagnostics, VioFrame);
    type Resources<'r> = tap_resources::Resources;

    fn new(config: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        crate::config::deny_unknown_keys(config, "cu_kornia_vio::VioTapSource", &[FRAMES_KEY])?;
        let frames = match config {
            Some(c) => c.get::<bool>(FRAMES_KEY)?.unwrap_or(true),
            None => true,
        };
        let tap = resources.tap;
        tap.arm(frames);
        Ok(Self {
            seen: tap.published(),
            tap,
            pool: CuHostMemoryPool::new(FEATURE_POOL_ID, FEATURE_BUFFERS, || {
                vec![FeaturePoint::default(); MAX_TAP_FEATURES]
            })?,
            emitted: 0,
            pool_exhausted: 0,
        })
    }

    fn process<'o>(&mut self, _ctx: &CuContext, output: &mut Self::Output<'o>) -> CuResult<()> {
        let (diag_out, frame_out) = output;
        diag_out.clear_payload();
        diag_out.tov = Tov::None;
        frame_out.clear_payload();
        frame_out.tov = Tov::None;
        let published = self.tap.published();
        if published == self.seen {
            return Ok(());
        }
        self.seen = published;
        let pool = &self.pool;
        let (diag, frame) = self.tap.take(|diag, frame, points| {
            let frame = frame.map(|pair| {
                let features = pooled_features(pool, points);
                (pair, features)
            });
            (diag, frame)
        });
        let Some((tov, diag)) = diag else {
            return Ok(());
        };
        diag_out.tov = tov;
        diag_out.set_payload(diag);
        if let Some((pair, features)) = frame {
            if features.is_none() {
                self.pool_exhausted += 1;
            }
            frame_out.tov = tov;
            frame_out.set_payload(VioFrame { pair, features });
        }
        self.emitted += 1;
        Ok(())
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        info!(
            "vio_tap: {} records emitted, {} overwritten unread, {} frames without keypoints \
             (feature pool exhausted)",
            self.emitted,
            self.tap.overwritten(),
            self.pool_exhausted
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu_stereo_payloads::{RectifiedStereo, TrackingState};
    use cu29::clock::CuTime;

    fn pair() -> StereoPair {
        StereoPair::new(
            2,
            1,
            CuHandle::new_detached(vec![1, 2]),
            CuHandle::new_detached(vec![3, 4]),
            RectifiedStereo::default(),
        )
        .expect("pair")
    }

    fn source(tap: &Arc<VioTap>) -> VioTapSource {
        VioTapSource::new(
            None,
            tap_resources::Resources {
                tap: Arc::clone(tap),
            },
        )
        .expect("source")
    }

    type Out = (CuMsg<VioDiagnostics>, CuMsg<VioFrame>);

    fn tick(src: &mut VioTapSource) -> Out {
        let mut out: Out = (CuMsg::new(None), CuMsg::new(None));
        src.process(&CuContext::new_with_clock(), &mut out)
            .expect("process");
        out
    }

    #[test]
    fn test_an_unarmed_tap_retains_nothing() {
        let tap = VioTap::new();
        tap.publish(Tov::None, VioDiagnostics::default(), &pair(), &[]);
        assert_eq!(tap.published(), 0);
        tap.take(|d, f, _| assert!(d.is_none() && f.is_none()));
    }

    #[test]
    fn test_each_record_is_emitted_once_with_its_tov() {
        let tap = Arc::new(VioTap::new());
        let mut src = source(&tap);
        assert!(tick(&mut src).0.payload().is_none(), "nothing written yet");
        let tov = Tov::Time(CuTime::from_nanos(42));
        let diag = VioDiagnostics {
            seq: 9,
            state: TrackingState::Lost,
            ..Default::default()
        };
        let feats = [FeaturePoint {
            x: 1.0,
            y: 2.0,
            class: 3,
        }];
        tap.publish(tov, diag, &pair(), &feats);
        let (d, f) = tick(&mut src);
        assert_eq!(d.tov, tov);
        assert_eq!(d.payload().map(|d| d.seq), Some(9));
        assert_eq!(f.tov, tov);
        let frame = f.payload().expect("frame retained by default");
        let got = frame
            .features
            .as_ref()
            .expect("pooled")
            .with_inner(|v| v.to_vec());
        assert_eq!(got, feats.to_vec(), "only this frame's keypoints");
        let (d, f) = tick(&mut src);
        assert!(
            d.payload().is_none() && f.payload().is_none(),
            "an empty tick sends nothing, never the last record again"
        );
    }

    #[test]
    fn test_frames_off_emits_diagnostics_alone() {
        let tap = Arc::new(VioTap::new());
        // From RON: `ComponentConfig::set` takes no bool.
        let text = r#"( tasks: [ ( id: "t", type: "t::T", config: { "frames": false } ) ],
                        cnx: [] )"#;
        let all = cu29::config::read_configuration_str(text.to_string(), None).expect("ron");
        let graph = all.get_graph(None).expect("graph");
        let cfg = graph
            .get_node(graph.get_node_id_by_name("t").expect("node"))
            .expect("node")
            .get_instance_config()
            .cloned()
            .expect("config");
        let mut src = VioTapSource::new(
            Some(&cfg),
            tap_resources::Resources {
                tap: Arc::clone(&tap),
            },
        )
        .expect("source");
        tap.publish(Tov::None, VioDiagnostics::default(), &pair(), &[]);
        let (d, f) = tick(&mut src);
        assert!(d.payload().is_some());
        assert!(f.payload().is_none(), "no camera buffer held");
    }

    #[test]
    fn test_an_unread_record_is_overwritten_and_counted() {
        let tap = Arc::new(VioTap::new());
        let mut src = source(&tap);
        for seq in 1..=3 {
            let d = VioDiagnostics {
                seq,
                ..Default::default()
            };
            tap.publish(Tov::None, d, &pair(), &[]);
        }
        assert_eq!(tap.overwritten(), 2);
        assert_eq!(tick(&mut src).0.payload().map(|d| d.seq), Some(3));
    }
}
