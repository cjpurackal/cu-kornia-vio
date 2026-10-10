//! Per-frame diagnostics: the tracker's state after a frame, and the record a status topic carries.
//!
//! [`tracking_state`] is the one mapping from the tracker's outcome to a [`TrackingState`]; the
//! record ([`frame_diagnostics`]) is everything else read off the tracker, all of it cached or
//! O(1) per frame.

use cu_stereo_payloads::{FusionCheck, TrackingState, VioDiagnostics};

use crate::track::{FrameReport, SystemMode, TrackStatus, Tracker};

/// What one frame produced, without its pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Measured against the map.
    Tracked,
    /// Propagated on the wheels.
    Coasting,
    /// Not placed; the wheels had nothing for it.
    NotLocalized,
    /// Not placed, no wheels.
    Untracked,
}

impl From<&TrackStatus> for Outcome {
    fn from(s: &TrackStatus) -> Self {
        match s {
            TrackStatus::Tracked(_) => Self::Tracked,
            TrackStatus::Coasting(_) => Self::Coasting,
            TrackStatus::NotLocalized(_) => Self::NotLocalized,
            TrackStatus::Untracked => Self::Untracked,
        }
    }
}

/// The tracker's state after a frame.
///
/// An unplaced frame is INITIALIZING before any map was anchored (since start or a requested
/// reset), RECOVERING while the world is being re-anchored (the frame anchored a map at a coast,
/// or the tracker is waiting to bootstrap a world that existed), and LOST otherwise: an
/// established world whose frame vision could not place. A coast is COASTING whatever caused it;
/// [`FailureReason`](cu_stereo_payloads::FailureReason) says whether vision failed.
pub fn tracking_state(
    outcome: Outcome,
    report: FrameReport,
    mode_after: SystemMode,
    anchored_since_reset: bool,
) -> TrackingState {
    match outcome {
        Outcome::Tracked => TrackingState::Tracking,
        Outcome::Coasting => TrackingState::Coasting,
        Outcome::NotLocalized | Outcome::Untracked => {
            if !anchored_since_reset {
                TrackingState::Initializing
            } else if report.reanchored || mode_after == SystemMode::Bootstrap {
                TrackingState::Recovering
            } else {
                TrackingState::Lost
            }
        }
    }
}

/// The frame bookkeeping the tracker cannot know: what the task saw of the stream.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamCounters {
    /// The frame's sequence number.
    pub seq: u64,
    /// Frames processed so far, this one included.
    pub frames: u64,
    /// Frames the stream skipped past the tracker, lifetime.
    pub skipped_frames: u64,
    /// The solve's wall-clock cost, milliseconds.
    pub solve_ms: f32,
}

/// The diagnostics record of the frame `tracker` just processed into `status`.
pub fn frame_diagnostics(
    tracker: &Tracker,
    status: &TrackStatus,
    stream: StreamCounters,
) -> VioDiagnostics {
    let report = tracker.frame_report();
    let stats = tracker.frame_stats();
    let (inertial_stage, gyro, accel) = tracker.inertial_diag();
    let f32x3 = |v: [f64; 3]| v.map(|c| c as f32);
    let sat = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    VioDiagnostics {
        seq: stream.seq,
        frames: stream.frames,
        skipped_frames: stream.skipped_frames,
        state: tracking_state(
            Outcome::from(status),
            report,
            tracker.mode(),
            tracker.anchored_since_reset(),
        ),
        failure: report.failure,
        keyframe: matches!(status, TrackStatus::Tracked(p) | TrackStatus::Coasting(p) if p.keyframe),
        relocalized: report.relocalized,
        keypoints: sat(stats.keypoints),
        stereo_matches: sat(stats.stereo_matches),
        inliers: sat(stats.inliers),
        consecutive_failures: stats.consecutive_failures,
        map_points: sat(tracker.live_map_points()),
        keyframes: sat(tracker.map().keyframes().len()),
        map_drops: tracker.map_drops(),
        reset_epoch: tracker.world_generation(),
        map_generation: tracker.map_generation(),
        solve_ms: stream.solve_ms,
        vision_degraded: tracker.vision_degraded(),
        inertial_stage,
        gyro_bias: f32x3(gyro),
        accel_bias: f32x3(accel),
        motion: tracker.last_motion_check(),
        fusion: fusion_check(tracker),
    }
}

/// The per-frame fusion's verdict on the frame `tracker` just processed.
fn fusion_check(tracker: &Tracker) -> FusionCheck {
    let f = tracker.fusion_stats();
    let report = tracker.frame_report();
    let (chi2, dof) = if report.fused {
        (f.last_chi2 as f32, f.last_dof)
    } else {
        (0.0, 0)
    };
    FusionCheck {
        frames: f.frames,
        rejects: f.rejects,
        wheel_dropped: f.wheel_dropped,
        chi2,
        dof,
        rejected: report.vision_rejected,
        cost_us: if report.fused { f.last_cost_us } else { 0 },
    }
}

/// The frames a stream skipped between two processed sequence numbers: `0` for consecutive ones,
/// and for a sequence that went BACKWARDS (a camera restart), which is a new stream, not a gap.
pub fn seq_gap(previous: Option<u64>, seq: u64) -> u64 {
    match previous {
        Some(p) if seq > p => seq - p - 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu_stereo_payloads::FailureReason;

    fn report(reanchored: bool) -> FrameReport {
        FrameReport {
            failure: FailureReason::LowPnpInliers,
            relocalized: false,
            reanchored,
            ..FrameReport::default()
        }
    }

    /// The whole table: each row is a state a reader of the status topic acts on differently.
    #[test]
    fn test_the_state_table() {
        use Outcome::*;
        use SystemMode::{Bootstrap, Tracking as Trk};
        let rows = [
            // (outcome, reanchored, mode after, anchored since reset) -> state
            (Tracked, false, Trk, true, TrackingState::Tracking),
            (Coasting, false, Trk, true, TrackingState::Coasting),
            (Coasting, true, Trk, true, TrackingState::Coasting),
            (
                Untracked,
                false,
                Bootstrap,
                false,
                TrackingState::Initializing,
            ),
            (
                NotLocalized,
                false,
                Bootstrap,
                false,
                TrackingState::Initializing,
            ),
            (Untracked, false, Trk, true, TrackingState::Lost),
            (NotLocalized, false, Trk, true, TrackingState::Lost),
            (Untracked, true, Trk, true, TrackingState::Recovering),
            (Untracked, false, Bootstrap, true, TrackingState::Recovering),
            (
                NotLocalized,
                false,
                Bootstrap,
                true,
                TrackingState::Recovering,
            ),
        ];
        for (outcome, reanchored, mode, anchored, want) in rows {
            assert_eq!(
                tracking_state(outcome, report(reanchored), mode, anchored),
                want,
                "{outcome:?} reanchored={reanchored} mode={mode:?} anchored={anchored}"
            );
        }
    }

    #[test]
    fn test_a_seq_gap_counts_only_forward_skips() {
        assert_eq!(seq_gap(None, 7), 0, "the first frame skips nothing");
        assert_eq!(seq_gap(Some(7), 8), 0);
        assert_eq!(seq_gap(Some(7), 10), 2);
        assert_eq!(seq_gap(Some(7), 7), 0, "a repeated frame is not a gap");
        assert_eq!(
            seq_gap(Some(70), 3),
            0,
            "a restarted camera is a new stream"
        );
    }
}
