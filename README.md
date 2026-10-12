# cu-kornia-vio

[Copper](https://github.com/copper-project/copper-rs) tasks wrapping
[kornia-slam](https://github.com/kornia/kornia-slam) stereo visual-inertial odometry.

A stereo source feeds `StereoVio`, which runs in the background and publishes one camera pose per
solved frame. IMU samples reach it through a shared resource bundle, so the tracker keeps a single
input edge and can run with `background: true`.

## Crates

- **`cu-kornia-vio`**: the tasks (`StereoVio`, `ImuFeed`, `WheelFeed`, `MountFeed`) and the
  `VioBus` resource bundle.
- **`cu-stereo-payloads`**: the wire types alone. It depends on cu29, the upstream copper
  payload crates, bincode and serde and nothing else, so a stereo driver can name the types it
  produces without depending on `cu-kornia-vio` and, through it, on kornia-slam. That is the whole
  reason it is a separate crate rather than a module.

Inside `cu-kornia-vio` there are two layers:

- **`track`**: the composition root over kornia-slam. kornia-slam ships stereo matching, the
  map, map-projection PnP and bundle adjustment; this is the orchestration over them, written for
  a live camera. It uses no copper runtime or graph types (no `CuMsg`, no tasks), only the cu29
  unit types, `CuTime` and the logging macros, so the estimation is testable without a graph.
- **`task`**: the `CuTask` shell. It owns only the payload-to-image conversion and the
  world-to-camera inversion and adds no estimation logic of its own. Anything it computed would be
  a second implementation to keep in step with `track`.

## Payload types

All from `cu-stereo-payloads`. Time rides on the message `tov`, never in the payload.

| Type | Carries | `tov` |
|---|---|---|
| `StereoPair` | two `CuImage` eyes (`CuHandle`-backed buffers) plus `RectifiedStereo`: intrinsics and baseline of the rectified pair | capture time of the pair |
| `ImuBatch<N>` | up to `N` `ImuSample`s (an upstream `cu_sensor_payloads::ImuPayload` plus its own capture time), oldest first, a cumulative `dropped` counter and a `camera_aligned` flag | `ImuBatch::tov()`, a `Tov::Range` over the samples |
| `VioPose` | a `cu_spatial_payloads::Pose<f64>` (camera in world) plus `VioStatus`: keyframe flag, reset epoch, landmark count, `localization` (`Tracking`, `Coasting`, `NotLocalized`); `covariance`, the pose's 6x6 covariance as the world left perturbation of the camera pose (`-1` in `[0][0]` for none; see `cu_kornia_vio::covariance`); and `map_points`, an optional map snapshot, `Option<CuHandle<Vec<Landmark>>>`. Without wheel odometry an untracked frame carries no payload | the `tov` of the frame it was computed from |
| `WheelOdom` | a planar wheel-odometry reading: the CUMULATIVE base pose `x`, `y`, `theta` in the odometry frame (REP-103) and the producer's row-major 3x3 `covariance` | capture time of the reading |
| `CameraMount` | `cam_in_base: Pose<f64>`, the rectified left camera's pose in the base frame, for a camera on a moving head | time the mount is valid from |
| `Landmark` | one map point: world-frame `position: [f32; 3]` in metres and its map `index: u32` | element of a `VioPose` snapshot, no time of its own |

`Landmark::index` is a `u32`, not a float in a fourth column: an `f32` is exact only to 2^24,
past which two points would share an index.

The eye geometry rides on `StereoPair`, so `StereoVio` is never told what the source already
knows: the tracker is built on the first frame, and every later frame must match its size.

## Using it

Neither crate is on crates.io yet (kornia-slam is still consumed from git), so depend on the repo.
Name both crates with the identical spec string; see [the spec-string rule](#the-spec-string-rule).

```toml
[dependencies]
cu-kornia-vio = { git = "https://github.com/kornia/cu-kornia-vio", branch = "main" }
cu-stereo-payloads = { git = "https://github.com/kornia/cu-kornia-vio", branch = "main" }
cu29 = { git = "https://github.com/copper-project/copper-rs", rev = "fe2061dc10539868334f6ded55a9e75feb0f2b62" }
```

A stereo driver that only produces frames needs `cu-stereo-payloads` alone. It must fill a
`StereoPair` with rectified GRAY8 eyes and stamp the message `tov` with the capture time; an
untimed frame (`Tov::None`) is refused.

## Wiring

```ron
resources: [ ( id: "bus", provider: "cu_kornia_vio::VioBus" ) ],
tasks: [
    ( id: "imu_feed", type: "cu_kornia_vio::ImuFeed<32>", resources: { "imu": "bus.imu" } ),
    ( id: "vio", type: "cu_kornia_vio::StereoVio", background: true,
      config: { "on_tracking_loss": "reset_map", "max_keyframes": 120,
                "orb_keypoints": 400, "pnp_lm_iterations": 5 },
      resources: { "imu": "bus.imu", "epoch": "bus.reset_epoch" } ),
],
cnx: [
    ( src: "stereo", dst: "vio", msg: "cu_stereo_payloads::StereoPair" ),
    ( src: "imu", dst: "imu_feed", msg: "cu_stereo_payloads::ImuBatch<32>" ),
    ( src: "vio", dst: "poses", msg: "cu_stereo_payloads::VioPose" ),
],
```

- **`VioBus`** is a resource bundle with three members: `imu`, a bounded queue of IMU samples,
  `reset_epoch`, a counter a caller bumps with `ResetEpoch::request` to drop the map on the next
  frame, and `wheel`, a bounded queue of wheel readings and camera mounts. Each bus instance is independent, so two estimators in one graph need two buses.
- **`ImuFeed<N>`** is an inline sink that pushes each `ImuBatch<N>` into the bus queue. Its `N`
  must match the source's batch capacity. The queue is armed only when `StereoVio` has an
  `inertial` config block; without one the push is a single atomic load.
- **`StereoVio`** takes a `StereoPair` and emits a `VioPose`, draining the bus queue once per
  frame. Unknown config keys are refused at startup rather than ignored. The recognised keys are
  `on_tracking_loss` (`keep_map`, the default, or `reset_map` for a long-running graph),
  `max_keyframes`, `orb_keypoints`, `search_radius_px`, `max_covisible_keyframes`,
  `pnp_lm_iterations`, `inertial`, `landmark_buffers`, `wheel_odom`, `per_frame_fusion`,
  `keyframe_policy` (insertion thresholds, parallax, a time bound) and `keyframe_culling`;
  `cu_kornia_vio::config::CONFIG_KEYS` lists them. Its `Freezable` is a no-op (the kornia-slam
  map has no serialised form), so a resim of a `background: true` graph reproduces it only from
  the start of a log; see Known limitations.

Why the IMU does not ride a second input: `CuAsyncTask`, which `background: true` wraps a task
in, accepts a single input message. And while a solve is running the background task refuses the
arriving input, so an IMU batch bundled into the frame payload would be dropped along with every
refused frame. Measured on a Jetson Orin with an OAK-D at 640x400 and 15 fps
(`orb_keypoints: 400`, `pnp_lm_iterations: 5`), tracking takes p50 53 ms / p99 91 ms, and a
frame that inserts a keyframe p50 93 ms, against a 66 ms frame interval, so poses come out at
about 10.8 Hz and about 28 % of frames are refused. Those two keys are what the Wiring snippet
sets; without them the tracker runs 1000 ORB keypoints (`TrackerConfig::new`) and kornia-slam's
50 PnP LM iterations, which are markedly slower and refuse more.

`examples/stereo_vio.ron` is a complete graph on synthetic input:

```sh
cargo run --release --example stereo_vio -- [ITERATIONS] [LOG_PATH]
cargo run --release --example stereo_vio_logreader -- <LOG_PATH> extract-copperlists
```

The second command dumps each copperlist as JSON: the `stereo` frame and the `vio` pose, each with
its `tov`. Because the tracker runs in the background, a pose lands one or more copperlists after
the frame it was computed from, and its `tov` is that frame's `tov`.

### Map points

`StereoVio` attaches a map snapshot, `VioPose::map_points`, to the pose of every frame that
inserted a keyframe, and to no other: a keyframe is when points are created, culled, fused and
moved by local bundle adjustment, while a plain tracking frame only reads the map.

A snapshot is a bounded, newest-window view, not the whole map: the live points among the newest
`MAX_LANDMARKS` (8192) map slots, refreshed on every keyframe, so its cost does not grow with the
map. Each snapshot is complete for that window, and a consumer REPLACES what it displays with
it. Culled and fused points therefore drop out, and after a `VioStatus::reset_epoch` change the
next snapshot describes the new world whether the map was rebuilt or kept. `Landmark::index` is
the point's map index, stable within an epoch: use it to correlate a point across snapshots, not
to accumulate them. `VioStatus::landmarks` counts every live point, so it can exceed a
snapshot's length once the map outgrows the window.

The buffers come from a `CuHostMemoryPool` owned by the task, `landmark_buffers` of them
(default 16, at most 256, 128 KiB each, all allocated at startup). A snapshot stays checked out
while anything holds its handle; when none is free the pose is published without one, and the
stop line counts how often.

## Inertial path

Off by default: without an `inertial` block `StereoVio` runs stereo-only and the IMU queue is
never armed. To enable it, add the block to the `vio` config. Every field is required and none
has a default, because a guessed extrinsic or noise density yields a plausible but wrong map:

```ron
config: {
    "inertial": (
        t_bc_rotation: [/* R_BC, row-major, rectified-left-camera axes into IMU axes */],
        t_bc_translation: [/* lever arm in metres, IMU origin relative to that camera */],
        gyro_noise: 0.0,        // rad/s/sqrt(Hz), must be > 0
        accel_noise: 0.0,       // m/s^2/sqrt(Hz), must be > 0
        gyro_bias_noise: 0.0,   // rad/s^2/sqrt(Hz), must be > 0
        accel_bias_noise: 0.0,  // m/s^3/sqrt(Hz), must be > 0
        rate_hz: 200.0,
    ),
},
```

The doc comments on `InertialRon` in `src/config.rs` explain each field and why no default is
defensible (the type is crate-private, so `cargo doc` does not render them).

### Initializing while parked

The default initializer needs the robot to move (>= 5 cm over >= 10 keyframes): it finds gravity
from how the IMU's integration differs from the visual trajectory. A stereo map is already
metric, so a still window is enough. Add `static_init` inside the `inertial` block:

```ron
"inertial": ( /* ... */, enable_inertial_ba: true,
    static_init: (enabled: true),   // optional: window_s, accel_rms_max, gyro_rms_max,
                                    // gyro_mean_max, gravity_tolerance
),
```

On each keyframe before initialization, the tracker looks for >= `window_s` (1.2 s) of contiguous
IMU edges ending at it in which the accelerometer and gyroscope spreads stay under four times
what the configured noise densities predict, the mean rate stays under a plausible gyro bias,
the wheels (when `wheel_odom` is on) report no net motion and the keyframes agree on the camera
pose. Finding one, it takes gravity from the mean specific force, the gyro bias from the mean
rate and zero velocity, and rotates the map onto gravity exactly as a motion initialization
does, bumping `reset_epoch`. The accelerometer bias across gravity is unobservable at rest and
stays at its prior; along gravity it is corrected by `|f| - g`. Keyframes of the window before
the still one get their finite-difference visual velocity. A window that is not still falls
through to the motion initializer on the same keyframe; one the map refuses is not retried for
`init_retry`. The camera must hold its attitude to 0.25 deg across the window, which is what
keeps a head pan or a slow turn the wheels cannot see out of the gyro bias.

After a static initialization the two refinements wait until the camera has moved 0.2 m or
turned 10 deg, and are then scheduled (5 s, 15 s) from that moment: on parked data they would only
rotate the world again and bump `reset_epoch` for nothing.

A map drop that keeps the world (with wheels, or a loss with `keep_map`) keeps the alignment: the
next map is inertial from its first keyframe, with no new rotation and no extra `reset_epoch`
bump. Its anchor velocity is measured from the last 0.2 s of wheel odometry at the re-bootstrap,
never carried from before the drop; with no wheel velocity available the alignment is dropped
and the inertial path initializes again. A drop that restarts the world at identity runs the
still window (or the motion solve) again.

## Wheel odometry

Off by default, and off means the tracker behaves exactly as without it. For a ground robot,
add a `wheel_odom` block with `enabled: true`, bind the bus's `wheel` on `StereoVio` (the binding
is optional otherwise), and feed it:

```ron
( id: "wheel_feed", type: "cu_kornia_vio::WheelFeed", resources: { "wheel": "bus.wheel" } ),
( id: "mount_feed", type: "cu_kornia_vio::MountFeed", resources: { "wheel": "bus.wheel" } ),
( id: "vio", type: "cu_kornia_vio::StereoVio", background: true,
  config: { "wheel_odom": (
      enabled: true,
      cam_in_base_rotation: [/* camera in base, row-major */],
      cam_in_base_translation: [/* metres */],
      translation_noise: 0.02,  // m per sqrt(m) of path
      rotation_noise: 0.05,     // rad per sqrt(rad) turned
      // optional: z_sigma, tilt_sigma (planar stiffness), huber_chi2, max_gap_s
  ) },
  resources: { "imu": "bus.imu", "epoch": "bus.reset_epoch", "wheel": "bus.wheel" } ),
```

With it on:

- each keyframe pair gets a kornia-slam `WheelOdomFactor` in local BA (an interval whose readings
  have not arrived yet waits for a later keyframe). With the inertial path on, the factors join
  the inertial solve once it is applied; before that, local BA stays visual;
- the per-frame prior is the camera motion the wheels measured, with the gyro's rotation when
  the inertial path is on, instead of constant velocity;
- while vision is lost the pose keeps moving on that prior and is published as `Coasting`, or as
  `NotLocalized` when no reading covers the frame;
- a map drop (`max_keyframes`, or a loss with `reset_map`) seeds the next map at the pose
  propagated on the wheels, so the world frame stays continuous. `reset_epoch` still bumps.

`MountFeed` is only needed when the mount moves (a tilting head); the config's `cam_in_base_*`
is used until the first `CameraMount` arrives.

## Per-frame fusion

Off by default. Without it the IMU and the wheels only seed each frame's PnP and meet vision in
keyframe local BA; the published pose is PnP's. With `"per_frame_fusion": (enabled: true)` each
tracked frame's pose is optimized against its map matches (Huber), the IMU preintegrated from the
last keyframe at its bias and velocity (rotation; position once gravity is settled: a still-window
initialization or the second refinement) and the wheels from the last keyframe (the BA factor and
noise model), weighted by the configured noise figures.

Two gates, at `chi2_gate` (a 6-DOF value scaled to each test's degrees of freedom; default
22.458, 99.9 %):

- the IMU against the wheels, at their joint optimum: when they disagree, the one vision agrees
  with less is dropped for the frame (and the wheels are also dropped when their rotation
  disagrees with the gyro's);
- vision against the remaining prediction: past the gate the frame publishes the prediction as
  `Coasting` (`FailureReason::VisionRejected`) and leaves the map untouched. After 15 rejections
  in a row the next frame's vision is accepted and the fusion waits for the next keyframe.

`VioDiagnostics::fusion` carries the counts (frames, rejects, expired runs, wheel and IMU drops),
this frame's chi2 and whether it was rejected. See the `fusion` module docs for the model and its
assumptions, none of which is measured on the robot yet.

## Documentation and development

```sh
cargo doc --workspace --no-deps --open   # API docs; every public item is documented
cargo test --workspace                   # unit tests, including golden wire-format fixtures
cargo clippy --workspace --all-targets -- -D warnings
```

## The spec-string rule

There is **no `[patch]` table**. Cargo keys a git source on the exact spec STRING, not on the
resolved commit, so two spellings of one dependency are two packages at the same sha, and the
type error names the same path twice. It is the most expensive mistake available here.

It applies in three directions:

- **copper**: every copper-rs crate (`cu29`, `cu-sensor-payloads`, `cu-spatial-payloads`) is
  pinned by the same `rev =`. A consumer must spell that identical rev, or it gets a second
  `CuMsg` and a second `CuImage`.
- **kornia**: the kornia-rs crates (`kornia-image`, `kornia-3d`, `kornia-algebra`,
  `kornia-imgproc`, and any other a consumer names) come from **crates.io at `"0.2.0"`**, the
  exact requirement kornia-slam uses. A git spelling of kornia-rs is a different
  source even at identical code (v0.1.15 at `c9f9fd7` is the same code as 0.2.0), and splits
  `Image` in two. `kornia-slam` AND `kornia-sensors` (the latter is a workspace member of the
  kornia-slam repo, not of kornia-rs) are on git `rev =` of kornia-slam `main`: v0.1.0 is on
  crates.io, but the local-BA window capture (#143) landed after it. They move to crates.io at
  the first release that carries it, and every consumer must move with them.
- **this repo's own two crates**: a consumer naming both `cu-kornia-vio` and `cu-stereo-payloads`
  must give them the identical URL and the identical `branch = "main"`. `cu-kornia-vio` reaches the
  payload crate by path, so it inherits the git source it was itself pulled from; pinning the
  payload line with `rev =` against a `branch =` on the other splits `StereoPair` in two. The
  instinct to pin a payload crate is exactly the trap here. The repo was renamed from
  `kornia/cufx-vio`, and GitHub still redirects that URL, so it resolves: a dependency spelled
  with the old URL next to one spelled with the new is the same split. Use
  `https://github.com/kornia/cu-kornia-vio` everywhere.

`Cargo.lock` is tracked so that a clean clone builds: it holds a kornia-rs / kornia-slam pair
known to compile together (kornia-rs 0.2.0 from crates.io, kornia-slam main at `a9f8c52`). A
consumer's own lock still wins, but a kornia-slam from before its move to crates.io kornia-rs
(`6820744`) names git kornia-rs and splits the graph, and one from before its map-API rework
does not compile against this crate.

## Known limitations

- **Background only.** `StereoVio` is designed for `background: true`: inline, every other task
  in the graph waits behind each solve, and a frame that inserts a keyframe takes p50 93 ms even
  tuned (see [Wiring](#wiring)) against a 66 ms frame interval.
- **No map persistence or resim.** The kornia-slam map has no serialised form, so `StereoVio`'s
  `Freezable` is a no-op. A resim reproduces the task only from the start of a log, never from a
  mid-log keyframe, and a map cannot be saved or reloaded across runs.
- **IMU replay is approximate.** Samples cross into the background worker through the bus queue,
  not a logged input of the tracker, so which samples a given solve saw depends on when the worker
  ran. The `ImuFeed` edge is logged, but a replay is not bit-identical on the inertial path.
- **The map is unbounded unless told otherwise.** Nothing culls keyframes by default, and with
  neither `max_keyframes` nor `on_tracking_loss: "reset_map"` set the map grows until per-frame
  cost stalls the tracker. Set `max_keyframes` on any long-running graph; the map is then dropped
  and rebuilt when it reaches the bound. With `"keyframe_culling": (enabled: true)` the tracker
  first removes redundant keyframes there (ORB-SLAM's rule, IMU and wheel edges merged) and drops
  the map only if that cannot make room. Culling waits for a settled inertial path, and while
  driving few keyframes are redundant: the room comes mostly from stops. With the inertial path
  on, that needs `max_merged_imu_span_s` of at least twice the parked keyframe spacing
  (`parked_keyframe_interval_s` plus 0.25 s for the frame that lands past it), which startup
  checks. Culling bounds keyframes, not landmark
  slots (kornia-slam never compacts those), so the map is still dropped once it holds
  `max_keyframes * orb_keypoints` of them.

## License

Apache-2.0; see [LICENSE](LICENSE). [NOTICE](NOTICE) credits ORB-SLAM3 (Campos et al., IEEE
T-RO 2021) as the algorithmic reference for the tracking pipeline design.
