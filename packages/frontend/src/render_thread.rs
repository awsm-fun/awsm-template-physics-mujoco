//! Render thread — hosts `awsm-renderer` on an `OffscreenCanvas`, loads the
//! editor-exported `scene.toml` through the **player** path, and drives the
//! MuJoCo sim instance in that scene from the sim worker's pose block.
//!
//! The MuJoCo side of this file is now about thirty lines, and that is the
//! point. The robot is *authored content*: exported from the compiled model by
//! `awsm-renderer-mujoco-export`, imported in the editor, and shipped in the
//! bundle like any other geometry — so this worker builds nothing. It only:
//!
//! 1. takes the sim instance the loader already resolved
//!    (`LoadedScene::mujoco` — geom_id→transform, plus the model fingerprint),
//!    and checks the sim is running the SAME model (content hash), refusing to
//!    drive the wrong robot rather than producing convincing nonsense;
//! 2. each frame, publishes its frame time to the sim worker (which steps to
//!    it), then takes the two pose slots around its display time out of the
//!    sim worker's `SharedArrayBuffer` pose block, blends them, and hands the
//!    result to `scene_loader::mujoco::apply_geom_poses` (timing and layout:
//!    see [`crate::protocol`]).
//!
//! The pose block is laid out as a **stream frame** in the documented
//! convention (7 f32 per geom, `[px,py,pz,qw,qx,qy,qz]`, geom-id indexed), so
//! nothing is reshaped on either side of the SAB — this worker could dump its
//! block verbatim as a capture file and the editor would bake it.
//!
//! Everything else is the stock player template: bundle load, orbit camera,
//! resize.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use awsm_renderer::AwsmRenderer;
use awsm_renderer_settings::{
    apply_anti_aliasing, post_to_page, scene_anti_aliasing, AaMode, SettingsToPage,
    SettingsToWorker,
};
use awsm_renderer_stats::{
    post_report, stats_logging, FrameInput, RendererTimings, StatsCollector, StatsToWorker,
};

use crate::protocol::{
    abs_ms_to_us, header, time_diff_us, CameraMsg, Layout, OrbitState, RenderMsg, CONTACT_STRIDE,
    JOINT_STRIDE, MAX_CONTACTS, POSE_SLOTS, POSE_STRIDE,
};
use crate::stats::{self, worker_now, Phase, SimReport, SimStats};

/// The boxed RAF callback, self-referenced so the render loop can reschedule
/// itself each frame (and stays alive for the worker's lifetime).
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Lens + clip planes for the one camera this app has.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 400.0;

/// The awsm-renderer release the vendored Basis transcoder files came with.
/// It is appended to their URLs as `?v=…`: those files keep their names from
/// one release to the next, so after a deployed upgrade a browser could
/// otherwise pair a cached old transcoder with the new worker script. Bump it
/// together with the awsm-renderer version in Cargo.toml.
const BASIS_VERSION: &str = "0.46.0";

/// The renderer, shared by the frame loop and the message handler. `None`
/// while an anti-aliasing change owns it across its awaits.
type RendererCell = Rc<RefCell<Option<AwsmRenderer>>>;

/// The orbit camera as drawn: a fixed target, and the [`OrbitState`] main
/// owns. Main posts its state once per main-thread frame; this keeps the last
/// few and each frame draws the newest one at least half a frame old (see
/// [`CameraMsg`]), so every frame shows exactly one main frame of input.
struct OrbitCamera {
    look_at: Vec3,
    state: OrbitState,
    /// `(absolute ms, state)` as main posted them, oldest first.
    posted: VecDeque<(f64, OrbitState)>,
}

impl OrbitCamera {
    /// States kept: a few frames' worth is all the selection ever looks at.
    const KEEP: usize = 4;

    fn post(&mut self, t_ms: f64, state: OrbitState) {
        if self.posted.len() == Self::KEEP {
            self.posted.pop_front();
        }
        self.posted.push_back((t_ms, state));
    }

    /// Pick the state to draw at frame time `now_ms` (absolute) on a display
    /// with frame period `period_ms`.
    fn select(&mut self, now_ms: f64, period_ms: f64) {
        let cutoff = now_ms - period_ms * 0.5;
        if let Some((_, s)) = self.posted.iter().rev().find(|(t, _)| *t <= cutoff) {
            self.state = *s;
        }
    }

    fn eye(&self) -> Vec3 {
        let (sy, cy) = self.state.yaw.sin_cos();
        let (sp, cp) = self.state.pitch.sin_cos();
        let r = self.state.radius;
        self.look_at + Vec3::new(r * cp * sy, r * sp, r * cp * cy)
    }
}

/// The live link between the sim worker's pose block and the scene's sim
/// instance. No geometry here — the instance came out of the bundle.
struct SimLink {
    /// The loader-resolved instance: geom_id→transform + the model fingerprint.
    instance: awsm_renderer_scene_loader::mujoco::MujocoInstance,
    layout: Layout,
    /// The sim's fixed step, µs.
    dt_us: f64,
    /// i32 and f32 views over the whole pose block.
    words: js_sys::Int32Array,
    floats: js_sys::Float32Array,
    /// Copies of the two slots around the display time.
    older: Vec<f32>,
    newer: Vec<f32>,
    /// The blended geom poses, then body poses, handed to the sinks.
    geoms: Vec<f32>,
    bodies_blend: Vec<f32>,
    /// The frame time published last frame (absolute µs): this frame displays
    /// `dt` before it.
    last_frame_us: Option<u32>,
    /// Whether the scene binds any node to a body (only flexes do).
    drive_bodies: bool,
    /// The contact-point debug overlay, when it was built.
    contacts: Option<ContactOverlay>,
    /// The joint-axis debug overlay, when it was built.
    joints: Option<JointOverlay>,
    /// The inertia-box debug overlay, when it was built.
    inertia: Option<InertiaOverlay>,
}

/// The inertia-box overlay: per body, the box that would have the same mass and
/// inertia, drawn in the body's INERTIAL frame.
///
/// This is the overlay that shows what the solver actually sees. A limb whose
/// visual mesh is slender but whose inertia box is fat is mis-specified, and
/// nothing else in the picture reveals that.
struct InertiaOverlay {
    region: js_sys::Float32Array,
    scratch: Vec<f32>,
    /// `body_id → ` (transform, half-extents). `None` for a massless body,
    /// which has no equivalent box — the world body most of all.
    boxes: Vec<Option<(awsm_renderer::transforms::TransformKey, Vec3)>>,
}

/// The joint-axis overlay: one bar through each hinge or slide joint's world
/// anchor, along its world axis.
///
/// Simpler than the contacts: a model's joint count is fixed, so there is no
/// pool, no live count and no visibility toggling — every bar that exists is
/// always drawn. Ball and free joints have no single axis and get no bar at all.
struct JointOverlay {
    region: js_sys::Float32Array,
    scratch: Vec<f32>,
    /// `joint_id → ` its bar, or `None` for a joint with no meaningful axis.
    bars: Vec<Option<awsm_renderer::transforms::TransformKey>>,
}

/// The contact-point overlay: a preallocated pool of spikes, one per possible
/// contact, parented into the sim's own frame so contact positions apply as raw
/// MuJoCo world coordinates — the same trick the geom nodes use.
///
/// DEV TOOLING. This is the only place in the whole feature where debug
/// visualisation lives: the renderer, the editor and the bundle format never
/// learn what a contact is. It exists here because this is the only side with
/// live sim data.
struct ContactOverlay {
    /// f32 view over the block's contact region.
    region: js_sys::Float32Array,
    scratch: Vec<f32>,
    /// One transform per pooled spike. The count varies every step and a
    /// renderer cannot mint nodes per frame, so the pool is fixed and the
    /// unused tail is hidden — the same shape the tendon channel uses.
    spikes: Vec<(
        awsm_renderer::transforms::TransformKey,
        awsm_renderer::meshes::MeshKey,
    )>,
    /// How many spikes are currently shown, so visibility is edge-triggered.
    shown: usize,
}

/// Worker entry: unpack the transferred `OffscreenCanvas`, the page origin and
/// the page's settings (`aa`, the anti-aliasing mode id), build the WebGPU
/// device, and kick off the async load+render.
pub fn start(payload: JsValue) -> Result<(), JsValue> {
    use awsm_renderer::core::renderer::{AwsmRendererWebGpuBuilder, DeviceRequestLimits};
    use awsm_renderer::web_global::navigator_gpu;
    use awsm_renderer_scene_loader::basis::{configure, BasisWorkerConfig};

    let canvas: web_sys::OffscreenCanvas =
        js_sys::Reflect::get(&payload, &JsValue::from_str("canvas"))?.unchecked_into();
    // The worker has a `blob:` base URL, so relative fetches can't resolve —
    // main passes the page origin so we can build the absolute scene.toml URL.
    let origin = js_sys::Reflect::get(&payload, &JsValue::from_str("origin"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    // The mode the page's Settings chose: the renderer is built with it, so
    // start-up compiles exactly the pipelines this session uses.
    let aa = js_sys::Reflect::get(&payload, &JsValue::from_str("aa"))
        .ok()
        .and_then(|v| v.as_string())
        .and_then(|id| AaMode::from_id(&id))
        .unwrap_or(AaMode::Msaa);
    // Provide the Basis (KTX2/BasisU) codec URLs — the crate hardcodes none.
    // Must be ABSOLUTE (our base is `blob:`); `app_base` keeps the deploy PATH
    // so a subpath deploy resolves.
    {
        let app_base = js_sys::Reflect::get(&payload, &JsValue::from_str("app_base"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let base = app_base.trim_end_matches('/');
        configure(BasisWorkerConfig::player(
            // `?v=` names the vendored build: these files keep their names
            // across releases, so a browser that cached an older transcoder
            // (wasm included: the worker carries the query over to it) would
            // otherwise pair it with the new JS. See `BASIS_VERSION`.
            format!("{base}/workers/basis-worker.js?v={BASIS_VERSION}"),
            format!("{base}/vendor/basis/basis_transcoder.js?v={BASIS_VERSION}"),
        ));
    }
    // Which bundle directory to load — main picked it from `?scene=` and passed
    // the NAME, since this worker's own base is a `blob:` with no query at all.
    let bundle_dir = js_sys::Reflect::get(&payload, &JsValue::from_str("bundle"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "bundle".to_string());
    // Debug overlays are opt-in (`?contacts` in the page URL); main reads it,
    // because this worker's own base is a `blob:` with no query at all.
    let flag = |key: &str| {
        js_sys::Reflect::get(&payload, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    };
    let overlays = OverlayToggles {
        contacts: flag("contacts"),
        joints: flag("joints"),
        inertia: flag("inertia"),
    };
    let canvas_handle = canvas.clone();
    post_progress("render worker: requesting WebGPU device…");
    let gpu =
        navigator_gpu().ok_or_else(|| JsValue::from_str("render worker: no navigator.gpu"))?;
    let gpu_builder = AwsmRendererWebGpuBuilder::new_with_offscreen_canvas(gpu, canvas)
        .with_device_request_limits(DeviceRequestLimits::max_all());

    wasm_bindgen_futures::spawn_local(async move {
        if let Err(err) = run(gpu_builder, canvas_handle, origin, bundle_dir, overlays, aa).await {
            tracing::error!("render thread: {err:?}");
            post_to_main(&RenderMsg::Error {
                message: format!("{err:?}"),
            });
        }
    });
    Ok(())
}

async fn run(
    gpu_builder: awsm_renderer::core::renderer::AwsmRendererWebGpuBuilder,
    canvas: web_sys::OffscreenCanvas,
    origin: String,
    bundle_dir: String,
    overlays: OverlayToggles,
    aa: AaMode,
) -> Result<(), JsValue> {
    use awsm_renderer::camera::CameraParams;
    use awsm_renderer::AwsmRendererBuilder;

    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(aa.to_anti_aliasing())
        // GPU frame timing for the Stats panel's GPU group: timestamp queries
        // on every pass, folded into one whole-frame span (cheap; silently
        // off without `timestamp-query`). The CPU tier stays `Off` until the
        // renderer group is expanded.
        .with_logging(stats_logging())
        .with_phase_handler(|phase| {
            use awsm_renderer::RendererLoadingPhase as P;
            let message = match phase {
                P::Init => "renderer init: GPU device + core resources…",
                P::CompilingShaders | P::BuildingPipelines | P::Ready => return,
            };
            post_progress(message);
        })
        .build()
        .await
        .map_err(|e| JsValue::from_str(&format!("build renderer: {e}")))?;
    post_progress(&format!(
        "render worker: WebGPU device + renderer ready (anti-aliasing: {})",
        AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
    ));

    // ── Warm-up + scene fetch, CONCURRENTLY ─────────────────────────────────
    let bundle_base = format!("{}/{bundle_dir}", origin.trim_end_matches('/'));
    let scene_url = format!("{bundle_base}/scene.toml");
    post_progress("compiling core render pipelines + fetching scene… (first visit can take a while — cached after)");
    let (compiled, scene) =
        futures::join!(renderer.ensure_config_pipelines(), fetch_scene(&scene_url));
    let compiled =
        compiled.map_err(|e| JsValue::from_str(&format!("ensure_config_pipelines: {e}")))?;
    let mut scene =
        scene.map_err(|e| JsValue::from_str(&format!("load scene {scene_url}: {e}")))?;
    // `load_scene_for_player` applies the scene's authored anti-aliasing
    // (`post_process.anti_aliasing`): make it the renderer's, the page's
    // choice, so the loader keeps it and nothing compiles twice.
    scene.post_process.anti_aliasing = scene_anti_aliasing(&renderer.anti_aliasing);
    post_progress(&format!(
        "core pipelines ready ({compiled}) · scene parsed ({} nodes)",
        scene.nodes.len()
    ));

    let assets = awsm_renderer_scene_loader::assets::HttpAssets::new(bundle_base.clone());
    let mut last_phase_line = String::new();
    let loaded = awsm_renderer_scene_loader::load_scene_for_player(
        &mut renderer,
        &scene,
        &assets,
        |phase| {
            let line = phase.label();
            if line != last_phase_line && !line.contains("0/0") {
                post_progress(&line);
                last_phase_line = line;
            }
        },
    )
    .await
    .map_err(|e| JsValue::from_str(&format!("load_scene_for_player: {e}")))?;

    // The loader builds the scene's particle emitters; the host ticks them
    // (as it does clips). Built before `loaded.mujoco` is moved out below.
    let mut particles = awsm_renderer_scene_loader::ParticleDriver::new(&loaded);

    // The robot is authored content: the loader already resolved its
    // geom_id→transform binding out of the bundle. Nothing is built here.
    let sim_instance = loaded.mujoco.into_iter().next();
    match &sim_instance {
        Some(i) => tracing::info!(
            "render thread: scene has sim instance {} ({} geoms, {} bound)",
            i.source.filename,
            i.geoms.len(),
            i.geoms.iter().flatten().count()
        ),
        None => tracing::warn!("render thread: bundle has no MuJoCo sim instance"),
    }

    let mut last_commit_line = String::new();
    renderer
        .commit_load(|stats| {
            let Some(line) = stats.phase_label() else {
                return;
            };
            if line != last_commit_line && !line.contains("0/0") {
                post_progress(&line);
                last_commit_line = line;
            }
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("commit_load: {e}")))?;
    renderer.update_transforms();
    post_progress("gpu commit complete — waiting for the MuJoCo model…");

    // ── Camera + shared state, then hand control to the event loop ──────────
    let camera = Rc::new(RefCell::new(OrbitCamera {
        look_at: Vec3::new(0.0, 0.8, 0.0),
        state: OrbitState::START,
        posted: VecDeque::new(),
    }));
    // If the page asked for TAA and this device cannot run it, the renderer
    // fell back to MSAA 4× while loading: tell the page (it stops offering
    // TAA and shows what runs).
    if aa.is_taa() {
        if let Some(reason) = renderer.taa_fallback_reason() {
            tracing::warn!("render thread: TAA unavailable, running MSAA 4×: {reason}");
            post_to_page(&SettingsToPage::AaCompileDone {
                ok: true,
                error: None,
                mode: AaMode::from_anti_aliasing(&renderer.anti_aliasing),
                taa_fallback: Some(reason.to_string()),
            });
        }
    }
    #[allow(clippy::arc_with_non_send_sync)]
    let cell: RendererCell = Rc::new(RefCell::new(Some(renderer)));
    let mirror: Rc<RefCell<Option<SimLink>>> = Rc::new(RefCell::new(None));
    // The instance the loader resolved, held until the sim worker announces its
    // model. There is exactly one in this template's scene; a multi-robot world
    // would key these by fingerprint instead.
    let mut pending = sim_instance;
    // The sim worker's model description, once main forwards it: linked by
    // the next frame (the frame loop owns the renderer).
    let model: Rc<RefCell<Option<JsValue>>> = Rc::new(RefCell::new(None));
    // The Stats panel's numbers (awsm-renderer-stats): main tells it which
    // groups are open and what the page measured, and forwards the sim's
    // reports; it posts a report about once a second.
    let collector = Rc::new(RefCell::new(StatsCollector::new(stats::config())));
    let sim_stats = Rc::new(RefCell::new(SimStats::default()));
    // The latest anti-aliasing change the page asked for and that has not been
    // applied yet (a newer one replaces it), and whether one is being applied.
    let pending_aa: Rc<Cell<Option<AaMode>>> = Rc::new(Cell::new(None));
    let busy = Rc::new(Cell::new(false));

    install_onmessage(
        model.clone(),
        camera.clone(),
        canvas.clone(),
        collector.clone(),
        sim_stats.clone(),
        pending_aa.clone(),
    )?;
    // Only now is the worker listening — main holds the model message until
    // this arrives.
    post_to_main(&RenderMsg::SceneReady);

    // ── Render loop ─────────────────────────────────────────────────────────
    #[allow(clippy::arc_with_non_send_sync)]
    let raf: RafCell = Rc::new(RefCell::new(None));
    let raf_init = raf.clone();
    let raf_run = raf.clone();
    let mut frame_count: u32 = 0;
    // This worker's clock origin: frame times are sent as absolute times, the
    // one timeline every thread of the page shares.
    let time_origin = stats::worker_time_origin();
    // The display's frame period (ms), smoothed: the camera's selection rule
    // needs half of it.
    let mut period_ms = 1000.0 / 60.0;
    let mut prev_vsync: Option<f64> = None;

    *raf_init.borrow_mut() = Some(Closure::new(move |vsync_ms: f64| {
        // An anti-aliasing change owns the renderer: skip the frame.
        if busy.get() {
            schedule(&raf_run);
            return;
        }
        // A pending Settings anti-aliasing change: apply it off the render
        // path. `apply_anti_aliasing` (set_anti_aliasing + commit_load) is
        // async and compiles the new mode's pipelines (the first switch in
        // each direction; later ones hit the renderer's cache). It owns the
        // renderer across its awaits, so no RefCell borrow is held over them;
        // `busy` keeps every frame off the renderer until it is back. The
        // page raises its "compiling pipelines" overlay from the events. The
        // sim keeps stepping meanwhile; the first frame after catches up.
        if let Some(mode) = pending_aa.take() {
            if let Some(mut renderer) = cell.borrow_mut().take() {
                busy.set(true);
                let (cell, busy) = (cell.clone(), busy.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    let result = apply_anti_aliasing(&mut renderer, mode, |event| {
                        post_to_page(&SettingsToPage::from(event))
                    })
                    .await;
                    match result {
                        Ok(()) => tracing::info!(
                            "render thread: anti-aliasing {}",
                            AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
                        ),
                        Err(message) => {
                            tracing::error!("render thread: {message}");
                            post_to_main(&RenderMsg::Error { message });
                        }
                    }
                    *cell.borrow_mut() = Some(renderer);
                    busy.set(false);
                });
            }
            schedule(&raf_run);
            return;
        }

        let frame_start = worker_now();
        let now_abs_ms = time_origin + vsync_ms;
        // This frame's real time step (0 on the first frame): it advances the
        // scene's clips and particles, and feeds the period estimate.
        let frame_dt_ms = prev_vsync
            .replace(vsync_ms)
            .map_or(0.0, |prev| (vsync_ms - prev).max(0.0));
        if frame_dt_ms > 2.0 && frame_dt_ms < 100.0 {
            period_ms += (frame_dt_ms - period_ms) * 0.1;
        }
        let mut guard = cell.borrow_mut();
        let Some(r) = guard.as_mut() else {
            schedule(&raf_run);
            return;
        };

        // The sim worker announced its model: bind its pose block to the
        // scene's sim instance.
        if let Some(data) = model.borrow_mut().take() {
            link_model(r, &mut pending, &data, overlays, &mirror);
        }

        // The renderer group's breakdown (our phases and awsm-renderer's CPU
        // scopes) is timed on every frame while it is expanded, never else.
        let detail = {
            let mut c = collector.borrow_mut();
            c.begin_renderer_frame(r, frame_start);
            c.renderer_detail()
        };
        let lap = |phase: Phase| {
            if detail {
                collector.borrow_mut().lap(phase as usize, worker_now());
            }
        };

        // Tell the sim this frame has started, first, so it steps while the
        // rest of the frame runs (see `protocol`, "Timing").
        let mut sim = mirror.borrow_mut();
        let display_us = sim
            .as_mut()
            .map(|m| m.begin_frame(abs_ms_to_us(now_abs_ms)));

        // The scene's own motion: authored clips, texture UV flows and
        // particle emitters, by this frame's real time step.
        if let Err(err) = r.update_animations(frame_dt_ms) {
            tracing::warn!("render thread: update_animations: {err}");
        }
        particles.tick(r, (frame_dt_ms / 1000.0) as f32);
        lap(Phase::Animate);

        // Then the sim's poses, at the time fixed by the previous frame. They
        // are written after the clips, so the sim wins on the nodes it drives.
        let mut late = false;
        if let (Some(m), Some(display_us)) = (sim.as_mut(), display_us) {
            late = apply_poses(r, m, display_us);
        }
        drop(sim);
        lap(Phase::Read);

        let mut cam = camera.borrow_mut();
        cam.select(now_abs_ms, period_ms);
        let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
        let mut params =
            CameraParams::perspective(CAMERA_FOV_Y_DEG.to_radians(), CAMERA_NEAR, CAMERA_FAR);
        params.focus_distance = cam.state.radius;
        let _ = r.set_camera(view, params);
        drop(cam);
        lap(Phase::Camera);

        r.update_transforms();
        lap(Phase::Transforms);
        if let Err(err) = r.render(None) {
            tracing::warn!("render thread: render error: {err}");
        }
        lap(Phase::Render);
        drop(guard);

        frame_count = frame_count.wrapping_add(1);
        let mut c = collector.borrow_mut();
        if frame_count == 3 {
            post_to_main(&RenderMsg::Ready);
            // The scene is on screen: the stats warm up, then count.
            post_report(c.mark_ready());
            sim_stats.borrow_mut().restart();
        }
        let input = FrameInput {
            late,
            ..FrameInput::new(vsync_ms, worker_now() - frame_start)
        };
        if let Some(window) = c.end_frame(input, &RendererTimings) {
            let section = sim_stats.borrow_mut().section(&window);
            post_report(c.report(&window, vec![section]));
        }
        drop(c);
        schedule(&raf_run);
    }));
    schedule(&raf_init);
    std::mem::forget(raf);
    Ok(())
}

/// Asks for the next animation frame for the render loop.
fn schedule(raf: &RafCell) {
    if let Some(cb) = raf.borrow().as_ref() {
        let _ = awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
    }
}

/// Bind the sim worker's model (`kind: "mujoco-model"`, forwarded by main) to
/// the scene's sim instance, once.
fn link_model(
    r: &mut AwsmRenderer,
    pending: &mut Option<awsm_renderer_scene_loader::mujoco::MujocoInstance>,
    data: &JsValue,
    overlays: OverlayToggles,
    mirror: &RefCell<Option<SimLink>>,
) {
    let Some(instance) = pending.take() else {
        tracing::warn!("render thread: no sim instance in the scene to drive");
        post_to_main(&RenderMsg::Error {
            message: "the loaded bundle has no MuJoCo sim instance — export a scene \
                      with one imported via import_mujoco_from_url"
                .to_string(),
        });
        return;
    };
    let bound = instance.geoms.iter().flatten().count();
    let total = instance.geoms.len();
    let label = instance
        .model_name
        .clone()
        .unwrap_or_else(|| instance.source.filename.clone());
    match link_sim(r, instance, data, overlays) {
        Ok(m) => {
            tracing::info!("render thread: driving {label} — {bound}/{total} geoms bound");
            post_progress(&format!("sim linked ({bound}/{total} geoms) — running"));
            *mirror.borrow_mut() = Some(m);
        }
        Err(err) => {
            tracing::error!("link_sim: {err:?}");
            post_to_main(&RenderMsg::Error {
                message: format!("link sim: {err:?}"),
            });
        }
    }
}

impl SimLink {
    /// Start of a frame at absolute time `frame_us`: publish it to the sim
    /// (time first, then the tick, so the sim never sees a tick without its
    /// time) and return the display time fixed by the PREVIOUS frame,
    /// `T_(k-1) - dt`, or `None` on the first frame.
    fn begin_frame(&mut self, frame_us: u32) -> Option<u32> {
        let display = self
            .last_frame_us
            .replace(frame_us)
            .map(|t| t.wrapping_sub(self.dt_us.round() as u32));
        let _ = js_sys::Atomics::store(&self.words, header::FRAME_TIME, frame_us as i32);
        let _ = js_sys::Atomics::store(&self.words, header::TIME_VALID, 1);
        let _ = js_sys::Atomics::add(&self.words, header::TICK, 1);
        let _ = js_sys::Atomics::notify(&self.words, header::TICK);
        display
    }

    /// Slot `n`'s seq and time, if it holds a complete state.
    fn slot_stamp(&self, n: u32) -> Option<(i32, u32)> {
        let base = self.layout.slot(n as usize % POSE_SLOTS) as u32;
        let seq = js_sys::Atomics::load(&self.words, base).ok()?;
        if seq == 0 || seq % 2 != 0 {
            return None;
        }
        let t = js_sys::Atomics::load(&self.words, base + 1).ok()? as u32;
        Some((seq, t))
    }
}

/// Copy slot `n` into `out` if it still holds the state stamped `seq` (the
/// writer may have lapped the ring and be rewriting it).
fn copy_slot(
    words: &js_sys::Int32Array,
    floats: &js_sys::Float32Array,
    layout: &Layout,
    n: u32,
    seq: i32,
    out: &mut [f32],
) -> bool {
    let base = layout.slot(n as usize % POSE_SLOTS);
    floats
        .subarray(base as u32, (base + layout.slot_words()) as u32)
        .copy_to(out);
    js_sys::Atomics::load(words, base as u32).ok() == Some(seq)
}

/// A published slot: its index, seq and time (absolute µs).
type Stamp = (u32, i32, u32);

/// Draw the sim at `display_us` (absolute µs; `None` = the newest state): find
/// the two published slots around it, blend them, and hand the result to the
/// renderer's pose sink. Returns `true` when the sim was late — no state at or
/// after the display time yet — and the newest one was held instead.
///
/// The writer may be lapping the ring while this reads, so every copy is
/// checked against its slot's seq; a slot that changed underneath is dropped
/// and the frame keeps last frame's poses. Nothing waits.
fn apply_poses(
    r: &mut awsm_renderer::AwsmRenderer,
    m: &mut SimLink,
    display_us: Option<u32>,
) -> bool {
    let published = js_sys::Atomics::load(&m.words, header::PUBLISHED).unwrap_or(0) as u32;
    if published == 0 {
        return false;
    }
    // The slot after the newest may be the one being written: stay clear of
    // it, which leaves POSE_SLOTS - 1 readable.
    let oldest = published.saturating_sub(POSE_SLOTS as u32 - 1);
    let newest = published - 1;

    // Walk back from the newest to the first slot at or before the display
    // time; the one after it is the other end of the pair.
    let mut late = false;
    let mut pair: Option<(Stamp, Option<Stamp>)> = None;
    match display_us {
        None => {
            if let Some((seq, t)) = m.slot_stamp(newest) {
                pair = Some(((newest, seq, t), None));
            }
        }
        Some(display) => {
            let mut after: Option<Stamp> = None;
            let mut n = newest;
            while let Some((seq, t)) = m.slot_stamp(n) {
                if time_diff_us(t, display) <= 0 {
                    pair = Some(((n, seq, t), after));
                    late = after.is_none();
                    break;
                }
                after = Some((n, seq, t));
                if n == oldest {
                    // Every readable state is after the display time (only at
                    // start-up): show the oldest.
                    pair = Some(((n, seq, t), None));
                    break;
                }
                n -= 1;
            }
        }
    }
    let Some(((n0, seq0, t0), next)) = pair else {
        return false;
    };
    if !copy_slot(&m.words, &m.floats, &m.layout, n0, seq0, &mut m.older) {
        return late;
    }
    let alpha_newer = match (next, display_us) {
        (Some((n1, seq1, t1)), Some(display)) => {
            if !copy_slot(&m.words, &m.floats, &m.layout, n1, seq1, &mut m.newer) {
                return late;
            }
            let span = time_diff_us(t1, t0).max(1) as f32;
            Some((time_diff_us(display, t0) as f32 / span).clamp(0.0, 1.0))
        }
        _ => None,
    };

    let geoms_at = m.layout.slot_geoms();
    let bodies_at = m.layout.slot_bodies();
    let ngeom = m.layout.ngeom * POSE_STRIDE;
    let nbody = m.layout.nbody * POSE_STRIDE;
    match alpha_newer {
        Some(alpha) => {
            blend_poses(
                &m.older[geoms_at..geoms_at + ngeom],
                &m.newer[geoms_at..geoms_at + ngeom],
                alpha,
                &mut m.geoms,
            );
            blend_poses(
                &m.older[bodies_at..bodies_at + nbody],
                &m.newer[bodies_at..bodies_at + nbody],
                alpha,
                &mut m.bodies_blend,
            );
        }
        None => {
            m.geoms
                .copy_from_slice(&m.older[geoms_at..geoms_at + ngeom]);
            m.bodies_blend
                .copy_from_slice(&m.older[bodies_at..bodies_at + nbody]);
        }
    }

    if let Err(err) = awsm_renderer_scene_loader::mujoco::apply_geom_poses(r, &m.instance, &m.geoms)
    {
        tracing::warn!("pose sink: {err}");
    }
    // The second real channel, right next to the first: this is the whole of
    // what it takes to drive a deformable. The flex is a skinned mesh, so
    // moving its joints IS deforming it.
    //
    // The sink resolves those joints through the loader's skin bridge, not
    // through the bone nodes' own scene transforms — a skin reads the rig glb's
    // baked joint transforms, and writing the scene ones moves transforms
    // nothing is skinned to (every frame applies, nothing errors, the cloth
    // never leaves its bind pose).
    if m.drive_bodies {
        if let Err(err) =
            awsm_renderer_scene_loader::mujoco::apply_body_poses(r, &m.instance, &m.bodies_blend)
        {
            tracing::warn!("body pose sink: {err}");
        }
    }
    apply_overlays(r, m);
    late
}

/// Blend two runs of `[px,py,pz, qw,qx,qy,qz]` poses: positions linearly,
/// rotations by normalized lerp along the shorter arc. Between two fixed steps
/// a few milliseconds apart this is indistinguishable from slerp.
fn blend_poses(a: &[f32], b: &[f32], alpha: f32, out: &mut [f32]) {
    for ((a, b), o) in a
        .as_chunks::<POSE_STRIDE>()
        .0
        .iter()
        .zip(b.as_chunks::<POSE_STRIDE>().0)
        .zip(out.as_chunks_mut::<POSE_STRIDE>().0)
    {
        for i in 0..3 {
            o[i] = a[i] + (b[i] - a[i]) * alpha;
        }
        let dot = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6];
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let mut len = 0.0;
        for i in 3..7 {
            o[i] = a[i] + (b[i] * sign - a[i]) * alpha;
            len += o[i] * o[i];
        }
        let len = len.sqrt();
        if len > 1e-12 {
            for v in &mut o[3..7] {
                *v /= len;
            }
        }
    }
}

/// The debug overlays, from the newest state (not interpolated). A torn read —
/// the sim was mid-publish — skips them for this frame.
fn apply_overlays(r: &mut awsm_renderer::AwsmRenderer, m: &mut SimLink) {
    if m.contacts.is_none() && m.joints.is_none() && m.inertia.is_none() {
        return;
    }
    let seq0 = js_sys::Atomics::load(&m.words, header::OVERLAY_SEQ).unwrap_or(1);
    if seq0 % 2 != 0 {
        return;
    }
    let ncon = js_sys::Atomics::load(&m.words, header::NCON)
        .unwrap_or(0)
        .max(0) as usize;
    if let Some(overlay) = &mut m.contacts {
        overlay.region.copy_to(&mut overlay.scratch[..]);
    }
    if let Some(overlay) = &mut m.joints {
        overlay.region.copy_to(&mut overlay.scratch[..]);
    }
    if let Some(overlay) = &mut m.inertia {
        overlay.region.copy_to(&mut overlay.scratch[..]);
    }
    let seq1 = js_sys::Atomics::load(&m.words, header::OVERLAY_SEQ).unwrap_or(-1);
    if seq0 != seq1 {
        return;
    }
    if let Some(overlay) = &mut m.contacts {
        overlay.apply(r, ncon);
    }
    if let Some(overlay) = &mut m.joints {
        overlay.apply(r);
    }
    if let Some(overlay) = &mut m.inertia {
        overlay.apply(r);
    }
}

impl InertiaOverlay {
    fn apply(&mut self, r: &mut awsm_renderer::AwsmRenderer) {
        for (b, entry) in self.boxes.iter().enumerate() {
            let Some((tk, half)) = entry else { continue };
            let o = b * POSE_STRIDE;
            let _ = r.transforms.set_local(
                *tk,
                awsm_renderer::transforms::Transform {
                    translation: Vec3::new(
                        self.scratch[o],
                        self.scratch[o + 1],
                        self.scratch[o + 2],
                    ),
                    // MuJoCo quaternions are [w, x, y, z]; glam's are [x, y, z, w].
                    rotation: Quat::from_xyzw(
                        self.scratch[o + 4],
                        self.scratch[o + 5],
                        self.scratch[o + 6],
                        self.scratch[o + 3],
                    ),
                    // The mesh is a UNIT cube, so the scale IS the half-extents.
                    scale: *half,
                },
            );
        }
    }
}

/// The half-extents of the box with mass `mass` and diagonal inertia `inertia`.
///
/// Inverting the box inertia formula: for half-extents (a, b, c),
/// `Ixx = m/3 (b² + c²)` and cyclically, so `a² = 3/(2m) (Iyy + Izz - Ixx)`.
/// A degenerate or inconsistent inertia (a shell, a point mass, a hand-authored
/// tensor that is not physically realisable) can drive that negative, which is
/// why it clamps at zero instead of taking a NaN square root.
fn inertia_half_extents(mass: f64, inertia: [f64; 3]) -> Option<Vec3> {
    if mass <= 0.0 {
        return None;
    }
    let k = 3.0 / (2.0 * mass);
    let half = |x: f64, y: f64, z: f64| (k * (y + z - x)).max(0.0).sqrt() as f32;
    Some(Vec3::new(
        half(inertia[0], inertia[1], inertia[2]),
        half(inertia[1], inertia[2], inertia[0]),
        half(inertia[2], inertia[0], inertia[1]),
    ))
}

impl JointOverlay {
    fn apply(&mut self, r: &mut awsm_renderer::AwsmRenderer) {
        for (j, bar) in self.bars.iter().enumerate() {
            let Some(tk) = bar else { continue };
            let o = j * JOINT_STRIDE;
            let anchor = Vec3::new(self.scratch[o], self.scratch[o + 1], self.scratch[o + 2]);
            let axis = Vec3::new(
                self.scratch[o + 3],
                self.scratch[o + 4],
                self.scratch[o + 5],
            );
            let rotation = if axis.length_squared() > 1e-9 {
                Quat::from_rotation_arc(Vec3::Y, axis.normalize())
            } else {
                Quat::IDENTITY
            };
            let _ = r.transforms.set_local(
                *tk,
                awsm_renderer::transforms::Transform {
                    // Centred on the anchor, unlike a contact spike: a joint
                    // axis runs BOTH ways through its pivot.
                    translation: anchor,
                    rotation,
                    scale: Vec3::ONE,
                },
            );
        }
    }
}

impl ContactOverlay {
    /// Place the live spikes and hide the rest.
    fn apply(&mut self, r: &mut awsm_renderer::AwsmRenderer, ncon: usize) {
        let live = ncon.min(self.spikes.len());
        for (i, (tk, mk)) in self.spikes.iter().enumerate() {
            if i < live {
                let o = i * CONTACT_STRIDE;
                let pos = Vec3::new(self.scratch[o], self.scratch[o + 1], self.scratch[o + 2]);
                let normal = Vec3::new(
                    self.scratch[o + 3],
                    self.scratch[o + 4],
                    self.scratch[o + 5],
                );
                // Force sets the spike's LENGTH, which is what turns a
                // contact-point overlay into a force overlay: a foot bearing
                // weight grows a long spike, a grazing touch keeps a stub.
                // Clamped at both ends — a zero-force contact still has to be
                // visible, and a landing impact must not skewer the scene.
                let force = self.scratch[o + 6];
                let len = (SPIKE_MIN + (force / FORCE_REF) * (SPIKE_MAX - SPIKE_MIN))
                    .clamp(SPIKE_MIN, SPIKE_MAX);
                // The spike mesh is built along +Y (meshgen's cylinder axis), so
                // the contact normal only has to be rotated onto it. A
                // degenerate normal would make `from_rotation_arc` produce NaN.
                let rotation = if normal.length_squared() > 1e-9 {
                    Quat::from_rotation_arc(Vec3::Y, normal.normalize())
                } else {
                    Quat::IDENTITY
                };
                let _ = r.transforms.set_local(
                    *tk,
                    awsm_renderer::transforms::Transform {
                        // Pushed half a spike along the normal so the spike
                        // stands ON the contact rather than straddling it.
                        translation: pos + rotation * Vec3::new(0.0, len * 0.5, 0.0),
                        rotation,
                        // Y only: the RADIUS stays constant so a light contact
                        // is a short spike, not an invisible hair. Scaling all
                        // three axes shrank a 25 N contact to about one pixel
                        // wide, which reads as "the overlay is broken".
                        scale: Vec3::new(1.0, len / SPIKE_MAX, 1.0),
                    },
                );
            }
            // Only spikes crossing the show/hide boundary are touched: hiding a
            // mesh re-syncs the spatial index, so re-asserting it every frame
            // for 256 spikes would churn the BVH for nothing.
            let was = i < self.shown;
            let now = i < live;
            if was != now {
                let _ = r.set_mesh_hidden(*mk, !now);
            }
        }
        self.shown = live;
    }
}

/// Contact spike length at zero force, metres — a contact carrying nothing is
/// still a contact and still has to be visible.
const SPIKE_MIN: f32 = 0.06;
/// Contact spike length at [`FORCE_REF`] and above, metres. Long enough to read
/// at humanoid scale, short enough not to bury the model it is annotating.
const SPIKE_MAX: f32 = 0.35;
/// The normal force, newtons, that draws a full-length spike.
///
/// Calibrated against the humanoid rather than guessed: at rest its contacts
/// carry 20-175 N each, and landing spikes to ~3600 N. Referencing the whole
/// body weight (400 N) put every resting contact within a hair of the minimum
/// and made the overlay look dead, so this sits inside the resting band and
/// lets impacts saturate.
const FORCE_REF: f32 = 150.0;

/// Which debug overlays the page opted into (`?contacts&joints&inertia` —
/// parsed by MAIN, since this worker's own base is a `blob:` with no query).
/// One value instead of three bools so the toggles travel the
/// entry → run → onmessage → link_sim chain as a unit.
#[derive(Clone, Copy, Default)]
struct OverlayToggles {
    contacts: bool,
    joints: bool,
    inertia: bool,
}

/// Install this worker's post-load `onmessage`: the forwarded MuJoCo model
/// description (`kind: "mujoco-model"`), the sim's Stats reports (`kind:
/// "sim-stats"`), the camera state, and the settings and stats kit's
/// messages (canvas resizes, the anti-aliasing mode, which Stats groups are
/// open, the page's own measurements).
fn install_onmessage(
    model: Rc<RefCell<Option<JsValue>>>,
    camera: Rc<RefCell<OrbitCamera>>,
    canvas: web_sys::OffscreenCanvas,
    collector: Rc<RefCell<StatsCollector>>,
    sim_stats: Rc<RefCell<SimStats>>,
    pending_aa: Rc<Cell<Option<AaMode>>>,
) -> Result<(), JsValue> {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    let cb = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        // The model description is a raw JS object tagged `kind` (it carries
        // typed arrays — not serde territory); so are the sim's stats.
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        match kind.as_deref() {
            Some("mujoco-model") => {
                *model.borrow_mut() = Some(data);
                return;
            }
            Some("sim-stats") => {
                sim_stats
                    .borrow_mut()
                    .sim_report(&SimReport::from_js(&data));
                return;
            }
            _ => {}
        }
        // The settings and stats kit's messages (`{type: …}`).
        if let Some(message) = SettingsToWorker::from_js(&data) {
            match message {
                Ok(SettingsToWorker::AntiAlias { mode }) => pending_aa.set(Some(mode)),
                Ok(SettingsToWorker::Resize { width, height }) => {
                    if width > 0 && height > 0 {
                        canvas.set_width(width);
                        canvas.set_height(height);
                    }
                }
                Err(err) => tracing::warn!("render thread: settings message: {err}"),
            }
            return;
        }
        if let Some(message) = StatsToWorker::from_js(&data) {
            match message {
                Ok(message) => collector.borrow_mut().handle(message),
                Err(err) => tracing::warn!("render thread: stats message: {err}"),
            }
            return;
        }
        if let Ok(CameraMsg::State { t_ms, state }) =
            serde_wasm_bindgen::from_value::<CameraMsg>(data)
        {
            camera.borrow_mut().post(t_ms, state);
        }
    });
    scope.set_onmessage(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
    Ok(())
}

/// Bind the sim worker's pose block to the sim instance the loader resolved.
///
/// The only thing this reads from the model message is the shared buffer and
/// the geom count — everything about *what* the robot is came out of the
/// bundle. The fingerprint check is the important part: a sim running a
/// different model would produce poses that are individually plausible and
/// collectively nonsense, which is close to undiagnosable, so a mismatch
/// refuses to bind rather than driving the wrong robot.
fn link_sim(
    r: &mut awsm_renderer::AwsmRenderer,
    instance: awsm_renderer_scene_loader::mujoco::MujocoInstance,
    data: &JsValue,
    overlays: OverlayToggles,
) -> Result<SimLink, JsValue> {
    let get = |key: &str| js_sys::Reflect::get(data, &JsValue::from_str(key));
    let ngeom = get("ngeom")?.as_f64().unwrap_or(0.0) as usize;
    if ngeom != instance.geoms.len() {
        return Err(JsValue::from_str(&format!(
            "the sim is running a {ngeom}-geom model but the scene's instance \
             ({}) has {} geoms — re-export the scene from the same model file",
            instance.source.filename,
            instance.geoms.len(),
        )));
    }
    // A sidecar-derived fingerprint would be even better, but the wasm module
    // only has the compiled model in memory; the geom count plus the scene's
    // own recorded hash is what is available on this side of the SAB.
    let sab = get("sab")?;
    let njnt = get("njnt")?.as_f64().unwrap_or(0.0) as usize;
    let nbody = get("nbody")?.as_f64().unwrap_or(0.0) as usize;
    let layout = Layout { ngeom, njnt, nbody };
    let timestep = get("timestep")?.as_f64().unwrap_or(0.005);
    tracing::info!(
        "contact overlay: requested={} root_transform={:?}",
        overlays.contacts,
        instance.root_transform.is_some()
    );
    let contacts = match (overlays.contacts, instance.root_transform) {
        (true, Some(root)) => match build_contact_overlay(r, &sab, &layout, root) {
            Ok(o) => Some(o),
            Err(err) => {
                // A debug overlay is never worth failing the run for.
                tracing::warn!("contact overlay: {err:?} — continuing without it");
                None
            }
        },
        (true, None) => {
            tracing::warn!("contact overlay: the sim instance has no resolved root transform");
            None
        }
        (false, _) => None,
    };
    let joints = match overlays.joints {
        true => match instance
            .root_transform
            .ok_or_else(|| JsValue::from_str("the sim instance has no resolved root transform"))
            .and_then(|root| build_joint_overlay(r, data, &sab, &layout, root))
        {
            Ok(o) => Some(o),
            Err(err) => {
                tracing::warn!("joint overlay: {err:?} — continuing without it");
                None
            }
        },
        false => None,
    };
    let inertia = match overlays.inertia {
        true => match instance
            .root_transform
            .ok_or_else(|| JsValue::from_str("the sim instance has no resolved root transform"))
            .and_then(|root| build_inertia_overlay(r, data, &sab, &layout, root))
        {
            Ok(o) => Some(o),
            Err(err) => {
                tracing::warn!("inertia overlay: {err:?} — continuing without it");
                None
            }
        },
        false => None,
    };
    // The body channel. Driven only when the scene actually binds nodes to
    // bodies — i.e. when the model has a deformable — since a model without one
    // would pay a sink call per frame that moves nothing.
    let drive_bodies = instance.bodies.iter().any(|b| !b.is_empty());
    if drive_bodies && instance.bodies.len() != nbody {
        // Same reasoning as the geom guard above: a length mismatch here would
        // deform the flex with some other body's frame, which looks like a
        // plausible wrong shape rather than an error.
        return Err(JsValue::from_str(&format!(
            "the sim is running a {nbody}-body model but the scene's instance \
             ({}) has {} bodies — re-export the scene from the same model file",
            instance.source.filename,
            instance.bodies.len(),
        )));
    }

    tracing::info!(
        "body channel: bound={} of {} instance bodies, sim nbody={}",
        instance.bodies.iter().filter(|b| !b.is_empty()).count(),
        instance.bodies.len(),
        nbody,
    );

    let words = js_sys::Int32Array::new(&sab);
    let floats = js_sys::Float32Array::new(&sab);
    Ok(SimLink {
        instance,
        layout,
        dt_us: timestep * 1_000_000.0,
        words,
        floats,
        older: vec![0.0; layout.slot_words()],
        newer: vec![0.0; layout.slot_words()],
        geoms: vec![0.0; ngeom * POSE_STRIDE],
        bodies_blend: vec![0.0; nbody * POSE_STRIDE],
        last_frame_us: None,
        drive_bodies,
        contacts,
        joints,
        inertia,
    })
}

/// Mint one translucent box per massive body, sized from its mass and inertia.
fn build_inertia_overlay(
    r: &mut awsm_renderer::AwsmRenderer,
    data: &JsValue,
    sab: &JsValue,
    layout: &Layout,
    root: awsm_renderer::transforms::TransformKey,
) -> Result<InertiaOverlay, JsValue> {
    let get = |key: &str| js_sys::Reflect::get(data, &JsValue::from_str(key));
    let nbody = layout.nbody;
    let f64s = |key: &str| -> Vec<f64> {
        get(key)
            .ok()
            .map(|v| v.unchecked_into::<js_sys::Float64Array>().to_vec())
            .unwrap_or_default()
    };
    let body_mass = f64s("body_mass");
    let body_inertia = f64s("body_inertia");

    let mut pbr = awsm_renderer::materials::pbr::PbrMaterial::new(
        // Translucent: a solid box per body would hide the robot it is
        // describing, which defeats the whole point of the overlay.
        awsm_renderer::materials::MaterialAlphaMode::Blend,
        false,
    );
    pbr.base_color_factor = [0.2, 0.9, 0.3, 0.28];
    pbr.metallic_factor = 0.0;
    pbr.roughness_factor = 1.0;
    let material = r.materials.insert(
        awsm_renderer::materials::Material::Pbr(Box::new(pbr)),
        &r.textures,
        &r.dynamic_materials,
        &r.extras_pool,
    );

    let mut boxes = Vec::with_capacity(nbody);
    for b in 0..nbody {
        let mass = body_mass.get(b).copied().unwrap_or(0.0);
        let inertia = [
            body_inertia.get(b * 3).copied().unwrap_or(0.0),
            body_inertia.get(b * 3 + 1).copied().unwrap_or(0.0),
            body_inertia.get(b * 3 + 2).copied().unwrap_or(0.0),
        ];
        let Some(half) = inertia_half_extents(mass, inertia) else {
            boxes.push(None);
            continue;
        };
        let tk = r
            .transforms
            .insert(awsm_renderer::transforms::Transform::default(), Some(root));
        // A UNIT cube (half-extent 1 on each axis) so the node's scale is
        // literally the half-extents.
        let mesh = awsm_renderer_scene_loader::mesh_data_to_raw(awsm_renderer_meshgen::box_mesh(
            Vec3::splat(2.0),
        ));
        r.add_raw_mesh(mesh, tk, material)
            .map_err(|e| JsValue::from_str(&format!("inertia box {b}: {e}")))?;
        boxes.push(Some((tk, half)));
    }

    let start = layout.inertia();
    let region =
        js_sys::Float32Array::new(sab).subarray(start as u32, (start + nbody * POSE_STRIDE) as u32);
    Ok(InertiaOverlay {
        region,
        scratch: vec![0.0; nbody * POSE_STRIDE],
        boxes,
    })
}

/// Mint one bar per hinge/slide joint, under the sim instance's root.
fn build_joint_overlay(
    r: &mut awsm_renderer::AwsmRenderer,
    data: &JsValue,
    sab: &JsValue,
    layout: &Layout,
    root: awsm_renderer::transforms::TransformKey,
) -> Result<JointOverlay, JsValue> {
    let get = |key: &str| js_sys::Reflect::get(data, &JsValue::from_str(key));
    let njnt = layout.njnt;
    let jnt_type: Vec<i32> = get("jnt_type")
        .ok()
        .map(|v| v.unchecked_into::<js_sys::Int32Array>().to_vec())
        .unwrap_or_default();

    let mut pbr = awsm_renderer::materials::pbr::PbrMaterial::new(
        awsm_renderer::materials::MaterialAlphaMode::Opaque,
        false,
    );
    // Blue, to read as a different KIND of annotation from the red contacts.
    pbr.base_color_factor = [0.15, 0.35, 1.0, 1.0];
    pbr.emissive_factor = [0.05, 0.15, 0.6];
    pbr.metallic_factor = 0.0;
    pbr.roughness_factor = 1.0;
    let material = r.materials.insert(
        awsm_renderer::materials::Material::Pbr(Box::new(pbr)),
        &r.textures,
        &r.dynamic_materials,
        &r.extras_pool,
    );

    let mut bars = Vec::with_capacity(njnt);
    for j in 0..njnt {
        // mjtJoint 2 = slide, 3 = hinge. A free joint has six degrees of
        // freedom and a ball joint three; neither has one axis to draw.
        let drawable = matches!(jnt_type.get(j).copied(), Some(2) | Some(3));
        if !drawable {
            bars.push(None);
            continue;
        }
        let tk = r
            .transforms
            .insert(awsm_renderer::transforms::Transform::default(), Some(root));
        let mesh = awsm_renderer_scene_loader::mesh_data_to_raw(
            awsm_renderer_meshgen::cylinder_mesh(0.008, JOINT_AXIS_LEN, 8),
        );
        r.add_raw_mesh(mesh, tk, material)
            .map_err(|e| JsValue::from_str(&format!("joint bar {j}: {e}")))?;
        bars.push(Some(tk));
    }

    let start = layout.joints();
    let region =
        js_sys::Float32Array::new(sab).subarray(start as u32, (start + njnt * JOINT_STRIDE) as u32);
    Ok(JointOverlay {
        region,
        scratch: vec![0.0; njnt * JOINT_STRIDE],
        bars,
    })
}

/// Length of a joint-axis bar, metres. Deliberately short: at humanoid scale a
/// long bar through every hinge hides the robot it is annotating.
const JOINT_AXIS_LEN: f32 = 0.16;

/// Mint the contact-spike pool, parented under the sim instance's root so
/// contact positions can be written as raw MuJoCo world coordinates.
fn build_contact_overlay(
    r: &mut awsm_renderer::AwsmRenderer,
    sab: &JsValue,
    layout: &Layout,
    root: awsm_renderer::transforms::TransformKey,
) -> Result<ContactOverlay, JsValue> {
    let mut pbr = awsm_renderer::materials::pbr::PbrMaterial::new(
        awsm_renderer::materials::MaterialAlphaMode::Opaque,
        false,
    );
    // Unlit-bright red: an overlay should never be mistaken for scene geometry
    // that happens to be lit oddly.
    pbr.base_color_factor = [1.0, 0.1, 0.05, 1.0];
    pbr.emissive_factor = [0.8, 0.05, 0.0];
    pbr.metallic_factor = 0.0;
    pbr.roughness_factor = 1.0;
    let material = r.materials.insert(
        awsm_renderer::materials::Material::Pbr(Box::new(pbr)),
        &r.textures,
        &r.dynamic_materials,
        &r.extras_pool,
    );

    let mut spikes = Vec::with_capacity(MAX_CONTACTS);
    for i in 0..MAX_CONTACTS {
        let tk = r
            .transforms
            .insert(awsm_renderer::transforms::Transform::default(), Some(root));
        // meshgen's cylinder runs along +Y, which is exactly the axis
        // `ContactOverlay::apply` rotates the contact normal onto.
        let mesh = awsm_renderer_scene_loader::mesh_data_to_raw(
            awsm_renderer_meshgen::cylinder_mesh(0.012, SPIKE_MAX, 8),
        );
        let mk = r
            .add_raw_mesh(mesh, tk, material)
            .map_err(|e| JsValue::from_str(&format!("contact spike {i}: {e}")))?;
        // The pool starts empty: every spike is hidden until a step reports it.
        r.set_mesh_hidden(mk, true)
            .map_err(|e| JsValue::from_str(&format!("contact spike {i}: {e}")))?;
        spikes.push((tk, mk));
    }

    let start = layout.contacts();
    let region = js_sys::Float32Array::new(sab)
        .subarray(start as u32, (start + MAX_CONTACTS * CONTACT_STRIDE) as u32);
    Ok(ContactOverlay {
        region,
        scratch: vec![0.0; MAX_CONTACTS * CONTACT_STRIDE],
        spikes,
        shown: 0,
    })
}

async fn fetch_scene(url: &str) -> Result<awsm_renderer_scene::Scene, String> {
    let text = gloo_net::http::Request::get(url)
        .cache(web_sys::RequestCache::NoCache)
        .send()
        .await
        .map_err(|e| format!("fetch: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read: {e}"))?;
    awsm_renderer_scene::project_dir::scene_from_toml(&text).map_err(|e| format!("parse: {e}"))
}

/// Post a human-readable load-progress line to main (the loading screen).
fn post_progress(message: &str) {
    post_to_main(&RenderMsg::Progress {
        message: message.to_string(),
    });
}

/// Serialize a [`RenderMsg`] and post it to the main thread.
fn post_to_main(msg: &RenderMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    match serde_wasm_bindgen::to_value(msg) {
        Ok(v) => {
            let _ = scope.post_message(&v);
        }
        Err(e) => tracing::error!("render thread: serialize RenderMsg: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_lerps_positions_and_nlerps_the_short_way() {
        // Identity, then 90° about z; the second one written with the sign
        // flipped (the same rotation) to exercise the hemisphere check.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let b = [2.0, 4.0, 6.0, -h, 0.0, 0.0, -h];
        let mut out = [0.0; 7];
        blend_poses(&a, &b, 0.5, &mut out);
        assert_eq!(&out[..3], &[1.0, 2.0, 3.0]);
        let q = Quat::from_xyzw(out[4], out[5], out[6], out[3]);
        let expected = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert!(q.angle_between(expected) < 1e-4);
    }
}
