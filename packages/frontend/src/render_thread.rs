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
//! 2. each frame, copies the latest stable snapshot out of the sim worker's
//!    `SharedArrayBuffer` pose block (seqlock — see [`crate::protocol`]) and
//!    hands it to `scene_loader::mujoco::apply_geom_poses`.
//!
//! The pose block is laid out as a **stream frame** in the documented
//! convention (7 f32 per geom, `[px,py,pz,qw,qx,qy,qz]`, geom-id indexed), so
//! nothing is reshaped on either side of the SAB — this worker could dump its
//! block verbatim as a capture file and the editor would bake it.
//!
//! Everything else is the stock player template: bundle load, orbit camera,
//! resize.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use crate::protocol::{
    body_offset, contact_offset, inertia_offset, joint_offset, CameraMsg, RenderMsg, ResizeMsg,
    CONTACT_STRIDE, JOINT_STRIDE, MAX_CONTACTS, POSE_HEADER, POSE_STRIDE,
};

/// The boxed RAF callback, self-referenced so the render loop can reschedule
/// itself each frame (and stays alive for the worker's lifetime).
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Lens + clip planes for the one camera this app has.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 400.0;

/// A minimal mouse-driven orbit camera: drag to orbit, wheel to dolly. State
/// lives here; main feeds it gesture deltas via [`CameraMsg`].
struct OrbitCamera {
    look_at: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitCamera {
    const SENSITIVITY: f32 = 0.008;
    const PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const PITCH_MIN: f32 = 0.05;
    const MIN_RADIUS: f32 = 1.5;
    const MAX_RADIUS: f32 = 30.0;

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * Self::SENSITIVITY;
        self.pitch = (self.pitch - dy * Self::SENSITIVITY).clamp(Self::PITCH_MIN, Self::PITCH_MAX);
    }

    fn zoom(&mut self, dy: f32) {
        self.radius = (self.radius * (1.0 + dy * 0.001)).clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }

    fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.look_at
            + Vec3::new(
                self.radius * cp * sy,
                self.radius * sp,
                self.radius * cp * cy,
            )
    }
}

/// The live link between the sim worker's pose block and the scene's sim
/// instance. No geometry here — the instance came out of the bundle.
struct SimLink {
    /// The loader-resolved instance: geom_id→transform + the model fingerprint.
    instance: awsm_renderer_scene_loader::mujoco::MujocoInstance,
    /// i32 view over the whole pose block (seq at 0, steps at 1).
    header: js_sys::Int32Array,
    /// f32 view over just the pose span (`POSE_HEADER..`).
    poses: js_sys::Float32Array,
    /// Copy target for one stable snapshot.
    scratch: Vec<f32>,
    /// Last successfully-applied seq — skip work when nothing new published.
    last_seq: i32,
    /// The contact-point debug overlay, when it was built.
    contacts: Option<ContactOverlay>,
    /// The joint-axis debug overlay, when it was built.
    joints: Option<JointOverlay>,
    /// The inertia-box debug overlay, when it was built.
    inertia: Option<InertiaOverlay>,
    /// Body world poses + their copy target — the channel that deforms flexes.
    ///
    /// `None` when the scene's instance binds no node to any body, which is the
    /// case for every model without a deformable (the humanoid included). The
    /// sim publishes the region regardless; nobody is listening.
    bodies: Option<(js_sys::Float32Array, Vec<f32>)>,
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

/// Worker entry: unpack the transferred `OffscreenCanvas` + page origin, build
/// the WebGPU device, and kick off the async load+render.
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
            format!("{base}/workers/basis-worker.js"),
            format!("{base}/vendor/basis/basis_transcoder.js"),
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
        if let Err(err) = run(gpu_builder, canvas_handle, origin, bundle_dir, overlays).await {
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
) -> Result<(), JsValue> {
    use awsm_renderer::camera::CameraParams;
    use awsm_renderer::AwsmRendererBuilder;

    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(awsm_renderer::anti_alias::AntiAliasing {
            msaa_sample_count: Some(4),
            smaa: false,
            mipmap: true,
        })
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
    post_progress("render worker: WebGPU device + renderer ready");

    // ── Warm-up + scene fetch, CONCURRENTLY ─────────────────────────────────
    let bundle_base = format!("{}/{bundle_dir}", origin.trim_end_matches('/'));
    let scene_url = format!("{bundle_base}/scene.toml");
    post_progress("compiling core render pipelines + fetching scene… (first visit can take a while — cached after)");
    let (compiled, scene) =
        futures::join!(renderer.ensure_config_pipelines(), fetch_scene(&scene_url));
    let compiled =
        compiled.map_err(|e| JsValue::from_str(&format!("ensure_config_pipelines: {e}")))?;
    let scene = scene.map_err(|e| JsValue::from_str(&format!("load scene {scene_url}: {e}")))?;
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
        radius: 6.0,
        yaw: 0.35,
        pitch: 0.35,
    }));
    #[allow(clippy::arc_with_non_send_sync)]
    let cell = Rc::new(RefCell::new(renderer));
    let mirror: Rc<RefCell<Option<SimLink>>> = Rc::new(RefCell::new(None));
    // The instance the loader resolved, held until the sim worker announces its
    // model. There is exactly one in this template's scene; a multi-robot world
    // would key these by fingerprint instead.
    let pending = Rc::new(RefCell::new(sim_instance));

    install_onmessage(
        cell.clone(),
        mirror.clone(),
        pending,
        camera.clone(),
        canvas.clone(),
        overlays,
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

    *raf_init.borrow_mut() = Some(Closure::new(move |_vsync_ms: f64| {
        let mut r = cell.borrow_mut();

        // Apply the newest stable sim snapshot (if the mirror exists yet and
        // the sim worker has published something new).
        if let Some(m) = mirror.borrow_mut().as_mut() {
            apply_poses(&mut r, m);
        }

        let cam = camera.borrow();
        let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
        let mut params =
            CameraParams::perspective(CAMERA_FOV_Y_DEG.to_radians(), CAMERA_NEAR, CAMERA_FAR);
        params.focus_distance = cam.radius;
        let _ = r.set_camera(view, params);
        drop(cam);

        r.update_transforms();
        if let Err(err) = r.render(None) {
            tracing::warn!("render thread: render error: {err}");
        }
        drop(r);

        frame_count = frame_count.wrapping_add(1);
        if frame_count == 3 {
            post_to_main(&RenderMsg::Ready);
        }
        if let Some(cb) = raf_run.borrow().as_ref() {
            let _ = awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }));
    if let Some(cb) = raf_init.borrow().as_ref() {
        awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref())?;
    }
    std::mem::forget(raf);
    Ok(())
}

/// Copy the latest stable pose snapshot out of the seqlock'd block and hand it
/// to the renderer's pose sink. On a torn read (seq odd / changed under us),
/// keep last frame's poses — the next frame catches up.
///
/// This is the entire per-frame cost of the integration: a seqlock read and one
/// sink call. Every geom-to-node decision was made at import time.
fn apply_poses(r: &mut awsm_renderer::AwsmRenderer, m: &mut SimLink) {
    let seq0 = js_sys::Atomics::load(&m.header, 0).unwrap_or(0);
    if seq0 == m.last_seq || seq0 % 2 != 0 || seq0 == 0 {
        return;
    }
    m.poses.copy_to(&mut m.scratch[..]);
    let seq1 = js_sys::Atomics::load(&m.header, 0).unwrap_or(-1);
    if seq0 != seq1 {
        return; // torn — writer was mid-publish
    }
    let ncon = js_sys::Atomics::load(&m.header, 3).unwrap_or(0).max(0) as usize;
    if let Some(overlay) = &mut m.contacts {
        overlay.region.copy_to(&mut overlay.scratch[..]);
    }
    if let Some(overlay) = &mut m.joints {
        overlay.region.copy_to(&mut overlay.scratch[..]);
    }
    if let Some(overlay) = &mut m.inertia {
        overlay.region.copy_to(&mut overlay.scratch[..]);
    }
    if let Some((region, scratch)) = &mut m.bodies {
        region.copy_to(&mut scratch[..]);
    }
    let seq1 = js_sys::Atomics::load(&m.header, 0).unwrap_or(-1);
    if seq0 != seq1 {
        return; // torn — writer was mid-publish
    }
    m.last_seq = seq0;
    if let Err(err) =
        awsm_renderer_scene_loader::mujoco::apply_geom_poses(r, &m.instance, &m.scratch)
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
    if let Some((_, scratch)) = &m.bodies {
        if let Err(err) =
            awsm_renderer_scene_loader::mujoco::apply_body_poses(r, &m.instance, scratch)
        {
            tracing::warn!("body pose sink: {err}");
        }
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
/// description (`kind: "mujoco-model"`), canvas resizes, and camera gestures.
fn install_onmessage(
    // The pose path never touches the renderer here — that is the point of the
    // migration — but the DEBUG overlay has to mint its own nodes, so the
    // handle is back for that one job.
    cell: Rc<RefCell<awsm_renderer::AwsmRenderer>>,
    mirror: Rc<RefCell<Option<SimLink>>>,
    pending: Rc<RefCell<Option<awsm_renderer_scene_loader::mujoco::MujocoInstance>>>,
    camera: Rc<RefCell<OrbitCamera>>,
    canvas: web_sys::OffscreenCanvas,
    overlays: OverlayToggles,
) -> Result<(), JsValue> {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    let cb = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        // The model description is a raw JS object tagged `kind` (it carries
        // typed arrays — not serde territory).
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() == Some("mujoco-model") {
            let Some(instance) = pending.borrow_mut().take() else {
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
            let mut r = cell.borrow_mut();
            match link_sim(&mut r, instance, &data, overlays) {
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
            return;
        }
        if let Ok(ResizeMsg::Canvas { width, height }) =
            serde_wasm_bindgen::from_value::<ResizeMsg>(data.clone())
        {
            if width > 0 && height > 0 {
                canvas.set_width(width);
                canvas.set_height(height);
            }
            return;
        }
        match serde_wasm_bindgen::from_value::<CameraMsg>(data) {
            Ok(CameraMsg::Orbit { dx, dy }) => camera.borrow_mut().orbit(dx, dy),
            Ok(CameraMsg::Zoom { dy }) => camera.borrow_mut().zoom(dy),
            Err(_) => {}
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
    let header = js_sys::Int32Array::new(&sab);
    let poses = js_sys::Float32Array::new(&sab).subarray(
        POSE_HEADER as u32,
        (POSE_HEADER + ngeom * POSE_STRIDE) as u32,
    );
    tracing::info!(
        "contact overlay: requested={} root_transform={:?}",
        overlays.contacts,
        instance.root_transform.is_some()
    );
    let contacts = match (overlays.contacts, instance.root_transform) {
        (true, Some(root)) => match build_contact_overlay(r, &sab, ngeom, root) {
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
            .and_then(|root| build_joint_overlay(r, data, &sab, ngeom, root))
        {
            Ok(o) => Some(o),
            Err(err) => {
                tracing::warn!("joint overlay: {err:?} — continuing without it");
                None
            }
        },
        false => None,
    };
    // The inertia region sits after the joint region, so its offset needs the
    // joint count even when the joint overlay itself was not built.
    let njnt = get("njnt")?.as_f64().unwrap_or(0.0) as usize;
    let inertia = match overlays.inertia {
        true => match instance
            .root_transform
            .ok_or_else(|| JsValue::from_str("the sim instance has no resolved root transform"))
            .and_then(|root| build_inertia_overlay(r, data, &sab, ngeom, njnt, root))
        {
            Ok(o) => Some(o),
            Err(err) => {
                tracing::warn!("inertia overlay: {err:?} — continuing without it");
                None
            }
        },
        false => None,
    };
    // The body channel. Built only when the scene actually binds nodes to
    // bodies — i.e. when the model has a deformable — since a model without one
    // would pay a copy per frame for a sink call that moves nothing.
    let nbody = get("nbody")?.as_f64().unwrap_or(0.0) as usize;
    let binds_bodies = instance.bodies.iter().any(|b| !b.is_empty());
    let bodies = if !binds_bodies {
        None
    } else if instance.bodies.len() != nbody {
        // Same reasoning as the geom guard above: a length mismatch here would
        // deform the flex with some other body's frame, which looks like a
        // plausible wrong shape rather than an error.
        return Err(JsValue::from_str(&format!(
            "the sim is running a {nbody}-body model but the scene's instance \
             ({}) has {} bodies — re-export the scene from the same model file",
            instance.source.filename,
            instance.bodies.len(),
        )));
    } else {
        let offset = body_offset(ngeom, njnt, nbody);
        Some((
            js_sys::Float32Array::new(&sab)
                .subarray(offset as u32, (offset + nbody * POSE_STRIDE) as u32),
            vec![0.0; nbody * POSE_STRIDE],
        ))
    };

    tracing::info!(
        "body channel: bound={} of {} instance bodies, sim nbody={}",
        instance.bodies.iter().filter(|b| !b.is_empty()).count(),
        instance.bodies.len(),
        nbody,
    );

    Ok(SimLink {
        instance,
        header,
        poses,
        scratch: vec![0.0; ngeom * POSE_STRIDE],
        last_seq: 0,
        contacts,
        joints,
        inertia,
        bodies,
    })
}

/// Mint one translucent box per massive body, sized from its mass and inertia.
fn build_inertia_overlay(
    r: &mut awsm_renderer::AwsmRenderer,
    data: &JsValue,
    sab: &JsValue,
    ngeom: usize,
    njnt: usize,
    root: awsm_renderer::transforms::TransformKey,
) -> Result<InertiaOverlay, JsValue> {
    let get = |key: &str| js_sys::Reflect::get(data, &JsValue::from_str(key));
    let nbody = get("nbody")?.as_f64().unwrap_or(0.0) as usize;
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

    let start = inertia_offset(ngeom, njnt);
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
    ngeom: usize,
    root: awsm_renderer::transforms::TransformKey,
) -> Result<JointOverlay, JsValue> {
    let get = |key: &str| js_sys::Reflect::get(data, &JsValue::from_str(key));
    let njnt = get("njnt")?.as_f64().unwrap_or(0.0) as usize;
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

    let start = joint_offset(ngeom);
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
    ngeom: usize,
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

    let start = contact_offset(ngeom);
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
