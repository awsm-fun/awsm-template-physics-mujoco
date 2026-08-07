//! Render thread — hosts `awsm-renderer` on an `OffscreenCanvas`, loads the
//! editor-exported `scene.toml` through the **player** path, then mirrors a
//! MuJoCo model into the scene and drives it from the sim worker's pose block.
//!
//! The MuJoCo side of this file is deliberately two small pieces — the shape
//! every external-sim integration reduces to:
//!
//! 1. **Mirror build** ([`build_mirror`]): the model description (geom types,
//!    sizes, colors — read once from the wasm module's `mjModel` and forwarded
//!    by main) becomes ordinary renderer content: one meshgen primitive + PBR
//!    material + transform node per geom, parented under a single *convention
//!    root* that performs the Z-up→Y-up frame change (and lifts the MuJoCo
//!    origin onto the scene's floor). The renderer never learns what MuJoCo is.
//! 2. **Pose apply** (in the RAF loop): each frame, copy the latest stable
//!    snapshot out of the sim worker's `SharedArrayBuffer` pose block
//!    (seqlock — see [`crate::protocol`]) and `set_local` every geom node.
//!    World poses go in as *locals* under the convention root, so no per-geom
//!    math happens here at all.
//!
//! Everything else is the stock player template: bundle load, orbit camera,
//! resize.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use crate::protocol::{CameraMsg, RenderMsg, ResizeMsg, POSE_HEADER, POSE_STRIDE};

/// The boxed RAF callback, self-referenced so the render loop can reschedule
/// itself each frame (and stays alive for the worker's lifetime).
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Lens + clip planes for the one camera this app has.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 400.0;

/// MuJoCo geom types (`mjtGeom`) — the ones this template mirrors.
const GEOM_PLANE: i32 = 0;
const GEOM_SPHERE: i32 = 2;
const GEOM_CAPSULE: i32 = 3;
const GEOM_ELLIPSOID: i32 = 4;
const GEOM_CYLINDER: i32 = 5;
const GEOM_BOX: i32 = 6;

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

/// The built MuJoCo mirror: one renderer node per mirrored geom plus the
/// typed-array views into the sim worker's pose block.
struct Mirror {
    /// Per-geom node (`None` = not mirrored: planes, collision-only groups).
    geom_nodes: Vec<Option<awsm_renderer::transforms::TransformKey>>,
    /// Per-geom constant node scale (unit-mesh geoms bake size here).
    geom_scales: Vec<Vec3>,
    /// i32 view over the whole pose block (seq at 0, steps at 1).
    header: js_sys::Int32Array,
    /// f32 view over just the pose span (`POSE_HEADER..`).
    poses: js_sys::Float32Array,
    /// Copy target for one stable snapshot.
    scratch: Vec<f32>,
    /// Last successfully-applied seq — skip work when nothing new published.
    last_seq: i32,
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
    let canvas_handle = canvas.clone();
    post_progress("render worker: requesting WebGPU device…");
    let gpu =
        navigator_gpu().ok_or_else(|| JsValue::from_str("render worker: no navigator.gpu"))?;
    let gpu_builder = AwsmRendererWebGpuBuilder::new_with_offscreen_canvas(gpu, canvas)
        .with_device_request_limits(DeviceRequestLimits::max_all());

    wasm_bindgen_futures::spawn_local(async move {
        if let Err(err) = run(gpu_builder, canvas_handle, origin).await {
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
    let bundle_base = format!("{}/bundle", origin.trim_end_matches('/'));
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

    // Where the MuJoCo world's floor (z=0) sits in the scene: on top of the
    // scene's largest static box collider (the tabletop), or world 0.
    let floor_top = scene_floor_top(&scene).unwrap_or(0.0);
    tracing::info!("render thread: MuJoCo origin will sit at y = {floor_top}");

    let assets = awsm_renderer_scene_loader::assets::HttpAssets::new(bundle_base.clone());
    let mut last_phase_line = String::new();
    awsm_renderer_scene_loader::load_scene_for_player(&mut renderer, &scene, &assets, |phase| {
        let line = phase.label();
        if line != last_phase_line && !line.contains("0/0") {
            post_progress(&line);
            last_phase_line = line;
        }
    })
    .await
    .map_err(|e| JsValue::from_str(&format!("load_scene_for_player: {e}")))?;

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
        look_at: Vec3::new(0.0, floor_top + 0.8, 0.0),
        radius: 6.0,
        yaw: 0.35,
        pitch: 0.35,
    }));
    #[allow(clippy::arc_with_non_send_sync)]
    let cell = Rc::new(RefCell::new(renderer));
    let mirror: Rc<RefCell<Option<Mirror>>> = Rc::new(RefCell::new(None));

    install_onmessage(
        cell.clone(),
        mirror.clone(),
        camera.clone(),
        canvas.clone(),
        floor_top,
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

/// Copy the latest stable pose snapshot out of the seqlock'd block and apply
/// it to every mirrored geom node. On a torn read (seq odd / changed under
/// us), keep last frame's poses — the next frame will catch up.
fn apply_poses(r: &mut awsm_renderer::AwsmRenderer, m: &mut Mirror) {
    let seq0 = js_sys::Atomics::load(&m.header, 0).unwrap_or(0);
    if seq0 == m.last_seq || seq0 % 2 != 0 || seq0 == 0 {
        return;
    }
    m.poses.copy_to(&mut m.scratch[..]);
    let seq1 = js_sys::Atomics::load(&m.header, 0).unwrap_or(-1);
    if seq0 != seq1 {
        return; // torn — writer was mid-publish
    }
    m.last_seq = seq0;
    for (g, node) in m.geom_nodes.iter().enumerate() {
        let Some(tk) = node else { continue };
        let s = &m.scratch[g * POSE_STRIDE..(g + 1) * POSE_STRIDE];
        let transform = awsm_renderer::transforms::Transform {
            translation: Vec3::new(s[0], s[1], s[2]),
            rotation: Quat::from_xyzw(s[3], s[4], s[5], s[6]).normalize(),
            scale: m.geom_scales[g],
        };
        if let Err(err) = r.transforms.set_local(*tk, transform) {
            tracing::warn!("set_local geom {g}: {err}");
        }
    }
}

/// Install this worker's post-load `onmessage`: the forwarded MuJoCo model
/// description (`kind: "mujoco-model"`), canvas resizes, and camera gestures.
fn install_onmessage(
    cell: Rc<RefCell<awsm_renderer::AwsmRenderer>>,
    mirror: Rc<RefCell<Option<Mirror>>>,
    camera: Rc<RefCell<OrbitCamera>>,
    canvas: web_sys::OffscreenCanvas,
    floor_top: f32,
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
            let mut r = cell.borrow_mut();
            match build_mirror(&mut r, &data, floor_top) {
                Ok(m) => {
                    let mirrored = m.geom_nodes.iter().flatten().count();
                    tracing::info!(
                        "render thread: MuJoCo mirror built — {mirrored}/{} geoms",
                        m.geom_nodes.len()
                    );
                    post_progress(&format!("MuJoCo mirror built ({mirrored} geoms) — running"));
                    *mirror.borrow_mut() = Some(m);
                }
                Err(err) => {
                    tracing::error!("build_mirror: {err:?}");
                    post_to_main(&RenderMsg::Error {
                        message: format!("build mirror: {err:?}"),
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

/// Build the renderer-side mirror of the model: one node + primitive mesh +
/// PBR material per mirrored geom, under a convention root that maps MuJoCo's
/// Z-up right-handed world into the scene's Y-up world (rotate −90° about X)
/// and lifts z=0 onto the scene floor.
fn build_mirror(
    r: &mut awsm_renderer::AwsmRenderer,
    data: &JsValue,
    floor_top: f32,
) -> Result<Mirror, JsValue> {
    let get = |key: &str| js_sys::Reflect::get(data, &JsValue::from_str(key));
    let ngeom = get("ngeom")?.as_f64().unwrap_or(0.0) as usize;
    if ngeom == 0 {
        return Err(JsValue::from_str("model has 0 geoms"));
    }
    let geom_type: Vec<i32> = get("geom_type")?.unchecked_into::<js_sys::Int32Array>().to_vec();
    let geom_group: Vec<i32> = get("geom_group")?
        .unchecked_into::<js_sys::Int32Array>()
        .to_vec();
    let geom_matid: Vec<i32> = get("geom_matid")?
        .unchecked_into::<js_sys::Int32Array>()
        .to_vec();
    let geom_size: Vec<f64> = get("geom_size")?
        .unchecked_into::<js_sys::Float64Array>()
        .to_vec();
    let geom_rgba: Vec<f32> = get("geom_rgba")?
        .unchecked_into::<js_sys::Float32Array>()
        .to_vec();
    let mat_rgba: Vec<f32> = get("mat_rgba")?
        .unchecked_into::<js_sys::Float32Array>()
        .to_vec();
    let sab: js_sys::SharedArrayBuffer = get("sab")?.unchecked_into();

    // The convention root: MuJoCo is Z-up right-handed; the scene is Y-up
    // right-handed. −90° about X maps +Z→+Y (up stays up); every geom pose is
    // then applied VERBATIM as a local transform under this node.
    let root = r.transforms.insert(
        awsm_renderer::transforms::Transform {
            translation: Vec3::new(0.0, floor_top, 0.0),
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            scale: Vec3::ONE,
        },
        None,
    );

    let mut geom_nodes = Vec::with_capacity(ngeom);
    let mut geom_scales = Vec::with_capacity(ngeom);
    let mut material_cache: std::collections::HashMap<
        [u32; 4],
        awsm_renderer::materials::MaterialKey,
    > = Default::default();

    for g in 0..ngeom {
        let ty = geom_type.get(g).copied().unwrap_or(-1);
        // MuJoCo convention: groups 0–2 are visible by default; higher groups
        // are collision/debug-only. Planes are the environment's job — the
        // scene bundle already has a floor.
        let group_visible = geom_group.get(g).map(|gr| (0..=2).contains(gr)).unwrap_or(true);
        let size = [
            geom_size.get(g * 3).copied().unwrap_or(0.1) as f32,
            geom_size.get(g * 3 + 1).copied().unwrap_or(0.1) as f32,
            geom_size.get(g * 3 + 2).copied().unwrap_or(0.1) as f32,
        ];
        let (mesh, scale) = match ty {
            _ if !group_visible => {
                geom_nodes.push(None);
                geom_scales.push(Vec3::ONE);
                continue;
            }
            GEOM_PLANE => {
                geom_nodes.push(None);
                geom_scales.push(Vec3::ONE);
                continue;
            }
            // size = [radius, _, _]
            GEOM_SPHERE => (
                awsm_renderer_meshgen::sphere_mesh(size[0], 32, 16),
                Vec3::ONE,
            ),
            // size = [radius, half-length, _]; MuJoCo capsules run along local Z.
            GEOM_CAPSULE => (capsule_mesh_z(size[0], size[1]), Vec3::ONE),
            // size = [rx, ry, rz]: a unit sphere scaled per-axis.
            GEOM_ELLIPSOID => (
                awsm_renderer_meshgen::sphere_mesh(1.0, 32, 16),
                Vec3::new(size[0], size[1], size[2]),
            ),
            // size = [radius, half-length, _]; along local Z (meshgen's is Y).
            GEOM_CYLINDER => (
                rotate_y_to_z(awsm_renderer_meshgen::cylinder_mesh(
                    size[0],
                    size[1] * 2.0,
                    32,
                )),
                Vec3::ONE,
            ),
            // size = half-extents.
            GEOM_BOX => (
                awsm_renderer_meshgen::box_mesh(Vec3::new(
                    size[0] * 2.0,
                    size[1] * 2.0,
                    size[2] * 2.0,
                )),
                Vec3::ONE,
            ),
            other => {
                tracing::warn!("geom {g}: unsupported type {other} — skipped");
                geom_nodes.push(None);
                geom_scales.push(Vec3::ONE);
                continue;
            }
        };

        // Color: the geom's material rgba when one is assigned, else its own.
        let rgba = {
            let matid = geom_matid.get(g).copied().unwrap_or(-1);
            let src = if matid >= 0 {
                &mat_rgba[(matid as usize * 4)..(matid as usize * 4 + 4)]
            } else {
                &geom_rgba[g * 4..g * 4 + 4]
            };
            [src[0], src[1], src[2], src[3]]
        };
        let mat_key = *material_cache
            .entry(rgba.map(f32::to_bits))
            .or_insert_with(|| {
                let mut pbr = awsm_renderer::materials::pbr::PbrMaterial::new(
                    awsm_renderer::materials::MaterialAlphaMode::Opaque,
                    false,
                );
                pbr.base_color_factor = rgba;
                pbr.metallic_factor = 0.0;
                pbr.roughness_factor = 0.7;
                r.materials.insert(
                    awsm_renderer::materials::Material::Pbr(Box::new(pbr)),
                    &r.textures,
                    &r.dynamic_materials,
                    &r.extras_pool,
                )
            });

        let tk = r.transforms.insert(
            awsm_renderer::transforms::Transform {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale,
            },
            Some(root),
        );
        r.add_raw_mesh(awsm_renderer_scene_loader::mesh_data_to_raw(mesh), tk, mat_key)
            .map_err(|e| JsValue::from_str(&format!("add_raw_mesh geom {g}: {e}")))?;
        geom_nodes.push(Some(tk));
        geom_scales.push(scale);
    }

    let header = js_sys::Int32Array::new(&sab);
    let poses = js_sys::Float32Array::new(&sab)
        .subarray(POSE_HEADER as u32, (POSE_HEADER + ngeom * POSE_STRIDE) as u32);
    Ok(Mirror {
        geom_nodes,
        geom_scales,
        header,
        poses,
        scratch: vec![0.0; ngeom * POSE_STRIDE],
        last_seq: 0,
    })
}

/// A capsule with its long axis along **Z** (MuJoCo's convention): two
/// hemisphere caps of `radius` around a cylindrical wall of half-length
/// `half_len`. Built as a lat-long sphere split at the equator with the two
/// halves pushed apart — the equator pair forms the wall, and the sphere
/// normals at the split are exactly the wall's radial normals. Same layout +
/// winding as `meshgen::sphere_mesh` (built along Y, then rotated Y→Z).
fn capsule_mesh_z(radius: f32, half_len: f32) -> awsm_renderer_meshgen::MeshData {
    use std::f32::consts::{PI, TAU};
    let radial = 24usize; // longitude segments
    let half_rings = 8usize; // latitude rings per hemisphere
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    // Rings: top hemisphere (theta 0..=π/2, offset +half_len along Y), then
    // bottom hemisphere (theta π/2..=π, offset −half_len). The consecutive
    // equator rings (same theta, different offset) become the cylinder wall.
    let mut ring = |theta: f32, offset: f32, v: f32| {
        let (sin_t, cos_t) = theta.sin_cos();
        for lon in 0..=radial {
            let u = lon as f32 / radial as f32;
            let phi = u * TAU;
            let (sin_p, cos_p) = phi.sin_cos();
            let n = [sin_t * cos_p, cos_t, sin_t * sin_p];
            positions.push([n[0] * radius, n[1] * radius + offset, n[2] * radius]);
            normals.push(n);
            uvs.push([u, v]);
        }
    };
    let total_rings = half_rings * 2 + 1; // +1: the duplicated equator
    let mut vi = 0.0;
    for i in 0..=half_rings {
        ring(
            PI * 0.5 * i as f32 / half_rings as f32,
            half_len,
            vi / total_rings as f32,
        );
        vi += 1.0;
    }
    for i in 0..=half_rings {
        ring(
            PI * 0.5 + PI * 0.5 * i as f32 / half_rings as f32,
            -half_len,
            vi / total_rings as f32,
        );
        vi += 1.0;
    }

    let stride = radial + 1;
    for lat in 0..(total_rings) {
        for lon in 0..radial {
            let a = (lat * stride + lon) as u32;
            let b = (lat * stride + lon + 1) as u32;
            let c = ((lat + 1) * stride + lon + 1) as u32;
            let d = ((lat + 1) * stride + lon) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    rotate_y_to_z(awsm_renderer_meshgen::MeshData {
        positions,
        normals: Some(normals),
        uvs: vec![uvs],
        colors: None,
        indices,
    })
}

/// Rotate a mesh +90° about X so its +Y axis becomes +Z (proper rotation —
/// winding and normals stay valid): `(x, y, z) → (x, −z, y)`.
fn rotate_y_to_z(mut mesh: awsm_renderer_meshgen::MeshData) -> awsm_renderer_meshgen::MeshData {
    for p in &mut mesh.positions {
        *p = [p[0], -p[2], p[1]];
    }
    if let Some(normals) = &mut mesh.normals {
        for n in normals {
            *n = [n[0], -n[2], n[1]];
        }
    }
    mesh
}

/// The scene's floor height: the top face of the largest (by horizontal area)
/// static box collider — the tabletop in the stock bundle. `None` when the
/// scene has no box colliders.
fn scene_floor_top(scene: &awsm_renderer_scene::Scene) -> Option<f32> {
    let mut best: Option<(f32, f32)> = None; // (area, top_y)
    for node in &scene.nodes {
        floor_walk(node, Mat4::IDENTITY, &mut best);
    }
    best.map(|(_, top)| top)
}

fn floor_walk(
    node: &awsm_renderer_scene::EditorNode,
    parent_world: Mat4,
    best: &mut Option<(f32, f32)>,
) {
    use awsm_renderer_scene::{ColliderShape, NodeKind};
    let t = &node.transform;
    let local = Mat4::from_scale_rotation_translation(
        Vec3::from_array(t.scale),
        Quat::from_array(t.rotation),
        Vec3::from_array(t.translation),
    );
    let world = parent_world * local;
    if let NodeKind::Collider(ColliderShape::Box { half_extents }) = &node.kind {
        let (scale, _, tr) = world.to_scale_rotation_translation();
        let area = (half_extents[0] * scale.x).abs() * (half_extents[2] * scale.z).abs();
        let top = tr.y + (half_extents[1] * scale.y).abs();
        if best.map(|(a, _)| area > a).unwrap_or(true) {
            *best = Some((area, top));
        }
    }
    for child in &node.children {
        floor_walk(child, world, best);
    }
}

/// Fetch + deserialize a same-origin player-bundle `scene.toml`.
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
