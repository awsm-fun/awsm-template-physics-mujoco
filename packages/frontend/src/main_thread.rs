//! Main thread — owns the DOM (built with Dominator) and orchestrates the two
//! workers.
//!
//! Flow:
//! 1. Build the canvas + HUD, size the canvas to native resolution, transfer
//!    it to an `OffscreenCanvas`, and spawn the **render** worker (a Rust
//!    thread sharing this wasm module + memory).
//! 2. Spawn the **MuJoCo** worker (`web/workers/mujoco-worker.js`, a plain JS
//!    module worker hosting the official `@mujoco/mujoco` Emscripten build —
//!    its own wasm module, its own heap).
//! 3. The MuJoCo worker loads the model and posts back a one-shot **model
//!    description** (`kind: "model"`: geom types/sizes/colors as typed
//!    arrays). Main allocates the `SharedArrayBuffer` pose block for it.
//! 4. When BOTH the model description and the render worker's `SceneReady`
//!    have arrived, main sends the pose block to the sim worker (`"start"` —
//!    it begins stepping + publishing) and forwards the model description +
//!    block to the render worker (`"mujoco-model"` — it builds the mirror).
//! 5. After that main is out of the loop: poses flow sim → render through
//!    shared memory only. Main just relays resize + camera gestures.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::Mutable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

use crate::bootstrap::spawn_shared_worker_transfer;
use crate::protocol::{pose_block_bytes, CameraMsg, RenderMsg, ResizeMsg};

/// Build + mount the DOM, then start the worker pipeline.
pub fn start() -> Result<(), JsValue> {
    let status = Mutable::new("booting…".to_string());

    let app = html!("div", {
        .child(html!("canvas" => HtmlCanvasElement, {
            .class("canvas")
            .after_inserted(clone!(status => move |canvas| {
                if let Err(e) = setup(canvas, status.clone()) {
                    status.set(format!("setup error: {e:?}"));
                    tracing::error!("main thread setup: {e:?}");
                }
            }))
        }))
        .child(html!("div", {
            .class("hud")
            .text("physics-mujoco — awsm-renderer scene + MuJoCo wasm sim\ndrag: orbit · wheel: zoom\n")
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
        }))
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

fn setup(canvas: HtmlCanvasElement, status: Mutable<String>) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;

    // Native-resolution backing store (CSS size × devicePixelRatio); must be
    // sized BEFORE the transfer.
    let dpr = window.device_pixel_ratio().max(1.0);
    let css_w = canvas.client_width().max(1) as f64;
    let css_h = canvas.client_height().max(1) as f64;
    canvas.set_width((css_w * dpr).round() as u32);
    canvas.set_height((css_h * dpr).round() as u32);
    let offscreen = canvas.transfer_control_to_offscreen()?;

    let base = page_base(&window);
    let app_base = app_base(&window);

    // ── Render worker ───────────────────────────────────────────────────────
    let payload = js_sys::Object::new();
    set(&payload, "canvas", &offscreen);
    set(&payload, "origin", &JsValue::from_str(&base));
    set(&payload, "app_base", &JsValue::from_str(&app_base));
    // Debug overlays are opt-in via the page URL, and only the main thread can
    // read it — the render worker's own base is a `blob:`.
    let search = window.location().search().unwrap_or_default();
    set(
        &payload,
        "contacts",
        &JsValue::from_bool(search.contains("contacts")),
    );
    set(
        &payload,
        "joints",
        &JsValue::from_bool(search.contains("joints")),
    );
    set(
        &payload,
        "inertia",
        &JsValue::from_bool(search.contains("inertia")),
    );
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    // The startup rendezvous (step 4 above): the model description and
    // `SceneReady` arrive in either order; whoever comes second triggers the
    // handoff.
    let model_msg: Rc<RefCell<Option<JsValue>>> = Rc::new(RefCell::new(None));
    let scene_ready = Rc::new(Cell::new(false));
    let render_ref: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));
    let mujoco_ref: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));

    let on_render_msg = Closure::<dyn FnMut(MessageEvent)>::new(clone!(
        status, model_msg, scene_ready, render_ref, mujoco_ref => move |e: MessageEvent| {
            match serde_wasm_bindgen::from_value::<RenderMsg>(e.data()) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::SceneReady) => {
                    scene_ready.set(true);
                    loading_log("scene ready — waiting on the MuJoCo model…");
                    maybe_start(&model_msg, &scene_ready, &render_ref, &mujoco_ref);
                }
                Ok(RenderMsg::Ready) => {
                    loading_log("first frames rendered — ready");
                    loading_done();
                    status.set("running — drag orbits, wheel zooms".into());
                }
                Ok(RenderMsg::Error { message }) => {
                    loading_log(&format!("ERROR: {message}"));
                    status.set(format!("render error: {message}"));
                }
                Err(_) => { /* not a RenderMsg (init-error blob) — ignore */ }
            }
        }
    ));

    loading_log("spawning render worker…");
    let render = spawn_shared_worker_transfer(
        "render",
        &payload,
        &transfer,
        on_render_msg.as_ref().unchecked_ref(),
    )?;
    on_render_msg.forget();
    *render_ref.borrow_mut() = Some(render.clone());

    // ── MuJoCo worker (plain JS module worker — its own wasm module) ────────
    loading_log("spawning MuJoCo worker…");
    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let ab = app_base.trim_end_matches('/');
    let mujoco = Worker::new_with_options(&format!("{ab}/workers/mujoco-worker.js"), &opts)?;
    let on_mujoco_msg = Closure::<dyn FnMut(MessageEvent)>::new(clone!(
        status, model_msg, scene_ready, render_ref, mujoco_ref => move |e: MessageEvent| {
            let data = e.data();
            let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            match kind.as_str() {
                "progress" => {
                    if let Ok(m) = js_sys::Reflect::get(&data, &JsValue::from_str("message")) {
                        loading_log(&m.as_string().unwrap_or_default());
                    }
                }
                "model" => {
                    let ngeom = js_sys::Reflect::get(&data, &JsValue::from_str("ngeom"))
                        .ok()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as usize;
                    loading_log(&format!("MuJoCo model loaded ({ngeom} geoms)"));
                    *model_msg.borrow_mut() = Some(data);
                    maybe_start(&model_msg, &scene_ready, &render_ref, &mujoco_ref);
                }
                "error" => {
                    let m = js_sys::Reflect::get(&data, &JsValue::from_str("message"))
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_default();
                    loading_log(&format!("MUJOCO ERROR: {m}"));
                    status.set(format!("mujoco error: {m}"));
                }
                _ => {}
            }
        }
    ));
    mujoco.set_onmessage(Some(on_mujoco_msg.as_ref().unchecked_ref()));
    on_mujoco_msg.forget();
    let init = js_sys::Object::new();
    set(&init, "kind", &JsValue::from_str("init"));
    set(
        &init,
        "mujoco_js",
        &JsValue::from_str(&format!("{ab}/vendor/mujoco/mujoco.js")),
    );
    set(
        &init,
        "mujoco_wasm",
        &JsValue::from_str(&format!("{ab}/vendor/mujoco/mujoco.wasm")),
    );
    // The model rides the MEDIA base (`base`), like the scene bundle.
    set(
        &init,
        "model_xml",
        &JsValue::from_str(&format!(
            "{}/mujoco/humanoid.xml",
            base.trim_end_matches('/')
        )),
    );
    mujoco.post_message(&init)?;
    *mujoco_ref.borrow_mut() = Some(mujoco);

    install_resize(&window, &canvas, &render)?;
    install_pointer(&canvas, render)?;

    status.set("loading scene…".into());
    Ok(())
}

/// Step-4 rendezvous: once the model description AND `SceneReady` are both in,
/// allocate the pose block, start the sim worker, and hand the render worker
/// the model + block. Idempotent via `take()` — runs exactly once.
fn maybe_start(
    model_msg: &Rc<RefCell<Option<JsValue>>>,
    scene_ready: &Rc<Cell<bool>>,
    render_ref: &Rc<RefCell<Option<Worker>>>,
    mujoco_ref: &Rc<RefCell<Option<Worker>>>,
) {
    if !scene_ready.get() {
        return;
    }
    let Some(model) = model_msg.borrow_mut().take() else {
        return;
    };
    let ngeom = js_sys::Reflect::get(&model, &JsValue::from_str("ngeom"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as usize;
    let njnt = js_sys::Reflect::get(&model, &JsValue::from_str("njnt"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as usize;
    let nbody = js_sys::Reflect::get(&model, &JsValue::from_str("nbody"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as usize;
    let sab = js_sys::SharedArrayBuffer::new(pose_block_bytes(ngeom, njnt, nbody) as u32);

    if let Some(mujoco) = mujoco_ref.borrow().as_ref() {
        let start = js_sys::Object::new();
        set(&start, "kind", &JsValue::from_str("start"));
        set(&start, "sab", &sab);
        let _ = mujoco.post_message(&start);
    }
    if let Some(render) = render_ref.borrow().as_ref() {
        // Forward the model description verbatim, retagged + carrying the block.
        set(
            &model.clone().unchecked_into::<js_sys::Object>(),
            "kind",
            &JsValue::from_str("mujoco-model"),
        );
        set(
            &model.clone().unchecked_into::<js_sys::Object>(),
            "sab",
            &sab,
        );
        let _ = render.post_message(&model);
    }
    loading_log("pose block allocated — sim + mirror starting");
}

/// Observe the canvas element's layout size and relay changes to the render
/// worker as [`ResizeMsg`]s in device pixels.
fn install_resize(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    render: &Worker,
) -> Result<(), JsValue> {
    let dpr = window.device_pixel_ratio().max(1.0);
    let render = render.clone();
    let cb = Closure::<dyn FnMut(js_sys::Array)>::new(move |entries: js_sys::Array| {
        let Some(entry) = entries
            .get(0)
            .dyn_into::<web_sys::ResizeObserverEntry>()
            .ok()
        else {
            return;
        };
        let rect = entry.content_rect();
        let width = (rect.width() * dpr).round() as u32;
        let height = (rect.height() * dpr).round() as u32;
        if width == 0 || height == 0 {
            return;
        }
        if let Ok(v) = serde_wasm_bindgen::to_value(&ResizeMsg::Canvas { width, height }) {
            let _ = render.post_message(&v);
        }
    });
    let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())?;
    observer.observe(canvas);
    cb.forget();
    std::mem::forget(observer);
    Ok(())
}

/// Pointer-drag orbits, wheel zooms — deltas forwarded to the render thread's
/// camera.
fn install_pointer(canvas: &HtmlCanvasElement, render: Worker) -> Result<(), JsValue> {
    let dragging = Rc::new(Cell::new(false));
    let last: Rc<Cell<(f32, f32)>> = Rc::new(Cell::new((0.0, 0.0)));

    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, last => move |e: web_sys::PointerEvent| {
            dragging.set(true);
            last.set((e.client_x() as f32, e.client_y() as f32));
        }),
    );
    canvas.add_event_listener_with_callback("pointerdown", down.as_ref().unchecked_ref())?;
    down.forget();

    let up = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging => move |_e: web_sys::PointerEvent| {
            dragging.set(false);
        }),
    );
    canvas.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("pointercancel", up.as_ref().unchecked_ref())?;
    up.forget();

    let render_move = render.clone();
    let mv = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, last => move |e: web_sys::PointerEvent| {
            if !dragging.get() {
                return;
            }
            let (lx, ly) = last.get();
            let (x, y) = (e.client_x() as f32, e.client_y() as f32);
            last.set((x, y));
            if let Ok(v) = serde_wasm_bindgen::to_value(&CameraMsg::Orbit { dx: x - lx, dy: y - ly }) {
                let _ = render_move.post_message(&v);
            }
        }),
    );
    canvas.add_event_listener_with_callback("pointermove", mv.as_ref().unchecked_ref())?;
    mv.forget();

    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
        e.prevent_default();
        if let Ok(v) = serde_wasm_bindgen::to_value(&CameraMsg::Zoom {
            dy: e.delta_y() as f32,
        }) {
            let _ = render.post_message(&v);
        }
    });
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();
    Ok(())
}

/// Append a line to the `#loading-log` panel (the pre-Ready progress feed).
pub fn loading_log(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(log) = document.get_element_by_id("loading-log") else {
        // Loading screen already dismissed — dev-console only.
        tracing::info!("{message}");
        return;
    };
    if let Ok(div) = document.create_element("div") {
        div.set_text_content(Some(message));
        let _ = log.append_child(&div);
    }
    tracing::info!("loading: {message}");
}

/// Fade out + disable the loading overlay (first frames are on screen).
fn loading_done() {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("loading"))
    {
        let _ = el.class_list().add_1("done");
    }
}

/// Paint a fatal boot error onto the loading screen (no renderer will come up).
pub fn fatal(message: &str) {
    loading_log(message);
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("loading-log"))
    {
        let _ = el.set_attribute("style", "color:#ff6a5e");
    }
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

/// The *media* base URL: `?media=` query param → compile-time `MEDIA_BASE`
/// (set by `task dev` to the side media server) → the page's own directory.
fn page_base(window: &web_sys::Window) -> String {
    let location = window.location();
    if let Ok(search) = location.search() {
        if let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search) {
            if let Some(media) = params.get("media") {
                return media.trim_end_matches('/').to_string();
            }
        }
    }
    if let Some(base) = option_env!("MEDIA_BASE") {
        return base.trim_end_matches('/').to_string();
    }
    app_base(window)
}

/// The *app* base URL: the directory this page is served from (where Trunk laid
/// down `workers/` + `vendor/`). Survives subpath deploys.
fn app_base(window: &web_sys::Window) -> String {
    let location = window.location();
    let origin = location.origin().unwrap_or_default();
    let path = location.pathname().unwrap_or_default();
    let dir = match path.rfind('/') {
        Some(idx) => &path[..idx],
        None => "",
    };
    format!("{origin}{dir}")
}
