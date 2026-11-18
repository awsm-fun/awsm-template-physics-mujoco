//! Main thread — owns the DOM (built with Dominator) and orchestrates the two
//! workers.
//!
//! Flow:
//! 1. Build the canvas + HUD, size the canvas to the chosen render resolution, transfer
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
//!    shared memory only, paced by the render worker's frame tick (see
//!    `protocol`, "Timing"). Main relays resizes, and owns the orbit camera:
//!    pointer events update it, and once per main-thread frame in which it
//!    changed, its state goes to the render worker (see [`CameraMsg`]).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::{Mutable, SignalExt};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

use crate::bootstrap::spawn_shared_worker_transfer;
use crate::protocol::{CameraMsg, Layout, OrbitState, ProfileMsg, RenderMsg, Scene};
use crate::stats::{SimReport, Stats};

/// Every `localStorage` key this app writes starts with this (the CDN origin is
/// shared with other experiments).
const STORAGE_PREFIX: &str = "awsm_mujoco_";

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
    // Take keyboard focus (the Settings modal closes on Escape). The page is
    // usually embedded in an iframe on awsm.fun; a click on the canvas focuses
    // it too, since no pointer handler here cancels the press.
    let _ = window.focus();
    maybe_reset_settings(&window);

    // The Stats chip + panel (see `crate::stats`), fed by both workers.
    let stats = Stats::install(&window, STORAGE_PREFIX);

    // Backing store at the chosen render resolution (a stored Settings choice,
    // else native capped at a pixel budget — see `resolution`). Must be sized
    // BEFORE the transfer.
    let resolution = crate::resolution::Resolution::new(&window, &canvas, "awsm_mujoco_res_scale");
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
    // Which of the two demo scenes to run (`?scene=flag`). Each is a MuJoCo
    // model plus the player bundle exported from it; the two must always be
    // chosen together, which is why one name picks both.
    let scene = Scene::from_query(&search);
    set(&payload, "bundle", &JsValue::from_str(scene.bundle_dir()));
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
        status, model_msg, scene_ready, render_ref, mujoco_ref, stats => move |e: MessageEvent| {
            match serde_wasm_bindgen::from_value::<RenderMsg>(e.data()) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::SceneReady) => {
                    scene_ready.set(true);
                    // The worker listens from now on: tell it which Stats
                    // groups are expanded (later changes follow, below).
                    if let Some(render) = render_ref.borrow().as_ref() {
                        post_profile(render, &stats);
                    }
                    loading_log("scene ready — waiting on the MuJoCo model…");
                    maybe_start(&model_msg, &scene_ready, &render_ref, &mujoco_ref);
                }
                Ok(RenderMsg::Ready) => {
                    loading_log("first frames rendered — ready");
                    loading_done();
                    stats.mark_ready();
                    status.set("running — drag orbits, wheel zooms".into());
                }
                Ok(RenderMsg::Stats(report)) => stats.window(report),
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
    wasm_bindgen_futures::spawn_local(stats.profile_signal().for_each(
        clone!(render, scene_ready, stats => move |_| {
            if scene_ready.get() {
                post_profile(&render, &stats);
            }
            async {}
        }),
    ));

    // ── MuJoCo worker (plain JS module worker — its own wasm module) ────────
    loading_log("spawning MuJoCo worker…");
    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let ab = app_base.trim_end_matches('/');
    let mujoco = Worker::new_with_options(&format!("{ab}/workers/mujoco-worker.js"), &opts)?;
    let on_mujoco_msg = Closure::<dyn FnMut(MessageEvent)>::new(clone!(
        status, model_msg, scene_ready, render_ref, mujoco_ref, stats => move |e: MessageEvent| {
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
                "stats" => stats.sim_window(sim_report(&data)),
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
            "{}/mujoco/{}",
            base.trim_end_matches('/'),
            scene.model_xml()
        )),
    );
    mujoco.post_message(&init)?;
    // The sim group's breakdown is measured only while it is expanded.
    wasm_bindgen_futures::spawn_local(stats.sim_profile_signal().for_each(
        clone!(mujoco => move |on| {
            let msg = js_sys::Object::new();
            set(&msg, "kind", &JsValue::from_str("profile"));
            set(&msg, "on", &JsValue::from_bool(on));
            let _ = mujoco.post_message(&msg);
            async {}
        }),
    ));
    *mujoco_ref.borrow_mut() = Some(mujoco);

    // Layout changes → ResizeMsg, plus the Settings modal (resolution slider).
    resolution.install(&window, &canvas, render.clone())?;
    install_pointer(&window, &canvas, render, stats)?;

    status.set("loading scene…".into());
    Ok(())
}

/// Tell the render worker which Stats groups are expanded.
fn post_profile(render: &Worker, stats: &Stats) {
    if let Ok(v) = serde_wasm_bindgen::to_value(&ProfileMsg::Set(stats.profile())) {
        let _ = render.post_message(&v);
    }
}

/// The MuJoCo worker's `{kind: "stats"}` message.
fn sim_report(data: &JsValue) -> SimReport {
    let num = |k: &str| {
        js_sys::Reflect::get(data, &JsValue::from_str(k))
            .ok()
            .and_then(|v| v.as_f64())
    };
    SimReport {
        span_ms: num("span_ms").unwrap_or(0.0),
        steps: num("steps").unwrap_or(0.0) as u32,
        busy_ms: num("busy_ms").unwrap_or(0.0),
        split: match (num("step_ms"), num("publish_ms"), num("split_steps")) {
            (Some(step), Some(publish), Some(n)) => Some(([step, publish], n as u32)),
            _ => None,
        },
    }
}

/// `?reset` in the URL forgets every stored setting of this app (the panel,
/// its groups, the render resolution), so the page boots with the defaults.
/// Runs before anything reads one.
fn maybe_reset_settings(window: &web_sys::Window) {
    let search = window.location().search().unwrap_or_default();
    let reset = web_sys::UrlSearchParams::new_with_str(&search)
        .map(|p| p.has("reset"))
        .unwrap_or(false);
    let Some(ls) = reset
        .then(|| window.local_storage().ok().flatten())
        .flatten()
    else {
        return;
    };
    let keys: Vec<String> = (0..ls.length().unwrap_or(0))
        .filter_map(|i| ls.key(i).ok().flatten())
        .filter(|k| k.starts_with(STORAGE_PREFIX))
        .collect();
    for key in keys {
        let _ = ls.remove_item(&key);
    }
    loading_log("?reset — stored settings cleared, booting with defaults");
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
    let sab = js_sys::SharedArrayBuffer::new(Layout { ngeom, njnt, nbody }.bytes() as u32);

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

/// Pointer-drag orbits, wheel zooms. Main owns the camera state: the events
/// update it as they arrive, and a main-thread animation-frame loop posts it to
/// the render worker, stamped with that frame's absolute time, whenever it
/// changed. The render worker picks which posted state to draw (see
/// [`CameraMsg`]).
fn install_pointer(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    render: Worker,
    stats: Rc<Stats>,
) -> Result<(), JsValue> {
    let camera = Rc::new(Cell::new(OrbitState::START));
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

    let mv = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, last, camera => move |e: web_sys::PointerEvent| {
            if !dragging.get() {
                return;
            }
            let (lx, ly) = last.get();
            let (x, y) = (e.client_x() as f32, e.client_y() as f32);
            last.set((x, y));
            let mut c = camera.get();
            c.orbit(x - lx, y - ly);
            camera.set(c);
        }),
    );
    canvas.add_event_listener_with_callback("pointermove", mv.as_ref().unchecked_ref())?;
    mv.forget();

    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(
        clone!(camera => move |e: web_sys::WheelEvent| {
            e.prevent_default();
            let mut c = camera.get();
            c.zoom(e.delta_y() as f32);
            camera.set(c);
        }),
    );
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();

    // Once per main-thread frame (after that frame's input events), post the
    // state if it changed.
    type Cb = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let time_origin = window.performance().map(|p| p.time_origin()).unwrap_or(0.0);
    let cb: Cb = Rc::new(RefCell::new(None));
    let again = cb.clone();
    let mut sent: Option<OrbitState> = None;
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        // This is the page's per-frame work (the Stats panel's page group).
        stats.time_camera(|| {
            let state = camera.get();
            if sent != Some(state) {
                sent = Some(state);
                let msg = CameraMsg::State {
                    t_ms: time_origin + t,
                    state,
                };
                if let Ok(v) = serde_wasm_bindgen::to_value(&msg) {
                    let _ = render.post_message(&v);
                }
            }
        });
        if let (Some(w), Some(f)) = (web_sys::window(), again.borrow().as_ref()) {
            let _ = w.request_animation_frame(f.as_ref().unchecked_ref());
        }
    }));
    if let Some(f) = cb.borrow().as_ref() {
        window.request_animation_frame(f.as_ref().unchecked_ref())?;
    }
    std::mem::forget(cb);
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
