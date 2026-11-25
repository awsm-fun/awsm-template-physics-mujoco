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
//!
//! Settings (bottom right: the render resolution and the anti-aliasing) and
//! the Stats panel are awsm-renderer-kit's panels: `SettingsPanel` and
//! `StatsPanel` here, the kit's messages to and from the render worker (see
//! [`WorkerLink`] and `crate::stats`).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::Mutable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

use awsm_renderer_settings::{
    probe_gpu, AaDefaults, AaMode, SettingsConfig, SettingsHooks, SettingsOptions, SettingsPanel,
    SettingsStore, SettingsToWorker, SettingsUi,
};
use awsm_renderer_stats::{StatsCollector, StatsHooks, StatsPanel, StatsToPage, StatsToWorker};

use crate::about::{About, Tab};
use crate::bootstrap::spawn_shared_worker_transfer;
use crate::protocol::{CameraMsg, Layout, OrbitState, RenderMsg, Scene};
use crate::stats as app_stats;

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
                wasm_bindgen_futures::spawn_local(async move {
                    if let Err(e) = setup(canvas, status.clone()).await {
                        status.set(format!("setup error: {e:?}"));
                        tracing::error!("main thread setup: {e:?}");
                    }
                });
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

/// This app's Settings: every anti-aliasing mode, TAA by default on a
/// desktop, MSAA 4× on a touch device (the fixed mode this template had
/// before the menu), Off on a software GPU, and the shared resolution policy
/// (awsm-renderer-settings).
fn settings_config() -> SettingsConfig {
    SettingsConfig {
        defaults: AaDefaults {
            coarse: AaMode::Msaa,
            ..AaDefaults::default()
        },
        ..SettingsConfig::default()
    }
}

/// The Settings modal's words and look: this template's close button and
/// hint.
fn settings_ui() -> SettingsUi {
    SettingsUi {
        close_class: "settings-close".into(),
        hint: Some("Lower the resolution if your framerate is low.".into()),
        ..SettingsUi::default()
    }
}

async fn setup(canvas: HtmlCanvasElement, status: Mutable<String>) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let body = dominator::body();
    // Take keyboard focus (the modals close on Escape). The page is
    // usually embedded in an iframe on awsm.fun; a click on the canvas focuses
    // it too, since no pointer handler here cancels the press.
    let _ = window.focus();
    // `?reset` forgets the stored settings (the resolution, the
    // anti-aliasing, the Stats panel and its groups), so the page boots with
    // the defaults. Before anything reads one.
    let store = SettingsStore::new(STORAGE_PREFIX);
    if store.reset_if_asked() {
        loading_log("?reset — stored settings cleared, booting with defaults");
    }
    // The kit's messages to the render worker; held until it listens. And
    // the MuJoCo worker's profiling switch, which follows the sim group.
    let link = Rc::new(WorkerLink::default());
    let sim_profile = Rc::new(SimProfile::default());

    // The Stats chip + panel (awsm-renderer-stats), fed by the render worker
    // (the sim's numbers reach it through main). Its `what do these mean?`
    // opens About on the Stats tab. Mounted between the Stats and Settings
    // buttons so the Tab order runs left to right.
    let about = About::new();
    let stats_panel: web_sys::HtmlElement = document.create_element("div")?.unchecked_into();
    stats_panel.set_class_name("stats");
    let stats_button: web_sys::HtmlElement = document.create_element("button")?.unchecked_into();
    stats_button.set_class_name("stats-btn");
    stats_button.set_attribute("type", "button")?;
    stats_button.set_text_content(Some("Stats"));
    body.append_child(&stats_panel)?;
    body.append_child(&stats_button)?;
    let stats = StatsPanel::mount(
        STORAGE_PREFIX,
        &stats_panel,
        Some(&stats_button),
        StatsHooks {
            post: Box::new(clone!(link, sim_profile => move |message| {
                if let StatsToWorker::StatsView { shown, open } = message {
                    sim_profile.set(*shown && open.iter().any(|id| id == app_stats::SIM));
                }
                link.post_stats(message);
            })),
            on_help: Some(Box::new(clone!(about => move || about.open(Tab::Stats)))),
            page_part: app_stats::PAGE_PART.to_string(),
            app_sections: None,
            section_rows: None,
            shown: None,
        },
    );
    // "starting…" in every group until the render worker reports.
    stats.handle(&StatsToPage::Stats(
        StatsCollector::new(app_stats::config()).placeholder(),
    ));
    about.install();

    // The Settings button (bottom right) and its modal (awsm-renderer-settings).
    let settings_button = document.create_element("button")?;
    settings_button.set_class_name("settings-btn");
    settings_button.set_attribute("type", "button")?;
    settings_button.set_text_content(Some("Settings"));
    body.append_child(&settings_button)?;
    let settings_root = document.create_element("div")?;
    body.append_child(&settings_root)?;
    // A software GPU starts at a lower resolution and without anti-aliasing,
    // so the adapter is asked first (a separate, lightweight request).
    let gpu = probe_gpu().await;
    tracing::info!(
        "gpu: fallback adapter {}, largest 2D texture {}",
        gpu.is_fallback,
        gpu.max_texture_dim
    );
    let settings = SettingsPanel::new(
        &settings_root,
        &canvas,
        store,
        &SettingsOptions::new(&settings_config(), Some(gpu)),
        settings_ui(),
        loading_log,
    );

    // Backing store at the chosen render resolution (a stored Settings choice,
    // else native capped at a pixel budget). Must be sized BEFORE the
    // transfer.
    let (width, height) = settings.backing();
    canvas.set_width(width);
    canvas.set_height(height);
    let offscreen = canvas.transfer_control_to_offscreen()?;

    let base = page_base(&window);
    let app_base = app_base(&window);

    // ── Render worker ───────────────────────────────────────────────────────
    let payload = js_sys::Object::new();
    set(&payload, "canvas", &offscreen);
    set(&payload, "origin", &JsValue::from_str(&base));
    set(&payload, "app_base", &JsValue::from_str(&app_base));
    // The renderer is built with the chosen mode.
    set(&payload, "aa", &JsValue::from_str(settings.aa().id()));
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
        status, model_msg, scene_ready, render_ref, mujoco_ref, stats, settings, link => move |e: MessageEvent| {
            let data = e.data();
            // The kit's messages first: the compile overlay, the Stats report.
            if settings.handle_js(&data) || stats.handle_js(&data) {
                return;
            }
            match serde_wasm_bindgen::from_value::<RenderMsg>(data) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::SceneReady) => {
                    scene_ready.set(true);
                    // The worker listens from now on: send it what the panels
                    // said meanwhile (which Stats groups are open, the
                    // canvas size, a Settings change).
                    if let Some(render) = render_ref.borrow().as_ref() {
                        link.listening(render.clone());
                    }
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
        status, model_msg, scene_ready, render_ref, mujoco_ref, link => move |e: MessageEvent| {
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
                // The sim's numbers go to the render worker, which puts the
                // sim group (and its lane of the frame budget) in its report.
                "stats" => {
                    set(&data.clone().unchecked_into::<js_sys::Object>(), "kind", &JsValue::from_str("sim-stats"));
                    link.post_raw(&data);
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
            "{}/mujoco/{}",
            base.trim_end_matches('/'),
            scene.model_xml()
        )),
    );
    mujoco.post_message(&init)?;
    // The sim group's breakdown is measured only while it is expanded.
    sim_profile.attach(mujoco.clone());
    *mujoco_ref.borrow_mut() = Some(mujoco);

    // Layout changes and the Settings modal → `resize` / `anti-alias`.
    settings.start(
        &canvas,
        SettingsHooks {
            on_resize: Box::new(clone!(link => move |width, height| {
                link.post_settings(SettingsToWorker::Resize { width, height });
            })),
            on_anti_alias: Box::new(clone!(link => move |mode| {
                link.post_settings(SettingsToWorker::AntiAlias { mode });
            })),
            open_button: Some(settings_button),
            extra_rows: Vec::new(),
        },
    );
    install_pointer(&window, &canvas, render, stats)?;

    status.set("loading scene…".into());
    Ok(())
}

/// The kit's messages from the panels to the render worker. The worker
/// listens only once its scene is loaded (`SceneReady`); until then the
/// latest of each kind waits here (the panel's last `stats-view`, the latest
/// canvas size, a Settings change), and the page's once-a-second samples are
/// dropped.
#[derive(Default)]
struct WorkerLink {
    worker: RefCell<Option<Worker>>,
    view: RefCell<Option<StatsToWorker>>,
    size: Cell<Option<(u32, u32)>>,
    aa: Cell<Option<AaMode>>,
}

impl WorkerLink {
    fn post_settings(&self, message: SettingsToWorker) {
        if let Some(worker) = self.worker.borrow().as_ref() {
            post_kit(worker, message.to_js());
            return;
        }
        match message {
            SettingsToWorker::Resize { width, height } => self.size.set(Some((width, height))),
            SettingsToWorker::AntiAlias { mode } => self.aa.set(Some(mode)),
        }
    }

    fn post_stats(&self, message: &StatsToWorker) {
        if let Some(worker) = self.worker.borrow().as_ref() {
            post_kit(worker, message.to_js());
        } else if matches!(message, StatsToWorker::StatsView { .. }) {
            *self.view.borrow_mut() = Some(message.clone());
        }
    }

    /// Any other message for the render worker, once it listens.
    fn post_raw(&self, message: &JsValue) {
        if let Some(worker) = self.worker.borrow().as_ref() {
            let _ = worker.post_message(message);
        }
    }

    /// The render worker listens from now on: send what waited.
    fn listening(&self, worker: Worker) {
        *self.worker.borrow_mut() = Some(worker);
        let view = self.view.borrow_mut().take();
        if let Some(view) = view {
            self.post_stats(&view);
        }
        if let Some((width, height)) = self.size.take() {
            self.post_settings(SettingsToWorker::Resize { width, height });
        }
        if let Some(mode) = self.aa.take() {
            self.post_settings(SettingsToWorker::AntiAlias { mode });
        }
    }
}

fn post_kit(worker: &Worker, message: Result<JsValue, String>) {
    match message {
        Ok(value) => {
            let _ = worker.post_message(&value);
        }
        Err(err) => tracing::error!("kit message: {err}"),
    }
}

/// `{kind: "profile", on}` to the MuJoCo worker: whether the sim group is
/// expanded on a shown panel (it splits its time into stepping and
/// publishing only then). Sent on every change, and once when the worker
/// starts.
#[derive(Default)]
struct SimProfile {
    worker: RefCell<Option<Worker>>,
    on: Cell<bool>,
    sent: Cell<Option<bool>>,
}

impl SimProfile {
    fn set(&self, on: bool) {
        self.on.set(on);
        self.send();
    }

    fn attach(&self, worker: Worker) {
        *self.worker.borrow_mut() = Some(worker);
        self.send();
    }

    fn send(&self) {
        let on = self.on.get();
        let Some(worker) = self.worker.borrow().clone() else {
            return;
        };
        if self.sent.replace(Some(on)) == Some(on) {
            return;
        }
        let msg = js_sys::Object::new();
        set(&msg, "kind", &JsValue::from_str("profile"));
        set(&msg, "on", &JsValue::from_bool(on));
        let _ = worker.post_message(&msg);
    }
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
    stats: StatsPanel,
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
    let performance = window.performance();
    let time_origin = performance.as_ref().map_or(0.0, |p| p.time_origin());
    let cb: Cb = Rc::new(RefCell::new(None));
    let again = cb.clone();
    let mut sent: Option<OrbitState> = None;
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        // This is the page's per-frame work (the Stats panel's page group).
        let t0 = performance.as_ref().map_or(0.0, |p| p.now());
        {
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
        }
        let t1 = performance.as_ref().map_or(0.0, |p| p.now());
        stats.page_frame(t1 - t0, None, true);
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
