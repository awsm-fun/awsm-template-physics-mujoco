//! **physics-mujoco** — a copyable template for driving an `awsm-renderer`
//! scene from **MuJoCo** physics running in the browser, across real wasm
//! threads:
//!
//! - **Main thread** ([`main_thread`]): owns the DOM (built with Dominator),
//!   spawns both workers, and brokers the one-time startup handshake.
//! - **Render thread** ([`render_thread`]): hosts `awsm-renderer` on an
//!   `OffscreenCanvas`, loads the editor-exported player bundle, mirrors the
//!   MuJoCo model's geoms as ordinary renderer nodes, and applies the sim's
//!   world poses every frame.
//! - **MuJoCo worker** (`web/workers/mujoco-worker.js` — plain JS): hosts the
//!   official `@mujoco/mujoco` Emscripten module (its OWN wasm module + heap,
//!   deliberately not linked into this Rust bundle), steps the sim in real
//!   time, and publishes geom world poses into a `SharedArrayBuffer` pose
//!   block ([`protocol`]).
//!
//! Main + render share one wasm module + `WebAssembly.Memory` (the
//! `wasm-bindgen-rayon` spawn pattern, [`bootstrap`]); the MuJoCo module is a
//! separate wasm instance and talks only through the pose block. The threaded
//! build profile (nightly + `+atomics` + `build-std`) and COOP/COEP headers
//! are what make the shared memory possible — see `Taskfile.yml`, `Trunk.toml`,
//! `rust-toolchain.toml`.
//!
//! The renderer knows nothing about MuJoCo: the mirror is plain nodes +
//! meshgen primitives, and poses are plain `set_local` calls — the same "pose
//! sink" shape any external sim integration reduces to.

pub mod bootstrap;
pub mod main_thread;
pub mod protocol;
pub mod render_thread;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

/// `true` when running inside a `DedicatedWorkerGlobalScope`.
pub fn is_worker_scope() -> bool {
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}

/// Single entry point. `wasm-bindgen` runs this automatically on every `init()`
/// (main thread *and* every worker). On the main thread it boots the app; in a
/// worker it does nothing — the worker's real work is triggered explicitly by
/// the bootstrap JS calling [`mt_worker_start`] after init returns.
#[wasm_bindgen(start)]
pub fn boot() -> Result<(), JsValue> {
    install_tracing();
    if is_worker_scope() {
        Ok(())
    } else {
        main_thread_boot()
    }
}

fn main_thread_boot() -> Result<(), JsValue> {
    tracing::info!("physics-mujoco: main-thread boot");
    let isolated = crossorigin_isolated();
    let has_sab = shared_array_buffer_available();
    tracing::info!("crossOriginIsolated = {isolated}, SharedArrayBuffer = {has_sab}");
    if !isolated || !has_sab {
        // Hard fail — this template is threads-only (the render worker shares
        // this module's memory, and the sim's pose block is a SAB).
        let msg = "Cross-origin isolation is OFF — multithreading is unavailable, \
                   so this build cannot run. It needs COOP: same-origin + COEP: \
                   require-corp (sent by `task dev`, or re-imposed by \
                   coi-serviceworker.js on static hosts like GitHub Pages).";
        tracing::error!("{msg}");
        main_thread::fatal(msg);
        return Err(JsValue::from_str(msg));
    }
    main_thread::start()
}

/// The worker-side entry point the bootstrap JS calls after init. Dispatches on
/// `role`; `payload` is the per-role data posted with the init message.
#[wasm_bindgen]
pub fn mt_worker_start(role: String, payload: JsValue) -> Result<(), JsValue> {
    install_tracing();
    match role.as_str() {
        "render" => render_thread::start(payload),
        other => {
            tracing::warn!("unknown worker role {other:?}");
            Ok(())
        }
    }
}

/// Install the browser-console tracing subscriber (idempotent — safe to call on
/// the main thread and in every worker).
pub fn install_tracing() {
    use tracing_subscriber::prelude::*;
    // Surface panic messages in the console — with `panic = abort` on wasm a
    // panic otherwise dies as an opaque `RuntimeError: unreachable`.
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&format!("PANIC: {info}")));
    }));
    // The default `fmt` time formatter calls `SystemTime::now()`, which panics
    // on wasm32; `without_time` strips it (the console prepends its own time).
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(tracing_web::MakeWebConsoleWriter::new())
        .with_target(false);
    let filter = tracing_subscriber::EnvFilter::new("info,physics_mujoco_frontend=debug");
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();
}

/// `globalThis.crossOriginIsolated` from whichever scope is active.
pub fn crossorigin_isolated() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("crossOriginIsolated"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// `typeof SharedArrayBuffer !== "undefined"` in the active scope.
pub fn shared_array_buffer_available() -> bool {
    js_sys::Reflect::has(&js_sys::global(), &JsValue::from_str("SharedArrayBuffer"))
        .unwrap_or(false)
}
