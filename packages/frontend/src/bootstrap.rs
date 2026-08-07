//! Worker spawning + the shared-memory bootstrap JS.
//!
//! Spawning a worker that shares this thread's `WebAssembly.Memory` is the
//! whole game for real wasm threads. The recipe (from `wasm-bindgen`'s
//! `raytrace-parallel` / `wasm-bindgen-rayon`):
//!
//! 1. Build a `Worker` from an inline blob URL (no separate `worker.js` — the
//!    source travels inside the wasm bundle).
//! 2. Post it `{ wasm_module, memory, glue_url, role, payload }`:
//!    - `wasm_module` = [`wasm_bindgen::module`] — the *compiled* artifact,
//!      structured-cloneable, so each worker skips re-compiling the multi-MB
//!      binary.
//!    - `memory` = [`wasm_bindgen::memory`] — the **shared**
//!      `WebAssembly.Memory` (shared because the bundle is built with
//!      `+atomics`). Passing it makes the worker attach to the *same* linear
//!      memory instead of allocating its own. This is what lets the physics
//!      thread write transforms the render thread reads, with no copy.
//! 3. Worker side: `await init({ module_or_path: wasm_module, memory })`, then
//!    call the role entry point [`crate::mt_worker_start`].

use wasm_bindgen::prelude::*;
use web_sys::js_sys;
use web_sys::{Blob, BlobPropertyBag, Url, Worker, WorkerOptions, WorkerType};

/// Spawn a worker that shares this thread's wasm module + linear memory,
/// tagged with `role` (read back by the bootstrap JS to pick an entry point)
/// and an arbitrary `payload` (delivered to the role entry point — pass
/// `JsValue::UNDEFINED` for none). `on_message` is installed as the worker's
/// `onmessage` so the spawner can observe what it posts back. The returned
/// [`Worker`] is also how the spawner posts *into* the worker later (e.g. the
/// main thread forwarding input commands to physics).
pub fn spawn_shared_worker(
    role: &str,
    payload: &JsValue,
    on_message: &js_sys::Function,
) -> Result<Worker, JsValue> {
    spawn_shared_worker_transfer(role, payload, &js_sys::Array::new(), on_message)
}

/// Like [`spawn_shared_worker`] but with a structured-clone `transfer` list
/// (e.g. an `OffscreenCanvas` handed to the render worker zero-copy).
pub fn spawn_shared_worker_transfer(
    role: &str,
    payload: &JsValue,
    transfer: &js_sys::Array,
    on_message: &js_sys::Function,
) -> Result<Worker, JsValue> {
    let blob_options = BlobPropertyBag::new();
    blob_options.set_type("application/javascript");
    let parts = js_sys::Array::new_with_length(1);
    parts.set(0, JsValue::from_str(WORKER_BOOTSTRAP_JS));
    let blob = Blob::new_with_str_sequence_and_options(&parts.into(), &blob_options)?;
    let blob_url = Url::create_object_url_with_blob(&blob)?;

    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let worker = Worker::new_with_options(&blob_url, &opts)?;
    let _ = Url::revoke_object_url(&blob_url);

    worker.set_onmessage(Some(on_message));
    let onerror = Closure::<dyn FnMut(JsValue)>::new(|err: JsValue| {
        // Pull the useful fields out of the ErrorEvent — logging the bare
        // object prints as `[object ErrorEvent]`, hiding the actual failure.
        let detail = err
            .dyn_ref::<web_sys::ErrorEvent>()
            .map(|e| format!("{} ({}:{})", e.message(), e.filename(), e.lineno()))
            .unwrap_or_default();
        web_sys::console::error_2(&JsValue::from_str(&format!("worker error: {detail}")), &err);
    });
    worker.set_onerror(Some(onerror.as_ref().unchecked_ref::<js_sys::Function>()));
    onerror.forget();

    let init_msg = js_sys::Object::new();
    set(&init_msg, "kind", &JsValue::from_str("awsm-mt-init"));
    set(&init_msg, "wasm_module", &wasm_bindgen::module());
    set(&init_msg, "memory", &wasm_bindgen::memory());
    set(&init_msg, "glue_url", &JsValue::from_str(&bundle_url()));
    set(&init_msg, "role", &JsValue::from_str(role));
    set(&init_msg, "payload", payload);
    if transfer.length() == 0 {
        worker.post_message(&init_msg)?;
    } else {
        worker.post_message_with_transfer(&init_msg, transfer)?;
    }

    Ok(worker)
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

/// Worker bootstrap JS. Attaches the worker to the **shared**
/// `WebAssembly.Memory` posted by the spawner, then dispatches to a role entry
/// point.
///
/// `init({ module_or_path, memory })` is the `wasm-bindgen` `--target web`
/// default export's options form. Passing `memory` is what makes the worker
/// share linear memory rather than instantiate a fresh one.
pub const WORKER_BOOTSTRAP_JS: &str = r#"
self.onmessage = async (e) => {
    const d = e.data;
    if (!d || d.kind !== "awsm-mt-init") return;
    const { wasm_module, memory, glue_url, role, payload } = d;
    try {
        // Stash the glue URL so a worker that itself spawns another worker can
        // recover it (no `document` in a worker).
        self.__awsm_glue_url = glue_url;
        const wbg = await import(glue_url);
        await wbg.default({ module_or_path: wasm_module, memory });
        // boot() ran during init (worker scope -> no-op). Now trigger the
        // role-specific work directly (a worker can't postMessage itself).
        wbg.mt_worker_start(role, payload);
    } catch (err) {
        self.postMessage({ kind: "awsm-mt-init-error", message: (err && err.message) ? err.message : String(err) });
    }
};
"#;

/// Recover the JS-glue bundle URL from the page (Trunk hashes the filename in
/// release builds, so it can't be hard-coded). Falls back to `import.meta.url`
/// outside a DOM context.
#[wasm_bindgen(inline_js = r#"
export function awsm_mt_bundle_url() {
    if (typeof self !== "undefined" && self.__awsm_glue_url) {
        return self.__awsm_glue_url;
    }
    if (typeof document !== "undefined") {
        const scripts = document.querySelectorAll("script[type=module]");
        for (const s of scripts) {
            const t = s.textContent || "";
            const m = t.match(/from\s+['"]([^'"]+\.js)['"]/);
            if (m) return new URL(m[1], location.href).href;
        }
    }
    return import.meta.url;
}
"#)]
extern "C" {
    fn awsm_mt_bundle_url() -> String;
}

/// The resolved JS-glue bundle URL (see [`awsm_mt_bundle_url`]).
pub fn bundle_url() -> String {
    awsm_mt_bundle_url()
}
