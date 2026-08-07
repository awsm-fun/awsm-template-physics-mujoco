// MuJoCo sim worker — hosts the official @mujoco/mujoco Emscripten build (its
// own wasm module + heap, separate from the app's Rust wasm) and publishes
// geom world poses into a SharedArrayBuffer pose block.
//
// Protocol (see src/protocol.rs for the block layout):
//   main → here  {kind:"init", mujoco_js, mujoco_wasm, model_xml}
//   here → main  {kind:"progress", message}
//   here → main  {kind:"model", ngeom, timestep, geom_type, geom_group,
//                 geom_matid, geom_size, geom_rgba, mat_rgba}   (typed-array copies)
//   main → here  {kind:"start", sab}       — begin stepping + publishing
//   here → main  {kind:"error", message}
//
// This is a MODULE worker (dynamic import of the ES-module mujoco.js). It is
// the template's "external sim": the renderer never sees MuJoCo, only poses.

let mujoco = null;
let model = null;
let data = null;
let timestep = 0.005;
let ngeom = 0;

// Pose block views (set on "start").
let header = null; // Int32Array  [seq, steps, ngeom, reserved]
let poses = null;  // Float32Array, 7 floats per geom after the header
let steps_total = 0;

const post = (obj) => self.postMessage(obj);
const progress = (message) => post({ kind: "progress", message: `mujoco: ${message}` });
const fail = (err) => {
    console.error("mujoco worker:", err);
    post({ kind: "error", message: (err && err.message) ? err.message : String(err) });
};

self.onmessage = async (e) => {
    const d = e.data;
    if (!d || !d.kind) return;
    try {
        if (d.kind === "init") await init(d);
        else if (d.kind === "start") start(d.sab);
    } catch (err) {
        fail(err);
    }
};

async function init({ mujoco_js, mujoco_wasm, model_xml }) {
    progress("loading wasm module…");
    const factory = (await import(mujoco_js)).default;
    mujoco = await factory({
        locateFile: (file) => (file.endsWith(".wasm") ? mujoco_wasm : file),
    });
    progress("wasm module ready — fetching model…");

    const xml = await (await fetch(model_xml, { cache: "no-cache" })).text();
    model = mujoco.MjModel.from_xml_string(xml);
    data = new mujoco.MjData(model);
    ngeom = model.ngeom;
    // model.opt is an embind struct wrapper; read the timestep defensively so
    // an API change degrades to the humanoid's authored 5 ms, not a crash.
    try {
        const t = model.opt && model.opt.timestep;
        if (typeof t === "number" && t > 0) timestep = t;
    } catch (_) { /* keep default */ }

    // Settle derived quantities (world poses at qpos0) so the first published
    // snapshot is the model's real initial pose, not zeros.
    mujoco.mj_forward(model, data);

    progress(`model ready — ${ngeom} geoms, dt ${(timestep * 1000).toFixed(1)} ms`);
    post({
        kind: "model",
        ngeom,
        timestep,
        // .slice() detaches plain copies safe to structured-clone.
        geom_type: model.geom_type.slice(),
        geom_group: model.geom_group.slice(),
        geom_matid: model.geom_matid.slice(),
        geom_size: model.geom_size.slice(),
        geom_rgba: model.geom_rgba.slice(),
        mat_rgba: model.mat_rgba.slice(),
    });
}

function start(sab) {
    header = new Int32Array(sab);
    poses = new Float32Array(sab, 4 * 4); // skip the 4-slot i32 header
    header[2] = ngeom;
    publish(); // the settled initial pose (mj_forward ran at init)
    progress("stepping");
    let last = performance.now();
    let acc = 0;
    const MAX_CATCHUP = 0.25; // s — cap the burst after a stall/hidden tab
    const tick = () => {
        const now = performance.now();
        acc = Math.min(acc + (now - last) / 1000, MAX_CATCHUP);
        last = now;
        let n = Math.floor(acc / timestep);
        if (n > 0) {
            acc -= n * timestep;
            for (let i = 0; i < n; i++) mujoco.mj_step(model, data);
            steps_total += n;
            publish();
        }
        setTimeout(tick, 2);
    };
    tick();
}

// Publish the current geom world poses under the seqlock: seq goes odd, poses
// are written, seq goes even. Single writer; the render thread is the reader.
function publish() {
    Atomics.store(header, 0, header[0] + 1); // odd — writing
    const xpos = data.geom_xpos;  // live f64 view, 3 per geom
    const xmat = data.geom_xmat;  // live f64 view, 9 per geom (row-major)
    for (let g = 0; g < ngeom; g++) {
        const o = g * 7;
        xposToPose(xpos, xmat, g, o);
    }
    Atomics.store(header, 1, steps_total);
    Atomics.store(header, 0, header[0] + 1); // even — stable
}

function xposToPose(xpos, xmat, g, o) {
    poses[o] = xpos[g * 3];
    poses[o + 1] = xpos[g * 3 + 1];
    poses[o + 2] = xpos[g * 3 + 2];
    // Row-major 3×3 → quaternion (glam order x,y,z,w).
    const m = g * 9;
    const m00 = xmat[m], m01 = xmat[m + 1], m02 = xmat[m + 2];
    const m10 = xmat[m + 3], m11 = xmat[m + 4], m12 = xmat[m + 5];
    const m20 = xmat[m + 6], m21 = xmat[m + 7], m22 = xmat[m + 8];
    const trace = m00 + m11 + m22;
    let x, y, z, w;
    if (trace > 0) {
        const s = Math.sqrt(trace + 1.0) * 2;
        w = 0.25 * s;
        x = (m21 - m12) / s;
        y = (m02 - m20) / s;
        z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2;
        w = (m21 - m12) / s;
        x = 0.25 * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2;
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25 * s;
        z = (m12 + m21) / s;
    } else {
        const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2;
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25 * s;
    }
    poses[o + 3] = x;
    poses[o + 4] = y;
    poses[o + 5] = z;
    poses[o + 6] = w;
}
