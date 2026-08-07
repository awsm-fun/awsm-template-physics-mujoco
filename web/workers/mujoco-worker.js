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
let njnt = 0;

// Pose block views (set on "start").
let header = null;   // Int32Array  [seq, steps, ngeom, ncon]
let poses = null;    // Float32Array, 7 floats per geom after the header
let contacts = null; // Float32Array, 7 floats per contact after the poses
let joints = null;   // Float32Array, 6 floats per joint after the contacts
let steps_total = 0;

// Must match protocol.rs. Contacts are a DEBUG overlay: the count varies every
// step while the block is fixed-size, so the region is preallocated and the
// live count rides in the header.
const MAX_CONTACTS = 256;
const CONTACT_STRIDE = 7;
const JOINT_STRIDE = 6;

// Scratch for mj_contactForce, which fills a 6-vector [force xyz, torque xyz]
// in the CONTACT's frame — so element 0 is already the normal component and
// needs no projection.
//
// It has to live in the MODULE's heap, not in a plain JS Float64Array: the
// binding accepts either, but only a view over the module's own memory is
// written THROUGH. A plain typed array is copied in and the result is
// discarded, silently, which reads exactly like "every contact carries zero
// force". Allocated once — a malloc per contact per step would be the only real
// cost in this whole overlay.
let forceBuf = null;

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
    njnt = model.njnt;
    // model.opt is an embind struct wrapper; read the timestep defensively so
    // an API change degrades to the humanoid's authored 5 ms, not a crash.
    try {
        const t = model.opt && model.opt.timestep;
        if (typeof t === "number" && t > 0) timestep = t;
    } catch (_) { /* keep default */ }

    // Settle derived quantities (world poses at qpos0) so the first published
    // snapshot is the model's real initial pose, not zeros.
    // 6 doubles for mj_contactForce's result (see `forceBuf`).
    forceBuf = new mujoco.DoubleBuffer(6);

    mujoco.mj_forward(model, data);

    progress(`model ready — ${ngeom} geoms, dt ${(timestep * 1000).toFixed(1)} ms`);
    post({
        kind: "model",
        ngeom,
        njnt,
        timestep,
        // .slice() detaches plain copies safe to structured-clone.
        geom_type: model.geom_type.slice(),
        geom_group: model.geom_group.slice(),
        geom_matid: model.geom_matid.slice(),
        geom_size: model.geom_size.slice(),
        geom_rgba: model.geom_rgba.slice(),
        mat_rgba: model.mat_rgba.slice(),
        // mjtJoint: 0 free, 1 ball, 2 slide, 3 hinge. Only slide and hinge have
        // a single axis worth drawing.
        jnt_type: model.jnt_type.slice(),
        jnt_group: model.jnt_group.slice(),
    });
}

function start(sab) {
    header = new Int32Array(sab);
    poses = new Float32Array(sab, 4 * 4, ngeom * 7); // skip the 4-slot i32 header
    contacts = new Float32Array(sab, (4 + ngeom * 7) * 4, MAX_CONTACTS * CONTACT_STRIDE);
    joints = new Float32Array(
        sab,
        (4 + ngeom * 7 + MAX_CONTACTS * CONTACT_STRIDE) * 4,
        njnt * JOINT_STRIDE,
    );
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
    publishContacts();
    publishJoints();
    Atomics.store(header, 1, steps_total);
    Atomics.store(header, 0, header[0] + 1); // even — stable
}

// Contact points + normals, inside the SAME seqlock as the poses: an overlay
// must never draw contacts from a different step than the bodies they touch.
//
// MuJoCo gives each contact a 3x3 frame whose FIRST ROW is the normal (rows 2
// and 3 are the tangents), pointing from geom1 toward geom2.
function publishContacts() {
    const ncon = Math.min(data.ncon | 0, MAX_CONTACTS);
    if (ncon === 0) {
        Atomics.store(header, 3, 0);
        return;
    }
    // `data.contact` is an embind binding, and embind exposes a sequence as
    // EITHER an indexable value or a vector with .get(). Handle both, and
    // release it if it owns wasm memory — this runs every published step, so a
    // leaked vector here would be a slow-motion OOM.
    const contact = data.contact;
    const at = typeof contact.get === "function" ? (i) => contact.get(i) : (i) => contact[i];
    let written = 0;
    for (let c = 0; c < ncon; c++) {
        const item = at(c);
        if (!item) break;
        const pos = item.pos;
        const frame = item.frame;
        const o = written * CONTACT_STRIDE;
        contacts[o] = pos[0];
        contacts[o + 1] = pos[1];
        contacts[o + 2] = pos[2];
        // MuJoCo's contact frame is a 3x3 whose FIRST ROW is the normal
        // (rows 2-3 are the tangents), pointing from geom1 toward geom2.
        contacts[o + 3] = frame[0];
        contacts[o + 4] = frame[1];
        contacts[o + 5] = frame[2];
        contacts[o + 6] = normalForce(c);
        written++;
        if (typeof item.delete === "function") item.delete();
    }
    if (typeof contact.delete === "function") contact.delete();
    Atomics.store(header, 3, written);
}

// Joint anchors and axes, world frame. Unlike contacts these are a FIXED-size
// set — one entry per joint, every step — so there is no count to publish.
function publishJoints() {
    const xanchor = data.xanchor; // 3 per joint
    const xaxis = data.xaxis;     // 3 per joint
    for (let j = 0; j < njnt; j++) {
        const o = j * JOINT_STRIDE;
        joints[o] = xanchor[j * 3];
        joints[o + 1] = xanchor[j * 3 + 1];
        joints[o + 2] = xanchor[j * 3 + 2];
        joints[o + 3] = xaxis[j * 3];
        joints[o + 4] = xaxis[j * 3 + 1];
        joints[o + 5] = xaxis[j * 3 + 2];
    }
}

// The normal force at contact `c`, newtons. mj_contactForce reports in the
// contact's own frame, whose first axis IS the normal, so element 0 is the
// answer with no projection.
//
// Probed once rather than assumed: the emscripten binding takes the output as a
// `val`, and whether that means "fills the array you pass" or "returns a new
// one" is not something the glue JS reveals.
function normalForce(c) {
    try {
        mujoco.mj_contactForce(model, data, c, forceBuf);
        // GetView() is the window onto the buffer's module-heap storage; the
        // wrapper object itself is not indexable, and reading it as if it were
        // yields a silent zero rather than an error.
        const out = forceBuf.GetView();
        const fn = Math.abs(out[0]);
        return fn;
    } catch (err) {
        if (!normalForce._warned) {
            normalForce._warned = true;
            console.warn("[contactforce] unavailable:", err && err.message);
        }
        return 0;
    }
}

function xposToPose(xpos, xmat, g, o) {
    poses[o] = xpos[g * 3];
    poses[o + 1] = xpos[g * 3 + 1];
    poses[o + 2] = xpos[g * 3 + 2];
    // Row-major 3x3 -> quaternion, written in MUJOCO's order [w,x,y,z].
    // That is deliberate: the pose block IS a stream frame in the documented
    // convention, so this worker could dump it verbatim as a capture file and
    // the renderer's pose sink takes it with no reshaping on either side.
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
    poses[o + 3] = w;
    poses[o + 4] = x;
    poses[o + 5] = y;
    poses[o + 6] = z;
}
