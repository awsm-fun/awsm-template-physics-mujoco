// MuJoCo sim worker — hosts the official @mujoco/mujoco Emscripten build (its
// own wasm module + heap, separate from the app's Rust wasm) and publishes
// geom world poses into a SharedArrayBuffer pose block.
//
// Timing (see "Timing" in src/protocol.rs): the render worker bumps a frame
// tick at the start of every frame and writes that frame's time; this worker
// wakes on the tick, steps MuJoCo at its fixed timestep as many times as the
// elapsed time pays for, and publishes one pose slot stamped with the time
// its state is due. The render worker interpolates between slots. With no tick
// for TICK_TIMEOUT_MS (no renderer yet, a hidden tab) it carries on, on its
// own clock, so a robotics run never stalls.
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
let nbody = 0;

// Pose block views (set on "start").
let words = null;    // Int32Array over the whole block (header + slot seq/t)
let floats = null;   // Float32Array over the whole block
let contacts = null; // Float32Array, the contact overlay
let joints = null;   // Float32Array, the joint overlay
let inertia = null;  // Float32Array, the inertia overlay
let steps_total = 0;
let published = 0;

// Must match protocol.rs.
const H_TICK = 0;
const H_FRAME_TIME = 1;
const H_TIME_VALID = 2;
const H_NGEOM = 3;
const H_STEPS = 4;
const H_PUBLISHED = 5;
const H_NCON = 6;
const H_OVERLAY_SEQ = 7;
const HEADER_WORDS = 16;
const POSE_SLOTS = 8;
const SLOT_HEADER = 2; // seq, t
const POSE_STRIDE = 7;
// Contacts are a DEBUG overlay: the count varies every step while the block is
// fixed-size, so the region is preallocated and the live count rides in the
// header.
const MAX_CONTACTS = 256;
const CONTACT_STRIDE = 7;
const JOINT_STRIDE = 6;

// No frame tick for this long (ms) means no renderer is drawing (not linked
// yet, or a hidden tab): step on this worker's own clock instead. Well above
// the slowest frame interval worth drawing, so a slow renderer never flips
// the clock source frame to frame.
const TICK_TIMEOUT_MS = 100;
// Most simulated time one wake may catch up (ms): after a stall the world
// slows down rather than freezing the worker in one enormous burst.
const MAX_CATCHUP_MS = 250;

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
    nbody = model.nbody;
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
        nbody,
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
        // The equivalent inertia BOX's half-extents are static, so they are
        // derived once on the render side from these rather than republished
        // every step.
        body_mass: model.body_mass.slice(),
        body_inertia: model.body_inertia.slice(),
    });
}

function slotWords() {
    return SLOT_HEADER + (ngeom + nbody) * POSE_STRIDE;
}

function start(sab) {
    words = new Int32Array(sab);
    floats = new Float32Array(sab);
    const contactsAt = HEADER_WORDS + POSE_SLOTS * slotWords();
    const jointsAt = contactsAt + MAX_CONTACTS * CONTACT_STRIDE;
    const inertiaAt = jointsAt + njnt * JOINT_STRIDE;
    contacts = floats.subarray(contactsAt, jointsAt);
    joints = floats.subarray(jointsAt, inertiaAt);
    inertia = floats.subarray(inertiaAt, inertiaAt + nbody * 7);
    Atomics.store(words, H_NGEOM, ngeom);
    progress("stepping");
    run();
}

// ── Clock ───────────────────────────────────────────────────────────────────
//
// Every time here is ABSOLUTE ms (performance.timeOrigin + now), the timeline
// the render worker's frame times are on too. `fedTo` is the time up to which
// the accumulator has been fed, whichever clock fed it, so switching between
// the render tick and this worker's own clock never counts a stretch twice.

const ownNow = () => performance.timeOrigin + performance.now();

// The render worker's latest frame time as absolute ms. The block holds it as
// wrapping microseconds; the full value is rebuilt around this worker's own
// clock, which is within a frame or two of it.
function frameTimeMs() {
    const own = ownNow();
    const ownUs = Math.round(own * 1000) % 0x1_0000_0000;
    const t = Atomics.load(words, H_FRAME_TIME) >>> 0;
    const diff = ((t - ownUs) | 0); // wrapping signed difference, µs
    return own + diff / 1000;
}

function run() {
    const stepMs = timestep * 1000;
    let fedTo = ownNow();
    let acc = 0; // ms of real time not yet stepped
    let seenTick = Atomics.load(words, H_TICK);

    // The settled initial pose (mj_forward ran at init), due now.
    publish(fedTo);

    const advance = (now) => {
        acc = Math.min(acc + Math.max(now - fedTo, 0), MAX_CATCHUP_MS);
        fedTo = Math.max(fedTo, now);
        let n = 0;
        while (acc >= stepMs) {
            mujoco.mj_step(model, data);
            acc -= stepMs;
            n++;
        }
        if (n > 0) {
            steps_total += n;
            // The state just stepped to is due at fedTo - acc: the time the
            // accumulator has been fed to, less what it still holds.
            publish(fedTo - acc);
        }
    };

    let lastTickAt = ownNow();
    const wake = () => {
        const tick = Atomics.load(words, H_TICK);
        if (tick !== seenTick && Atomics.load(words, H_TIME_VALID) === 1) {
            seenTick = tick;
            lastTickAt = ownNow();
            advance(frameTimeMs());
        } else if (ownNow() - lastTickAt >= TICK_TIMEOUT_MS) {
            // No renderer is drawing: keep the world going on our own clock.
            seenTick = tick;
            advance(ownNow());
        }
        wait();
    };

    // Atomics.waitAsync where the browser has it; otherwise a short timer
    // that polls the tick.
    const wait = typeof Atomics.waitAsync === "function"
        ? () => {
            const r = Atomics.waitAsync(words, H_TICK, seenTick, TICK_TIMEOUT_MS);
            if (r.async) r.value.then(wake);
            else wake();
        }
        : () => setTimeout(wake, 4);
    wait();
}

// Publish the current state into the next pose slot, stamped `dueMs` (absolute
// ms): slot seq goes odd, poses and time are written, seq goes even, then the
// published count moves on. Single writer; the render worker is the reader.
// The debug overlays follow under their own seqlock, outside the slot, so the
// contact walk never holds a pose slot open.
function publish(dueMs) {
    const slot = published % POSE_SLOTS;
    const base = HEADER_WORDS + slot * slotWords();
    const seq = Atomics.load(words, base);
    Atomics.store(words, base, seq + 1); // odd — writing
    words[base + 1] = Math.round(dueMs * 1000) % 0x1_0000_0000 | 0;
    const xpos = data.geom_xpos;  // live f64 view, 3 per geom
    const xmat = data.geom_xmat;  // live f64 view, 9 per geom (row-major)
    const geomsAt = base + SLOT_HEADER;
    for (let g = 0; g < ngeom; g++) {
        matToPose(xpos, xmat, g, floats, geomsAt + g * 7);
    }
    publishBodies(geomsAt + ngeom * 7);
    Atomics.store(words, base, seq + 2); // even — stable
    published++;
    Atomics.store(words, H_PUBLISHED, published);
    Atomics.store(words, H_STEPS, steps_total);

    const oseq = Atomics.load(words, H_OVERLAY_SEQ);
    Atomics.store(words, H_OVERLAY_SEQ, oseq + 1);
    publishContacts();
    publishJoints();
    publishInertia();
    Atomics.store(words, H_OVERLAY_SEQ, oseq + 2);
}

// Contact points + normals (debug overlay, newest state only).
//
// MuJoCo gives each contact a 3x3 frame whose FIRST ROW is the normal (rows 2
// and 3 are the tangents), pointing from geom1 toward geom2.
function publishContacts() {
    const ncon = Math.min(data.ncon | 0, MAX_CONTACTS);
    if (ncon === 0) {
        Atomics.store(words, H_NCON, 0);
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
    Atomics.store(words, H_NCON, written);
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

// Body WORLD frames — the channel that deforms flexes.
//
// A deformable imports as an ordinary skinned mesh whose joints ARE the bodies
// its cage rides, so this is all the sim has to publish to drive one: nbody
// frames, no vertices. Note this is the body frame, NOT the inertial frame
// published just above.
//
// `xquat` is already a quaternion in MuJoCo's [w,x,y,z] order — the very layout
// the pose block wants — so unlike the geom and inertia frames there is no
// matrix to convert.
function publishBodies(at) {
    const xpos = data.xpos;   // 3 per body
    const xquat = data.xquat; // 4 per body, [w,x,y,z]
    for (let b = 0; b < nbody; b++) {
        const o = at + b * 7;
        floats[o] = xpos[b * 3];
        floats[o + 1] = xpos[b * 3 + 1];
        floats[o + 2] = xpos[b * 3 + 2];
        floats[o + 3] = xquat[b * 4];
        floats[o + 4] = xquat[b * 4 + 1];
        floats[o + 5] = xquat[b * 4 + 2];
        floats[o + 6] = xquat[b * 4 + 3];
    }
}

// Body INERTIAL frames — the frame the equivalent inertia box is drawn in,
// which is NOT the body frame: it is offset to the centre of mass and rotated
// onto the principal axes.
function publishInertia() {
    const xipos = data.xipos; // 3 per body
    const ximat = data.ximat; // 9 per body, row-major
    for (let b = 0; b < nbody; b++) {
        matToPose(xipos, ximat, b, inertia, b * 7);
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

// Write entry `i` of a MuJoCo (positions, row-major 3x3 matrices) pair into
// `target` at float offset `o` as a 7-float pose. Shared by the geom poses and
// the inertia-box frames, which are the same shape.
function matToPose(xpos, xmat, i, target, o) {
    const g = i;
    const poses = target;
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
