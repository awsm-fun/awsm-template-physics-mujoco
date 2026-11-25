//! The wire between the three players:
//!
//! - **MuJoCo worker ↔ render worker**: one plain `SharedArrayBuffer`, the
//!   **pose block** (layout below). The render worker writes its frame time and
//!   bumps a frame tick at the start of every frame; the sim worker wakes on
//!   that tick, runs as many fixed `dt` steps as the elapsed time pays for, and
//!   publishes the result, stamped with the simulation time it shows, into a
//!   small ring of pose slots. The render worker draws every frame at one
//!   display time, interpolating between the two slots around it. This is the
//!   template's stand-in for the renderer-side "pose sink": the renderer itself
//!   knows nothing about MuJoCo or transports, it just gets world poses applied
//!   to nodes.
//! - **main ↔ render worker**: small serde messages (progress, the camera
//!   state) and the settings and stats kit's messages (resize, the
//!   anti-aliasing mode and its compile progress, the Stats report and which
//!   Stats groups are open: plain objects with a `type`, see
//!   awsm-renderer-settings and awsm-renderer-stats) over `postMessage`.
//! - **main ↔ MuJoCo worker**: raw JS objects (`{kind: ...}`) — see
//!   `web/workers/mujoco-worker.js`; the model description crosses once at
//!   startup, tagged `kind: "model"`, and is forwarded to the render worker
//!   verbatim as `kind: "mujoco-model"`. The sim's Stats numbers come back
//!   about once a second (`kind: "stats"`); main forwards them to the render
//!   worker as `kind: "sim-stats"` (see `crate::stats`).
//!
//! ## Timing: one clock, fixed steps, interpolated display
//!
//! The rule (Glenn Fiedler's "Fix Your Timestep"): the sim plays every step at a
//! fixed `dt`, as many per wake as real time has paid for; the display only ever
//! shows the latest state or a blend of the latest two.
//!
//! Every time below is **absolute**: `performance.timeOrigin + now` in
//! microseconds, kept as a wrapping `u32` (compare with a wrapping signed
//! difference). Workers have different `timeOrigin`s; the sum is one clock.
//!
//! 1. Render frame `k` starts at rAF time `T_k`. Before drawing, it writes `T_k`
//!    to `FRAME_TIME`, sets `TIME_VALID`, bumps `TICK` and notifies.
//! 2. The sim wakes (`Atomics.waitAsync` on `TICK`), feeds the accumulator with
//!    the time since the last time it fed it, steps while it holds a whole
//!    `dt`, then publishes one slot stamped `t = T_k - accumulator`: the time
//!    at which the state it holds is due. With no tick for 100 ms (no renderer
//!    yet, or a hidden tab) it falls back to its own clock, on the same absolute
//!    timeline, so a robotics run keeps going and nothing is counted twice.
//! 3. Frame `k + 1` draws at display time `D = T_k - dt`: between the newest
//!    slot at or before `D` and the one after it. That is exactly
//!    `lerp(latest-1, latest, accumulator / dt)`, it advances by exactly the
//!    frame interval every frame, and it is fixed before the sim's next burst
//!    starts, so nothing the sim writes meanwhile can move it. If the newest
//!    slot is still at or before `D`, the sim was late: the frame holds the
//!    newest pose and the Stats panel counts it.
//!
//! The cost is one frame of latency (the sim runs alongside the renderer, one
//! frame behind) plus at most one step.
//!
//! ## Pose block layout (the SharedArrayBuffer)
//!
//! ```text
//! header, HEADER_WORDS i32 words:
//!   [TICK]          render → sim: bumped at the start of every render frame
//!   [FRAME_TIME]    render → sim: that frame's rAF time, absolute µs (u32)
//!   [TIME_VALID]    render → sim: 1 once FRAME_TIME has ever been written
//!   [NGEOM]         sim: geom count
//!   [STEPS]         sim: total steps simulated (monotonic)
//!   [PUBLISHED]     sim: slots published so far; the newest is (n-1) % POSE_SLOTS
//!   [NCON]          sim: live contact count in the overlay region
//!   [OVERLAY_SEQ]   sim: seqlock over the overlay region (odd while writing)
//! POSE_SLOTS slots of slot_words(ngeom, nbody) words each:
//!   i32 seq         odd while the slot is being written
//!   u32 t           the absolute time (µs) this state is due
//!   f32 ngeom * 7   geom world poses  [px,py,pz, qw,qx,qy,qz]  (MuJoCo frame)
//!   f32 nbody * 7   body world poses  [px,py,pz, qw,qx,qy,qz]  (drive flexes)
//! overlay region (DEBUG; latest state only, under OVERLAY_SEQ):
//!   f32 MAX_CONTACTS * 7   contact world position, unit NORMAL, NORMAL FORCE (N)
//!   f32 njnt * 6           joint world ANCHOR + unit AXIS
//!   f32 nbody * 7          body INERTIAL frame (position + quaternion)
//! ```
//!
//! A slot is written whole by the single writer (seq odd → data → seq even),
//! then `PUBLISHED` is bumped. The reader copies a slot and re-checks its seq;
//! a changed seq means the writer lapped it, and the reader falls back to an
//! older pair rather than waiting. Eight slots hold about 25 steps of the
//! humanoid (5 ms each) — far more than the frame plus one burst the reader
//! needs.
//!
//! ## The overlay region is DEBUG tooling
//!
//! Contacts, joint axes and inertia boxes live only in this template — never in
//! the renderer, the editor, or a bundle — and are drawn from the newest state,
//! not interpolated. The contact count varies every step while the block must
//! be a fixed size, so the region is preallocated to [`MAX_CONTACTS`] and the
//! live count rides in the header. A torn overlay read is simply skipped for
//! that frame.
//!
//! Quaternions are in MUJOCO's `[w, x, y, z]` order, not glam's — the poses ARE
//! stream frames in the documented pose-sink convention, so a producer could
//! dump one verbatim as a capture file and the sink takes it with no reshaping.
//!
//! All values little-endian f32/i32; MuJoCo's f64 state is narrowed to f32 by
//! the worker.

use serde::{Deserialize, Serialize};

/// f32 slots per geom or body pose: pos xyz + quat wxyz.
pub const POSE_STRIDE: usize = 7;

/// Header word indices (i32), shared with `mujoco-worker.js`.
pub mod header {
    pub const TICK: u32 = 0;
    pub const FRAME_TIME: u32 = 1;
    pub const TIME_VALID: u32 = 2;
    pub const NGEOM: u32 = 3;
    pub const STEPS: u32 = 4;
    pub const PUBLISHED: u32 = 5;
    pub const NCON: u32 = 6;
    pub const OVERLAY_SEQ: u32 = 7;
}
/// i32 words before the pose ring.
pub const HEADER_WORDS: usize = 16;
/// Pose slots in the ring.
pub const POSE_SLOTS: usize = 8;
/// Words before the poses inside a slot: seq, t.
pub const SLOT_HEADER: usize = 2;

/// f32 slots per contact in the overlay region: world position, unit normal,
/// and the normal force in newtons.
pub const CONTACT_STRIDE: usize = 7;

/// How many contacts the block reserves room for.
///
/// A humanoid on the ground runs a handful; a pile of objects can spike. This
/// caps the overlay's cost without ever throttling the sim, which does not know
/// the overlay exists.
pub const MAX_CONTACTS: usize = 256;

/// f32 slots per joint in the overlay region: world anchor + unit axis.
pub const JOINT_STRIDE: usize = 6;

/// Where everything sits in the pose block, for one model. All offsets are in
/// 4-byte words from the start of the block.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub ngeom: usize,
    pub njnt: usize,
    pub nbody: usize,
}

impl Layout {
    /// Words per pose slot.
    pub fn slot_words(&self) -> usize {
        SLOT_HEADER + (self.ngeom + self.nbody) * POSE_STRIDE
    }

    /// Start of slot `i`.
    pub fn slot(&self, i: usize) -> usize {
        HEADER_WORDS + i * self.slot_words()
    }

    /// Start of the geom poses inside a slot, relative to the slot.
    pub fn slot_geoms(&self) -> usize {
        SLOT_HEADER
    }

    /// Start of the body poses inside a slot, relative to the slot.
    ///
    /// These are BODY frames (`xpos`/`xquat`) — a second real pose channel. A
    /// deformable (a MuJoCo *flex*) imports as an ordinary skinned mesh whose
    /// joints are the bodies its cage rides, so driving one is just
    /// `apply_body_poses` alongside `apply_geom_poses`: no vertex ever crosses
    /// the wire, only `nbody` frames.
    pub fn slot_bodies(&self) -> usize {
        SLOT_HEADER + self.ngeom * POSE_STRIDE
    }

    /// Start of the contact overlay.
    pub fn contacts(&self) -> usize {
        HEADER_WORDS + POSE_SLOTS * self.slot_words()
    }

    /// Start of the joint overlay: `njnt` entries, all written every publish.
    pub fn joints(&self) -> usize {
        self.contacts() + MAX_CONTACTS * CONTACT_STRIDE
    }

    /// Start of the inertia overlay: one INERTIAL frame per body (offset to the
    /// centre of mass, rotated onto the principal axes — not the body frame).
    pub fn inertia(&self) -> usize {
        self.joints() + self.njnt * JOINT_STRIDE
    }

    /// Byte size of the whole block.
    pub fn bytes(&self) -> usize {
        (self.inertia() + self.nbody * POSE_STRIDE) * 4
    }
}

/// `a - b` for wrapping `u32` microsecond times: positive when `a` is later.
pub fn time_diff_us(a: u32, b: u32) -> i32 {
    a.wrapping_sub(b) as i32
}

/// An absolute time in ms (`timeOrigin + now`) as the block's wrapping µs.
pub fn abs_ms_to_us(ms: f64) -> u32 {
    ((ms * 1000.0).round() as u64 & 0xffff_ffff) as u32
}

/// The orbit camera's state: owned by the main thread (which has the pointer
/// events), drawn by the render thread. See [`CameraMsg::State`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct OrbitState {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
}

impl OrbitState {
    const SENSITIVITY: f32 = 0.008;
    const PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const PITCH_MIN: f32 = 0.05;
    const MIN_RADIUS: f32 = 1.5;
    const MAX_RADIUS: f32 = 30.0;

    /// Where the camera starts.
    pub const START: Self = Self {
        yaw: 0.35,
        pitch: 0.35,
        radius: 6.0,
    };

    /// A drag of `dx, dy` CSS pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * Self::SENSITIVITY;
        self.pitch = (self.pitch - dy * Self::SENSITIVITY).clamp(Self::PITCH_MIN, Self::PITCH_MAX);
    }

    /// A wheel movement of `dy` CSS pixels.
    pub fn zoom(&mut self, dy: f32) {
        self.radius = (self.radius * (1.0 + dy * 0.001)).clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }
}

/// Render-worker → main messages (loading progress + lifecycle).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RenderMsg {
    /// Human-readable loading-screen line.
    Progress {
        message: String,
    },
    /// The scene bundle is loaded + committed; safe to forward the MuJoCo
    /// model description (before this, the worker's onmessage isn't listening).
    SceneReady,
    /// First real frames have been presented.
    Ready,
    Error {
        message: String,
    },
}

/// Main → render worker: the orbit camera's state.
///
/// Main owns the camera (it has the pointer events) and posts its state once
/// per main-thread animation frame in which it changed, stamped with that
/// frame's absolute rAF time in ms (`timeOrigin + t`). The render thread keeps
/// the last few and, each frame, draws the newest one stamped at least half a
/// frame before its own frame time: the previous main frame's, which has
/// certainly arrived. Every render frame therefore shows exactly one main
/// frame's worth of input — no frame gets two drags and the next none, however
/// the two threads' callbacks happen to interleave.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CameraMsg {
    State { t_ms: f64, state: OrbitState },
}

/// Which demo scene the page is running (`?scene=flag`; default humanoid).
///
/// The template ships TWO, because no single model shows both halves of the
/// integration. The humanoid is a rigid-body ragdoll and the subject of every
/// debug overlay — contacts, joint axes, inertia boxes. The flag is a
/// **deformable** (a MuJoCo *flex*), which imports as an ordinary skinned mesh
/// and is driven through the body channel; it has almost no contacts and would
/// make a poor showcase for the overlays, which is why it is a second scene
/// rather than a replacement.
///
/// A scene is a MuJoCo model AND the player bundle exported from that same
/// model. Mixing them is exactly how you get an instance whose geom count
/// disagrees with the sim, so ONE name chooses both.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Scene {
    #[default]
    Humanoid,
    Flag,
}

impl Scene {
    /// Pick the scene out of the page's query string.
    pub fn from_query(search: &str) -> Self {
        if search.contains("scene=flag") {
            Self::Flag
        } else {
            Self::Humanoid
        }
    }

    /// This scene's MJCF, relative to `media/mujoco/`.
    pub fn model_xml(self) -> &'static str {
        match self {
            Self::Humanoid => "humanoid.xml",
            Self::Flag => "flag.xml",
        }
    }

    /// The media directory holding this scene's exported player bundle.
    pub fn bundle_dir(self) -> &'static str {
        match self {
            Self::Humanoid => "bundle",
            Self::Flag => "bundle-flag",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_compare_across_the_wrap() {
        let before = u32::MAX - 500;
        let after = before.wrapping_add(1_000);
        assert_eq!(time_diff_us(after, before), 1_000);
        assert_eq!(time_diff_us(before, after), -1_000);
        // A real absolute time keeps its sub-millisecond part.
        let ms = 1_763_300_000_123.456;
        let us = abs_ms_to_us(ms);
        assert_eq!(time_diff_us(abs_ms_to_us(ms + 4.166), us), 4_166);
    }

    #[test]
    fn orbit_state_clamps_like_the_renderer_did() {
        let mut s = OrbitState::START;
        s.orbit(0.0, -100_000.0);
        assert!((s.pitch - (std::f32::consts::FRAC_PI_2 - 0.01)).abs() < 1e-6);
        // Reversing from the clamp moves at once: no dead zone.
        s.orbit(0.0, 10.0);
        assert!(s.pitch < std::f32::consts::FRAC_PI_2 - 0.01);
        s.zoom(1e9);
        assert_eq!(s.radius, 30.0);
    }

    #[test]
    fn layout_regions_follow_each_other() {
        let l = Layout {
            ngeom: 3,
            njnt: 2,
            nbody: 4,
        };
        assert_eq!(l.slot(1) - l.slot(0), l.slot_words());
        assert_eq!(l.contacts(), l.slot(POSE_SLOTS));
        assert_eq!(l.bytes(), (l.inertia() + 4 * POSE_STRIDE) * 4);
    }
}
