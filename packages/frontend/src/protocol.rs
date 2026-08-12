//! The wire between the three players:
//!
//! - **MuJoCo worker → render worker**: a plain `SharedArrayBuffer` pose block
//!   (layout below) — the sim publishes every geom's world pose after each
//!   step batch; the render thread reads the latest stable snapshot per frame.
//!   This is the template's stand-in for the renderer-side "pose sink": the
//!   renderer itself knows nothing about MuJoCo or transports, it just gets
//!   world poses applied to nodes.
//! - **main ↔ render worker**: small serde messages (progress, resize, camera
//!   gestures) over `postMessage` — none of them are per-frame hot paths.
//! - **main ↔ MuJoCo worker**: raw JS objects (`{kind: ...}`) — see
//!   `web/workers/mujoco-worker.js`; the model description crosses once at
//!   startup, tagged `kind: "model"`, and is forwarded to the render worker
//!   verbatim as `kind: "mujoco-model"`.
//!
//! ## Pose block layout (the SharedArrayBuffer)
//!
//! ```text
//! i32[0]  seq        seqlock: writer makes it ODD, writes poses, makes it EVEN
//! i32[1]  steps      total sim steps published (monotonic; readers may show it)
//! i32[2]  ngeom
//! i32[3]  ncon       LIVE contact count this frame, capped at MAX_CONTACTS
//! f32[4 + g*7 + 0..3]  geom g world position  (MuJoCo frame, metres)
//! f32[4 + g*7 + 3..7]  geom g world rotation  (quaternion, MuJoCo order w,x,y,z)
//! f32[C + c*7 + 0..3]  contact c world position  (MuJoCo frame, metres)
//! f32[C + c*7 + 3..6]  contact c world NORMAL    (unit, MuJoCo frame)
//! f32[C + c*7 + 6]     contact c NORMAL FORCE    (newtons)
//! f32[J + j*6 + 0..3]  joint j world ANCHOR      (MuJoCo frame, metres)
//! f32[J + j*6 + 3..6]  joint j world AXIS        (unit, MuJoCo frame)
//! f32[I + b*7 + 0..3]  body b INERTIAL position  (MuJoCo frame, metres)
//! f32[I + b*7 + 3..7]  body b INERTIAL rotation  (quaternion, MuJoCo w,x,y,z)
//! f32[B + b*7 + 0..3]  body b WORLD position     (MuJoCo frame, metres)
//! f32[B + b*7 + 3..7]  body b WORLD rotation     (quaternion, MuJoCo w,x,y,z)
//!     where C = 4 + ngeom*7, J = C + MAX_CONTACTS*7, I = J + njnt*6,
//!           B = I + nbody*7
//! ```
//!
//! ## The contact region is a DEBUG overlay
//!
//! Contacts are dev tooling and live only in this template — never in the
//! renderer, the editor, or a bundle. They are published in the same seqlock as
//! the poses so an overlay can never draw contacts from a different step than
//! the bodies they belong to.
//!
//! The count varies every step while the block must be a fixed size, so the
//! region is preallocated to [`MAX_CONTACTS`] and the live count rides in the
//! header — the same shape the renderer's tendon channel uses, and for the same
//! reason. A step with more contacts than the cap publishes the first
//! `MAX_CONTACTS`; that is a visualisation losing detail, not the sim losing
//! anything.
//!
//! Quaternions are in MUJOCO's `[w, x, y, z]` order, not glam's — the block IS a
//! stream frame in the documented pose-sink convention, so a producer could dump
//! it verbatim as a capture file and the sink takes it with no reshaping.
//!
//! All values little-endian f32/i32; MuJoCo's f64 state is narrowed to f32 by
//! the worker, so the f64 question never reaches the renderer. The seqlock is
//! single-writer (the sim worker) / single-reader (the render worker): the
//! reader copies the pose span, then re-checks `seq` — if it changed or is odd,
//! the copy is torn and it retries (or keeps last frame's poses).

use serde::{Deserialize, Serialize};

/// f32 slots per geom in the pose block: pos xyz + quat xyzw.
pub const POSE_STRIDE: usize = 7;
/// i32/f32 header slots before the pose span.
pub const POSE_HEADER: usize = 4;

/// f32 slots per contact in the debug region: world position, unit normal, and
/// the normal force in newtons.
pub const CONTACT_STRIDE: usize = 7;

/// How many contacts the block reserves room for.
///
/// A humanoid on the ground runs a handful; a pile of objects can spike. This
/// caps the overlay's cost (and the block at ~6 KB) without ever throttling the
/// sim, which does not know the overlay exists.
pub const MAX_CONTACTS: usize = 256;

/// f32 slots per joint in the debug region: world anchor + unit axis.
pub const JOINT_STRIDE: usize = 6;

/// f32 index where the contact region starts, for `ngeom` geoms.
pub fn contact_offset(ngeom: usize) -> usize {
    POSE_HEADER + ngeom * POSE_STRIDE
}

/// f32 index where the joint region starts.
///
/// Unlike contacts, the joint count is FIXED by the model, so this region needs
/// no live count and no pool slack — `njnt` entries, every one of them written
/// every step.
pub fn joint_offset(ngeom: usize) -> usize {
    contact_offset(ngeom) + MAX_CONTACTS * CONTACT_STRIDE
}

/// f32 index where the inertia region starts.
///
/// One entry per body, same shape as a geom pose — the equivalent inertia BOX
/// is a rigid frame like any other. Its half-extents are static, derived once
/// from the model's mass and inertia, so they never cross the wire.
pub fn inertia_offset(ngeom: usize, njnt: usize) -> usize {
    joint_offset(ngeom) + njnt * JOINT_STRIDE
}

/// f32 index where the BODY region starts.
///
/// Unlike the three regions before it, this one is not debug tooling — it is a
/// second real pose channel. A deformable (a MuJoCo *flex*) imports as an
/// ordinary skinned mesh whose joints are the bodies its cage rides, so driving
/// one is just `apply_body_poses` alongside `apply_geom_poses`: no vertex ever
/// crosses the wire, only `nbody` frames.
///
/// These are BODY frames (`xpos`/`xquat`), which is not what the inertia region
/// carries — that one holds the *inertial* frames, offset to each centre of mass
/// and rotated onto the principal axes. Feeding one to the other's consumer
/// would deform the mesh plausibly and wrongly.
pub fn body_offset(ngeom: usize, njnt: usize, nbody: usize) -> usize {
    inertia_offset(ngeom, njnt) + nbody * POSE_STRIDE
}

/// Byte size of the pose block, including the debug regions and the body poses.
pub fn pose_block_bytes(ngeom: usize, njnt: usize, nbody: usize) -> usize {
    (body_offset(ngeom, njnt, nbody) + nbody * POSE_STRIDE) * 4
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

/// Main → render worker: canvas backing-store resize (device pixels).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResizeMsg {
    Canvas { width: u32, height: u32 },
}

/// Main → render worker: orbit-camera gestures (main owns the DOM events; the
/// camera itself lives on the render thread).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CameraMsg {
    Orbit { dx: f32, dy: f32 },
    Zoom { dy: f32 },
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
