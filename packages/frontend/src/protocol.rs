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
//! i32[3]  reserved
//! f32[4 + g*7 + 0..3]  geom g world position  (MuJoCo frame, metres)
//! f32[4 + g*7 + 3..7]  geom g world rotation  (quaternion, glam order x,y,z,w)
//! ```
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

/// Byte size of the pose block for `ngeom` geoms.
pub fn pose_block_bytes(ngeom: usize) -> usize {
    (POSE_HEADER + ngeom * POSE_STRIDE) * 4
}

/// Render-worker → main messages (loading progress + lifecycle).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RenderMsg {
    /// Human-readable loading-screen line.
    Progress { message: String },
    /// The scene bundle is loaded + committed; safe to forward the MuJoCo
    /// model description (before this, the worker's onmessage isn't listening).
    SceneReady,
    /// First real frames have been presented.
    Ready,
    Error { message: String },
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
