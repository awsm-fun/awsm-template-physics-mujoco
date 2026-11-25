//! What this app adds to the Stats panel (top right; the Stats chip, bottom
//! left, shows and hides it).
//!
//! awsm-renderer-stats does the measuring and the formatting: the frame
//! windows, the warm-up, the smoothing, the frame budget, the display's
//! refresh rate and missed frames, the renderer's CPU scopes and GPU passes
//! (its `StatsCollector`, in the render worker), and the panel that shows
//! the report (its `StatsPanel`, on the main thread). The panel tells the
//! render worker which groups are open (`stats-view`) and what only the page
//! can measure (`stats-page`: its own frame gaps and work); the render worker
//! posts a report about once a second.
//!
//! One group per thread, so that every number sits under the thread that
//! spent it, and every number says its unit and what it is per:
//!
//! ```text
//! frame budget               the display's refresh period, and what each lane
//!                            (the GPU, each thread) uses of it per frame.
//! display                    the display's refresh rate (measured on the main
//!                            thread, from its own animation frames), the
//!                            frames the render worker presented per second,
//!                            and the refreshes that went by without a new
//!                            frame.
//! GPU                        the GPU's time per frame, from the renderer's
//!                            timestamp queries (`n/a` without them). Its rows
//!                            are the passes.
//! renderer (render worker)   the render worker's CPU time per frame: its five
//!                            phases ([`Phase`]), and under `render (encode +
//!                            submit)` awsm-renderer's own CPU scopes.
//! sim (MuJoCo, worker)       this file: the MuJoCo worker's CPU time per fixed
//!                            step, its steps per second, the frames the render
//!                            worker drew before the sim had published the
//!                            moment they show (late), and stepping vs
//!                            publishing. A lane of the frame budget.
//! page (main thread)         the main thread's own work per frame (`camera
//!                            frame loop`).
//! ```
//!
//! **The sim's numbers.** The MuJoCo worker measures its own time (two clock
//! reads per wake; the stepping vs publishing split only while the sim group
//! is open, `{kind: "profile"}` from main) and reports about once a second
//! (`{kind: "stats"}`). Main forwards each report to the render worker
//! (`{kind: "sim-stats"}`), which smooths it here and puts the sim group in
//! its report, so the sim joins the frame budget like every other thread.
//!
//! **Late frames.** Each frame draws the sim one frame behind, between the two
//! published states around its display time (see `protocol`, "Timing"). A
//! frame that finds the sim has not yet published a state past that time holds
//! the newest one instead: the picture pauses for a frame. Late frames in
//! `LATE_WINDOWS` windows in a row mean MuJoCo is not keeping up with real
//! time: the row and the sim header turn red.

use awsm_renderer_stats::{
    duration_row, smooth, step_headline, AppSection, NamedMeans, SectionSpec, StatRow, StatsConfig,
    Window,
};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::js_sys;

/// The frame function's phases, in the order they run.
#[derive(Clone, Copy)]
pub enum Phase {
    /// Ticking the sim, then the scene's authored clips, texture UV flows and
    /// particles (`update_animations`, `ParticleDriver::tick`).
    Animate,
    /// Reading and blending the sim's pose slots, the debug overlays: written
    /// after `Animate`, so the sim wins on its nodes.
    Read,
    Camera,
    Transforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
}

const PHASE_NAMES: [&str; 5] = [
    "animations + particles",
    "read app state + interpolate",
    "set camera",
    "update transforms",
    "render (encode + submit)",
];

/// The sim group's id (the page stores whether it is open under it, and
/// tells the MuJoCo worker to split its time while it is).
pub const SIM: &str = "sim";
const SIM_LABEL: &str = "sim (MuJoCo, worker)";

/// The page's work, as the page group names it.
pub const PAGE_PART: &str = "camera frame loop";

/// The panel's configuration, the same on both threads (the main thread
/// shows the "starting…" report from it before the render worker is up).
pub fn config() -> StatsConfig {
    StatsConfig {
        phases: PHASE_NAMES.map(String::from).to_vec(),
        render_phase: Some(Phase::Render as usize),
        sections: vec![SectionSpec::new(SIM, SIM_LABEL, Some("sim"))],
        // The renderer draws on every refresh: no pacing to report.
        pacing_row: false,
        ..StatsConfig::default()
    }
}

/// The MuJoCo worker's report (`{kind: "stats"}`, about once a second).
#[derive(Clone, Debug, PartialEq)]
pub struct SimReport {
    pub span_ms: f64,
    pub steps: u32,
    /// Its CPU time for those steps: stepping and publishing.
    pub busy_ms: f64,
    /// While the sim group is open: `[simulate, publish]` ms over that many
    /// steps.
    pub split: Option<([f64; 2], u32)>,
}

impl SimReport {
    /// From the raw message (its fields are snake_case numbers).
    pub fn from_js(data: &JsValue) -> Self {
        let num = |k: &str| {
            js_sys::Reflect::get(data, &JsValue::from_str(k))
                .ok()
                .and_then(|v| v.as_f64())
        };
        Self {
            span_ms: num("span_ms").unwrap_or(0.0),
            steps: num("steps").unwrap_or(0.0) as u32,
            busy_ms: num("busy_ms").unwrap_or(0.0),
            split: match (num("step_ms"), num("publish_ms"), num("split_steps")) {
                (Some(step), Some(publish), Some(n)) => Some(([step, publish], n as u32)),
                _ => None,
            },
        }
    }
}

/// The sim group: the MuJoCo worker's reports, smoothed.
#[derive(Default)]
pub struct SimStats {
    /// Whether the collector is counting (past its warm-up): reports before
    /// that are start-up, not steady state, and are left out.
    live: bool,
    steps: Option<f64>,
    step_ms: Option<f64>,
    /// The worker's busy time per ms of real time.
    load: Option<f64>,
    /// Presented frames per second, for the sim's time per frame drawn.
    fps: Option<f64>,
    split: NamedMeans,
}

impl SimStats {
    /// Counting starts over (the scene is on screen; the collector warms up).
    pub fn restart(&mut self) {
        *self = Self::default();
    }

    /// One report from the MuJoCo worker.
    pub fn sim_report(&mut self, s: &SimReport) {
        if !self.live || s.steps == 0 || s.span_ms <= 0.0 {
            return;
        }
        let steps = f64::from(s.steps);
        self.steps = Some(smooth(self.steps, steps * 1000.0 / s.span_ms));
        self.step_ms = Some(smooth(self.step_ms, s.busy_ms / steps));
        self.load = Some(smooth(self.load, s.busy_ms / s.span_ms));
        let split = s
            .split
            .filter(|(_, n)| *n > 0)
            .map(|(ms, n)| {
                let n = f64::from(n);
                vec![
                    ("simulate (mj_step)".to_string(), ms[0] / n),
                    ("publish".to_string(), ms[1] / n),
                ]
            })
            .unwrap_or_default();
        self.split.update(split);
    }

    /// The group for one window.
    pub fn section(&mut self, window: &Window) -> AppSection {
        self.live = true;
        if window.span_ms > 0.0 {
            self.fps = Some(smooth(
                self.fps,
                f64::from(window.frames) * 1000.0 / window.span_ms,
            ));
        }
        // The sim's busy share of real time, times the real time per frame
        // drawn: its time per step × steps per frame.
        let per_frame = self
            .load
            .zip(self.fps.filter(|f| *f > 0.0))
            .map(|(load, fps)| load * 1000.0 / fps);
        let mut rows = vec![
            StatRow::new(
                "steps",
                self.steps
                    .map_or("—".to_string(), |r| format!("{r:.0} steps/s")),
            ),
            window.late.row("late frames"),
        ];
        rows.extend(
            self.split
                .entries()
                .iter()
                .map(|(name, ms)| duration_row(name.clone(), *ms, "step")),
        );
        SectionSpec::new(SIM, SIM_LABEL, Some("sim"))
            .group()
            .headline(step_headline(self.step_ms))
            .bad(window.late.bad)
            .rows(rows)
            .budget_ms(per_frame)
    }
}

/// `performance.timeOrigin` in the render worker: added to a frame timestamp,
/// it gives the absolute time every thread of the page agrees on.
pub fn worker_time_origin() -> f64 {
    js_sys::global()
        .dyn_into::<web_sys::WorkerGlobalScope>()
        .ok()
        .and_then(|g| g.performance())
        .map(|p| p.time_origin())
        .unwrap_or(0.0)
}

/// `performance.now()` in the render worker.
pub fn worker_now() -> f64 {
    js_sys::global()
        .dyn_into::<web_sys::WorkerGlobalScope>()
        .ok()
        .and_then(|g| g.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use awsm_renderer_stats::LateStats;

    fn window() -> Window {
        Window {
            seconds: 1.0,
            frames: 60,
            span_ms: 1000.0,
            late: LateStats::default(),
        }
    }

    #[test]
    fn the_sim_joins_the_budget_once_counting() {
        let mut sim = SimStats::default();
        let report = SimReport {
            span_ms: 1000.0,
            steps: 500,
            busy_ms: 120.0,
            split: Some(([100.0, 20.0], 500)),
        };
        // Before the first window (warm-up): left out.
        sim.sim_report(&report);
        assert_eq!(sim.section(&window()).headline, "—");
        sim.sim_report(&report);
        let group = sim.section(&window());
        assert_eq!(group.headline, "0.24 ms / step");
        assert_eq!(group.rows[0].value, "500 steps/s");
        assert_eq!(group.rows[2].label, "simulate (mj_step)");
        assert_eq!(group.rows[2].value, "0.20 ms / step");
        // 120 ms per second over 60 frames: 2 ms per frame drawn.
        assert!((group.budget_ms.unwrap() - 2.0).abs() < 1e-9);
    }
}
