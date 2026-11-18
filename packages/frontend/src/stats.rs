//! The Stats panel (top right) and the Stats chip (bottom left) that shows and
//! hides it, plus the render worker's half of what it measures.
//!
//! One group per thread, so that every number sits under the thread that
//! spent it, and every number says its unit and what it is per:
//!
//! ```text
//! display                    the display's refresh rate (measured here, from
//!                            this thread's own animation frames), the frames
//!                            the render worker presented per second, and the
//!                            refreshes that went by without a new frame.
//! GPU                        the GPU's time per frame, from the renderer's
//!                            timestamp queries (`n/a` without them). Its rows
//!                            are the passes; on GPUs that run passes
//!                            overlapped (Apple's) they add up to more than
//!                            the frame.
//! renderer (render worker)   the render worker's CPU time per frame: its four
//!                            phases, awsm-renderer's own CPU scopes under
//!                            `render (encode + submit)`, and what timing those
//!                            scopes adds (`timing the passes`).
//! sim (MuJoCo, worker)       the MuJoCo worker's CPU time per fixed step, its
//!                            steps per second, the frames the render worker
//!                            drew before the sim had published the moment
//!                            they show (late), and stepping vs publishing.
//! page (main thread)         this thread's own work per frame: its two
//!                            animation-frame loops, timed apart (`camera
//!                            frame loop`, `refresh probe`).
//! ```
//!
//! A group's header always shows its headline number. Expanding a group shows
//! its breakdown, and only then is the breakdown measured: the renderer group
//! times the frame function's phases and switches awsm-renderer's CPU tier to
//! `SubFrame` on one frame in [`PASS_TIMING_EVERY`] (`Off` otherwise; see
//! [`PassSampler`] for why); the GPU group passes on the per-pass GPU times
//! (the `Frame` tier, always on for the headline, already times every pass);
//! the sim group splits the MuJoCo worker's time into stepping and publishing.
//! While the panel is hidden every group counts as collapsed. Main tells the
//! render worker with a [`ProfileMsg`](crate::protocol::ProfileMsg) and the
//! MuJoCo worker with `{kind: "profile"}`; the numbers come back about once a
//! second, in the reports both already send.
//!
//! **Late frames.** Each frame draws the sim one frame behind, between the two
//! published states around its display time (see `protocol`, "Timing"). A
//! frame that finds the sim has not yet published a state past that time holds
//! the newest one instead: the picture pauses for a frame. Late frames in
//! [`LATE_WINDOWS`] windows in a row mean MuJoCo is not keeping up with real
//! time: the row and the sim header turn red.
//!
//! **Display rate.** No browser API reports the refresh rate, so the main
//! thread runs its own (otherwise empty) `requestAnimationFrame` loop and
//! measures it: the main thread does almost nothing here, so its frames land on
//! every refresh. The period is the mean of the recent frame intervals that sit
//! within 15% of their median, which ignores the odd late frame.
//!
//! **Missed frames.** The render worker reports, about once a second, how many
//! frames it presented and how much time (`span_ms`, from rAF timestamp to rAF
//! timestamp) they covered. That span lasted `round(span / period)` refreshes;
//! every refresh beyond the frames presented showed the previous picture again
//! — a missed frame, which is what makes motion hitch. A long gap (a hidden
//! tab, a debugger pause) is left out of the span together with its frame, so
//! pauses never count as misses.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use awsm_renderer::profiling::{
    cpu_timing_stats, gpu_frame_timing, gpu_timing_stats, TimingStat, GPU_FRAME_SCOPE,
};
use dominator::{clone, html, Dom};
use futures_signals::map_ref;
use futures_signals::signal::{Mutable, Signal, SignalExt};
use futures_signals::signal_vec::SignalVecExt;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::protocol::{FrameReport, Profile};

// ── Render worker side ──────────────────────────────────────────────────────

/// Roughly how often the render worker reports (ms of presented frames).
const REPORT_SPAN_MS: f64 = 1000.0;
/// A frame interval longer than this is a pause, not a slow frame: it is left
/// out of the window (with its frame), so it never counts as missed frames.
const GAP_MS: f64 = 250.0;

/// The frame function's phases, in the order they run.
#[derive(Clone, Copy)]
pub enum Phase {
    /// Everything up to the transforms being written: ticking the sim,
    /// reading and blending its pose slots, the debug overlays.
    Read,
    Camera,
    Transforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
}
pub const PHASES: usize = 4;
const PHASE_NAMES: [&str; PHASES] = [
    "read app state + interpolate",
    "set camera",
    "update transforms",
    "render (encode + submit)",
];
/// awsm-renderer's own scope around the whole of `render()`: the same time as
/// our render phase, so it is not listed among the passes.
const WHOLE_RENDER_SCOPE: &str = "Render";
/// The per-pass CPU scopes are on for one frame in this many (see
/// [`PassSampler`]).
const PASS_TIMING_EVERY: u64 = 16;
/// The scope cost in use is the median of this many estimates, one per report
/// that had frames of both kinds: a few seconds' worth.
const SCOPE_ESTIMATES: usize = 5;

/// Times the phases of one frame. Only built while the renderer group is open.
pub struct PhaseClock {
    t: f64,
    ms: [f64; PHASES],
    scopes: bool,
}

impl PhaseClock {
    /// Whether awsm-renderer's per-pass CPU scopes are on for this frame.
    pub fn scopes(&self) -> bool {
        self.scopes
    }
}

/// Charge the time since the last lap to `phase` (a no-op without a clock).
pub fn lap(clock: &mut Option<PhaseClock>, phase: Phase) {
    if let Some(c) = clock {
        let now = worker_now();
        c.ms[phase as usize] += now - c.t;
        c.t = now;
    }
}

/// The renderer group's breakdown, while it is expanded.
///
/// awsm-renderer's per-pass CPU scopes are costly here: each looks the clock
/// up through the global object at both ends, which in a worker takes far
/// longer than many of the passes it times. So that looking does not change
/// what is looked at:
///
/// * the scopes are on for one frame in [`PASS_TIMING_EVERY`] only, and the
///   phases are taken from the other frames, which are undisturbed;
/// * what the scopes cost is measured, as the difference between the render
///   phase with and without them, and reported as a row of its own. One
///   estimate is made per report and the median of the last few is used, so
///   that a frame that hitched (a pipeline compile, a collection) does not
///   pass for the scopes' doing;
/// * each pass has the scope's share of its reading taken out. A scope's
///   reading runs from the end of its first lookup to the end of its second,
///   so it contains one of the two: half of what the scope costs.
///
/// Everything is kept as running totals; a report sends the means of the
/// change since the last one (3–4 timed frames a second at 60 fps).
#[derive(Default)]
struct PassSampler {
    /// Over the frames WITHOUT the scopes: each phase, and the frames.
    phase_ms: [f64; PHASES],
    plain_frames: u64,
    /// The frames WITH them so far, and the render phase on those since the
    /// last report.
    timed_frames: u64,
    window_timed_render_ms: f64,
    /// Per scope: the renderer's totals as last read, and our running total
    /// with the scope's own share taken out.
    passes: HashMap<&'static str, PassTotal>,
    /// The latest estimates of what one scope costs, in milliseconds, and the
    /// running total of what the scopes cost by them.
    scope_estimates: VecDeque<f64>,
    overhead_ms: f64,
    /// The totals as of the last report, and the pass means it sent (kept for
    /// a window that happens to hold no timed frame).
    reported: Reported,
    pass_means: Vec<(String, f64)>,
}

#[derive(Default)]
struct PassTotal {
    seen_ms: f64,
    seen_count: u64,
    net_ms: f64,
    reported_ms: f64,
    reported_count: u64,
}

#[derive(Default)]
struct Reported {
    phase_ms: [f64; PHASES],
    plain_frames: u64,
    timed_frames: u64,
    overhead_ms: f64,
}

impl PassSampler {
    /// Start from the renderer's totals as they stand: they are cumulative,
    /// and may hold an earlier expansion's.
    fn new() -> Self {
        let passes = cpu_timing_stats()
            .into_iter()
            .map(|(name, s)| {
                let pass = PassTotal {
                    seen_ms: s.total_ms,
                    seen_count: s.count,
                    reported_count: s.count,
                    ..PassTotal::default()
                };
                (name, pass)
            })
            .collect();
        Self {
            passes,
            ..Self::default()
        }
    }

    /// Whether the next frame runs with the scopes on.
    fn timing_scopes(&self) -> bool {
        (self.plain_frames + self.timed_frames).is_multiple_of(PASS_TIMING_EVERY)
    }

    fn add(&mut self, clock: PhaseClock) {
        if clock.scopes {
            self.window_timed_render_ms += clock.ms[Phase::Render as usize];
            self.timed_frames += 1;
        } else {
            for (sum, ms) in self.phase_ms.iter_mut().zip(clock.ms) {
                *sum += ms;
            }
            self.plain_frames += 1;
        }
    }

    /// Fold in what the scopes read since the last report, and put this
    /// window's means into `report`.
    fn report(&mut self, report: &mut FrameReport) {
        let mut window = Vec::new();
        let mut scope_runs = 0;
        for (name, stat) in cpu_timing_stats() {
            let pass = self.passes.entry(name).or_default();
            let runs = stat.count.saturating_sub(pass.seen_count);
            let read_ms = stat.total_ms - pass.seen_ms;
            pass.seen_count = stat.count;
            pass.seen_ms = stat.total_ms;
            scope_runs += runs;
            window.push((name, runs as f64, read_ms));
        }

        // What one scope costs: what a frame with the scopes took over one
        // without in this window, shared among the scopes it ran.
        let r = &self.reported;
        let plain = self.plain_frames - r.plain_frames;
        let timed = self.timed_frames - r.timed_frames;
        let render = Phase::Render as usize;
        if plain > 0 && timed > 0 && scope_runs > 0 {
            let plain_ms = (self.phase_ms[render] - r.phase_ms[render]) / plain as f64;
            let timed_ms = self.window_timed_render_ms / timed as f64;
            let estimate = (timed_ms - plain_ms) * timed as f64 / scope_runs as f64;
            if self.scope_estimates.len() == SCOPE_ESTIMATES {
                self.scope_estimates.pop_front();
            }
            self.scope_estimates.push_back(estimate);
        }
        self.window_timed_render_ms = 0.0;
        let scope_ms = median(&self.scope_estimates).max(0.0);
        for (name, runs, read_ms) in window {
            self.overhead_ms += runs * scope_ms;
            if let Some(pass) = self.passes.get_mut(name) {
                pass.net_ms += (read_ms - runs * scope_ms / 2.0).max(0.0);
            }
        }

        let r = &mut self.reported;
        if plain > 0 {
            let n = plain as f64;
            let mut phases = [0.0; PHASES];
            for (i, mean) in phases.iter_mut().enumerate() {
                *mean = (self.phase_ms[i] - r.phase_ms[i]) / n;
            }
            report.phases_ms = Some(phases);
        }
        // Passes are per TIMED frame: only those ran them.
        if timed > 0 {
            let n = timed as f64;
            self.pass_means = self
                .passes
                .iter_mut()
                .filter(|(name, _)| **name != WHOLE_RENDER_SCOPE)
                .filter_map(|(name, pass)| {
                    let ran = pass.seen_count > pass.reported_count;
                    let mean = (pass.net_ms - pass.reported_ms) / n;
                    pass.reported_ms = pass.net_ms;
                    pass.reported_count = pass.seen_count;
                    ran.then(|| (name.to_string(), mean))
                })
                .collect();
            r.timed_frames = self.timed_frames;
        }
        report.cpu_passes = self.pass_means.clone();
        // Over every frame: what looking adds to the renderer's time per frame.
        let all = plain + timed;
        if all > 0 {
            report.pass_timing_ms = Some((self.overhead_ms - r.overhead_ms) / all as f64);
        }
        r.phase_ms = self.phase_ms;
        r.plain_frames = self.plain_frames;
        r.overhead_ms = self.overhead_ms;
    }
}

/// The middle value (the upper of the two for an even count), or 0 for none.
fn median(values: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

/// Running `(name, total_ms)` per awsm-renderer scope.
type Totals = Vec<(&'static str, f64)>;

fn totals(stats: Vec<(&'static str, TimingStat)>, skip: &str) -> Totals {
    stats
        .into_iter()
        .filter(|(name, _)| *name != skip)
        .map(|(name, s)| (name, s.total_ms))
        .collect()
}

/// Each scope's time since `base`, divided by `per`; scopes that did not run
/// are left out.
fn means(now: &Totals, base: &Totals, per: f64) -> Vec<(String, f64)> {
    now.iter()
        .filter_map(|(name, total)| {
            let before = base.iter().find(|(n, _)| n == name).map_or(0.0, |b| b.1);
            (*total > before).then(|| (name.to_string(), (total - before) / per))
        })
        .collect()
}

/// The GPU's whole-frame `(total_ms, sampled frames)` so far.
fn gpu_frames() -> Option<(f64, u64)> {
    gpu_frame_timing().map(|s| (s.total_ms, s.count))
}

/// Counts presented frames on the render worker. Call [`FrameWindow::frame`]
/// once per frame; post what it returns.
pub struct FrameWindow {
    prev_t: Option<f64>,
    frames: u32,
    span_ms: f64,
    cpu_ms: f64,
    late: u32,
    gpu_available: bool,
    /// The GPU whole-frame totals at the window's start.
    gpu_base: Option<(f64, u64)>,
    profile: Profile,
    /// The renderer group's breakdown, while it is expanded.
    sampler: Option<PassSampler>,
    /// GPU group: the pass totals and the sampled-frame count at the window's
    /// start.
    gpu_pass_base: (u64, Totals),
}

impl FrameWindow {
    /// `gpu_available`: the renderer has timestamp queries.
    pub fn new(gpu_available: bool) -> Self {
        Self {
            prev_t: None,
            frames: 0,
            span_ms: 0.0,
            cpu_ms: 0.0,
            late: 0,
            gpu_available,
            gpu_base: None,
            profile: Profile::default(),
            sampler: None,
            gpu_pass_base: (0, Vec::new()),
        }
    }

    /// Which groups are open (see [`Profile`]): start (or drop) the baselines
    /// of a group that opened (or closed).
    pub fn set_profile(&mut self, p: Profile) {
        if p.renderer != self.profile.renderer {
            self.sampler = p.renderer.then(PassSampler::new);
        }
        if p.gpu != self.profile.gpu {
            self.gpu_pass_base = if p.gpu {
                (
                    gpu_frames().map_or(0, |f| f.1),
                    totals(gpu_timing_stats(), GPU_FRAME_SCOPE),
                )
            } else {
                (0, Vec::new())
            };
        }
        self.profile = p;
    }

    /// Start a frame at `t` (`performance.now()`): its phase clock while the
    /// renderer group is open. The caller switches awsm-renderer's CPU tier to
    /// [`PhaseClock::scopes`] (`Off` without a clock).
    pub fn begin_frame(&mut self, t: f64) -> Option<PhaseClock> {
        self.sampler.as_ref().map(|s| PhaseClock {
            t,
            ms: [0.0; PHASES],
            scopes: s.timing_scopes(),
        })
    }

    /// One presented frame: `t` is its rAF timestamp, `cpu_ms` the CPU time
    /// the frame took, `late` whether it had to hold the sim's newest state,
    /// `phases` its phase timings while the renderer group is open. Returns a
    /// report when a window is full.
    pub fn frame(
        &mut self,
        t: f64,
        cpu_ms: f64,
        late: bool,
        phases: Option<PhaseClock>,
    ) -> Option<FrameReport> {
        // Pauses or not: the renderer's scopes count every frame they ran.
        if let (Some(s), Some(p)) = (self.sampler.as_mut(), phases) {
            s.add(p);
        }
        let prev = self.prev_t.replace(t)?; // the first frame only starts the clock
        let dt = t - prev;
        if !(dt > 0.0 && dt <= GAP_MS) {
            return None; // a pause: neither the gap nor this frame is counted
        }
        self.frames += 1;
        self.span_ms += dt;
        self.cpu_ms += cpu_ms;
        self.late += u32::from(late);
        if self.span_ms < REPORT_SPAN_MS {
            return None;
        }

        // GPU means are per SAMPLED frame: the readback skips frames while
        // one is in flight.
        let gpu_now = gpu_frames();
        let gpu_ms = match (self.gpu_base, gpu_now) {
            (Some((t0, c0)), Some((t1, c1))) if c1 > c0 => Some((t1 - t0) / (c1 - c0) as f64),
            _ => None,
        };
        self.gpu_base = gpu_now;
        let mut report = FrameReport {
            frames: self.frames,
            span_ms: self.span_ms,
            cpu_ms: self.cpu_ms,
            late: self.late,
            gpu_available: self.gpu_available,
            gpu_ms,
            ..FrameReport::default()
        };
        if self.profile.gpu {
            let sampled = gpu_now.map_or(0, |f| f.1);
            let now = totals(gpu_timing_stats(), GPU_FRAME_SCOPE);
            let (c0, base) = &self.gpu_pass_base;
            if sampled > *c0 {
                report.gpu_passes = means(&now, base, (sampled - c0) as f64);
            }
            self.gpu_pass_base = (sampled, now);
        }
        if let Some(s) = self.sampler.as_mut() {
            s.report(&mut report);
        }
        self.frames = 0;
        self.span_ms = 0.0;
        self.cpu_ms = 0.0;
        self.late = 0;
        Some(report)
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

// ── Main thread side ────────────────────────────────────────────────────────

/// Smoothing weight of each new window: a real change shows within 2–3
/// windows, while the readout holds steady against noise.
const EMA_ALPHA: f64 = 0.35;
/// Frames that must be presented after `Ready` before anything is counted:
/// start-up is not steady state (caches are cold, the wasm engine is still
/// optimizing), and those first windows read badly.
const WARMUP_FRAMES: u32 = 60;
/// Main-thread frame intervals kept for the refresh estimate, and how many are
/// needed before there is one.
const REFRESH_SAMPLES: usize = 120;
const REFRESH_MIN_SAMPLES: usize = 30;
/// Intervals within this share of the median count as single refreshes.
const REFRESH_INLIER: f64 = 0.15;
/// Late frames in this many windows in a row turn the sim red: a steady
/// condition, not a one-off such as a tab switch.
const LATE_WINDOWS: u32 = 3;

/// The MuJoCo worker's report (`{kind: "stats"}`, about once a second).
pub struct SimReport {
    pub span_ms: f64,
    pub steps: u32,
    /// Its CPU time for those steps: stepping and publishing.
    pub busy_ms: f64,
    /// While the sim group is open: `[simulate, publish]` ms over that many
    /// steps.
    pub split: Option<([f64; 2], u32)>,
}

/// One row of a group: label left, value right.
#[derive(Clone)]
struct Row {
    label: String,
    value: String,
    /// A breakdown of the row above (the renderer's scopes under `render`).
    nested: bool,
    bad: bool,
}

impl Row {
    fn new(label: impl Into<String>, value: String) -> Self {
        Self {
            label: label.into(),
            value,
            nested: false,
            bad: false,
        }
    }
}

/// One collapsible group of the panel.
struct Group {
    id: &'static str,
    label: &'static str,
    open: Mutable<bool>,
    headline: Mutable<String>,
    bad: Mutable<bool>,
    rows: Mutable<Vec<Row>>,
}

impl Group {
    fn new(window: &web_sys::Window, prefix: &str, id: &'static str, label: &'static str) -> Self {
        Self {
            id,
            label,
            open: Mutable::new(stored_bool(window, &open_key(prefix, id)) == Some(true)),
            headline: Mutable::new(String::new()),
            bad: Mutable::new(false),
            rows: Mutable::new(Vec::new()),
        }
    }

    fn set(&self, headline: String, rows: Vec<Row>, bad: bool) {
        self.headline.set_neq(headline);
        self.bad.set_neq(bad);
        self.rows.set(rows);
    }

    fn dom(&self, prefix: &'static str) -> Dom {
        let (id, label) = (self.id, self.label);
        let open = self.open.clone();
        let rows_id = format!("stats-rows-{id}");
        html!("div", {
            .child(html!("button", {
                .class("stats-group")
                .attr("type", "button")
                .attr("aria-controls", &rows_id)
                .class_signal("stats-bad", self.bad.signal())
                .attr_signal("aria-expanded", open.signal().map(|o| if o { "true" } else { "false" }))
                .child(html!("span", {
                    .text_signal(open.signal().map(move |o| {
                        format!("{} {label}", if o { '▾' } else { '▸' })
                    }))
                }))
                .text(" ") // for a screen reader; the layout ignores it
                .child(html!("span", {
                    .text_signal(self.headline.signal_cloned())
                }))
                .event(clone!(open => move |_: dominator::events::Click| {
                    let next = !open.get();
                    open.set_neq(next);
                    if let Some(w) = web_sys::window() {
                        store_bool(&w, &open_key(prefix, id), next);
                    }
                }))
            }))
            .child(html!("div", {
                .class("stats-rows")
                .attr("id", &rows_id)
                .prop_signal("hidden", open.signal().map(|o| !o))
                .children_signal_vec(self.rows.signal_cloned().to_signal_vec().map(|row| {
                    html!("div", {
                        .class("stats-row")
                        .apply_if(row.nested, |d| d.class("stats-nested"))
                        .apply_if(row.bad, |d| d.class("stats-bad"))
                        .child(html!("span", { .text(&row.label) }))
                        .child(html!("span", { .text(&row.value) }))
                    })
                }))
            }))
        })
    }
}

/// `<prefix>stats_open_<id>`: whether a group is expanded.
fn open_key(prefix: &str, id: &str) -> String {
    format!("{prefix}stats_open_{id}")
}

/// The main thread's own per-frame work since the last report, for the page
/// group.
#[derive(Default)]
struct PageWork {
    frames: Cell<u32>,
    camera_ms: Cell<f64>,
    refresh_ms: Cell<f64>,
}

/// The panel's state; owned by the main thread.
pub struct Stats {
    panel: Mutable<bool>,
    display: Group,
    gpu: Group,
    renderer: Group,
    sim: Group,
    page: Group,
    state: RefCell<State>,
    refresh: Rc<RefCell<VecDeque<f64>>>,
    work: Rc<PageWork>,
    perf: Option<web_sys::Performance>,
}

#[derive(Default)]
struct State {
    ready: bool,
    warm_frames: u32,
    /// Shown in place of every figure until there are numbers.
    placeholder: Option<&'static str>,
    fps_ema: Option<f64>,
    cpu_ema: Option<f64>,
    gpu_ema: Option<f64>,
    gpu_available: bool,
    step_ema: Option<f64>,
    steps_ema: Option<f64>,
    page_ema: Option<f64>,
    /// The page's two animation-frame loops, per frame.
    page_rows: Vec<(String, f64)>,
    missed_rate: Option<f64>,
    /// Signed running sum of (refreshes - frames): a window whose ends fall
    /// unevenly can read ±1, and the next one cancels it.
    missed_sum: f64,
    late_rate: f64,
    late_total: u64,
    late_windows: u32,
    /// The breakdowns, smoothed per name. An entry missing from a report is no
    /// longer measured (its group closed, or the pass did not run) and drops
    /// out rather than show a stale figure.
    phases: Vec<(String, f64)>,
    cpu_passes: Vec<(String, f64)>,
    pass_timing: Option<f64>,
    gpu_passes: Vec<(String, f64)>,
    sim_phases: Vec<(String, f64)>,
}

impl Stats {
    /// Mount the panel and its chip, and start measuring the display.
    /// `prefix` is this app's `localStorage` prefix: `<prefix>stats` remembers
    /// whether the panel is open, `<prefix>stats_open_<group>` each group.
    pub fn install(window: &web_sys::Window, prefix: &'static str) -> Rc<Self> {
        let group = |id, label| Group::new(window, prefix, id, label);
        let stats = Rc::new(Self {
            // Off by default: on a phone the panel covers a lot of the screen.
            // A stored "open" is honoured only on a desktop; on a touch device
            // the chip still opens it for the session.
            panel: Mutable::new(
                !coarse_pointer(window)
                    && stored_bool(window, &format!("{prefix}stats")) == Some(true),
            ),
            display: group("display", "display"),
            gpu: group("gpu", "GPU"),
            renderer: group("renderer", "renderer (render worker)"),
            sim: group("sim", "sim (MuJoCo, worker)"),
            page: group("page", "page (main thread)"),
            state: RefCell::new(State {
                placeholder: Some("starting…"),
                ..State::default()
            }),
            refresh: Rc::new(RefCell::new(VecDeque::with_capacity(REFRESH_SAMPLES))),
            work: Rc::new(PageWork::default()),
            perf: window.performance(),
        });
        stats.show(&stats.state.borrow());

        let panel = stats.panel.clone();
        dominator::append_dom(
            &dominator::body(),
            html!("div", {
                .child(html!("div", {
                    .class("stats")
                    .visible_signal(panel.signal())
                    .children(stats.groups().map(|g| g.dom(prefix)))
                }))
                .child(html!("button", {
                    .class("stats-btn")
                    .class_signal("active", panel.signal())
                    .text("Stats")
                    .event(clone!(panel => move |_: dominator::events::Click| {
                        let next = !panel.get();
                        panel.set_neq(next);
                        if let Some(w) = web_sys::window() {
                            store_bool(&w, &format!("{prefix}stats"), next);
                        }
                    }))
                }))
            }),
        );

        measure_refresh(
            stats.refresh.clone(),
            stats.work.clone(),
            stats.perf.clone(),
        );
        stats
    }

    fn groups(&self) -> [&Group; 5] {
        [
            &self.display,
            &self.gpu,
            &self.renderer,
            &self.sim,
            &self.page,
        ]
    }

    /// Which render-worker groups are expanded (none while the panel is
    /// hidden).
    pub fn profile(&self) -> Profile {
        let panel = self.panel.get();
        Profile {
            renderer: panel && self.renderer.open.get(),
            gpu: panel && self.gpu.open.get(),
        }
    }

    /// [`Self::profile`] as it changes.
    pub fn profile_signal(&self) -> impl Signal<Item = Profile> {
        map_ref! {
            let panel = self.panel.signal(),
            let renderer = self.renderer.open.signal(),
            let gpu = self.gpu.open.signal() =>
            Profile { renderer: *panel && *renderer, gpu: *panel && *gpu }
        }
        .dedupe()
    }

    /// Whether the sim group is expanded (and the panel shown), as it changes.
    pub fn sim_profile_signal(&self) -> impl Signal<Item = bool> {
        map_ref! {
            let panel = self.panel.signal(),
            let sim = self.sim.open.signal() => *panel && *sim
        }
        .dedupe()
    }

    /// Run the page's per-frame camera work, timed for the page group.
    pub fn time_camera(&self, f: impl FnOnce()) {
        let t0 = now(&self.perf);
        f();
        let w = &self.work;
        w.camera_ms.set(w.camera_ms.get() + now(&self.perf) - t0);
    }

    /// The renderer is presenting real frames: start the warm-up.
    pub fn mark_ready(&self) {
        let mut st = self.state.borrow_mut();
        st.ready = true;
        st.placeholder = Some("warming up…");
        self.show(&st);
    }

    /// One report from the render worker.
    pub fn window(&self, r: FrameReport) {
        let mut st = self.state.borrow_mut();
        // The page's work since the last report, warm or not.
        let page_frames = self.work.frames.replace(0);
        let page_ms = [
            self.work.camera_ms.replace(0.0),
            self.work.refresh_ms.replace(0.0),
        ];
        if !st.ready || r.frames == 0 || r.span_ms <= 0.0 {
            return;
        }
        if st.warm_frames < WARMUP_FRAMES {
            st.warm_frames += r.frames;
            return;
        }
        st.placeholder = None;
        let frames = f64::from(r.frames);

        st.fps_ema = smooth(st.fps_ema, frames * 1000.0 / r.span_ms);
        if let Some(p) = refresh_period(&self.refresh.borrow()) {
            let missed = (r.span_ms / p).round() - frames;
            st.missed_sum += missed;
            st.missed_rate = Some(missed.max(0.0) * 1000.0 / r.span_ms);
        }
        st.gpu_available = r.gpu_available;
        if let Some(ms) = r.gpu_ms {
            st.gpu_ema = smooth(st.gpu_ema, ms);
        }
        st.cpu_ema = smooth(st.cpu_ema, r.cpu_ms / frames);
        if page_frames > 0 {
            let n = f64::from(page_frames);
            st.page_ema = smooth(st.page_ema, (page_ms[0] + page_ms[1]) / n);
            let rows = vec![
                ("camera frame loop".to_string(), page_ms[0] / n),
                ("refresh probe".to_string(), page_ms[1] / n),
            ];
            st.page_rows = smooth_named(&st.page_rows, rows);
        }

        st.late_total += u64::from(r.late);
        st.late_rate = f64::from(r.late) * 1000.0 / r.span_ms;
        st.late_windows = if r.late > 0 { st.late_windows + 1 } else { 0 };

        let phases: Vec<(String, f64)> = r
            .phases_ms
            .map(|p| PHASE_NAMES.iter().map(|n| n.to_string()).zip(p).collect())
            .unwrap_or_default();
        st.phases = smooth_named(&st.phases, phases);
        st.cpu_passes = smooth_named(&st.cpu_passes, r.cpu_passes);
        st.pass_timing = r.pass_timing_ms.and_then(|ms| smooth(st.pass_timing, ms));
        st.gpu_passes = smooth_named(&st.gpu_passes, r.gpu_passes);
        self.show(&st);
    }

    /// One report from the MuJoCo worker.
    pub fn sim_window(&self, s: SimReport) {
        let mut st = self.state.borrow_mut();
        if st.placeholder.is_some() || s.steps == 0 || s.span_ms <= 0.0 {
            return;
        }
        let steps = f64::from(s.steps);
        st.steps_ema = smooth(st.steps_ema, steps * 1000.0 / s.span_ms);
        st.step_ema = smooth(st.step_ema, s.busy_ms / steps);
        let split: Vec<(String, f64)> = s
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
        st.sim_phases = smooth_named(&st.sim_phases, split);
        self.show(&st);
    }

    fn show(&self, st: &State) {
        if let Some(p) = st.placeholder {
            for g in self.groups() {
                let headline = if g.id == "display" { "—" } else { p };
                g.set(headline.to_string(), Vec::new(), false);
            }
            return;
        }

        let hz = refresh_period(&self.refresh.borrow()).map_or("measuring…".to_string(), |p| {
            format!("{:.0} Hz", 1000.0 / p)
        });
        let fps = st.fps_ema.unwrap_or(0.0);
        self.display.set(
            format!("{hz} · {fps:.0} fps"),
            vec![
                Row::new("refresh rate", hz),
                Row::new("frames presented", format!("{fps:.0} / s")),
                Row::new(
                    "missed frames",
                    st.missed_rate.map_or("—".to_string(), |r| {
                        format!("{r:.0} / s · {} total", st.missed_sum.max(0.0).round())
                    }),
                ),
            ],
            false,
        );

        self.gpu.set(
            match (st.gpu_ema, st.gpu_available) {
                (Some(ms), _) => per_frame(ms),
                (None, false) => "n/a".to_string(),
                (None, true) => "—".to_string(),
            },
            slowest_first(&st.gpu_passes)
                .map(|(name, ms)| Row::new(name, per_frame(*ms)))
                .collect(),
            false,
        );

        // The phases in the order they run, with awsm-renderer's CPU scopes
        // listed under the phase that runs them.
        let mut rows = Vec::new();
        for (name, ms) in &st.phases {
            rows.push(Row::new(name, per_frame(*ms)));
            if name != PHASE_NAMES[Phase::Render as usize] || st.cpu_passes.is_empty() {
                continue;
            }
            let nested = |label: &str, ms: f64| Row {
                nested: true,
                ..Row::new(label, per_frame(ms))
            };
            rows.extend(slowest_first(&st.cpu_passes).map(|(n, ms)| nested(n, *ms)));
            // What render() spends that no scope accounts for: uploads,
            // culling, bookkeeping.
            let in_passes: f64 = st.cpu_passes.iter().map(|p| p.1).sum();
            rows.push(nested("outside the passes", (ms - in_passes).max(0.0)));
        }
        // The one part of the headline that is there because the group is
        // open: the per-pass scopes are costly, so they run on some frames.
        if let Some(ms) = st.pass_timing {
            rows.push(Row::new(
                "timing the passes (only while open)",
                per_frame(ms),
            ));
        }
        self.renderer
            .set(st.cpu_ema.map_or("—".to_string(), per_frame), rows, false);

        let late = st.late_windows >= LATE_WINDOWS;
        let mut rows = vec![
            Row::new(
                "steps",
                st.steps_ema
                    .map_or("—".to_string(), |r| format!("{r:.0} steps/s")),
            ),
            Row {
                bad: late,
                ..Row::new(
                    "late frames",
                    format!("{:.0} / s · {} total", st.late_rate, st.late_total),
                )
            },
        ];
        rows.extend(
            st.sim_phases
                .iter()
                .map(|(name, ms)| Row::new(name, per_step(*ms))),
        );
        self.sim
            .set(st.step_ema.map_or("—".to_string(), per_step), rows, late);

        self.page.set(
            st.page_ema.map_or("—".to_string(), per_frame),
            st.page_rows
                .iter()
                .map(|(name, ms)| Row::new(name, per_frame(*ms)))
                .collect(),
            false,
        );
    }
}

fn smooth(ema: Option<f64>, v: f64) -> Option<f64> {
    Some(ema.map_or(v, |e| e + (v - e) * EMA_ALPHA))
}

/// `now`, each entry smoothed against the same name in `prev`.
fn smooth_named(prev: &[(String, f64)], now: Vec<(String, f64)>) -> Vec<(String, f64)> {
    now.into_iter()
        .map(|(name, v)| {
            let ema = prev.iter().find(|(n, _)| *n == name).map(|p| p.1);
            let v = smooth(ema, v).unwrap_or(v);
            (name, v)
        })
        .collect()
}

fn slowest_first(entries: &[(String, f64)]) -> impl Iterator<Item = &(String, f64)> {
    let mut sorted: Vec<&(String, f64)> = entries.iter().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    sorted.into_iter()
}

/// A duration with its unit: microseconds below a tenth of a millisecond.
fn duration(ms: f64) -> String {
    if ms < 0.09995 {
        let us = ms * 1000.0;
        if us < 9.95 {
            format!("{us:.1} µs")
        } else {
            format!("{us:.0} µs")
        }
    } else if ms < 9.995 {
        format!("{ms:.2} ms")
    } else {
        format!("{ms:.1} ms")
    }
}

fn per_frame(ms: f64) -> String {
    format!("{} / frame", duration(ms))
}

fn per_step(ms: f64) -> String {
    format!("{} / step", duration(ms))
}

fn now(perf: &Option<web_sys::Performance>) -> f64 {
    perf.as_ref().map_or(0.0, |p| p.now())
}

/// The main thread's own rAF loop: records frame intervals for
/// [`refresh_period`] and counts (and times) the page's frames. Runs for the
/// page's lifetime.
fn measure_refresh(
    samples: Rc<RefCell<VecDeque<f64>>>,
    work: Rc<PageWork>,
    perf: Option<web_sys::Performance>,
) {
    type Cb = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let cb: Cb = Rc::new(RefCell::new(None));
    let again = cb.clone();
    let mut prev: Option<f64> = None;
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        let t0 = now(&perf);
        if let Some(p) = prev.replace(t) {
            let dt = t - p;
            // A hidden tab runs no frames; the gap is not a refresh interval.
            if dt > 0.0 && dt <= GAP_MS {
                let mut s = samples.borrow_mut();
                if s.len() == REFRESH_SAMPLES {
                    s.pop_front();
                }
                s.push_back(dt);
            }
        }
        if let (Some(w), Some(f)) = (web_sys::window(), again.borrow().as_ref()) {
            let _ = w.request_animation_frame(f.as_ref().unchecked_ref());
        }
        work.frames.set(work.frames.get() + 1);
        work.refresh_ms.set(work.refresh_ms.get() + now(&perf) - t0);
    }));
    if let (Some(w), Some(f)) = (web_sys::window(), cb.borrow().as_ref()) {
        let _ = w.request_animation_frame(f.as_ref().unchecked_ref());
    }
    std::mem::forget(cb);
}

/// The display's refresh period (ms) from recent main-thread frame intervals:
/// the mean of those within [`REFRESH_INLIER`] of the median.
fn refresh_period(samples: &VecDeque<f64>) -> Option<f64> {
    if samples.len() < REFRESH_MIN_SAMPLES {
        return None;
    }
    let mut sorted: Vec<f64> = samples.iter().copied().collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    let (sum, n) = sorted
        .iter()
        .filter(|d| (**d - median).abs() <= median * REFRESH_INLIER)
        .fold((0.0, 0u32), |(s, n), d| (s + d, n + 1));
    (n > 0).then_some(sum / f64::from(n))
}

/// A touch / `pointer: coarse` device (phones, tablets).
fn coarse_pointer(window: &web_sys::Window) -> bool {
    window
        .match_media("(pointer: coarse)")
        .ok()
        .flatten()
        .map(|m| m.matches())
        .unwrap_or(false)
}

fn stored_bool(window: &web_sys::Window, key: &str) -> Option<bool> {
    let ls = window.local_storage().ok().flatten()?;
    match ls.get_item(key).ok().flatten()?.as_str() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

fn store_bool(window: &web_sys::Window, key: &str, val: bool) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(key, if val { "1" } else { "0" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_ignores_late_frames() {
        let mut s: VecDeque<f64> = (0..40).map(|_| 8.333).collect();
        s.push_back(16.7);
        s.push_back(25.0);
        let p = refresh_period(&s).unwrap();
        assert!((p - 8.333).abs() < 1e-6);
    }

    #[test]
    fn durations_read_the_same_everywhere() {
        assert_eq!(duration(0.0084), "8.4 µs");
        assert_eq!(duration(0.084), "84 µs");
        assert_eq!(duration(0.61), "0.61 ms");
        assert_eq!(duration(4.2), "4.20 ms");
        assert_eq!(duration(12.34), "12.3 ms");
    }

    #[test]
    fn the_scopes_run_on_one_frame_in_sixteen() {
        let mut s = PassSampler::default();
        let mut timed = 0;
        for _ in 0..32 {
            let scopes = s.timing_scopes();
            timed += u32::from(scopes);
            s.add(PhaseClock {
                t: 0.0,
                ms: [0.0; PHASES],
                scopes,
            });
        }
        assert_eq!((timed, s.timed_frames, s.plain_frames), (2, 2, 30));
    }

    #[test]
    fn frame_window_reports_and_skips_pauses() {
        let mut w = FrameWindow::new(false);
        let mut t = 0.0;
        assert!(w.frame(t, 1.0, false, None).is_none());
        // A pause is neither a frame nor span.
        t += 5000.0;
        assert!(w.frame(t, 1.0, false, None).is_none());
        let mut out = None;
        for _ in 0..50 {
            t += 20.0;
            if let Some(m) = w.frame(t, 1.0, false, None) {
                out = Some(m);
            }
        }
        let r = out.expect("a report");
        assert_eq!(r.frames, 50);
        assert!((r.span_ms - 1000.0).abs() < 1e-6);
        // The renderer group is closed: no breakdown.
        assert!(r.phases_ms.is_none() && r.cpu_passes.is_empty());
    }
}
