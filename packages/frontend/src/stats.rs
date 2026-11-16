//! The Stats panel (top right) and the Stats chip (bottom left) that shows and
//! hides it, plus the render worker's half of what it measures.
//!
//! What it shows:
//!
//! ```text
//! render
//!   display       120 Hz          the display's refresh rate (measured)
//!   fps           118             frames the render worker presented per second
//!   missed        2/s · 37 total  refreshes that showed no new frame
//!   cpu time      1.3 ms          the render worker's CPU time per frame
//! ```
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

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::Mutable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::protocol::RenderMsg;

// ── Render worker side ──────────────────────────────────────────────────────

/// Roughly how often the render worker reports (ms of presented frames).
const REPORT_SPAN_MS: f64 = 1000.0;
/// A frame interval longer than this is a pause, not a slow frame: it is left
/// out of the window (with its frame), so it never counts as missed frames.
const GAP_MS: f64 = 250.0;

/// Counts presented frames on the render worker. Call [`FrameWindow::frame`]
/// once per frame; post what it returns.
#[derive(Default)]
pub struct FrameWindow {
    prev_t: Option<f64>,
    frames: u32,
    span_ms: f64,
    cpu_ms: f64,
}

impl FrameWindow {
    pub fn new() -> Self {
        Self::default()
    }

    /// One presented frame: `t` is its rAF timestamp, `cpu_ms` the CPU time
    /// the frame took. Returns a [`RenderMsg::Stats`] when a window is full.
    pub fn frame(&mut self, t: f64, cpu_ms: f64) -> Option<RenderMsg> {
        let prev = self.prev_t.replace(t)?; // the first frame only starts the clock
        let dt = t - prev;
        if !(dt > 0.0 && dt <= GAP_MS) {
            return None; // a pause: neither the gap nor this frame is counted
        }
        self.frames += 1;
        self.span_ms += dt;
        self.cpu_ms += cpu_ms;
        if self.span_ms < REPORT_SPAN_MS {
            return None;
        }
        let msg = RenderMsg::Stats {
            frames: self.frames,
            span_ms: self.span_ms,
            cpu_ms: self.cpu_ms,
        };
        self.frames = 0;
        self.span_ms = 0.0;
        self.cpu_ms = 0.0;
        Some(msg)
    }
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

/// Smoothing weight of each new window for fps and cpu time: a real change
/// shows within 2–3 windows, while the readout holds steady against noise.
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

/// The panel's state; owned by the main thread.
pub struct Stats {
    text: Mutable<String>,
    state: RefCell<State>,
    refresh: Rc<RefCell<VecDeque<f64>>>,
}

#[derive(Default)]
struct State {
    ready: bool,
    warm_frames: u32,
    fps_ema: Option<f64>,
    cpu_ema: Option<f64>,
    missed_rate: Option<f64>,
    /// Signed running sum of (refreshes - frames): a window whose ends fall
    /// unevenly can read ±1, and the next one cancels it.
    missed_sum: f64,
}

impl Stats {
    /// Mount the panel and its chip, and start measuring the display.
    /// `storage_key` remembers whether the panel is open.
    pub fn install(window: &web_sys::Window, storage_key: &'static str) -> Rc<Self> {
        let stats = Rc::new(Self {
            text: Mutable::new("render\n  starting…".to_string()),
            state: RefCell::new(State::default()),
            refresh: Rc::new(RefCell::new(VecDeque::with_capacity(REFRESH_SAMPLES))),
        });

        // Off by default: on a phone the panel covers a lot of the screen. A
        // stored "open" is honoured only on a desktop; on a touch device the
        // chip still opens it for the session.
        let open =
            Mutable::new(!coarse_pointer(window) && stored_bool(window, storage_key) == Some(true));
        dominator::append_dom(
            &dominator::body(),
            html!("div", {
                .child(html!("div", {
                    .class("stats")
                    .visible_signal(open.signal())
                    .text_signal(stats.text.signal_cloned())
                }))
                .child(html!("button", {
                    .class("stats-btn")
                    .class_signal("active", open.signal())
                    .text("Stats")
                    .event(clone!(open => move |_: dominator::events::Click| {
                        let next = !open.get();
                        open.set_neq(next);
                        if let Some(w) = web_sys::window() {
                            store_bool(&w, storage_key, next);
                        }
                    }))
                }))
            }),
        );

        measure_refresh(stats.refresh.clone());
        stats
    }

    /// The renderer is presenting real frames: start the warm-up.
    pub fn mark_ready(&self) {
        self.state.borrow_mut().ready = true;
        self.text.set("render\n  warming up…".to_string());
    }

    /// One [`RenderMsg::Stats`] window from the render worker.
    pub fn window(&self, frames: u32, span_ms: f64, cpu_ms: f64) {
        let mut st = self.state.borrow_mut();
        if !st.ready || frames == 0 || span_ms <= 0.0 {
            return;
        }
        if st.warm_frames < WARMUP_FRAMES {
            st.warm_frames += frames;
            return;
        }
        let smooth = |ema: Option<f64>, v: f64| Some(ema.map_or(v, |e| e + (v - e) * EMA_ALPHA));
        st.fps_ema = smooth(st.fps_ema, frames as f64 * 1000.0 / span_ms);
        st.cpu_ema = smooth(st.cpu_ema, cpu_ms / frames as f64);

        let period = refresh_period(&self.refresh.borrow());
        let display = match period {
            Some(p) => {
                let missed = (span_ms / p).round() - frames as f64;
                st.missed_sum += missed;
                st.missed_rate = Some(missed.max(0.0) * 1000.0 / span_ms);
                format!("{:.0} Hz", 1000.0 / p)
            }
            None => "measuring…".to_string(),
        };
        let missed = match st.missed_rate {
            Some(rate) => format!(
                "{:.0}/s · {:.0} total",
                rate,
                st.missed_sum.max(0.0).round()
            ),
            None => "—".to_string(),
        };
        let ms = |v: Option<f64>| v.map_or("—".to_string(), |v| format!("{v:.1} ms"));
        self.text.set(format!(
            "render\n  display       {display}\n  fps           {:.0}\n  missed        {missed}\n  cpu time      {}",
            st.fps_ema.unwrap_or(0.0),
            ms(st.cpu_ema),
        ));
    }
}

/// The main thread's own rAF loop: records frame intervals for
/// [`refresh_period`]. Runs for the page's lifetime.
fn measure_refresh(samples: Rc<RefCell<VecDeque<f64>>>) {
    type Cb = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let cb: Cb = Rc::new(RefCell::new(None));
    let again = cb.clone();
    let mut prev: Option<f64> = None;
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
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
    (n > 0).then(|| sum / n as f64)
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
    fn frame_window_reports_and_skips_pauses() {
        let mut w = FrameWindow::new();
        let mut t = 0.0;
        assert!(w.frame(t, 1.0).is_none());
        // A pause is neither a frame nor span.
        t += 5000.0;
        assert!(w.frame(t, 1.0).is_none());
        let mut out = None;
        for _ in 0..50 {
            t += 20.0;
            if let Some(m) = w.frame(t, 1.0) {
                out = Some(m);
            }
        }
        match out {
            Some(RenderMsg::Stats {
                frames, span_ms, ..
            }) => {
                assert_eq!(frames, 50);
                assert!((span_ms - 1000.0).abs() < 1e-6);
            }
            other => panic!("expected stats, got {other:?}"),
        }
    }
}
