//! Render resolution — the one fill-rate lever — and the Settings modal that
//! exposes it.
//!
//! The canvas renders at `devicePixelRatio × scale` device pixels. The user
//! owns the scale (the bottom-right **Settings** slider, persisted per app);
//! we pick only the DEFAULT, and only while they haven't chosen: native
//! resolution capped at a pixel budget ([`DEFAULT_PIXEL_BUDGET`]), lower on a
//! touch device, never below an effective DPR of 1. The renderer's per-frame
//! GPU cost scales with pixel count, and native on a Retina laptop (≈4.3 MP for
//! a 1200×900 window) can miss a 120 Hz display's 8.3 ms budget — the budget
//! is in pixels rather than DPR, so a small window keeps native. It never goes
//! below an effective DPR of 1 (softer than CSS resolution reads as blurry), so
//! a DPR-1 display keeps native by default.
//!
//! Main owns layout, so it sizes the backing store (before the canvas is
//! transferred) and relays every later change to the render worker as a
//! [`ResizeMsg`]; the worker applies it to its `OffscreenCanvas`.

use std::cell::Cell;
use std::rc::Rc;

use dominator::{clone, html, with_node};
use futures_signals::signal::{Mutable, SignalExt};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, HtmlInputElement, Worker};

use crate::protocol::ResizeMsg;

/// Floor of the slider (50% of native). The slider's `min` mirrors it.
const MIN_SCALE: f64 = 0.5;
/// Ceiling: native resolution. We never supersample.
const MAX_SCALE: f64 = 1.0;
/// Default-scale ceiling on a touch / `pointer: coarse` device (phones,
/// tablets): high DPR × a mobile GPU.
const COARSE_START_SCALE: f64 = 0.6;
/// Backing-store pixels the DEFAULT resolution allows (see the module docs).
const DEFAULT_PIXEL_BUDGET: f64 = 2_800_000.0;
/// Max backing-store dimension (WebGPU guarantees at least this 2D texture
/// size), so a huge display can't ask for a canvas the GPU can't allocate.
const MAX_TEX: u32 = 8192;

/// Everything needed to size the backing store. `Copy` so it lives in a `Cell`.
#[derive(Clone, Copy)]
struct ResState {
    scale: f64,
    css_w: f64,
    css_h: f64,
    dpr: f64,
    /// Device-class ceiling for the DEFAULT scale (never over a user choice).
    auto_cap: f64,
}

impl ResState {
    /// Backing-store size in device pixels: `css × dpr × scale`, each axis
    /// clamped to `[1, MAX_TEX]`.
    fn backing(&self) -> (u32, u32) {
        let s = (self.dpr * self.scale).max(0.01);
        let w = ((self.css_w * s).round() as u32).clamp(1, MAX_TEX);
        let h = ((self.css_h * s).round() as u32).clamp(1, MAX_TEX);
        (w, h)
    }

    /// The default scale for the current size: native, capped by the pixel
    /// budget and the device-class ceiling, floored at an effective DPR of 1
    /// (and the slider's range).
    fn auto_scale(&self) -> f64 {
        let native_px = self.css_w * self.css_h * self.dpr * self.dpr;
        let budget = if native_px > DEFAULT_PIXEL_BUDGET {
            (DEFAULT_PIXEL_BUDGET / native_px)
                .sqrt()
                .max(1.0 / self.dpr)
        } else {
            MAX_SCALE
        };
        // Round DOWN to the slider's 5% step so the thumb, its label and the
        // real scale agree (the browser would snap an 81% value to 80).
        let stepped = (budget.min(self.auto_cap) * 20.0).floor() / 20.0;
        stepped.clamp(MIN_SCALE, MAX_SCALE)
    }
}

/// The resolution controller: create it before transferring the canvas
/// ([`Resolution::new`] sizes the backing store), then [`Resolution::install`]
/// once the render worker exists.
pub struct Resolution {
    state: Rc<Cell<ResState>>,
    /// Whether the user has made an explicit choice (then no auto-default
    /// ever overrides it).
    user_pref: Rc<Cell<bool>>,
    /// Slider position + label, in percent.
    pct: Mutable<u32>,
    storage_key: &'static str,
}

impl Resolution {
    /// Pick the starting scale (stored choice, else the budgeted default) and
    /// size `canvas`'s backing store to it. Call BEFORE the canvas is
    /// transferred to the worker. `storage_key` is this app's `localStorage`
    /// key — every experiment shares the CDN origin, so it must be unique.
    pub fn new(
        window: &web_sys::Window,
        canvas: &HtmlCanvasElement,
        storage_key: &'static str,
    ) -> Self {
        let mut st = ResState {
            scale: MAX_SCALE,
            css_w: canvas.client_width().max(1) as f64,
            css_h: canvas.client_height().max(1) as f64,
            dpr: window.device_pixel_ratio().max(1.0),
            auto_cap: if coarse_pointer(window) {
                COARSE_START_SCALE
            } else {
                MAX_SCALE
            },
        };
        let stored = stored_scale(window, storage_key);
        st.scale = stored.unwrap_or_else(|| st.auto_scale());
        let (w, h) = st.backing();
        canvas.set_width(w);
        canvas.set_height(h);
        Self {
            state: Rc::new(Cell::new(st)),
            user_pref: Rc::new(Cell::new(stored.is_some())),
            pct: Mutable::new(pct_of(st.scale)),
            storage_key,
        }
    }

    /// Wire the live parts to the render worker: a `ResizeObserver` on the
    /// canvas (re-derives the backing store, and the default scale while the
    /// user hasn't chosen) and the bottom-right Settings button + modal.
    pub fn install(
        self,
        window: &web_sys::Window,
        canvas: &HtmlCanvasElement,
        render: Worker,
    ) -> Result<(), JsValue> {
        let Self {
            state,
            user_pref,
            pct,
            storage_key,
        } = self;

        // ── Layout changes ──────────────────────────────────────────────────
        let cb = Closure::<dyn FnMut(js_sys::Array)>::new(clone!(
            state, user_pref, pct, render, window => move |entries: js_sys::Array| {
                let Ok(entry) = entries.get(0).dyn_into::<web_sys::ResizeObserverEntry>() else {
                    return;
                };
                let rect = entry.content_rect();
                if rect.width() <= 0.0 || rect.height() <= 0.0 {
                    return;
                }
                let mut st = state.get();
                // DPR can change when the window moves between monitors.
                st.dpr = window.device_pixel_ratio().max(1.0);
                st.css_w = rect.width();
                st.css_h = rect.height();
                if !user_pref.get() {
                    st.scale = st.auto_scale();
                    pct.set_neq(pct_of(st.scale));
                }
                state.set(st);
                post_resize(&render, &st);
            }
        ));
        let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())?;
        observer.observe(canvas);
        // Both live for the page's lifetime.
        cb.forget();
        std::mem::forget(observer);

        // ── Settings button + modal ─────────────────────────────────────────
        let open = Mutable::new(false);
        let button = html!("button", {
            .class("settings-btn")
            .text("Settings")
            .event(clone!(open => move |_: dominator::events::Click| open.set_neq(true)))
        });
        let resolution_row = html!("div", {
            .class("settings-row")
            .child(html!("label", {
                .class("settings-label")
                .text_signal(pct.signal().map(|p| format!("Resolution — {p}%")))
            }))
            .child(html!("input" => HtmlInputElement, {
                .class("res-slider")
                .attr("type", "range")
                // Mirror MIN_SCALE / MAX_SCALE (as percent).
                .attr("min", "50")
                .attr("max", "100")
                .attr("step", "5")
                .attr("aria-label", "render resolution")
                .prop_signal("value", pct.signal().map(|p| p.to_string()))
                .with_node!(el => {
                    .event(clone!(state, user_pref, pct, window, render => move |_: dominator::events::Input| {
                        let p = el.value().parse::<f64>().unwrap_or(100.0);
                        let scale = (p / 100.0).clamp(MIN_SCALE, MAX_SCALE);
                        let mut st = state.get();
                        st.scale = scale;
                        state.set(st);
                        user_pref.set(true);
                        pct.set_neq(pct_of(scale));
                        store_scale(&window, storage_key, scale);
                        post_resize(&render, &st);
                    }))
                })
            }))
        });
        let modal = html!("div", {
            .class("settings-overlay")
            .visible_signal(open.signal())
            .event(clone!(open => move |e: dominator::events::Click| {
                let on_backdrop = e
                    .dyn_target::<web_sys::Element>()
                    .map(|el| el.class_list().contains("settings-overlay"))
                    .unwrap_or(false);
                if on_backdrop {
                    open.set_neq(false);
                }
            }))
            .global_event(clone!(open => move |e: dominator::events::KeyDown| {
                if e.key() == "Escape" {
                    open.set_neq(false);
                }
            }))
            .child(html!("div", {
                .class("settings-modal")
                .child(html!("button", {
                    .class("settings-close")
                    .attr("aria-label", "Close")
                    .text("×")
                    .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
                }))
                .child(html!("h2", { .text("Settings") }))
                .child(resolution_row)
                .child(html!("p", {
                    .class("settings-hint")
                    .text("Lower the resolution if your framerate is low.")
                }))
            }))
        });
        dominator::append_dom(&dominator::body(), button);
        dominator::append_dom(&dominator::body(), modal);
        Ok(())
    }
}

/// Post the backing-store size for `st` to the render worker.
fn post_resize(render: &Worker, st: &ResState) {
    let (width, height) = st.backing();
    if let Ok(v) = serde_wasm_bindgen::to_value(&ResizeMsg::Canvas { width, height }) {
        let _ = render.post_message(&v);
    }
}

/// Percent form of a scale fraction (0.6 → 60) — the slider's unit.
fn pct_of(scale: f64) -> u32 {
    (scale * 100.0).round() as u32
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

/// The persisted resolution scale, if the user has set one (clamped to range).
fn stored_scale(window: &web_sys::Window, key: &str) -> Option<f64> {
    let ls = window.local_storage().ok().flatten()?;
    let raw = ls.get_item(key).ok().flatten()?;
    raw.parse::<f64>()
        .ok()
        .map(|v| v.clamp(MIN_SCALE, MAX_SCALE))
}

/// Persist the user's resolution scale (only slider drags call this — the
/// default is recomputed each load, never written).
fn store_scale(window: &web_sys::Window, key: &str, scale: f64) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(key, &format!("{scale:.2}"));
    }
}
