//! The About button (bottom centre) and its modal, with two tabs: About (what
//! the demo is) and Stats (what the Stats panel's numbers mean; the panel's
//! `what do these mean?` button opens it there).
//!
//! The camera listens on the canvas element, and the modal is not inside it,
//! so drags and the wheel in the modal never reach the camera. Nothing on the
//! page listens for keys other than Escape, which closes the modal.

use std::rc::Rc;

use dominator::{clone, html, Dom};
use futures_signals::signal::{Mutable, SignalExt};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

const REPO: &str = "https://github.com/awsm-fun/awsm-template-physics-mujoco";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    About,
    Stats,
}

const TABS: [Tab; 2] = [Tab::About, Tab::Stats];

impl Tab {
    fn name(self) -> &'static str {
        match self {
            Tab::About => "about",
            Tab::Stats => "stats",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Tab::About => "About",
            Tab::Stats => "Stats",
        }
    }

    fn tab_id(self) -> String {
        format!("about-tab-{}", self.name())
    }

    fn panel_id(self) -> String {
        format!("about-panel-{}", self.name())
    }

    /// The tab a Left / Right / Home / End key moves to.
    fn after_key(self, key: &str) -> Option<Tab> {
        let i = TABS.iter().position(|t| *t == self)?;
        let n = TABS.len();
        let next = match key {
            "ArrowRight" => (i + 1) % n,
            "ArrowLeft" => (i + n - 1) % n,
            "Home" => 0,
            "End" => n - 1,
            _ => return None,
        };
        Some(TABS[next])
    }
}

pub struct About {
    open: Mutable<bool>,
    tab: Mutable<Tab>,
}

impl About {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            open: Mutable::new(false),
            tab: Mutable::new(Tab::About),
        })
    }

    /// Open the modal on `tab`, with the keyboard on that tab.
    pub fn open(&self, tab: Tab) {
        self.tab.set_neq(tab);
        self.open.set_neq(true);
        // The overlay is shown by its signal, which runs after this returns:
        // focus once it is on screen (a hidden element cannot take focus).
        let focus = Closure::once_into_js(move || focus_tab(tab));
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback(focus.unchecked_ref());
        }
    }

    /// Mount the button and the modal.
    pub fn install(self: &Rc<Self>) {
        let about = self.clone();
        dominator::append_dom(
            &dominator::body(),
            html!("button", {
                .class("about-btn")
                .attr("type", "button")
                .text("About")
                .event(clone!(about => move |_: dominator::events::Click| about.open(Tab::About)))
            }),
        );
        dominator::append_dom(&dominator::body(), self.modal());
    }

    fn modal(self: &Rc<Self>) -> Dom {
        let open = self.open.clone();
        html!("div", {
            .class("about-overlay")
            .visible_signal(open.signal())
            .event(clone!(open => move |e: dominator::events::Click| {
                let on_backdrop = e
                    .dyn_target::<web_sys::Element>()
                    .is_some_and(|el| el.class_list().contains("about-overlay"));
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
                .class("about-modal")
                .attr("role", "dialog")
                .attr("aria-modal", "true")
                .attr("aria-labelledby", "about-title")
                .child(html!("button", {
                    .class("about-close")
                    .attr("type", "button")
                    .attr("aria-label", "Close")
                    .text("×")
                    .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
                }))
                .child(html!("header", {
                    .class("about-header")
                    .child(html!("h2", {
                        .attr("id", "about-title")
                        .text("MuJoCo Physics Template")
                    }))
                    .child(html!("a", {
                        .class("about-github")
                        .attr("href", REPO)
                        .attr("target", "_blank")
                        .attr("rel", "noopener")
                        .text("View the template on GitHub ↗")
                    }))
                }))
                .child(self.tabs())
                .child(self.panel(Tab::About, about_tab()))
                .child(self.panel(Tab::Stats, stats_tab()))
            }))
        })
    }

    /// The tablist: a click selects, the arrow keys (and Home / End) select
    /// and focus. Only the selected tab is in the Tab order.
    fn tabs(&self) -> Dom {
        let tab = self.tab.clone();
        html!("div", {
            .class("about-tabs")
            .attr("role", "tablist")
            .attr("aria-label", "About this demo")
            .event(clone!(tab => move |e: dominator::events::KeyDown| {
                if let Some(next) = tab.get().after_key(&e.key()) {
                    e.prevent_default();
                    tab.set_neq(next);
                    focus_tab(next);
                }
            }))
            .children(TABS.map(|t| {
                let selected = tab.signal().map(move |s| s == t);
                html!("button", {
                    .class("about-tab")
                    .attr("type", "button")
                    .attr("role", "tab")
                    .attr("id", &t.tab_id())
                    .attr("aria-controls", &t.panel_id())
                    .attr_signal("aria-selected", selected.map(|s| if s { "true" } else { "false" }))
                    .attr_signal("tabindex", tab.signal().map(move |s| if s == t { "0" } else { "-1" }))
                    .text(t.label())
                    .event(clone!(tab => move |_: dominator::events::Click| tab.set_neq(t)))
                })
            }))
        })
    }

    fn panel(&self, t: Tab, children: Vec<Dom>) -> Dom {
        html!("div", {
            .class("about-panel")
            .attr("role", "tabpanel")
            .attr("id", &t.panel_id())
            .attr("aria-labelledby", &t.tab_id())
            .attr("tabindex", "0")
            .prop_signal("hidden", self.tab.signal().map(move |s| s != t))
            .children(children)
        })
    }
}

fn focus_tab(tab: Tab) {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&tab.tab_id()))
        .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let _ = el.focus();
    }
}

// ── The text ────────────────────────────────────────────────────────────────

fn about_tab() -> Vec<Dom> {
    vec![
        html!("p", {
            .text("A humanoid ragdoll simulated by MuJoCo, DeepMind's physics engine, \
                   compiled to wasm and running in your browser, and a template to start \
                   your own from.")
        }),
        h3("Controls"),
        html!("table", {
            .class("about-keys")
            .child(key_row(vec![html!("span", { .text("Drag") })], "orbit the camera"))
            .child(key_row(vec![html!("span", { .text("Wheel") })], "zoom"))
            .child(key_row(vec![code("?scene=flag")], "a cloth flag waving in the wind instead"))
            .child(key_row(
                vec![code("?contacts"), text(" "), code("?joints"), text(" "), code("?inertia")],
                "debug overlays: contact forces, joint axes, inertia boxes (they stack)",
            ))
            .child(key_row(
                vec![html!("span", { .text("Stats, Settings") })],
                "what each thread is doing; the render resolution",
            ))
        }),
        h3("How it works"),
        html!("p", {
            .text("Three threads. This page owns the canvas and your pointer. MuJoCo (the \
                   official @mujoco/mujoco wasm build) runs in a worker at the model's fixed \
                   timestep. A Rust WebGPU renderer (")
            .child(link("https://scene.awsm.fun", "AwsmRenderer"))
            .text(") in another worker draws it, blending between steps so motion stays \
                   smooth at any refresh rate. The two workers meet in shared memory: the \
                   poses never go through postMessage.")
        }),
        h3("Make it yours"),
        html!("p", {
            .text("Read ")
            .child(code("packages/frontend/src/main_thread.rs"))
            .text(", then ")
            .child(code("protocol.rs"))
            .text(" (the shared pose block). To play your own MuJoCo model, see \
                   \"Swapping in your own scene\" in the README.")
        }),
        html!("p", {
            .class("about-more")
            .text("The README has the rest. Part of the ")
            .child(link("https://awsm.fun", "Awsm"))
            .text(" project.")
        }),
    ]
}

fn stats_tab() -> Vec<Dom> {
    vec![
        html!("p", {
            .text("The Stats button (bottom left) opens a panel: a frame budget first, then \
                   one group per thread. Each group's header shows its main number; click a \
                   group to open its breakdown. A breakdown is only measured while its group \
                   is open, so the panel costs nothing when it is closed.")
        }),
        html!("p", {
            .child(code("ms / frame"))
            .text(" is time per rendered frame. ")
            .child(code("µs / step"))
            .text(" is time per simulation step (1000 µs = 1 ms). ")
            .child(code("/ s"))
            .text(" is per second.")
        }),
        h3("frame budget"),
        dl(vec![
            (
                "budget",
                vec![text(
                    "The time one display refresh allows: 16.7 ms at 60 Hz, 8.3 ms at 120 Hz. \
                 It comes from your display's measured refresh rate.",
                )],
            ),
            (
                "headroom",
                vec![
                    text(
                        "The budget minus the largest lane below. The GPU and the threads work \
                      in parallel, so their times do not add up: each one has to fit the \
                      budget on its own. If it reads ",
                    ),
                    code("over by …"),
                    text(", that lane cannot keep up with the display."),
                ],
            ),
            (
                "the lanes",
                vec![text(
                    "One row for the GPU and one per thread: the time it uses per frame, out of \
                 the budget. Here: the GPU, the renderer, the sim and the page. The sim's \
                 lane is its time per step times the steps in a frame.",
                )],
            ),
            (
                "latency (renderer + GPU), reference only",
                vec![text(
                    "Roughly how long one frame takes from the renderer starting it to the GPU \
                 finishing it. Not what the budget is checked against.",
                )],
            ),
        ]),
        html!("p", {
            .text("Missed frames while there is headroom mean something is waiting rather \
                   than working, for example the CPU waiting on the GPU.")
        }),
        h3("display"),
        dl(vec![
            (
                "refresh rate",
                vec![text(
                    "How often your display refreshes, as the browser paces this page.",
                )],
            ),
            (
                "frames presented",
                vec![text(
                    "New frames shown per second (fps). Never more than the refresh rate.",
                )],
            ),
            (
                "missed frames",
                vec![text(
                    "Refreshes that went by without a new frame. A steady count is visible \
                 stutter: lower the resolution in Settings.",
                )],
            ),
        ]),
        h3("GPU"),
        dl(vec![
            (
                "GPU time per frame",
                vec![
                    text("How long the GPU works on a frame, from timestamp queries. Reads "),
                    code("n/a"),
                    text(" where the browser or the GPU does not offer them."),
                ],
            ),
            (
                "the rows",
                vec![text(
                    "One per render pass. Some GPUs (Apple's) run passes overlapped, so the rows \
                 can add up to more than the frame.",
                )],
            ),
        ]),
        h3("renderer (render worker)"),
        dl(vec![
            (
                "the header",
                vec![text("The render thread's CPU time per frame.")],
            ),
            (
                "read app state + interpolate",
                vec![text(
                    "Reads the two latest simulation steps and blends between them for this \
                 frame's exact moment.",
                )],
            ),
            (
                "set camera, update transforms",
                vec![text(
                    "Sets the view, and turns the new poses into world matrices.",
                )],
            ),
            ("render (encode + submit)", render_dd()),
        ]),
        h3("sim (MuJoCo, worker)"),
        dl(vec![
            (
                "the header",
                vec![text("CPU time of one fixed simulation step.")],
            ),
            (
                "steps",
                vec![text(
                    "Steps per second. Fixed (the model's rate: 200 for the humanoid, 500 for \
                 the flag), whatever the display does; the renderer blends between steps.",
                )],
            ),
            (
                "late frames",
                vec![text(
                    "Frames drawn before the sim had finished the moment they show; such a \
                 frame holds the previous pose. The row and the header turn red if it keeps \
                 happening.",
                )],
            ),
            (
                "simulate (mj_step), publish",
                vec![text(
                    "The parts of a step, in order: MuJoCo's own step, then copying the poses \
                 into shared memory.",
                )],
            ),
        ]),
        h3("page (main thread)"),
        dl(vec![(
            "camera frame loop, refresh probe",
            vec![text(
                "The page's own work per frame: sending the camera, and measuring the refresh \
             rate. It should stay tiny; the heavy work is in the workers.",
            )],
        )]),
    ]
}

/// The render phase's breakdown, the same in every template.
fn render_dd() -> Vec<Dom> {
    vec![
        text("The renderer's own work, split into "),
        code("prepare"),
        text(" (uploading this frame's data to the GPU), "),
        code("passes"),
        text(" (encoding each render pass), "),
        code("submit"),
        text(" (handing the frame to the GPU) and "),
        code("outside the scopes"),
        text(
            " (the little that is left untimed). The smallest rows are folded into \
              \"N more under 1 µs\".",
        ),
    ]
}

fn h3(s: &str) -> Dom {
    html!("h3", { .text(s) })
}

fn text(s: &str) -> Dom {
    dominator::text(s)
}

fn code(s: &str) -> Dom {
    html!("code", { .text(s) })
}

fn link(href: &str, label: &str) -> Dom {
    html!("a", {
        .attr("href", href)
        .attr("target", "_blank")
        .attr("rel", "noopener")
        .text(label)
    })
}

fn key_row(key: Vec<Dom>, does: &str) -> Dom {
    html!("tr", {
        .child(html!("th", { .children(key) }))
        .child(html!("td", { .text(does) }))
    })
}

/// A group's terms, each the label as the panel prints it.
fn dl(entries: Vec<(&str, Vec<Dom>)>) -> Dom {
    html!("dl", {
        .children(entries.into_iter().flat_map(|(term, def)| {
            [
                html!("dt", { .text(term) }),
                html!("dd", { .children(def) }),
            ]
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_wrap_and_home_end_jump() {
        assert_eq!(Tab::About.after_key("ArrowRight"), Some(Tab::Stats));
        assert_eq!(Tab::Stats.after_key("ArrowRight"), Some(Tab::About));
        assert_eq!(Tab::About.after_key("ArrowLeft"), Some(Tab::Stats));
        assert_eq!(Tab::Stats.after_key("Home"), Some(Tab::About));
        assert_eq!(Tab::About.after_key("End"), Some(Tab::Stats));
        assert_eq!(Tab::About.after_key("Enter"), None);
    }
}
