/*! coi-serviceworker v0.1.7 - Guido Zuidhof and contributors, licensed under MIT */
// Re-imposes the COOP/COEP headers this threaded wasm build needs
// (`crossOriginIsolated` → SharedArrayBuffer → shared `WebAssembly.Memory`) on
// static hosts that cannot send custom HTTP headers — notably GitHub Pages.
// `task dev`/Trunk already send the headers, so on a cross-origin-isolated page
// this script registers nothing and is a no-op. See Trunk.toml / src/lib.rs.
let coepCredentialless = false;
if (typeof window === 'undefined') {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

    self.addEventListener("message", (ev) => {
        if (!ev.data) {
            return;
        } else if (ev.data.type === "deregister") {
            self.registration
                .unregister()
                .then(() => {
                    return self.clients.matchAll();
                })
                .then(clients => {
                    clients.forEach((client) => client.navigate(client.url));
                });
        } else if (ev.data.type === "coepCredentialless") {
            coepCredentialless = ev.data.value;
        }
    });

    self.addEventListener("fetch", function (event) {
        const r = event.request;
        if (r.cache === "only-if-cached" && r.mode !== "same-origin") {
            return;
        }

        const request = (coepCredentialless && r.mode === "no-cors")
            ? new Request(r, {
                credentials: "omit",
            })
            : r;
        event.respondWith(
            fetch(request)
                .then((response) => {
                    if (response.status === 0) {
                        return response;
                    }

                    const newHeaders = new Headers(response.headers);
                    newHeaders.set("Cross-Origin-Embedder-Policy",
                        coepCredentialless ? "credentialless" : "require-corp"
                    );
                    if (!coepCredentialless) {
                        newHeaders.set("Cross-Origin-Resource-Policy", "cross-origin");
                    }
                    newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");

                    return new Response(response.body, {
                        status: response.status,
                        statusText: response.statusText,
                        headers: newHeaders,
                    });
                })
                .catch((e) => console.error(e))
        );
    });

} else {
    (() => {
        const reloadedBySelf = window.sessionStorage.getItem("coiReloadedBySelf");
        window.sessionStorage.removeItem("coiReloadedBySelf");
        const coepDegrading = (reloadedBySelf == "coepdegrade");

        // You can customize the behavior of this script through a global `coi` variable.
        const coi = {
            shouldRegister: () => !reloadedBySelf,
            shouldDeregister: () => false,
            coepCredentialless: () => true,
            coepDegrade: () => true,
            doReload: () => window.location.reload(),
            quiet: false,
            ...window.coi
        };

        const n = navigator;
        const controlling = n.serviceWorker && n.serviceWorker.controller;

        // First-visit fix: on a brand-new registration the worker activates but
        // does NOT control the page that registered it, so this load stays
        // un-isolated (SharedArrayBuffer unavailable, threaded wasm can't boot)
        // until a manual refresh. Reload ourselves the moment the worker takes
        // control so the very first visit becomes cross-origin isolated. The
        // sessionStorage flag (cleared on the next load, line ~69) prevents a
        // reload loop and defers to the explicit reloads below.
        if (n.serviceWorker) {
            n.serviceWorker.addEventListener("controllerchange", () => {
                if (window.crossOriginIsolated) return;
                if (window.sessionStorage.getItem("coiReloadedBySelf")) return;
                window.sessionStorage.setItem("coiReloadedBySelf", "controllerchange");
                !coi.quiet && console.log("Reloading page now that the COOP/COEP Service Worker controls it.");
                coi.doReload();
            });
        }

        // Record the failure if the page is served by serviceWorker.
        if (controlling && !window.crossOriginIsolated) {
            window.sessionStorage.setItem("coiCoepHasFailed", "true");
        }
        const coepHasFailed = window.sessionStorage.getItem("coiCoepHasFailed");

        if (controlling) {
            // Reload only on the first failure.
            const reloadToDegrade = coi.coepDegrade() && !(
                coepDegrading || window.crossOriginIsolated
            );
            n.serviceWorker.controller.postMessage({
                type: "coepCredentialless",
                value: (reloadToDegrade || coepHasFailed && coi.coepDegrade())
                    ? false
                    : coi.coepCredentialless(),
            });
            if (reloadToDegrade) {
                !coi.quiet && console.log("Reloading page to degrade COEP.");
                window.sessionStorage.setItem("coiReloadedBySelf", "coepdegrade");
                coi.doReload("coepdegrade");
            }
        } else if (window.crossOriginIsolated) {
            // Already isolated (e.g. Trunk/`task dev` sent the headers) — nothing to do.
        } else if (coi.shouldRegister()) {
            if (!window.isSecureContext) {
                !coi.quiet && console.log("COOP/COEP Service Worker not registered, a secure context is required.");
            } else {
                n.serviceWorker.register(window.document.currentScript.src).then(
                    (registration) => {
                        !coi.quiet && console.log("COOP/COEP Service Worker registered", registration.scope);

                        registration.addEventListener("updatefound", () => {
                            !coi.quiet && console.log("Reloading page to make use of updated COOP/COEP Service Worker.");
                            window.sessionStorage.setItem("coiReloadedBySelf", "updatefound");
                            coi.doReload();
                        });

                        // If the registration is active, but it's not controlling the page
                        if (registration.active && !n.serviceWorker.controller) {
                            !coi.quiet && console.log("Reloading page to make use of COOP/COEP Service Worker.");
                            window.sessionStorage.setItem("coiReloadedBySelf", "notcontrolling");
                            coi.doReload();
                        }
                    },
                    (err) => {
                        !coi.quiet && console.error("COOP/COEP Service Worker failed to register:", err);
                    }
                );
            }
        }
    })();
}
