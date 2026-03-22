/**
 * Bottom-right toasts: Flask flashes, page JSON (initial_toasts), faiss template.
 */
(function () {
  var AUTO_MS = 4800;
  var LEAVE_MS = 280;

  function getRoot() {
    return document.getElementById("toast-root");
  }

  function removeToast(el) {
    if (!el || !el.parentNode) return;
    el.classList.add("toast--leave");
    setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, LEAVE_MS);
  }

  function wireDismissDelegation(root) {
    if (!root || root._toastDismissWired) return;
    root._toastDismissWired = true;
    root.addEventListener("click", function (e) {
      var btn = e.target.closest(".toast-dismiss");
      if (!btn || !root.contains(btn)) return;
      var toast = btn.closest(".toast");
      if (toast) removeToast(toast);
    });
  }

  function addDismissButton(toast) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "toast-dismiss";
    btn.setAttribute("aria-label", "Dismiss");
    btn.innerHTML = "&times;";
    toast.insertBefore(btn, toast.firstChild);
  }

  function showToastPlain(message, category, options) {
    options = options || {};
    var root = getRoot();
    if (!root) return;
    var toast = document.createElement("div");
    toast.className = "toast toast--" + (category || "info");
    if (options.wide) toast.classList.add("toast--wide");
    toast.setAttribute("role", category === "error" ? "alert" : "status");
    var body = document.createElement("div");
    body.className = "toast-body";
    body.textContent = message;
    toast.appendChild(body);
    addDismissButton(toast);
    root.appendChild(toast);
    wireDismissDelegation(root);
    requestAnimationFrame(function () {
      toast.classList.add("toast--in");
    });
    if (!options.persistent) {
      setTimeout(function () {
        removeToast(toast);
      }, options.durationMs != null ? options.durationMs : AUTO_MS);
    }
  }

  function showToastHtml(html, category, options) {
    options = options || {};
    var root = getRoot();
    if (!root) return;
    var toast = document.createElement("div");
    toast.className = "toast toast--" + (category || "info");
    if (options.wide) toast.classList.add("toast--wide");
    if (options.persistent) toast.classList.add("toast--persistent");
    toast.setAttribute("role", category === "error" ? "alert" : "status");
    var body = document.createElement("div");
    body.className = "toast-body";
    body.innerHTML = html;
    toast.appendChild(body);
    addDismissButton(toast);
    root.appendChild(toast);
    wireDismissDelegation(root);
    requestAnimationFrame(function () {
      toast.classList.add("toast--in");
    });
    if (!options.persistent) {
      setTimeout(function () {
        removeToast(toast);
      }, options.durationMs != null ? options.durationMs : AUTO_MS);
    }
  }

  function parseJsonScript(id) {
    var el = document.getElementById(id);
    if (!el) return null;
    var raw = el.textContent.trim();
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  function run() {
    var root = getRoot();
    if (!root) return;
    wireDismissDelegation(root);

    var flashes = parseJsonScript("flash-data");
    if (Array.isArray(flashes)) {
      flashes.forEach(function (pair) {
        if (!pair || pair.length < 2) return;
        var cat = pair[0];
        var msg = pair[1];
        if (msg != null && String(msg).length) showToastPlain(String(msg), cat || "info", {});
      });
    }

    var pageToasts = parseJsonScript("page-toasts");
    if (Array.isArray(pageToasts)) {
      pageToasts.forEach(function (item) {
        if (!item || typeof item !== "object") return;
        var cat = item.category || "info";
        if (item.html) {
          showToastHtml(String(item.html), cat, { wide: !!item.wide });
        } else if (item.message) {
          showToastPlain(String(item.message), cat, {});
        }
      });
    }

    var faissTpl = document.getElementById("faiss-toast-template");
    if (faissTpl && faissTpl.content) {
      var node = faissTpl.content.cloneNode(true);
      var first = node.querySelector(".toast");
      root.appendChild(node);
      if (first) {
        requestAnimationFrame(function () {
          first.classList.add("toast--in");
        });
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
