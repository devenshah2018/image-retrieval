/**
 * Architecture diagram: node tilt + SVG pipeline highlights + placeholder sphere drag.
 */
(function () {
  var root = document.querySelector(".train-arch");
  if (!root) return;
  /** Notebooks pipeline: no pointer parallax — keeps the UI stable and enterprise-sharp. */
  var notebooksStatic = root.classList.contains("train-arch--notebooks");

  /* Pills: one detail panel at a time; default none expanded */
  var pills = root.querySelectorAll(".train-arch-pill[data-pipeline-step]");
  var panels = root.querySelectorAll(".train-arch-detail__panel[data-panel-step]");
  if (pills.length && panels.length) {
    function clearPipeline() {
      pills.forEach(function (btn) {
        btn.setAttribute("aria-pressed", "false");
        btn.classList.remove("is-active");
      });
      panels.forEach(function (p) {
        p.hidden = true;
      });
    }
    function openPipelineStep(step) {
      clearPipeline();
      pills.forEach(function (btn) {
        if (btn.getAttribute("data-pipeline-step") === step) {
          btn.setAttribute("aria-pressed", "true");
          btn.classList.add("is-active");
        }
      });
      panels.forEach(function (p) {
        if (p.getAttribute("data-panel-step") === step) {
          p.hidden = false;
        }
      });
    }
    pills.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var step = btn.getAttribute("data-pipeline-step");
        if (btn.classList.contains("is-active")) {
          clearPipeline();
          return;
        }
        openPipelineStep(step);
      });
    });
  }

  var reduceMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var nodes = root.querySelectorAll(".train-arch-node");
  nodes.forEach(function (el) {
    el.setAttribute("tabindex", "0");
    var tilt = el.querySelector(".train-arch-node__tilt");

    el.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        el.classList.toggle("train-arch-node--pressed");
        window.setTimeout(function () {
          el.classList.remove("train-arch-node--pressed");
        }, 160);
      }
    });

    if (!tilt) return;

    var skipParallax =
      reduceMotion ||
      el.classList.contains("train-arch-node--globe") ||
      notebooksStatic;

    el.addEventListener("mousemove", function (ev) {
      if (skipParallax) return;
      var r = el.getBoundingClientRect();
      if (r.width < 8 || r.height < 8) return;
      var x = (ev.clientX - r.left) / r.width - 0.5;
      var y = (ev.clientY - r.top) / r.height - 0.5;
      tilt.style.setProperty("--tilt-x", (x * 8).toFixed(2) + "px");
      tilt.style.setProperty("--tilt-y", (y * 6).toFixed(2) + "px");
    });

    el.addEventListener("mouseleave", function () {
      tilt.style.removeProperty("--tilt-x");
      tilt.style.removeProperty("--tilt-y");
    });
  });

  /* Highlight SVG pipeline blocks on hover (train_quick.py / model.py diagrams) */
  root.querySelectorAll(".js-arch-viz").forEach(function (viz) {
    var hits = viz.querySelectorAll(".train-arch-viz__hit, [data-arch-piece]");
    hits.forEach(function (el) {
      el.addEventListener("pointerenter", function () {
        el.classList.add("is-active");
      });
      el.addEventListener("pointerleave", function () {
        el.classList.remove("is-active");
      });
    });
  });

  /* Placeholder category space: drag to rotate wireframe */
  var ph = root.querySelector(".js-arch-sphere-placeholder");
  if (ph && !reduceMotion) {
    var rotateEl = ph.querySelector(".js-arch-placeholder-rotate") || ph.querySelector("svg");
    var ry = 0;
    var dragging = false;

    ph.addEventListener("pointerdown", function (e) {
      dragging = true;
      ph.classList.add("is-dragging");
      ph.setPointerCapture(e.pointerId);
    });

    ph.addEventListener("pointermove", function (e) {
      if (!dragging || !rotateEl) return;
      ry += e.movementX * 0.65;
      rotateEl.style.transform = "rotateY(" + ry + "deg)";
    });

    function endDrag() {
      dragging = false;
      ph.classList.remove("is-dragging");
    }

    ph.addEventListener("pointerup", endDrag);
    ph.addEventListener("pointerleave", endDrag);
    ph.addEventListener("pointercancel", endDrag);
  }
})();
