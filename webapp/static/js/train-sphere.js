/**
 * 3D "watch sphere" — categories on a Fibonacci sphere, drag to rotate (rotateX / rotateY).
 */
(function () {
  var dataEl = document.getElementById("train-sphere-data");
  var inner = document.getElementById("train-sphere-inner");
  var viewport = document.getElementById("train-sphere-viewport");
  if (!dataEl || !inner || !viewport) return;

  var categories;
  try {
    categories = JSON.parse(dataEl.textContent || "[]");
  } catch (e) {
    return;
  }
  if (!categories.length) return;

  /* Radius scales with √N; tighter in the process strip (shorter viewport). */
  var compact = viewport.classList.contains("train-sphere-viewport--process");
  var R = compact
    ? 72 + Math.sqrt(categories.length) * 6.2
    : 90 + Math.sqrt(categories.length) * 8.2;
  var golden = Math.PI * (3 - Math.sqrt(5));

  function fibonacciSphere(n) {
    var pts = [];
    for (var i = 0; i < n; i++) {
      var y = n > 1 ? 1 - (i / (n - 1)) * 2 : 0;
      var yClamped = Math.max(-1, Math.min(1, y));
      var r = Math.sqrt(Math.max(0, 1 - yClamped * yClamped));
      var theta = golden * i;
      pts.push({
        x: Math.cos(theta) * r * R,
        y: yClamped * R,
        z: Math.sin(theta) * r * R,
      });
    }
    return pts;
  }

  var pts = fibonacciSphere(categories.length);
  var nodes = [];

  /** Align local +Z with the radial direction from center (badge sits on the sphere surface). */
  function radialTransform(p) {
    var len = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) || 1;
    var nx = p.x / len;
    var ny = p.y / len;
    var nz = p.z / len;
    var ryDeg = Math.atan2(nx, nz) * (180 / Math.PI);
    var rxDeg = -Math.asin(Math.max(-1, Math.min(1, ny))) * (180 / Math.PI);
    return (
      "translate(-50%, -50%) translate3d(" +
      p.x.toFixed(2) +
      "px," +
      p.y.toFixed(2) +
      "px," +
      p.z.toFixed(2) +
      "px) rotateY(" +
      ryDeg.toFixed(2) +
      "deg) rotateX(" +
      rxDeg.toFixed(2) +
      "deg)"
    );
  }

  /**
   * Same axis order as CSS: rotateX(rx) rotateY(ry) applies R_y first then R_x (right-to-left).
   * Used to rotate world outward normals for visibility.
   */
  function rotateParentNormal(nx, ny, nz, rxDeg, ryDeg) {
    var rx = (rxDeg * Math.PI) / 180;
    var ry = (ryDeg * Math.PI) / 180;
    var cx = Math.cos(rx);
    var sx = Math.sin(rx);
    var cy = Math.cos(ry);
    var sy = Math.sin(ry);
    var x1 = cy * nx + sy * nz;
    var y1 = ny;
    var z1 = -sy * nx + cy * nz;
    var x2 = x1;
    var y2 = cx * y1 - sx * z1;
    var z2 = sx * y1 + cx * z1;
    return { x: x2, y: y2, z: z2 };
  }

  categories.forEach(function (label, i) {
    var el = document.createElement("span");
    el.className = "train-sphere-node";
    el.textContent = String(label);
    el.setAttribute("role", "listitem");
    nodes.push(el);
    inner.appendChild(el);
  });

  var rx = -18;
  var ry = 0;
  var dragging = false;
  var lastX = 0;
  var lastY = 0;

  function apply() {
    inner.style.transform =
      "rotateX(" + rx + "deg) rotateY(" + ry + "deg)";
    nodes.forEach(function (el, i) {
      var p = pts[i];
      el.style.transform = radialTransform(p);
      var len = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) || 1;
      var nw = rotateParentNormal(p.x / len, p.y / len, p.z / len, rx, ry);
      var t = nw.z;
      if (t <= 0) {
        el.style.opacity = String(Math.max(0, 0.06 + t * 0.05));
      } else {
        el.style.opacity = String(Math.min(1, 0.28 + 0.72 * t));
      }
    });
  }

  apply();

  function onDown(e) {
    dragging = true;
    lastX = e.clientX ?? e.touches[0].clientX;
    lastY = e.clientY ?? e.touches[0].clientY;
    viewport.style.cursor = "grabbing";
  }

  function onMove(e) {
    if (!dragging) return;
    var x = e.clientX ?? e.touches[0].clientX;
    var y = e.clientY ?? e.touches[0].clientY;
    var dx = x - lastX;
    var dy = y - lastY;
    lastX = x;
    lastY = y;
    ry += dx * 0.45;
    rx -= dy * 0.45;
    rx = Math.max(-72, Math.min(72, rx));
    apply();
  }

  function onUp() {
    dragging = false;
    viewport.style.cursor = "grab";
  }

  viewport.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  viewport.addEventListener(
    "touchstart",
    function (e) {
      onDown(e);
    },
    { passive: true }
  );
  window.addEventListener(
    "touchmove",
    function (e) {
      if (dragging) e.preventDefault();
      onMove(e);
    },
    { passive: false }
  );
  window.addEventListener("touchend", onUp);
})();
