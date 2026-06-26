// Emergent World — 3D voxel web client (Frontend).
//
// Renders the REAL simulation streamed read-only over SSE (/api/stream): a
// `snapshot` builds the voxel terrain + entities; a `delta` per tick updates
// them. Objects are drawn with per-category InstancedMesh so thousands of
// plants/berries stay fast (no lag as the world fills). Water is one flat
// sea-level surface. Click anything to inspect it.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as BufferGeometryUtils from "three/addons/utils/BufferGeometryUtils.js";

const MAX_H = 10; // voxel column height for elevation == 1.0 (gentle hills)
const TERRAIN = { SOIL: 0, ROCK: 1, WATER: 2, SAND: 3 };
const TERRAIN_NAME = { 0: "soil", 1: "rock", 2: "water", 3: "sand" };

// ---- scene -----------------------------------------------------------------

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x8fbcd4);
scene.fog = new THREE.Fog(0x8fbcd4, 60, 240);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  2000,
);
camera.position.set(60, 70, 60);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;

// floating marker shown above the currently-followed agent (visibility).
// Drawn on top (depthTest off, high renderOrder) so it is never hidden behind
// hills or other agents — you can always spot who the camera is tracking.
const followMarker = new THREE.Mesh(
  new THREE.ConeGeometry(0.7, 1.5, 4),
  new THREE.MeshBasicMaterial({
    color: 0xffe14a,
    depthTest: false,
    transparent: true,
  }),
);
followMarker.rotation.x = Math.PI; // point down at the agent
followMarker.renderOrder = 999;
followMarker.visible = false;
scene.add(followMarker);

const hemi = new THREE.HemisphereLight(0xbfd9ff, 0x4a4030, 0.9);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff2d8, 1.2);
sun.position.set(50, 80, 30);
scene.add(sun);
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);

function frameCamera() {
  const span = Math.max(gridW, gridH, 16);
  controls.target.set(0, MAX_H * 0.3, 0);
  camera.position.set(span * 0.55, span * 0.6, span * 0.55);
  camera.far = span * 8;
  camera.updateProjectionMatrix();
  if (scene.fog) {
    scene.fog.near = span * 0.9;
    scene.fog.far = span * 3.5;
  }
  controls.update();
}

// ---- module state ----------------------------------------------------------

let gridW = 0;
let gridH = 0;
let elevation = null;
let terrainCodes = null;
let fertilityGrid = null;
let moistureGrid = null;
let seaLevel = 1;

const agents = new Map(); // id -> {id, group, target, energyBar, bob, yaw, targetYaw, data}
let burning = new Set();
let signalGroup = new THREE.Group();
scene.add(signalGroup);
let followId = null; // id of the agent the camera follows (null = free-fly)
let bornReady = false;
let weatherRaining = false;

const hud = {
  tick: document.getElementById("hud-tick"),
  agents: document.getElementById("hud-agents"),
  objects: document.getElementById("hud-objects"),
  season: document.getElementById("hud-season"),
  weather: document.getElementById("hud-weather"),
  brain: document.getElementById("hud-brain"),
  conn: document.getElementById("hud-conn"),
};

const inspectorEl = document.getElementById("inspector");
const inspectTitle = document.getElementById("inspect-title");
const inspectBody = document.getElementById("inspect-body");
let selected = null; // {kind:'agent'|'object'|'tile', id?, x?, y?}
let selectedId = null;

// ---- helpers ---------------------------------------------------------------

function decodeGrid(packed) {
  const bin = atob(packed.b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}
function colBlocks(x, y) {
  if (!elevation) return 1;
  return Math.max(1, Math.round((elevation[y * gridW + x] / 255) * MAX_H));
}
function worldX(x) {
  return x + 0.5 - gridW / 2;
}
function worldZ(y) {
  return y + 0.5 - gridH / 2;
}
function surfaceY(x, y) {
  if (x < 0 || y < 0 || x >= gridW || y >= gridH) return 1;
  return colBlocks(x, y);
}
function lineageColor(lineage) {
  if (lineage < 0) return new THREE.Color(0x9aa0a6);
  const h = ((lineage * 47) % 360) / 360;
  return new THREE.Color().setHSL(h < 0 ? h + 1 : h, 0.55, 0.55);
}

const TERRAIN_COLOR = {
  [TERRAIN.SOIL]: new THREE.Color(0x4f9d3a),
  [TERRAIN.ROCK]: new THREE.Color(0x8a8d92),
  [TERRAIN.WATER]: new THREE.Color(0x2f6fb0),
  [TERRAIN.SAND]: new THREE.Color(0xd7c489),
};

// ---- terrain + flat water --------------------------------------------------

let terrainMesh = null;
let waterMesh = null;

function buildTerrain(pack) {
  gridW = pack.width;
  gridH = pack.height;
  elevation = decodeGrid(pack.elevation);
  terrainCodes = decodeGrid(pack.terrain);
  fertilityGrid = decodeGrid(pack.fertility);
  moistureGrid = decodeGrid(pack.moisture);

  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
  }
  if (waterMesh) {
    scene.remove(waterMesh);
    waterMesh.geometry.dispose();
    waterMesh.material.dispose();
  }

  const n = gridW * gridH;
  const box = new THREE.BoxGeometry(1, 1, 1);
  terrainMesh = new THREE.InstancedMesh(box, new THREE.MeshLambertMaterial({}), n);

  // sea level = mean height of water columns (a single flat surface)
  let wsum = 0;
  let wcount = 0;
  for (let i = 0; i < n; i++) {
    if (terrainCodes[i] === TERRAIN.WATER) {
      wsum += Math.max(1, Math.round((elevation[i] / 255) * MAX_H));
      wcount++;
    }
  }
  seaLevel = wcount ? Math.max(1, Math.round(wsum / wcount) + 0.5) : 1;

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const p = new THREE.Vector3();
  const col = new THREE.Color();

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const idx = y * gridW + x;
      const h = Math.max(1, Math.round((elevation[idx] / 255) * MAX_H));
      const code = terrainCodes[idx];
      const baseCode = code === TERRAIN.WATER ? TERRAIN.SAND : code; // lakebed
      col.copy(TERRAIN_COLOR[baseCode] || TERRAIN_COLOR[TERRAIN.SOIL]);
      if (baseCode === TERRAIN.SOIL || baseCode === TERRAIN.SAND) {
        col.multiplyScalar(0.7 + 0.3 * (fertilityGrid[idx] / 255));
      }
      p.set(worldX(x), h / 2, worldZ(y));
      s.set(1, h, 1);
      m.compose(p, q, s);
      terrainMesh.setMatrixAt(idx, m);
      terrainMesh.setColorAt(idx, col);
    }
  }
  terrainMesh.instanceMatrix.needsUpdate = true;
  if (terrainMesh.instanceColor) terrainMesh.instanceColor.needsUpdate = true;
  scene.add(terrainMesh);

  // one flat translucent water surface over the whole map at sea level
  const waterGeo = new THREE.PlaneGeometry(gridW, gridH);
  waterMesh = new THREE.Mesh(
    waterGeo,
    new THREE.MeshLambertMaterial({
      color: 0x2f6fb0,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
    }),
  );
  waterMesh.rotation.x = -Math.PI / 2;
  waterMesh.position.set(0, seaLevel, 0);
  scene.add(waterMesh);

  ObjLayer.reset();
  frameCamera();
}

// ---- objects: per-category InstancedMesh layer (perf) ----------------------

function _treeGeometry() {
  // Bigger than life so trees read clearly on the large map and stand out
  // against the green soil (brown trunk + tall dark-green canopy).
  const trunk = new THREE.CylinderGeometry(0.18, 0.26, 1.2, 6).translate(0, 0.6, 0);
  const foliage = new THREE.ConeGeometry(0.85, 2.0, 8).translate(0, 2.1, 0);
  return BufferGeometryUtils.mergeGeometries([trunk, foliage], false);
}

// Sizes are deliberately a bit oversized so seeds/berries/trees are visible
// from a distance on the 160×160 world (true positions, exaggerated markers).
const OBJ_CATS = {
  food: { geo: () => new THREE.IcosahedronGeometry(0.38, 0), color: 0xff2d2d },
  toxic: { geo: () => new THREE.IcosahedronGeometry(0.38, 0), color: 0xb02ee0 },
  plant: { geo: _treeGeometry, color: 0x1f7a2e },
  seed: { geo: () => new THREE.BoxGeometry(0.34, 0.34, 0.34), color: 0xf0d27a },
  fertilizer: { geo: () => new THREE.BoxGeometry(0.34, 0.22, 0.34), color: 0x6b4a2b },
  hazard: { geo: () => new THREE.ConeGeometry(0.34, 0.7, 4), color: 0x3a2a2a },
  other: { geo: () => new THREE.BoxGeometry(0.32, 0.32, 0.32), color: 0xb0b0b0 },
};

function categoryOf(o) {
  const cat = (o.category || "").toLowerCase();
  const tid = (o.type_id || "").toLowerCase();
  if (tid.includes("night") || tid.includes("toxic")) return "toxic";
  if (cat.includes("food") || tid.includes("berry") || tid.includes("fruit"))
    return "food";
  if (cat.includes("plant") || tid.includes("tree") || tid.includes("shrub"))
    return "plant";
  if (cat.includes("seed")) return "seed";
  if (cat.includes("fertil")) return "fertilizer";
  if (cat.includes("hazard") || tid.includes("thorn")) return "hazard";
  return "other";
}

// One growable InstancedMesh per category with slot bookkeeping.
const ObjLayer = {
  cats: {},
  byId: new Map(), // objId -> {cat, slot}
  data: new Map(), // objId -> object view (for inspector)

  reset() {
    for (const k of Object.keys(this.cats)) {
      const c = this.cats[k];
      if (c.mesh) scene.remove(c.mesh);
    }
    this.cats = {};
    this.byId.clear();
    this.data.clear();
  },

  _ensure(cat) {
    if (this.cats[cat]) return this.cats[cat];
    const spec = OBJ_CATS[cat] || OBJ_CATS.other;
    const c = { spec, capacity: 256, count: 0, slotToId: [], mesh: null };
    c.mesh = new THREE.InstancedMesh(
      spec.geo(),
      new THREE.MeshLambertMaterial({ color: spec.color }),
      c.capacity,
    );
    c.mesh.count = 0;
    c.mesh.userData.cat = cat;
    scene.add(c.mesh);
    this.cats[cat] = c;
    return c;
  },

  _grow(c) {
    const newCap = c.capacity * 2;
    const mesh = new THREE.InstancedMesh(c.mesh.geometry, c.mesh.material, newCap);
    const m = new THREE.Matrix4();
    for (let i = 0; i < c.count; i++) {
      c.mesh.getMatrixAt(i, m);
      mesh.setMatrixAt(i, m);
    }
    mesh.count = c.count;
    mesh.userData.cat = c.mesh.userData.cat;
    scene.remove(c.mesh);
    scene.add(mesh);
    c.mesh = mesh;
    c.capacity = newCap;
  },

  _matrix(o) {
    const m = new THREE.Matrix4();
    const cat = categoryOf(o);
    let sc = 1;
    if (cat === "plant") sc = 0.5 + 0.7 * (o.value || 0); // grow with maturity
    const yOff = cat === "plant" ? 0 : 0.3;
    m.compose(
      new THREE.Vector3(worldX(o.x), surfaceY(o.x, o.y) + yOff, worldZ(o.y)),
      new THREE.Quaternion(),
      new THREE.Vector3(sc, sc, sc),
    );
    return m;
  },

  upsert(o) {
    this.data.set(o.id, o);
    const cat = categoryOf(o);
    const existing = this.byId.get(o.id);
    if (existing && existing.cat === cat) {
      const c = this.cats[cat];
      c.mesh.setMatrixAt(existing.slot, this._matrix(o));
      c.mesh.instanceMatrix.needsUpdate = true;
      return;
    }
    if (existing) this.remove(o.id, true); // category changed → re-add
    const c = this._ensure(cat);
    if (c.count >= c.capacity) this._grow(c);
    const slot = c.count++;
    c.slotToId[slot] = o.id;
    c.mesh.setMatrixAt(slot, this._matrix(o));
    c.mesh.count = c.count;
    c.mesh.instanceMatrix.needsUpdate = true;
    this.byId.set(o.id, { cat, slot });
  },

  remove(id, keepData) {
    const rec = this.byId.get(id);
    if (!rec) return;
    const c = this.cats[rec.cat];
    const last = c.count - 1;
    const m = new THREE.Matrix4();
    if (rec.slot !== last) {
      c.mesh.getMatrixAt(last, m);
      c.mesh.setMatrixAt(rec.slot, m);
      const movedId = c.slotToId[last];
      c.slotToId[rec.slot] = movedId;
      const movedRec = this.byId.get(movedId);
      if (movedRec) movedRec.slot = rec.slot;
    }
    c.count = last;
    c.mesh.count = last;
    c.slotToId.length = last;
    c.mesh.instanceMatrix.needsUpdate = true;
    this.byId.delete(id);
    if (!keepData) this.data.delete(id);
  },

  positionOf(id) {
    const o = this.data.get(id);
    if (!o) return null;
    return new THREE.Vector3(worldX(o.x), surfaceY(o.x, o.y) + 0.3, worldZ(o.y));
  },

  count() {
    return this.byId.size;
  },

  raycast(raycaster) {
    for (const k of Object.keys(this.cats)) {
      const c = this.cats[k];
      if (!c.count) continue;
      const hits = raycaster.intersectObject(c.mesh, false);
      if (hits.length && hits[0].instanceId != null) {
        const id = c.slotToId[hits[0].instanceId];
        if (id !== undefined) return id;
      }
    }
    return null;
  },
};

// ---- agents ----------------------------------------------------------------

function makeAgentModel(lineage) {
  const group = new THREE.Group();
  const color = lineageColor(lineage);
  const bob = new THREE.Group();
  group.add(bob);
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 0.6, 0.6),
    new THREE.MeshLambertMaterial({ color }),
  );
  body.position.y = 0.3;
  bob.add(body);
  const head = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 0.34, 0.34),
    new THREE.MeshLambertMaterial({ color: color.clone().offsetHSL(0, 0, 0.12) }),
  );
  head.position.y = 0.78;
  bob.add(head);
  const snout = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.12, 0.22),
    new THREE.MeshLambertMaterial({ color: 0x222222 }),
  );
  snout.position.set(0, 0.78, 0.24);
  bob.add(snout);
  const bar = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 0.08, 0.08),
    new THREE.MeshBasicMaterial({ color: 0x6bd06b }),
  );
  bar.position.y = 1.15;
  group.add(bar);
  body.userData.agentGroup = group;
  head.userData.agentGroup = group;
  group.userData.energyBar = bar;
  group.userData.bob = bob;
  group.userData.phase = Math.random() * Math.PI * 2;
  return group;
}

function upsertAgent(a) {
  let rec = agents.get(a.id);
  if (!rec) {
    const group = makeAgentModel(a.lineage);
    scene.add(group);
    rec = {
      id: a.id,
      group,
      target: new THREE.Vector3(),
      energyBar: group.userData.energyBar,
      bob: group.userData.bob,
      yaw: 0,
      targetYaw: 0,
      data: {},
    };
    rec.group.position.set(worldX(a.x), surfaceY(a.x, a.y), worldZ(a.y));
    agents.set(a.id, rec);
    if (bornReady) spawnBurst(rec.group.position, 0x8ef0a0, 8);
  }
  rec.target.set(worldX(a.x), surfaceY(a.x, a.y), worldZ(a.y));
  const [dx, dy] = a.dir;
  rec.targetYaw = Math.atan2(dx, dy);
  const e = Math.max(0, Math.min(1, a.energy));
  rec.energyBar.scale.x = Math.max(0.02, e);
  rec.energyBar.material.color.setHSL(0.33 * e, 0.7, 0.5);
  rec.group.visible = a.alive !== false;
  rec.data = {
    id: a.id,
    energy: e,
    age: a.age,
    action: a.action,
    inv: a.inv,
    has_food: a.has_food,
    has_seed: a.has_seed,
    lineage: a.lineage,
    generation: a.generation,
    cohort: a.cohort,
    x: a.x,
    y: a.y,
  };
  if (selected && selected.kind === "agent" && selected.id === a.id) renderInspector();
}

// ---- particles (birth/death/consume/fire/rain) -----------------------------

const particles = [];
const _particleGeo = new THREE.SphereGeometry(0.09, 6, 6);
function spawnBurst(pos, colorHex, count = 8, speed = 2.0) {
  for (let i = 0; i < count; i++) {
    const m = new THREE.Mesh(
      _particleGeo,
      new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 1 }),
    );
    m.position.copy(pos);
    m.position.y += 0.5;
    scene.add(m);
    particles.push({
      mesh: m,
      life: 0.8,
      maxLife: 0.8,
      vel: new THREE.Vector3(
        (Math.random() - 0.5) * speed,
        Math.random() * speed * 0.9 + 0.5,
        (Math.random() - 0.5) * speed,
      ),
    });
  }
}
function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life -= dt;
    if (p.life <= 0) {
      scene.remove(p.mesh);
      p.mesh.material.dispose();
      particles.splice(i, 1);
      continue;
    }
    p.vel.y -= 6 * dt;
    p.mesh.position.addScaledVector(p.vel, dt);
    const f = p.life / p.maxLife;
    p.mesh.material.opacity = f;
    p.mesh.scale.setScalar(0.5 + f);
  }
}

const rainDrops = [];
const _rainGeo = new THREE.BoxGeometry(0.04, 0.5, 0.04);
const _rainMat = new THREE.MeshBasicMaterial({
  color: 0x9ec4ff,
  transparent: true,
  opacity: 0.5,
});
function updateRain(dt) {
  if (weatherRaining && rainDrops.length < 300) {
    for (let i = 0; i < 10; i++) {
      const m = new THREE.Mesh(_rainGeo, _rainMat);
      m.position.set(
        (Math.random() - 0.5) * gridW,
        MAX_H + 6 + Math.random() * 6,
        (Math.random() - 0.5) * gridH,
      );
      scene.add(m);
      rainDrops.push({ mesh: m, vy: -18 - Math.random() * 6 });
    }
  }
  for (let i = rainDrops.length - 1; i >= 0; i--) {
    const r = rainDrops[i];
    r.mesh.position.y += r.vy * dt;
    if (r.mesh.position.y < 0) {
      scene.remove(r.mesh);
      rainDrops.splice(i, 1);
    }
  }
}

// ---- signals ---------------------------------------------------------------

function updateSignals(signals) {
  scene.remove(signalGroup);
  signalGroup = new THREE.Group();
  const geo = new THREE.PlaneGeometry(1, 1);
  for (const [x, y, v] of signals) {
    const base = Math.min(0.7, 0.15 + 0.6 * v);
    const m = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        color: 0x66e0ff,
        transparent: true,
        opacity: base,
        depthWrite: false,
      }),
    );
    m.userData.base = base;
    m.rotation.x = -Math.PI / 2;
    m.position.set(worldX(x), surfaceY(x, y) + 0.05, worldZ(y));
    signalGroup.add(m);
  }
  scene.add(signalGroup);
}

// ---- sky / weather ---------------------------------------------------------

function updateSky(sky) {
  if (!sky) return;
  const t = sky.time_of_day ?? 0;
  const light = sky.light ?? 1;
  const ang = t * Math.PI * 2 - Math.PI / 2;
  sun.position.set(Math.cos(ang) * 80, Math.max(10, Math.sin(ang) * 90), 30);
  sun.intensity = 0.5 + 0.9 * light;
  hemi.intensity = 0.5 + 0.5 * light;
  ambient.intensity = 0.3 + 0.25 * light;
  const night = new THREE.Color(0x1b2b4a);
  const day = new THREE.Color(0x8fbcd4);
  let bg = night.clone().lerp(day, light);
  if (sky.drought) bg = bg.lerp(new THREE.Color(0xb98a4a), 0.35);
  if (sky.raining) bg = bg.lerp(new THREE.Color(0x4a5566), 0.45);
  scene.background = bg;
  if (scene.fog) scene.fog.color = bg;
  weatherRaining = !!sky.raining;
  hud.season.textContent = (sky.season ?? 0).toFixed(2);
  hud.weather.textContent = sky.raining ? "rain" : sky.drought ? "drought" : "clear";
}

// ---- inspector -------------------------------------------------------------

function showInspector(title, rows) {
  inspectorEl.classList.remove("hidden");
  inspectTitle.textContent = title;
  inspectBody.innerHTML = rows
    .map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`)
    .join("");
}
function clearInspector() {
  selected = null;
  selectedId = null;
  inspectorEl.classList.add("hidden");
}
function renderInspector() {
  if (!selected) return;
  if (selected.kind === "agent") {
    const rec = agents.get(selected.id);
    if (!rec) return clearInspector();
    const d = rec.data;
    const carry =
      d.inv > 0
        ? `${d.inv}${d.has_food ? " 🍒" : ""}${d.has_seed ? " 🌱" : ""}`
        : "empty";
    showInspector(`agent #${d.id}`, [
      ["cohort", d.cohort || "default"],
      ["energy", (d.energy * 100).toFixed(0) + "%"],
      ["age", (d.age * 100).toFixed(0) + "% of max"],
      ["last action", d.action || "—"],
      ["carrying", carry],
      ["lineage", d.lineage],
      ["generation", d.generation],
      ["position", `${d.x}, ${d.y}`],
    ]);
  } else if (selected.kind === "object") {
    const o = ObjLayer.data.get(selected.id);
    if (!o) return clearInspector();
    const cat = (o.category || "").toLowerCase();
    const valLabel = cat.includes("food")
      ? "freshness"
      : cat.includes("plant")
        ? "maturity"
        : cat.includes("seed")
          ? "viability"
          : "value";
    showInspector(`${o.type_id || o.category} #${o.id}`, [
      ["category", o.category],
      ["type", o.type_id || "—"],
      [valLabel, ((o.value || 0) * 100).toFixed(0) + "%"],
      ["planted by agent", o.planted ? "yes" : "no"],
      ["on fire", burning.has(o.id) ? "yes" : "no"],
      ["position", `${o.x}, ${o.y}`],
    ]);
  } else if (selected.kind === "tile") {
    const { x, y } = selected;
    if (!terrainCodes) return;
    const i = y * gridW + x;
    showInspector(`tile ${x}, ${y}`, [
      ["terrain", TERRAIN_NAME[terrainCodes[i]] || "?"],
      ["elevation", (elevation[i] / 255).toFixed(2)],
      ["fertility", (fertilityGrid[i] / 255).toFixed(2)],
      ["moisture", (moistureGrid[i] / 255).toFixed(2)],
    ]);
  }
}

// ---- snapshot / delta ------------------------------------------------------

function applySnapshot(snap) {
  bornReady = false;
  buildTerrain(snap.terrain);
  for (const a of agents.values()) scene.remove(a.group);
  agents.clear();
  for (const o of snap.objects) ObjLayer.upsert(o);
  for (const a of snap.agents) upsertAgent(a);
  burning = new Set(snap.burning || []);
  updateSky(snap.sky);
  hud.objects.textContent = ObjLayer.count();
  hud.tick.textContent = snap.tick;
  const out = snap.brain_output_size;
  const isV3 = snap.brain_class === "BrainV3";
  const ver = out === 9 ? "v3.5" : isV3 ? "v3" : out === 8 ? "v2" : "—";
  const feats = [];
  if (snap.signal_enabled) feats.push("signal");
  if (snap.transfer_enabled) feats.push("trade");
  hud.brain.textContent = ver + (feats.length ? " · " + feats.join("+") : "");
  bornReady = true;
}

function applyDelta(d) {
  for (const o of d.objects || []) ObjLayer.upsert(o);
  for (const id of d.removed_objects || []) {
    const pos = ObjLayer.positionOf(id);
    if (pos) spawnBurst(pos, 0xffb24a, 5, 1.4);
    ObjLayer.remove(id);
  }
  for (const a of d.agents || []) upsertAgent(a);
  for (const id of d.removed_agents || []) {
    const r = agents.get(id);
    if (r) {
      spawnBurst(r.group.position, 0x9aa0a6, 12, 2.2);
      scene.remove(r.group);
      agents.delete(id);
      if (selected && selected.kind === "agent" && selected.id === id) clearInspector();
    }
  }
  if (d.burning) burning = new Set(d.burning);
  updateSignals(d.signals || []);
  updateSky(d.sky);
  hud.tick.textContent = d.tick;
  hud.agents.textContent = agents.size;
  hud.objects.textContent = ObjLayer.count();
}

// ---- input -----------------------------------------------------------------

const keys = new Set();
window.addEventListener("keydown", (e) => {
  keys.add(e.key.toLowerCase());
  if (e.key.toLowerCase() === "f") cycleFollow();
  if (e.key.toLowerCase() === "r") resetCamera();
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

function resetCamera() {
  followId = null;
  frameCamera();
}
function nearestAgentId(toVec) {
  let best = null;
  let bestD = Infinity;
  for (const rec of agents.values()) {
    const d = rec.group.position.distanceToSquared(toVec);
    if (d < bestD) {
      bestD = d;
      best = rec.id;
    }
  }
  return best;
}
// F cycles to the NEXT living agent (by id), then back to free-fly.
function cycleFollow() {
  const ids = Array.from(agents.keys()).sort((a, b) => a - b);
  if (!ids.length) {
    followId = null;
    return;
  }
  if (followId == null) {
    followId = ids[0];
  } else {
    const i = ids.indexOf(followId);
    const next = i < 0 ? 0 : i + 1;
    followId = next >= ids.length ? null : ids[next]; // wrap → free-fly
  }
  if (followId != null) {
    selected = { kind: "agent", id: followId };
    selectedId = followId;
    renderInspector();
  } else {
    clearInspector();
  }
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let dragMoved = false;
canvas.addEventListener("pointerdown", () => (dragMoved = false));
canvas.addEventListener("pointermove", () => (dragMoved = true));
canvas.addEventListener("pointerup", (e) => {
  if (dragMoved) return;
  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const agentMeshes = [];
  for (const rec of agents.values())
    rec.group.traverse((o) => o.isMesh && agentMeshes.push(o));
  let hits = raycaster.intersectObjects(agentMeshes, false);
  if (hits.length) {
    let g = hits[0].object;
    while (g && !g.userData.agentGroup) g = g.parent;
    for (const rec of agents.values()) {
      if (rec.group === g.userData.agentGroup) {
        selected = { kind: "agent", id: rec.id };
        selectedId = rec.id;
        renderInspector();
        return;
      }
    }
  }
  const oid = ObjLayer.raycast(raycaster);
  if (oid != null) {
    selected = { kind: "object", id: oid };
    renderInspector();
    return;
  }
  if (terrainMesh) {
    hits = raycaster.intersectObject(terrainMesh, false);
    if (hits.length && hits[0].instanceId != null) {
      const idx = hits[0].instanceId;
      selected = { kind: "tile", x: idx % gridW, y: Math.floor(idx / gridW) };
      renderInspector();
      return;
    }
  }
  clearInspector();
});

function applyFreeFly(dt) {
  const speed = 28 * dt;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const move = new THREE.Vector3();
  if (keys.has("w")) move.add(forward);
  if (keys.has("s")) move.addScaledVector(forward, -1);
  if (keys.has("d")) move.add(right);
  if (keys.has("a")) move.addScaledVector(right, -1);
  if (keys.has("e")) move.y += 1;
  if (keys.has("q")) move.y -= 1;
  if (move.lengthSq() > 0) {
    move.normalize().multiplyScalar(speed);
    camera.position.add(move);
    controls.target.add(move);
  }
}

// ---- animation -------------------------------------------------------------

const clock = new THREE.Clock();
let elapsed = 0;
const _chase = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, clock.getDelta());
  elapsed += dt;

  for (const rec of agents.values()) {
    rec.group.position.lerp(rec.target, 0.18);
    let dyaw = rec.targetYaw - rec.yaw;
    dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
    rec.yaw += dyaw * 0.25;
    rec.group.rotation.y = rec.yaw;
    if (rec.bob) rec.bob.position.y = Math.sin(elapsed * 4 + rec.group.userData.phase) * 0.05;
  }

  updateParticles(dt);
  updateRain(dt);

  // fire: flame particles at burning object positions
  if (burning.size && Math.random() < 0.5) {
    for (const id of burning) {
      const pos = ObjLayer.positionOf(id);
      if (pos && Math.random() < 0.2) spawnBurst(pos, 0xff7a1e, 1, 1.6);
    }
  }

  const pulse = 0.75 + 0.25 * Math.sin(elapsed * 3);
  signalGroup.children.forEach((m) => (m.material.opacity = m.userData.base * pulse));

  if (waterMesh) waterMesh.position.y = seaLevel + Math.sin(elapsed * 1.5) * 0.05;

  if (followId != null) {
    let rec = agents.get(followId);
    if (!rec) {
      // the followed agent died — hand off to the nearest living one so the
      // camera (and marker) keep tracking instead of freezing on a corpse.
      followId = nearestAgentId(controls.target);
      rec = followId != null ? agents.get(followId) : null;
      if (selected && selected.kind === "agent") {
        selected.id = followId;
        selectedId = followId;
        followId != null ? renderInspector() : clearInspector();
      }
    }
    if (rec) {
      controls.target.lerp(rec.group.position, 0.2);
      _chase.copy(rec.group.position).add(new THREE.Vector3(6, 7, 6));
      camera.position.lerp(_chase, 0.12);
      // bob + spin a bright marker above the followed agent for visibility
      followMarker.visible = true;
      followMarker.position.set(
        rec.group.position.x,
        rec.group.position.y + 2.6 + Math.sin(elapsed * 4) * 0.2,
        rec.group.position.z,
      );
      followMarker.rotation.y = elapsed * 2;
    } else {
      followMarker.visible = false;
    }
  } else {
    followMarker.visible = false;
    applyFreeFly(dt);
  }

  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---- SSE -------------------------------------------------------------------

function connect() {
  const es = new EventSource("/api/stream");
  es.addEventListener("snapshot", (ev) => {
    hud.conn.textContent = "live";
    hud.conn.className = "ok";
    applySnapshot(JSON.parse(ev.data));
  });
  es.addEventListener("delta", (ev) => applyDelta(JSON.parse(ev.data)));
  es.onerror = () => {
    hud.conn.textContent = "reconnecting…";
    hud.conn.className = "err";
  };
}

connect();
animate();
