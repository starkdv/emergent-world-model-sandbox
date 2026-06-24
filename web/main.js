// Emergent World — 3D voxel web client (Frontend phase F2/F3b).
//
// Consumes the F0 state bridge over the F3a SSE stream (/api/stream):
//   - a `snapshot` event builds the voxel terrain (blocky columns from the W2
//     elevation grid) and the initial entities;
//   - a `delta` event per tick upserts/removes agents & objects, updates the
//     pheromone glow, and animates the day/night sky.
// Agents are grounded on their tile's surface height and tween between ticks so
// movement over elevation reads smoothly. Distinct models per object category;
// agents tinted per lineage. Camera: orbit + free-fly + follow.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MAX_H = 18; // voxel column height for elevation == 1.0
const TERRAIN = { SOIL: 0, ROCK: 1, WATER: 2, SAND: 3 };

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
  1000,
);
camera.position.set(40, 50, 40);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;

const hemi = new THREE.HemisphereLight(0xbfd9ff, 0x4a4030, 0.9);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff2d8, 1.2);
sun.position.set(50, 80, 30);
scene.add(sun);
// always-on ambient so geometry stays readable even at deep night
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);

function frameCamera() {
  // place the camera relative to the world size so it isn't tiny / clipped
  const span = Math.max(gridW, gridH, 16);
  controls.target.set(0, MAX_H * 0.3, 0);
  camera.position.set(span * 0.55, span * 0.6, span * 0.55);
  camera.far = span * 6;
  camera.updateProjectionMatrix();
  if (scene.fog) {
    scene.fog.near = span * 0.9;
    scene.fog.far = span * 3.2;
  }
  controls.update();
}

// ---- module state ----------------------------------------------------------

let gridW = 0;
let gridH = 0;
let elevation = null; // Uint8Array [h*w]
let terrainCodes = null; // Uint8Array [h*w]
const agents = new Map(); // id -> { group, target:THREE.Vector3, energyBar }
const objects = new Map(); // id -> THREE.Object3D
let burning = new Set(); // object ids currently on fire (W3)
let signalGroup = new THREE.Group();
scene.add(signalGroup);
let followIds = [];
let followIdx = -1;
let bornReady = false; // suppress birth bursts while applying the initial snapshot

// ---- particle bursts (lifecycle effects, derived from deltas) --------------
const particles = []; // { mesh, life, maxLife, vel }
const _particleGeo = new THREE.SphereGeometry(0.09, 6, 6);

function spawnBurst(pos, colorHex, count = 8, speed = 2.0) {
  for (let i = 0; i < count; i++) {
    const mat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 1,
      depthWrite: false,
    });
    const m = new THREE.Mesh(_particleGeo, mat);
    m.position.copy(pos);
    m.position.y += 0.5;
    const vel = new THREE.Vector3(
      (Math.random() - 0.5) * speed,
      Math.random() * speed * 0.9 + 0.5,
      (Math.random() - 0.5) * speed,
    );
    scene.add(m);
    particles.push({ mesh: m, life: 0.8, maxLife: 0.8, vel });
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
    p.vel.y -= 6 * dt; // gravity
    p.mesh.position.addScaledVector(p.vel, dt);
    const f = p.life / p.maxLife;
    p.mesh.material.opacity = f;
    p.mesh.scale.setScalar(0.5 + f);
  }
}

const hud = {
  tick: document.getElementById("hud-tick"),
  agents: document.getElementById("hud-agents"),
  objects: document.getElementById("hud-objects"),
  season: document.getElementById("hud-season"),
  weather: document.getElementById("hud-weather"),
  conn: document.getElementById("hud-conn"),
};

const inspectorEl = document.getElementById("inspector");
const ins = {
  id: document.getElementById("ins-id"),
  energy: document.getElementById("ins-energy"),
  lineage: document.getElementById("ins-lineage"),
  gen: document.getElementById("ins-gen"),
  pos: document.getElementById("ins-pos"),
};
let selectedId = null;

function updateInspector(d) {
  if (!d) {
    inspectorEl.classList.add("hidden");
    return;
  }
  inspectorEl.classList.remove("hidden");
  ins.id.textContent = d.id;
  ins.energy.textContent = (d.energy * 100).toFixed(0) + "%";
  ins.lineage.textContent = d.lineage;
  ins.gen.textContent = d.generation;
  ins.pos.textContent = `${d.x},${d.y}`;
}

// ---- helpers ---------------------------------------------------------------

function decodeGrid(packed) {
  // base64 -> Uint8Array, row-major [shape[0]=h, shape[1]=w]
  const bin = atob(packed.b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

function colBlocks(x, y) {
  if (!elevation) return 1;
  const e = elevation[y * gridW + x] / 255;
  return Math.max(1, Math.round(e * MAX_H));
}

// world grid (x,y) -> centered three position helpers
function worldX(x) {
  return x + 0.5 - gridW / 2;
}
function worldZ(y) {
  return y + 0.5 - gridH / 2;
}
function surfaceY(x, y) {
  return colBlocks(x, y); // top of the column (base at 0)
}

const TERRAIN_COLOR = {
  [TERRAIN.SOIL]: new THREE.Color(0x4f9d3a),
  [TERRAIN.ROCK]: new THREE.Color(0x8a8d92),
  [TERRAIN.WATER]: new THREE.Color(0x2f6fb0),
  [TERRAIN.SAND]: new THREE.Color(0xd7c489),
};

function lineageColor(lineage) {
  // stable hue from the lineage id
  const h = ((lineage * 47) % 360) / 360;
  return new THREE.Color().setHSL(h < 0 ? h + 1 : h, 0.55, 0.55);
}

// ---- terrain ---------------------------------------------------------------

let terrainMesh = null;
let waterMesh = null;

function buildTerrain(pack) {
  gridW = pack.width;
  gridH = pack.height;
  elevation = decodeGrid(pack.elevation);
  terrainCodes = decodeGrid(pack.terrain);
  const fertility = decodeGrid(pack.fertility);

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

  const box = new THREE.BoxGeometry(1, 1, 1);
  const landMat = new THREE.MeshLambertMaterial({ vertexColors: true });
  const n = gridW * gridH;
  terrainMesh = new THREE.InstancedMesh(box, landMat, n);

  const waterBox = new THREE.BoxGeometry(1, 1, 1);
  const waterMat = new THREE.MeshLambertMaterial({
    color: 0x2f6fb0,
    transparent: true,
    opacity: 0.6,
  });
  // count water tiles
  let waterCount = 0;
  for (let i = 0; i < n; i++) if (terrainCodes[i] === TERRAIN.WATER) waterCount++;
  waterMesh = new THREE.InstancedMesh(waterBox, waterMat, Math.max(1, waterCount));

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const p = new THREE.Vector3();
  const col = new THREE.Color();
  let wi = 0;

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const idx = y * gridW + x;
      const h = Math.max(1, Math.round((elevation[idx] / 255) * MAX_H));
      const code = terrainCodes[idx];

      // land column (full height; water gets a dirt/sand base under the water)
      const baseCode = code === TERRAIN.WATER ? TERRAIN.SAND : code;
      col.copy(TERRAIN_COLOR[baseCode] || TERRAIN_COLOR[TERRAIN.SOIL]);
      // shade by fertility for soil/sand
      if (baseCode === TERRAIN.SOIL || baseCode === TERRAIN.SAND) {
        const f = 0.7 + 0.3 * (fertility[idx] / 255);
        col.multiplyScalar(f);
      }
      p.set(worldX(x), h / 2, worldZ(y));
      s.set(1, h, 1);
      m.compose(p, q, s);
      terrainMesh.setMatrixAt(idx, m);
      terrainMesh.setColorAt(idx, col);

      if (code === TERRAIN.WATER) {
        const wh = Math.max(1, h + 1); // a shallow translucent layer on top
        p.set(worldX(x), wh - 0.25, worldZ(y));
        s.set(1, 0.5, 1);
        m.compose(p, q, s);
        waterMesh.setMatrixAt(wi++, m);
      }
    }
  }
  terrainMesh.instanceMatrix.needsUpdate = true;
  if (terrainMesh.instanceColor) terrainMesh.instanceColor.needsUpdate = true;
  waterMesh.instanceMatrix.needsUpdate = true;
  scene.add(terrainMesh);
  scene.add(waterMesh);

  frameCamera();
}

// ---- entity models ---------------------------------------------------------

function makeAgentModel(lineage) {
  const group = new THREE.Group();
  const color = lineageColor(lineage);
  // inner group bobs/yaws; the energy bar stays put on the outer group
  const bob = new THREE.Group();
  group.add(bob);
  const bodyMat = new THREE.MeshLambertMaterial({ color });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.6), bodyMat);
  body.position.y = 0.3;
  bob.add(body);
  const head = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 0.34, 0.34),
    new THREE.MeshLambertMaterial({ color: color.clone().offsetHSL(0, 0, 0.12) }),
  );
  head.position.y = 0.78;
  bob.add(head);
  // facing marker (snout) — points +Z, the yaw reference
  const snout = new THREE.Mesh(
    new THREE.BoxGeometry(0.12, 0.12, 0.22),
    new THREE.MeshLambertMaterial({ color: 0x222222 }),
  );
  snout.position.set(0, 0.78, 0.24);
  bob.add(snout);
  // energy bar
  const bar = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 0.08, 0.08),
    new THREE.MeshBasicMaterial({ color: 0x6bd06b }),
  );
  bar.position.y = 1.15;
  group.add(bar);
  // make the whole agent pickable → store a back-reference for raycasting
  body.userData.agentGroup = group;
  head.userData.agentGroup = group;
  group.userData.energyBar = bar;
  group.userData.bob = bob;
  group.userData.phase = Math.random() * Math.PI * 2;
  return group;
}

// distinct per-category object models
function makeObjectModel(o) {
  const cat = (o.category || "object").toLowerCase();
  const tid = (o.type_id || "").toLowerCase();
  let geo, mat;
  if (cat.includes("food") || tid.includes("berry") || tid.includes("fruit")) {
    geo = new THREE.IcosahedronGeometry(0.22, 0);
    const toxic = tid.includes("night") || tid.includes("toxic");
    mat = new THREE.MeshLambertMaterial({ color: toxic ? 0x9b30c0 : 0xe23b3b });
  } else if (cat.includes("plant") || tid.includes("tree") || tid.includes("shrub")) {
    geo = new THREE.ConeGeometry(0.3, 0.7, 6);
    mat = new THREE.MeshLambertMaterial({ color: 0x2f8f3a });
  } else if (cat.includes("seed")) {
    geo = new THREE.BoxGeometry(0.18, 0.18, 0.18);
    mat = new THREE.MeshLambertMaterial({ color: 0xcdb079 });
  } else if (cat.includes("fertil")) {
    geo = new THREE.BoxGeometry(0.26, 0.16, 0.26);
    mat = new THREE.MeshLambertMaterial({ color: 0x6b4a2b });
  } else if (cat.includes("hazard") || tid.includes("thorn")) {
    geo = new THREE.ConeGeometry(0.26, 0.5, 4);
    mat = new THREE.MeshLambertMaterial({ color: 0x3a2a2a });
  } else {
    geo = new THREE.BoxGeometry(0.24, 0.24, 0.24);
    mat = new THREE.MeshLambertMaterial({ color: 0xb0b0b0 });
  }
  return new THREE.Mesh(geo, mat);
}

function placeOnSurface(obj3d, x, y, yOffset = 0) {
  obj3d.position.set(worldX(x), surfaceY(x, y) + yOffset, worldZ(y));
}

// ---- snapshot / delta application ------------------------------------------

function applySnapshot(snap) {
  bornReady = false; // no birth bursts for the initial population
  buildTerrain(snap.terrain);
  // clear entities
  for (const a of agents.values()) scene.remove(a.group);
  agents.clear();
  for (const o of objects.values()) scene.remove(o);
  objects.clear();
  for (const o of snap.objects) upsertObject(o);
  for (const a of snap.agents) upsertAgent(a);
  applyBurning(snap.burning || []);
  updateSky(snap.sky);
  hud.objects.textContent = objects.size;
  hud.tick.textContent = snap.tick;
  bornReady = true;
}

function applyBurning(ids) {
  const next = new Set(ids);
  // newly extinguished → clear emissive
  for (const id of burning) {
    if (!next.has(id)) {
      const m = objects.get(id);
      if (m && m.material.emissive) m.material.emissive.setHex(0x000000);
    }
  }
  // currently burning → emissive orange (intensity flickers in animate)
  for (const id of next) {
    const m = objects.get(id);
    if (m && m.material.emissive) m.material.emissive.setHex(0xff5a1e);
  }
  burning = next;
}

function upsertObject(o) {
  let mesh = objects.get(o.id);
  if (!mesh) {
    mesh = makeObjectModel(o);
    objects.set(o.id, mesh);
    scene.add(mesh);
  }
  placeOnSurface(mesh, o.x, o.y, 0.3);
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
    if (bornReady) spawnBurst(rec.group.position, 0x8ef0a0, 8); // birth pop
  }
  rec.target.set(worldX(a.x), surfaceY(a.x, a.y), worldZ(a.y));
  // yaw from direction (dx,dy): face +Z when dir=(0,1)
  const [dx, dy] = a.dir;
  rec.targetYaw = Math.atan2(dx, dy);
  // energy bar scale + color
  const e = Math.max(0, Math.min(1, a.energy));
  rec.energyBar.scale.x = Math.max(0.02, e);
  rec.energyBar.material.color.setHSL(0.33 * e, 0.7, 0.5);
  rec.group.visible = a.alive !== false;
  rec.data = {
    id: a.id,
    energy: e,
    lineage: a.lineage,
    generation: a.generation,
    x: a.x,
    y: a.y,
  };
  if (selectedId === a.id) updateInspector(rec.data);
}

function applyDelta(d) {
  for (const o of d.objects || []) upsertObject(o);
  for (const id of d.removed_objects || []) {
    const m = objects.get(id);
    if (m) {
      spawnBurst(m.position, 0xffb24a, 5, 1.4); // consumed/decayed puff
      scene.remove(m);
      objects.delete(id);
    }
  }
  for (const a of d.agents || []) upsertAgent(a);
  for (const id of d.removed_agents || []) {
    const r = agents.get(id);
    if (r) {
      spawnBurst(r.group.position, 0x9aa0a6, 12, 2.2); // death puff
      scene.remove(r.group);
      agents.delete(id);
      if (selectedId === id) {
        selectedId = null;
        updateInspector(null);
      }
    }
  }
  applyBurning(d.burning || []);
  updateSignals(d.signals || []);
  updateSky(d.sky);
  hud.tick.textContent = d.tick;
  hud.agents.textContent = agents.size;
  hud.objects.textContent = objects.size;
}

function updateSignals(signals) {
  // rebuild the glow group each delta (sparse; the field decays to zero)
  scene.remove(signalGroup);
  signalGroup = new THREE.Group();
  const geo = new THREE.PlaneGeometry(1, 1);
  for (const [x, y, v] of signals) {
    const base = Math.min(0.7, 0.15 + 0.6 * v);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x66e0ff,
      transparent: true,
      opacity: base,
      depthWrite: false,
    });
    const m = new THREE.Mesh(geo, mat);
    m.userData.base = base;
    m.rotation.x = -Math.PI / 2;
    m.position.set(worldX(x), surfaceY(x, y) + 0.05, worldZ(y));
    signalGroup.add(m);
  }
  scene.add(signalGroup);
}

function updateSky(sky) {
  if (!sky) return;
  const t = sky.time_of_day ?? 0; // 0..1
  const light = sky.light ?? 1;
  // sun arcs across the sky over a day
  const ang = t * Math.PI * 2 - Math.PI / 2;
  sun.position.set(Math.cos(ang) * 80, Math.max(10, Math.sin(ang) * 90), 30);
  sun.intensity = 0.5 + 0.9 * light;
  hemi.intensity = 0.5 + 0.5 * light;
  ambient.intensity = 0.3 + 0.25 * light;
  // background: dusk blue -> day blue by light (night floor kept visible)
  const night = new THREE.Color(0x1b2b4a);
  const day = new THREE.Color(0x8fbcd4);
  const bg = night.clone().lerp(day, light);
  scene.background = bg;
  if (scene.fog) scene.fog.color = bg;
  hud.season.textContent = (sky.season ?? 0).toFixed(2);
  let w = "clear";
  if (sky.raining) w = "rain";
  else if (sky.drought) w = "drought";
  hud.weather.textContent = w;
}

// ---- input: free-fly + follow ---------------------------------------------

const keys = new Set();
window.addEventListener("keydown", (e) => {
  keys.add(e.key.toLowerCase());
  if (e.key.toLowerCase() === "f") cycleFollow();
  if (e.key.toLowerCase() === "r") resetCamera();
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

function resetCamera() {
  followIdx = -1;
  frameCamera();
}

function cycleFollow() {
  followIds = Array.from(agents.keys());
  if (followIds.length === 0) return;
  followIdx = (followIdx + 1) % (followIds.length + 1);
  if (followIdx === followIds.length) followIdx = -1; // wrap to free cam
  if (followIdx >= 0) selectedId = followIds[followIdx];
}

// click-to-inspect (raycast against agent meshes)
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let dragMoved = false;
canvas.addEventListener("pointerdown", () => (dragMoved = false));
canvas.addEventListener("pointermove", () => (dragMoved = true));
canvas.addEventListener("pointerup", (e) => {
  if (dragMoved) return; // it was an orbit drag, not a click
  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const meshes = [];
  for (const rec of agents.values()) rec.group.traverse((o) => o.isMesh && meshes.push(o));
  const hits = raycaster.intersectObjects(meshes, false);
  if (hits.length > 0) {
    let g = hits[0].object;
    while (g && !g.userData.agentGroup) g = g.parent;
    const grp = g && g.userData.agentGroup;
    if (grp) {
      // find the rec for this group
      for (const rec of agents.values()) {
        if (rec.group === grp) {
          selectedId = rec.id;
          updateInspector(rec.data);
          break;
        }
      }
      return;
    }
  }
  selectedId = null;
  updateInspector(null);
});

function applyFreeFly(dt) {
  const speed = 20 * dt;
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

// ---- animation loop --------------------------------------------------------

const clock = new THREE.Clock();
let elapsed = 0;
const _chaseTmp = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, clock.getDelta());
  elapsed += dt;

  // tween agents toward targets (smooth motion over elevation), smooth yaw,
  // and a gentle idle bob on the inner group
  for (const rec of agents.values()) {
    rec.group.position.lerp(rec.target, 0.18);
    // shortest-arc yaw lerp
    let dyaw = rec.targetYaw - rec.yaw;
    dyaw = Math.atan2(Math.sin(dyaw), Math.cos(dyaw));
    rec.yaw += dyaw * 0.25;
    rec.group.rotation.y = rec.yaw;
    if (rec.bob) {
      rec.bob.position.y = Math.sin(elapsed * 4 + rec.group.userData.phase) * 0.05;
    }
  }

  updateParticles(dt);

  // fire flicker + occasional flame particles on burning objects
  if (burning.size) {
    const flick = 0.6 + 0.4 * Math.abs(Math.sin(elapsed * 12));
    for (const id of burning) {
      const m = objects.get(id);
      if (!m || !m.material.emissive) continue;
      m.material.emissiveIntensity = flick;
      if (Math.random() < 0.15) spawnBurst(m.position, 0xff7a1e, 1, 1.6);
    }
  }

  // signal glow pulse
  const pulse = 0.75 + 0.25 * Math.sin(elapsed * 3);
  signalGroup.children.forEach((m) => (m.material.opacity = m.userData.base * pulse));

  if (followIdx >= 0 && followIds[followIdx] !== undefined) {
    const rec = agents.get(followIds[followIdx]);
    if (rec) {
      controls.target.lerp(rec.group.position, 0.2);
      // chase: keep the camera trailing the agent at a fixed offset
      _chaseTmp.copy(rec.group.position).add(new THREE.Vector3(6, 7, 6));
      camera.position.lerp(_chaseTmp, 0.06);
    }
  } else {
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

// ---- SSE connection --------------------------------------------------------

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
