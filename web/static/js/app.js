/*
 * Main application entry point for the Emergent World-Model Sandbox web UI.
 *
 * Wires together Three.js rendering (world3d.js), the DOM panels (ui.js) and
 * the JSON API (net.js):
 *   - sets up scene, camera, lights and OrbitControls
 *   - polls server state and interpolates agents for smooth motion
 *   - handles mouse picking (hover tooltip + click inspector) and the spawn tool
 *   - binds keyboard shortcuts (Space / S / R / G / T)
 *
 * Author: Karan Vasa
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Net } from "./net.js";
import { World3D } from "./world3d.js";
import { UI } from "./ui.js";
import { iconForType, AGENT_ICON } from "./icons.js";

let scene, camera, renderer, controls, world3d, ui;
let meta = null;
let latestState = null;
let spawnType = null;

// Frame timing + FPS smoothing.
let lastFrame = performance.now();
let fps = 0;

// Mouse picking.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let groundPlane = null;
let hoverInfo = null;
let mouseClientX = 0, mouseClientY = 0;

// State polling cadence (Hz). Interpolation smooths motion between polls.
const POLL_HZ = 20;

async function init() {
  meta = await Net.meta();

  // ---- Scene ----
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x12121a);
  scene.fog = new THREE.Fog(0x12121a, 60, 220);

  const W = meta.world.width, H = meta.world.height;

  // ---- Camera ----
  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, Math.max(W, H) * 0.75, Math.max(W, H) * 0.75);

  // ---- Renderer ----
  renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById("viewport"),
    antialias: true,
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // ---- Controls ----
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.target.set(0, 0, 0);

  // ---- Lights ----
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xfff2d8, 1.1);
  sun.position.set(W * 0.4, Math.max(W, H), H * 0.3);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const d = Math.max(W, H) * 0.7;
  sun.shadow.camera.left = -d; sun.shadow.camera.right = d;
  sun.shadow.camera.top = d; sun.shadow.camera.bottom = -d;
  sun.shadow.camera.far = Math.max(W, H) * 3;
  scene.add(sun);
  const fill = new THREE.HemisphereLight(0x88aaff, 0x332211, 0.4);
  scene.add(fill);

  // ---- World ----
  world3d = new World3D(scene, meta);
  const terrain = await Net.terrain();
  world3d.buildTerrain(terrain);

  // Invisible ground plane for tile picking.
  const planeGeo = new THREE.PlaneGeometry(W, H);
  const planeMat = new THREE.MeshBasicMaterial({ visible: false });
  groundPlane = new THREE.Mesh(planeGeo, planeMat);
  groundPlane.rotation.x = -Math.PI / 2;
  scene.add(groundPlane);

  // ---- UI ----
  ui = new UI(meta, {
    onControl: handleControl,
    onSpawnSelect: (t) => { spawnType = t; },
    onToggle: handleToggle,
  });

  // ---- Events ----
  window.addEventListener("resize", onResize);
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("click", onClick);
  window.addEventListener("keydown", onKey);

  document.getElementById("loading").classList.add("hidden");

  // Kick off loops.
  pollLoop();
  animate();
}

// ----------------------------------------------------------------------
// Control + toggle handlers
// ----------------------------------------------------------------------

async function handleControl(payload) {
  const res = await Net.control(payload);
  if (payload.cmd === "reset") {
    // Refresh meta + terrain after a world rebuild.
    meta = await Net.meta();
    const terrain = await Net.terrain();
    world3d.buildTerrain(terrain);
  }
  return res;
}

function handleToggle(name, on) {
  if (name === "grid") world3d.setGrid(on);
  else if (name === "trails") world3d.setTrails(on);
  else if (name === "height") world3d.setHeight(on);
}

// ----------------------------------------------------------------------
// Networking loop (poll state, refresh terrain on version change)
// ----------------------------------------------------------------------

async function pollLoop() {
  while (true) {
    try {
      const state = await Net.state();
      latestState = state;
      world3d.syncObjects(state.objects);
      world3d.syncAgents(state.agents);
      world3d.updateTrails();
      ui.updateStats(state, fps);
      ui.pushGraph(state);

      // Terrain changed (sand spread / reclaim / reset)?
      if (state.terrain_version !== world3d.terrainVersion) {
        const terrain = await Net.terrain();
        world3d.buildTerrain(terrain);
      }
    } catch (err) {
      // Server momentarily busy — retry on next tick.
    }
    await sleep(1000 / POLL_HZ);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ----------------------------------------------------------------------
// Render loop
// ----------------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  fps = Math.round(0.9 * fps + 0.1 * (1 / Math.max(1e-3, dt)));

  controls.update();
  world3d.interpolate(dt);
  updateHover();
  renderer.render(scene, camera);
}

// ----------------------------------------------------------------------
// Picking: hover tooltip + click inspector / spawn
// ----------------------------------------------------------------------

function setPointerFromEvent(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

function pickEntity() {
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(world3d.getPickables(), true);
  for (const h of hits) {
    let obj = h.object;
    while (obj && !obj.userData?.kind) obj = obj.parent;
    if (obj?.userData?.kind) return obj.userData;
  }
  return null;
}

function pickTile() {
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(groundPlane);
  if (!hit.length) return null;
  const p = hit[0].point;
  const x = Math.floor(p.x + world3d.offX);
  const y = Math.floor(p.z + world3d.offZ);
  if (x < 0 || y < 0 || x >= meta.world.width || y >= meta.world.height) return null;
  return { x, y };
}

function onPointerMove(e) {
  setPointerFromEvent(e);
  mouseClientX = e.clientX;
  mouseClientY = e.clientY;
}

function updateHover() {
  const entity = pickEntity();
  if (entity) {
    hoverInfo = entity;
    if (entity.kind === "agent") {
      const a = (latestState?.agents || []).find((x) => x.id === entity.id);
      if (a) {
        ui.showTooltip(
          `<div class="t-title"><img src="${AGENT_ICON}" width="16" height="16" style="vertical-align:middle">
             Agent #${a.id}</div>
           <div class="muted">Energy ${a.e}/${a.me} · gen ${a.gen}</div>`,
          mouseClientX, mouseClientY
        );
      }
    } else {
      const def = meta.object_types[entity.typeId];
      const url = iconForType(entity.typeId, entity.cat).url;
      ui.showTooltip(
        `<div class="t-title"><img src="${url}" width="16" height="16" style="vertical-align:middle">
           ${def ? def.display_name : entity.typeId}</div>
         <div class="muted">${entity.cat}</div>`,
        mouseClientX, mouseClientY
      );
    }
    renderer.domElement.style.cursor = "pointer";
    return;
  }

  // Tile hover (only show when spawn mode active to reduce noise).
  if (spawnType) {
    const tile = pickTile();
    if (tile) {
      ui.showTooltip(
        `<div class="t-title">Spawn here</div><div class="muted">(${tile.x}, ${tile.y})</div>`,
        mouseClientX, mouseClientY
      );
      renderer.domElement.style.cursor = "crosshair";
      hoverInfo = null;
      return;
    }
  }

  hoverInfo = null;
  ui.hideTooltip();
  renderer.domElement.style.cursor = "grab";
}

async function onClick(e) {
  setPointerFromEvent(e);

  // Spawn mode takes priority.
  if (spawnType) {
    const tile = pickTile();
    if (tile) {
      await Net.control({ cmd: "spawn", type_id: spawnType, x: tile.x, y: tile.y });
    }
    return;
  }

  const entity = pickEntity();
  if (entity) {
    if (entity.kind === "agent") {
      const d = await Net.inspectAgent(entity.id);
      if (d && !d.error) ui.showAgent(d);
    } else {
      const d = await Net.inspectObject(entity.id);
      if (d && !d.error) ui.showObject(d);
    }
    return;
  }

  const tile = pickTile();
  if (tile) {
    const d = await Net.inspectTile(tile.x, tile.y);
    if (d && !d.error) ui.showTile(d);
  }
}

// ----------------------------------------------------------------------
// Keyboard shortcuts
// ----------------------------------------------------------------------

function onKey(e) {
  if (e.target.tagName === "INPUT") return;
  switch (e.key) {
    case " ":
      e.preventDefault();
      handleControl({ cmd: "toggle" });
      break;
    case "s": case "S":
      handleControl({ cmd: "step", ticks: 1 });
      break;
    case "r": case "R":
      if (meta.reset_available && confirm("Reset the world?")) handleControl({ cmd: "reset" });
      break;
    case "g": case "G": {
      const cb = document.getElementById("tg-grid");
      cb.checked = !cb.checked;
      handleToggle("grid", cb.checked);
      break;
    }
    case "t": case "T": {
      const cb = document.getElementById("tg-trails");
      cb.checked = !cb.checked;
      handleToggle("trails", cb.checked);
      break;
    }
    case "Escape":
      ui.closeInspector();
      ui.clearSpawnSelection();
      spawnType = null;
      break;
  }
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

init().catch((err) => {
  console.error(err);
  document.querySelector(".loading-text").textContent =
    "Failed to connect to simulation. Is the server running?";
});
