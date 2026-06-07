/*
 * 3D world renderer (Three.js) for the Emergent World-Model Sandbox.
 *
 * Responsible for building and updating the scene:
 *   - terrain  : an InstancedMesh of tile boxes coloured + raised by type
 *   - objects  : per-object meshes keyed by id (berry, seed, plant, …)
 *   - agents   : oriented cones, colour-coded by energy, smoothly interpolated
 *   - trails   : optional fading dots tracing recent agent positions
 *
 * Every registered object type is rendered with a distinct mesh derived from
 * its category and registry colour, so custom YAML objects appear correctly
 * without any client changes.
 *
 * Author: Karan Vasa
 */

import * as THREE from "three";

// Terrain visual profile per type: [heightScale, yColorTint].
const TERRAIN_HEIGHT = { soil: 0.4, rock: 1.4, water: 0.18, sand: 0.5 };
const TERRAIN_Y = { soil: 0.0, rock: 0.0, water: -0.18, sand: 0.0 };

const _color = new THREE.Color();
const _m4 = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _scale = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

export class World3D {
  /**
   * @param {THREE.Scene} scene
   * @param {Object} meta - payload from /api/meta
   */
  constructor(scene, meta) {
    this.scene = scene;
    this.meta = meta;
    this.width = meta.world.width;
    this.height = meta.world.height;
    this.offX = this.width / 2;
    this.offZ = this.height / 2;

    this.objectTypes = meta.object_types || {};
    this.terrainPalette = meta.terrain_palette || {};
    this.terrainCodeName = {}; // code → name
    for (const [name, code] of Object.entries(meta.terrain_codes || {})) {
      this.terrainCodeName[code] = name;
    }

    // Live mesh registries.
    this.objectMeshes = new Map(); // id → THREE.Mesh
    this.agentMeshes = new Map(); // id → { mesh, target:Vec3, targetRot:number }
    this.pickables = []; // meshes raycastable for selection

    // Shared geometries (created once, reused per mesh).
    this._geo = {
      food: new THREE.SphereGeometry(0.32, 14, 12),
      seed: new THREE.OctahedronGeometry(0.28),
      plant: new THREE.ConeGeometry(0.32, 0.9, 8),
      fertilizer: new THREE.BoxGeometry(0.45, 0.4, 0.45),
      tool: new THREE.BoxGeometry(0.4, 0.5, 0.4),
      generic: new THREE.IcosahedronGeometry(0.32, 0),
      agent: new THREE.ConeGeometry(0.36, 0.95, 5),
    };

    this.terrainMesh = null;
    this.terrainVersion = -1;
    this.gridHelper = null;
    this.trailsEnabled = false;
    this.heightEnabled = true;
    this._trailGroup = new THREE.Group();
    this.scene.add(this._trailGroup);
    this._agentHistory = new Map(); // id → [Vec3,…]
  }

  /** Convert world (x, y) to scene coordinates (X, Z), centred on origin. */
  worldToScene(x, y) {
    return [x - this.offX + 0.5, y - this.offZ + 0.5];
  }

  // ------------------------------------------------------------------
  // Terrain
  // ------------------------------------------------------------------

  /**
   * Build (or rebuild) the terrain InstancedMesh from a terrain payload.
   * @param {Object} terrain - payload from /api/terrain
   */
  buildTerrain(terrain) {
    if (this.terrainMesh) {
      this.scene.remove(this.terrainMesh);
      this.terrainMesh.dispose?.();
      this.terrainMesh.geometry.dispose();
      this.terrainMesh.material.dispose();
    }

    const count = this.width * this.height;
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.95,
      metalness: 0.0,
      flatShading: true,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, count);
    mesh.castShadow = false;
    mesh.receiveShadow = true;

    const types = terrain.types;
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const i = y * this.width + x;
        const name = this.terrainCodeName[types[i]] || "soil";
        const h = this.heightEnabled ? TERRAIN_HEIGHT[name] ?? 0.4 : 0.3;
        const yBase = this.heightEnabled ? TERRAIN_Y[name] ?? 0.0 : 0.0;
        const [sx, sz] = this.worldToScene(x, y);
        _scale.set(0.98, h, 0.98);
        _q.identity();
        _v.set(sx, yBase + h / 2 - 0.5, sz);
        _m4.compose(_v, _q, _scale);
        mesh.setMatrixAt(i, _m4);

        const rgb = this.terrainPalette[name] || [100, 100, 100];
        _color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
        // Subtle per-tile variation for visual richness.
        const jitter = 0.9 + ((x * 13 + y * 7) % 7) * 0.03;
        _color.multiplyScalar(jitter);
        mesh.setColorAt(i, _color);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    this.terrainMesh = mesh;
    this.terrainVersion = terrain.version;
    this.scene.add(mesh);
  }

  /** Toggle the wireframe grid overlay. */
  setGrid(on) {
    if (on && !this.gridHelper) {
      const size = Math.max(this.width, this.height);
      this.gridHelper = new THREE.GridHelper(size, size, 0x444455, 0x2a2a38);
      this.gridHelper.position.y = 0.02;
      this.scene.add(this.gridHelper);
    } else if (!on && this.gridHelper) {
      this.scene.remove(this.gridHelper);
      this.gridHelper.geometry.dispose();
      this.gridHelper.material.dispose();
      this.gridHelper = null;
    }
  }

  /** Toggle 3D terrain heights (forces a terrain rebuild on next sync). */
  setHeight(on) {
    this.heightEnabled = on;
    this.terrainVersion = -1; // force rebuild
  }

  // ------------------------------------------------------------------
  // Objects
  // ------------------------------------------------------------------

  _objColor(typeId) {
    const def = this.objectTypes[typeId];
    const rgb = def ? def.color : [200, 200, 200];
    return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
  }

  _makeObjectMesh(rec) {
    const cat = rec.cat;
    let geo = this._geo.generic;
    if (cat === "food") geo = this._geo.food;
    else if (cat === "seed") geo = this._geo.seed;
    else if (cat === "plant") geo = this._geo.plant;
    else if (cat === "fertilizer") geo = this._geo.fertilizer;
    else if (cat === "tool") geo = this._geo.tool;

    const col = this._objColor(rec.t);
    const mat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.6,
      metalness: 0.1,
      emissive: col.clone().multiplyScalar(0.0),
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.userData = { kind: "object", id: rec.id, typeId: rec.t, cat };
    mesh.castShadow = true;
    return mesh;
  }

  /**
   * Reconcile object meshes against the latest state snapshot.
   * @param {Array} records - state.objects
   */
  syncObjects(records) {
    const seen = new Set();
    for (const rec of records) {
      // Terrain-layer objects (sand) are represented by the terrain colour.
      if (rec.cat === "terrain") continue;
      seen.add(rec.id);

      let mesh = this.objectMeshes.get(rec.id);
      if (!mesh) {
        mesh = this._makeObjectMesh(rec);
        this.objectMeshes.set(rec.id, mesh);
        this.scene.add(mesh);
      }

      const [sx, sz] = this.worldToScene(rec.x, rec.y);
      let yPos = 0.4;
      const cat = rec.cat;

      if (cat === "plant") {
        const g = rec.growth ?? 1;
        const s = 0.5 + 0.7 * Math.min(1, g);
        mesh.scale.set(s, s, s);
        yPos = 0.4 + (0.45 * s);
        mesh.material.emissive.copy(this._objColor(rec.t)).multiplyScalar(rec.mature ? 0.35 : 0.05);
      } else if (cat === "seed") {
        const g = rec.growth ?? 0;
        // Natural tan → sprouting green; agent-planted seeds glow gold.
        const base = this._objColor(rec.t);
        if (rec.planted) {
          mesh.material.color.setRGB(1.0, 0.78, 0.0).lerp(new THREE.Color(0.47, 0.78, 0.31), g);
          mesh.material.emissive.setRGB(1.0, 0.85, 0.2).multiplyScalar(0.5);
        } else {
          mesh.material.color.copy(base).lerp(new THREE.Color(0.47, 0.78, 0.31), g);
          mesh.material.emissive.setScalar(0.0);
        }
        yPos = 0.55;
        mesh.rotation.y += 0.02;
      } else if (cat === "food") {
        const fresh = rec.fresh ?? 1;
        mesh.material.emissive.copy(this._objColor(rec.t)).multiplyScalar(0.15 * fresh);
        yPos = 0.55;
      } else if (cat === "fertilizer") {
        yPos = 0.5;
      }

      mesh.position.set(sx, yPos, sz);
    }

    // Remove meshes for objects no longer present.
    for (const [id, mesh] of this.objectMeshes) {
      if (!seen.has(id)) {
        this.scene.remove(mesh);
        mesh.material.dispose();
        this.objectMeshes.delete(id);
      }
    }
  }

  // ------------------------------------------------------------------
  // Agents
  // ------------------------------------------------------------------

  _energyColor(ratio) {
    if (ratio > 0.7) return new THREE.Color(0.31, 0.86, 0.31);
    if (ratio > 0.4) return new THREE.Color(1.0, 0.78, 0.0);
    if (ratio > 0.2) return new THREE.Color(1.0, 0.39, 0.0);
    return new THREE.Color(1.0, 0.2, 0.2);
  }

  _dirToRot(dx, dy) {
    // Cone points +Y by default; we tilt it forward and rotate about Y so the
    // tip indicates the facing direction on the ground plane (X, Z=y).
    return Math.atan2(dx, dy);
  }

  /**
   * Reconcile agent meshes against the latest state snapshot.
   * @param {Array} agents - state.agents
   */
  syncAgents(agents) {
    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      let entry = this.agentMeshes.get(a.id);
      const [sx, sz] = this.worldToScene(a.x, a.y);

      if (!entry) {
        const mat = new THREE.MeshStandardMaterial({
          color: 0x50dc64,
          roughness: 0.4,
          metalness: 0.2,
          emissive: new THREE.Color(0x103010),
        });
        const mesh = new THREE.Mesh(this._geo.agent, mat);
        // Lay the cone on its side so the tip points along +Z; the parent group's
        // Y-rotation then aims that tip in the agent's facing direction.
        mesh.rotation.x = Math.PI / 2;
        mesh.castShadow = true;
        const group = new THREE.Group();
        group.add(mesh);
        group.position.set(sx, 0.7, sz);
        group.userData = { kind: "agent", id: a.id };
        this.scene.add(group);
        this.pickables.push(group);
        entry = {
          group,
          mesh,
          target: new THREE.Vector3(sx, 0.7, sz),
          targetRot: this._dirToRot(a.dx, a.dy),
        };
        this.agentMeshes.set(a.id, entry);
      }

      entry.target.set(sx, 0.7, sz);
      entry.targetRot = this._dirToRot(a.dx, a.dy);
      const ratio = a.e / Math.max(1e-6, a.me);
      entry.mesh.material.color.copy(this._energyColor(ratio));
      entry.mesh.material.emissive.copy(this._energyColor(ratio)).multiplyScalar(0.2);
    }

    // Remove meshes for agents that died.
    for (const [id, entry] of this.agentMeshes) {
      if (!seen.has(id)) {
        this.scene.remove(entry.group);
        entry.mesh.material.dispose();
        this.agentMeshes.delete(id);
        const idx = this.pickables.indexOf(entry.group);
        if (idx >= 0) this.pickables.splice(idx, 1);
        this._agentHistory.delete(id);
      }
    }
  }

  /**
   * Per-frame interpolation toward target positions for smooth movement.
   * @param {number} dt - frame delta seconds
   */
  interpolate(dt) {
    const k = Math.min(1, dt * 8.0);
    for (const entry of this.agentMeshes.values()) {
      entry.group.position.lerp(entry.target, k);
      // Shortest-arc rotation toward target heading.
      let cur = entry.group.rotation.y;
      let diff = entry.targetRot - cur;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      entry.group.rotation.y = cur + diff * k;
    }
  }

  // ------------------------------------------------------------------
  // Trails
  // ------------------------------------------------------------------

  setTrails(on) {
    this.trailsEnabled = on;
    if (!on) {
      this._trailGroup.clear();
      this._agentHistory.clear();
    }
  }

  /** Record current agent positions and draw fading trail dots. */
  updateTrails() {
    if (!this.trailsEnabled) return;
    this._trailGroup.clear();
    const dotGeo = new THREE.SphereGeometry(0.12, 6, 6);
    for (const [id, entry] of this.agentMeshes) {
      let hist = this._agentHistory.get(id);
      if (!hist) {
        hist = [];
        this._agentHistory.set(id, hist);
      }
      hist.push(entry.target.clone());
      if (hist.length > 20) hist.shift();
      for (let i = 0; i < hist.length; i++) {
        const alpha = i / hist.length;
        const mat = new THREE.MeshBasicMaterial({
          color: 0x64c8ff,
          transparent: true,
          opacity: 0.1 + alpha * 0.4,
        });
        const dot = new THREE.Mesh(dotGeo, mat);
        dot.position.copy(hist[i]).setY(0.3);
        this._trailGroup.add(dot);
      }
    }
  }

  // ------------------------------------------------------------------
  // Picking + highlight
  // ------------------------------------------------------------------

  /** Return the list of pickable meshes (agents + objects). */
  getPickables() {
    return [...this.pickables, ...this.objectMeshes.values()];
  }
}
