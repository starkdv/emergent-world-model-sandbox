/*
 * 3D world renderer (Three.js) for the Emergent World-Model Sandbox.
 *
 * Objects and agents are rendered as TEXTURED IMAGE SPRITES loaded from real
 * SVG art assets (web/static/assets/) — not primitive geometry. Every
 * registered object type maps to a bespoke icon (or a category fallback,
 * tinted by its registry colour for custom YAML types), so the world reads as
 * a real simulation: berries look like berries, plants like plants, agents
 * like little creatures.
 *
 * Responsibilities:
 *   - terrain : an InstancedMesh of tile boxes (the ground) coloured by type
 *   - objects : camera-facing image sprites keyed by object id
 *   - agents  : image-sprite creatures (energy-tinted) + a flat facing arrow
 *   - trails  : optional fading dots tracing recent agent positions
 *
 * Author: Karan Vasa
 */

import * as THREE from "three";
import { iconForType, VARIANT, AGENT_ICON, ARROW_ICON } from "./icons.js";

// Terrain visual profile per type.
const TERRAIN_HEIGHT = { soil: 0.4, rock: 1.4, water: 0.18, sand: 0.5 };
const TERRAIN_Y = { soil: 0.0, rock: 0.0, water: -0.18, sand: 0.0 };

const _color = new THREE.Color();
const _m4 = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _scale = new THREE.Vector3();

/** Caches textures by URL so each SVG asset is loaded only once. */
class TextureCache {
  constructor() {
    this.loader = new THREE.TextureLoader();
    this.cache = new Map();
  }
  get(url) {
    let tex = this.cache.get(url);
    if (!tex) {
      tex = this.loader.load(url);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.LinearFilter;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.generateMipmaps = true;
      tex.anisotropy = 4;
      this.cache.set(url, tex);
    }
    return tex;
  }
}

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
    this.terrainCodeName = {};
    for (const [name, code] of Object.entries(meta.terrain_codes || {})) {
      this.terrainCodeName[code] = name;
    }

    this.tex = new TextureCache();

    // Live registries.
    this.objectMeshes = new Map(); // id → THREE.Sprite
    this.agentMeshes = new Map(); // id → { group, sprite, arrow, target, targetRot }
    this.pickables = []; // agent groups raycastable for selection

    // Shared geometry for the flat facing arrow under each agent.
    this._arrowGeo = new THREE.PlaneGeometry(0.7, 0.7);

    this.terrainMesh = null;
    this.terrainVersion = -1;
    this.tileTopY = null; // per-tile ground-surface Y (row-major)
    this.gridHelper = null;
    this.trailsEnabled = false;
    this.heightEnabled = true;
    this._trailGroup = new THREE.Group();
    this.scene.add(this._trailGroup);
    this._agentHistory = new Map();
  }

  /** Convert world (x, y) to scene coordinates (X, Z), centred on origin. */
  worldToScene(x, y) {
    return [x - this.offX + 0.5, y - this.offZ + 0.5];
  }

  /** Ground-surface Y for a world tile (top of its terrain box). */
  topYAt(x, y) {
    if (!this.tileTopY) return 0;
    const i = y * this.width + x;
    return this.tileTopY[i] ?? 0;
  }

  // ------------------------------------------------------------------
  // Terrain (the ground — kept as instanced boxes)
  // ------------------------------------------------------------------

  buildTerrain(terrain) {
    if (this.terrainMesh) {
      this.scene.remove(this.terrainMesh);
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
    mesh.receiveShadow = true;

    this.tileTopY = new Float32Array(count);
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
        this.tileTopY[i] = yBase + h - 0.5; // top surface

        const rgb = this.terrainPalette[name] || [100, 100, 100];
        _color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
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

  setHeight(on) {
    this.heightEnabled = on;
    this.terrainVersion = -1; // force rebuild on next sync
  }

  // ------------------------------------------------------------------
  // Objects — image sprites
  // ------------------------------------------------------------------

  _registryColor(typeId) {
    const def = this.objectTypes[typeId];
    const rgb = def ? def.color : [200, 200, 200];
    return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
  }

  /** Pick the icon URL for an object record, honouring growth-state variants. */
  _objectIcon(rec) {
    if (rec.cat === "seed" && rec.planted) {
      return { url: VARIANT.seed_planted, specific: true };
    }
    if (rec.cat === "plant" && rec.mature) {
      return { url: VARIANT.plant_mature, specific: true };
    }
    return iconForType(rec.t, rec.cat);
  }

  _makeSprite(url, tintColor) {
    const mat = new THREE.SpriteMaterial({
      map: this.tex.get(url),
      transparent: true,
      alphaTest: 0.25,
      depthWrite: true,
    });
    if (tintColor) mat.color.copy(tintColor);
    const sprite = new THREE.Sprite(mat);
    sprite.center.set(0.5, 0.0); // anchor bottom to the ground
    return sprite;
  }

  syncObjects(records) {
    const seen = new Set();
    for (const rec of records) {
      if (rec.cat === "terrain") continue; // sand shown by the ground tile
      seen.add(rec.id);

      const icon = this._objectIcon(rec);
      const tint = icon.specific ? null : this._registryColor(rec.t);

      let sprite = this.objectMeshes.get(rec.id);
      if (!sprite) {
        sprite = this._makeSprite(icon.url, tint);
        sprite.userData = { kind: "object", id: rec.id, typeId: rec.t, cat: rec.cat };
        sprite.userData.curUrl = icon.url;
        this.objectMeshes.set(rec.id, sprite);
        this.scene.add(sprite);
      } else if (sprite.userData.curUrl !== icon.url) {
        // Growth-state changed (e.g. seed germinated, plant matured).
        sprite.material.map = this.tex.get(icon.url);
        sprite.material.color.copy(tint || new THREE.Color(0xffffff));
        sprite.material.needsUpdate = true;
        sprite.userData.curUrl = icon.url;
      }

      // Size: plants grow with maturity; food pulses subtly with freshness.
      let s = 0.95;
      if (rec.cat === "plant") s = 0.7 + 0.7 * Math.min(1, rec.growth ?? 1);
      else if (rec.cat === "seed") s = 0.7;
      else if (rec.cat === "food") s = 0.8 + 0.15 * (rec.fresh ?? 1);
      sprite.scale.set(s, s, 1);

      const [sx, sz] = this.worldToScene(rec.x, rec.y);
      sprite.position.set(sx, this.topYAt(rec.x, rec.y) + 0.02, sz);
    }

    for (const [id, sprite] of this.objectMeshes) {
      if (!seen.has(id)) {
        this.scene.remove(sprite);
        sprite.material.dispose();
        this.objectMeshes.delete(id);
      }
    }
  }

  // ------------------------------------------------------------------
  // Agents — creature sprite (energy-tinted) + flat facing arrow
  // ------------------------------------------------------------------

  _energyColor(ratio) {
    if (ratio > 0.7) return new THREE.Color(0.31, 0.86, 0.31);
    if (ratio > 0.4) return new THREE.Color(1.0, 0.78, 0.0);
    if (ratio > 0.2) return new THREE.Color(1.0, 0.39, 0.0);
    return new THREE.Color(1.0, 0.2, 0.2);
  }

  // Heading so the flat arrow (image points +Y, laid flat → -Z) aims along
  // the agent's facing vector (dx, dy) in the scene XZ plane.
  _dirToRot(dx, dy) {
    return Math.atan2(dx, dy) + Math.PI;
  }

  syncAgents(agents) {
    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      const [sx, sz] = this.worldToScene(a.x, a.y);
      const topY = this.topYAt(a.x, a.y);
      const ratio = a.e / Math.max(1e-6, a.me);

      let entry = this.agentMeshes.get(a.id);
      if (!entry) {
        const group = new THREE.Group();

        const sprite = this._makeSprite(AGENT_ICON, this._energyColor(ratio));
        sprite.scale.set(1.1, 1.1, 1);
        group.add(sprite);

        const arrow = new THREE.Mesh(
          this._arrowGeo,
          new THREE.MeshBasicMaterial({
            map: this.tex.get(ARROW_ICON),
            transparent: true,
            alphaTest: 0.3,
            depthWrite: false,
          })
        );
        arrow.rotation.x = -Math.PI / 2; // lay flat on the ground
        arrow.position.y = 0.03;
        arrow.renderOrder = 1;
        group.add(arrow);

        group.position.set(sx, topY + 0.02, sz);
        group.userData = { kind: "agent", id: a.id };
        this.scene.add(group);
        this.pickables.push(group);

        entry = {
          group,
          sprite,
          arrow,
          target: new THREE.Vector3(sx, topY + 0.02, sz),
          targetRot: this._dirToRot(a.dx, a.dy),
        };
        this.agentMeshes.set(a.id, entry);
      }

      entry.target.set(sx, topY + 0.02, sz);
      entry.targetRot = this._dirToRot(a.dx, a.dy);
      entry.sprite.material.color.copy(this._energyColor(ratio));
    }

    for (const [id, entry] of this.agentMeshes) {
      if (!seen.has(id)) {
        this.scene.remove(entry.group);
        entry.sprite.material.dispose();
        entry.arrow.material.dispose();
        this.agentMeshes.delete(id);
        const idx = this.pickables.indexOf(entry.group);
        if (idx >= 0) this.pickables.splice(idx, 1);
        this._agentHistory.delete(id);
      }
    }
  }

  /** Per-frame interpolation toward target positions for smooth movement. */
  interpolate(dt) {
    const k = Math.min(1, dt * 8.0);
    for (const entry of this.agentMeshes.values()) {
      entry.group.position.lerp(entry.target, k);
      // Rotate only the flat arrow toward the heading (sprite is a billboard).
      let cur = entry.arrow.rotation.z;
      let diff = entry.targetRot - cur;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      entry.arrow.rotation.z = cur + diff * k;
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
  // Picking
  // ------------------------------------------------------------------

  /** Return the list of pickable objects (agent groups + object sprites). */
  getPickables() {
    return [...this.pickables, ...this.objectMeshes.values()];
  }
}
