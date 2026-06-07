/*
 * 3D world renderer (Three.js) for the Emergent World-Model Sandbox.
 *
 * Real game-development asset pipeline:
 *   - Agents and objects can be rendered as real glTF (.glb) MODELS loaded via
 *     GLTFLoader, driven by web/static/assets/manifest.json. Agents default to
 *     three.js's CC0 "RobotExpressive" model (animated) loaded from the same CDN
 *     as the three.js runtime.
 *   - Anything without a model (or whose model fails to load) falls back
 *     automatically to the shipped SVG sprite icons — so the world always
 *     renders, and dropping CC0 model packs into assets/models/ upgrades the
 *     visuals with no code changes.
 *
 * Terrain is the coloured instanced-box ground.
 *
 * Author: Karan Vasa
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { clone as cloneSkinned } from "three/addons/utils/SkeletonUtils.js";
import { iconForType, VARIANT, AGENT_ICON, ARROW_ICON } from "./icons.js";

const TERRAIN_HEIGHT = { soil: 0.4, rock: 1.4, water: 0.18, sand: 0.5 };
const TERRAIN_Y = { soil: 0.0, rock: 0.0, water: -0.18, sand: 0.0 };
const ASSET_BASE = "/static/assets/";

const _color = new THREE.Color();
const _m4 = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _scale = new THREE.Vector3();

/** Resolve a manifest asset path: absolute URLs as-is, else under /static/assets/. */
function resolveUrl(p) {
  if (!p) return p;
  return /^https?:\/\//i.test(p) ? p : ASSET_BASE + p.replace(/^\/+/, "");
}

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
    this.gltfLoader = new GLTFLoader();

    // Model prototypes (loaded from manifest).
    this.agentProto = null; // { scene, animations, cfg }
    this.objProtoByType = new Map(); // type_id → { scene, cfg }
    this.objProtoByCat = new Map(); // category → { scene, cfg }

    this.objectMeshes = new Map(); // id → Object3D (sprite or model root)
    this.agentMeshes = new Map(); // id → entry
    this.pickables = [];

    this._arrowGeo = new THREE.PlaneGeometry(0.7, 0.7);
    this._discGeo = new THREE.CircleGeometry(0.5, 24);

    this.terrainMesh = null;
    this.terrainVersion = -1;
    this.tileTopY = null;
    this.gridHelper = null;
    this.trailsEnabled = false;
    this.heightEnabled = true;
    this._trailGroup = new THREE.Group();
    this.scene.add(this._trailGroup);
    this._agentHistory = new Map();
  }

  worldToScene(x, y) {
    return [x - this.offX + 0.5, y - this.offZ + 0.5];
  }

  topYAt(x, y) {
    if (!this.tileTopY) return 0;
    return this.tileTopY[y * this.width + x] ?? 0;
  }

  // ------------------------------------------------------------------
  // Asset loading (manifest + glTF models). Always resolves; failures
  // simply leave the affected entity on its sprite fallback.
  // ------------------------------------------------------------------

  async loadAssets() {
    let manifest = null;
    try {
      const res = await fetch("/static/assets/manifest.json", { cache: "no-store" });
      manifest = await res.json();
    } catch (e) {
      console.warn("No asset manifest; using sprite icons only.", e);
      return;
    }

    // Agent model.
    const ag = manifest.agent;
    if (ag && ag.model) {
      try {
        const gltf = await this.gltfLoader.loadAsync(resolveUrl(ag.model));
        this.agentProto = { scene: gltf.scene, animations: gltf.animations, cfg: ag };
        console.info("Loaded agent model:", ag.model);
      } catch (e) {
        console.warn("Agent model failed to load; using sprite agents.", e);
      }
    }

    // Object models (by exact type, then by category).
    const objs = manifest.objects || {};
    await this._loadObjMap(objs.by_type, this.objProtoByType);
    await this._loadObjMap(objs.by_category, this.objProtoByCat);
  }

  async _loadObjMap(map, store) {
    if (!map) return;
    for (const [key, cfg] of Object.entries(map)) {
      if (!cfg || !cfg.model) continue;
      try {
        const gltf = await this.gltfLoader.loadAsync(resolveUrl(cfg.model));
        store.set(key, { scene: gltf.scene, cfg });
      } catch (e) {
        console.warn(`Object model for '${key}' failed; using sprite.`, e);
      }
    }
  }

  // ------------------------------------------------------------------
  // Terrain
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
        this.tileTopY[i] = yBase + h - 0.5;

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
    this.terrainVersion = -1;
  }

  // ------------------------------------------------------------------
  // Sprites (fallback art)
  // ------------------------------------------------------------------

  _registryColor(typeId) {
    const def = this.objectTypes[typeId];
    const rgb = def ? def.color : [200, 200, 200];
    return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
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
    sprite.center.set(0.5, 0.0);
    return sprite;
  }

  _objectIcon(rec) {
    if (rec.cat === "seed" && rec.planted) return { url: VARIANT.seed_planted, specific: true };
    if (rec.cat === "plant" && rec.mature) return { url: VARIANT.plant_mature, specific: true };
    return iconForType(rec.t, rec.cat);
  }

  // ------------------------------------------------------------------
  // Objects (model if available, else sprite)
  // ------------------------------------------------------------------

  _objProtoFor(rec) {
    return this.objProtoByType.get(rec.t) || this.objProtoByCat.get(rec.cat) || null;
  }

  _makeObjectDisplay(rec) {
    const proto = this._objProtoFor(rec);
    if (proto) {
      const root = proto.scene.clone(true);
      root.userData.isModel = true;
      root.userData.baseScale = proto.cfg.scale ?? 1;
      root.userData.yOffset = proto.cfg.yOffset ?? 0;
      root.traverse((o) => {
        if (o.isMesh) o.castShadow = true;
      });
      return root;
    }
    const icon = this._objectIcon(rec);
    const tint = icon.specific ? null : this._registryColor(rec.t);
    const sprite = this._makeSprite(icon.url, tint);
    sprite.userData.isModel = false;
    sprite.userData.curUrl = icon.url;
    return sprite;
  }

  syncObjects(records) {
    const seen = new Set();
    for (const rec of records) {
      if (rec.cat === "terrain") continue;
      seen.add(rec.id);

      let disp = this.objectMeshes.get(rec.id);
      if (!disp) {
        disp = this._makeObjectDisplay(rec);
        disp.userData.kind = "object";
        disp.userData.id = rec.id;
        disp.userData.typeId = rec.t;
        disp.userData.cat = rec.cat;
        this.objectMeshes.set(rec.id, disp);
        this.scene.add(disp);
      }

      const [sx, sz] = this.worldToScene(rec.x, rec.y);
      const topY = this.topYAt(rec.x, rec.y);

      if (disp.userData.isModel) {
        let s = disp.userData.baseScale;
        if (rec.cat === "plant") s *= 0.5 + 0.6 * Math.min(1, rec.growth ?? 1);
        disp.scale.setScalar(s);
        disp.position.set(sx, topY + (disp.userData.yOffset || 0), sz);
      } else {
        // Sprite: handle growth-state variant swaps.
        const icon = this._objectIcon(rec);
        if (disp.userData.curUrl !== icon.url) {
          disp.material.map = this.tex.get(icon.url);
          const tint = icon.specific ? new THREE.Color(0xffffff) : this._registryColor(rec.t);
          disp.material.color.copy(tint);
          disp.material.needsUpdate = true;
          disp.userData.curUrl = icon.url;
        }
        let s = 0.95;
        if (rec.cat === "plant") s = 0.7 + 0.7 * Math.min(1, rec.growth ?? 1);
        else if (rec.cat === "seed") s = 0.7;
        else if (rec.cat === "food") s = 0.8 + 0.15 * (rec.fresh ?? 1);
        disp.scale.set(s, s, 1);
        disp.position.set(sx, topY + 0.02, sz);
      }
    }

    for (const [id, disp] of this.objectMeshes) {
      if (!seen.has(id)) {
        this.scene.remove(disp);
        this._disposeObject(disp);
        this.objectMeshes.delete(id);
      }
    }
  }

  _disposeObject(obj) {
    if (obj.isSprite) {
      obj.material.dispose();
    } else {
      obj.traverse((o) => {
        if (o.isMesh) o.material?.dispose?.();
      });
    }
  }

  // ------------------------------------------------------------------
  // Agents (animated model if available, else sprite creature)
  // ------------------------------------------------------------------

  _energyColor(ratio) {
    if (ratio > 0.7) return new THREE.Color(0.31, 0.86, 0.31);
    if (ratio > 0.4) return new THREE.Color(1.0, 0.78, 0.0);
    if (ratio > 0.2) return new THREE.Color(1.0, 0.39, 0.0);
    return new THREE.Color(1.0, 0.2, 0.2);
  }

  _spriteHeading(dx, dy) {
    return Math.atan2(dx, dy) + Math.PI; // for the flat arrow decal
  }
  _modelHeading(dx, dy) {
    return Math.atan2(dx, dy); // glTF models face +Z
  }

  _makeEnergyDisc(ratio) {
    const disc = new THREE.Mesh(
      this._discGeo,
      new THREE.MeshBasicMaterial({
        color: this._energyColor(ratio),
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
      })
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = 0.03;
    disc.renderOrder = 1;
    return disc;
  }

  _makeAgentEntry(a) {
    const [sx, sz] = this.worldToScene(a.x, a.y);
    const topY = this.topYAt(a.x, a.y);
    const ratio = a.e / Math.max(1e-6, a.me);
    const group = new THREE.Group();
    group.position.set(sx, topY + 0.02, sz);
    group.userData = { kind: "agent", id: a.id };

    if (this.agentProto) {
      const model = cloneSkinned(this.agentProto.scene);
      const cfg = this.agentProto.cfg;
      model.scale.setScalar(cfg.scale ?? 1);
      model.position.y = cfg.yOffset ?? 0;
      model.traverse((o) => {
        if (o.isMesh) o.castShadow = true;
      });
      model.rotation.y = this._modelHeading(a.dx, a.dy);

      let mixer = null;
      const clips = this.agentProto.animations || [];
      if (clips.length) {
        mixer = new THREE.AnimationMixer(model);
        const name = cfg.animation;
        const clip =
          (name && THREE.AnimationClip.findByName(clips, name)) || clips[0];
        if (clip) mixer.clipAction(clip).play();
      }

      const disc = this._makeEnergyDisc(ratio);
      group.add(model);
      group.add(disc);

      this.scene.add(group);
      this.pickables.push(group);
      return {
        group,
        model,
        mixer,
        disc,
        isModel: true,
        target: new THREE.Vector3(sx, topY + 0.02, sz),
        targetRot: this._modelHeading(a.dx, a.dy),
      };
    }

    // Sprite fallback: creature billboard + flat facing arrow.
    const sprite = this._makeSprite(AGENT_ICON, this._energyColor(ratio));
    sprite.scale.set(1.1, 1.1, 1);
    const arrow = new THREE.Mesh(
      this._arrowGeo,
      new THREE.MeshBasicMaterial({
        map: this.tex.get(ARROW_ICON),
        transparent: true,
        alphaTest: 0.3,
        depthWrite: false,
      })
    );
    arrow.rotation.x = -Math.PI / 2;
    arrow.position.y = 0.03;
    arrow.renderOrder = 1;
    group.add(sprite);
    group.add(arrow);
    this.scene.add(group);
    this.pickables.push(group);
    return {
      group,
      sprite,
      arrow,
      isModel: false,
      target: new THREE.Vector3(sx, topY + 0.02, sz),
      targetRot: this._spriteHeading(a.dx, a.dy),
    };
  }

  syncAgents(agents) {
    const seen = new Set();
    for (const a of agents) {
      seen.add(a.id);
      let entry = this.agentMeshes.get(a.id);
      if (!entry) {
        entry = this._makeAgentEntry(a);
        this.agentMeshes.set(a.id, entry);
      }
      const [sx, sz] = this.worldToScene(a.x, a.y);
      const topY = this.topYAt(a.x, a.y);
      const ratio = a.e / Math.max(1e-6, a.me);
      entry.target.set(sx, topY + 0.02, sz);
      if (entry.isModel) {
        entry.targetRot = this._modelHeading(a.dx, a.dy);
        entry.disc.material.color.copy(this._energyColor(ratio));
      } else {
        entry.targetRot = this._spriteHeading(a.dx, a.dy);
        entry.sprite.material.color.copy(this._energyColor(ratio));
      }
    }

    for (const [id, entry] of this.agentMeshes) {
      if (!seen.has(id)) {
        this.scene.remove(entry.group);
        if (entry.mixer) entry.mixer.stopAllAction();
        this._disposeObject(entry.group);
        this.agentMeshes.delete(id);
        const idx = this.pickables.indexOf(entry.group);
        if (idx >= 0) this.pickables.splice(idx, 1);
        this._agentHistory.delete(id);
      }
    }
  }

  /** Per-frame interpolation + animation-mixer updates. */
  interpolate(dt) {
    const k = Math.min(1, dt * 8.0);
    for (const entry of this.agentMeshes.values()) {
      entry.group.position.lerp(entry.target, k);
      const target = entry.isModel ? entry.model : entry.arrow;
      const axis = entry.isModel ? "y" : "z";
      let cur = target.rotation[axis];
      let diff = entry.targetRot - cur;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      target.rotation[axis] = cur + diff * k;
      if (entry.mixer) entry.mixer.update(dt);
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

  getPickables() {
    return [...this.pickables, ...this.objectMeshes.values()];
  }
}
