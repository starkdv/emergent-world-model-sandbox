/*
 * DOM UI layer for the Emergent World-Model Sandbox web client.
 *
 * Builds and updates every HTML panel:
 *   - top-bar live stats (tick, agents, generation, FPS…)
 *   - the Object Registry panel (a UI card for EVERY registered object type)
 *   - the Spawn tool, Legend, Inspector, and live population/energy graph
 *
 * Author: Karan Vasa
 */

import { iconForType } from "./icons.js";

const $ = (id) => document.getElementById(id);
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

/**
 * Build an <img> tag for an object's real icon (shared art assets). Category
 * fallback icons get a coloured ring so custom types stay distinguishable.
 */
function iconImg(typeId, category, color, size = 22) {
  const { url, specific } = iconForType(typeId, category);
  const ring = !specific && color ? `box-shadow:0 0 0 2px ${rgb(color)} inset;` : "";
  return `<img class="obj-icon" src="${url}" width="${size}" height="${size}" style="${ring}" alt="" draggable="false">`;
}

export class UI {
  constructor(meta, callbacks) {
    this.meta = meta;
    this.cb = callbacks; // { onControl, onSpawnSelect }
    this.spawnType = null;
    this.history = []; // rolling [{pop, energy}]
    this._buildRegistry();
    this._buildSpawn();
    this._buildLegend();
    this._wireControls();
  }

  // ------------------------------------------------------------------
  // Top-bar stats
  // ------------------------------------------------------------------

  updateStats(state, fps) {
    const s = state.stats || {};
    const statusColor = state.paused ? "var(--warn)" : "var(--good)";
    const statusText = state.paused ? "PAUSED" : "RUNNING";
    $("stats").innerHTML = `
      <div class="stat"><span class="k">Status</span>
        <span class="v"><span class="dot" style="background:${statusColor}"></span>${statusText}</span></div>
      <div class="stat"><span class="k">Tick</span><span class="v">${state.tick.toLocaleString()}</span></div>
      <div class="stat"><span class="k">Agents</span><span class="v">${state.agent_count}</span></div>
      <div class="stat"><span class="k">Generation</span><span class="v">${s.max_generation ?? 0}</span></div>
      <div class="stat"><span class="k">Avg Energy</span><span class="v">${s.avg_energy ?? 0}</span></div>
      <div class="stat"><span class="k">Food</span><span class="v">${state.counts.total_food}</span></div>
      <div class="stat"><span class="k">Plants</span><span class="v">${state.counts.total_plants}</span></div>
      <div class="stat"><span class="k">Sim TPS</span><span class="v">${state.sim_tps}</span></div>
      <div class="stat"><span class="k">Render FPS</span><span class="v">${fps}</span></div>
    `;
    // Keep Play/Pause button label in sync with server state.
    $("btn-play").textContent = state.paused ? "▶ Play" : "⏸ Pause";
  }

  // ------------------------------------------------------------------
  // Object registry — a card per registered object type
  // ------------------------------------------------------------------

  _kv(k, v) {
    return `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  }

  _sect(title, rows) {
    if (!rows.length) return "";
    return `<div class="sect"><div class="sect-title">${title}</div>${rows.join("")}</div>`;
  }

  _objectDetailHTML(d) {
    const parts = [];
    if (d.edible) {
      parts.push(this._sect("Edible", [
        this._kv("Calories", d.edible.calories),
        this._kv("Toxicity", d.edible.toxicity),
        this._kv("Freshness", d.edible.freshness),
      ]));
    }
    if (d.seed) {
      parts.push(this._sect("Seed", [
        this._kv("Grows into", d.seed.grows_into || "—"),
        this._kv("Grow time", `${d.seed.grow_time} ticks`),
        this._kv("Req. fertility", d.seed.required_fertility),
        this._kv("Req. moisture", d.seed.required_moisture),
        this._kv("Max age", d.seed.max_age),
      ]));
    }
    if (d.plant) {
      parts.push(this._sect("Plant", [
        this._kv("Mature age", d.plant.mature_age),
        this._kv("Max age", d.plant.max_age),
        this._kv("Produces", d.plant.produces || "—"),
        this._kv("Spawn rate", d.plant.spawn_rate),
      ]));
    }
    if (d.fertilizer) {
      parts.push(this._sect("Fertilizer", [
        this._kv("Boost", `+${d.fertilizer.fertility_boost}`),
        this._kv("Duration", d.fertilizer.duration),
        this._kv("Radius", `${d.fertilizer.radius} tiles`),
      ]));
    }
    if (d.tool) {
      parts.push(this._sect("Tool", [
        this._kv("Effect", d.tool.effect_type || "—"),
        this._kv("Efficiency", d.tool.efficiency),
      ]));
    }
    if (d.tile_effect) {
      const te = d.tile_effect;
      parts.push(this._sect("Tile Effect", [
        this._kv("Germination ×", te.germination_multiplier),
        this._kv("Growth ×", te.growth_multiplier),
        this._kv("Spawn rate ×", te.spawn_rate_multiplier),
        this._kv("Spreads", te.spread_type_id || "no"),
        this._kv("Spread chance", te.spread_chance),
        this._kv("Converts to", te.converts_terrain || "—"),
        this._kv("Reclaim to", te.reclaim_terrain || "—"),
      ]));
    }
    parts.push(this._sect("Physics", [
      this._kv("Decay rate", d.physics.decay_rate),
      this._kv("Decomposes into", d.physics.decompose_into || "—"),
      this._kv("Nutrient return", d.physics.nutrient_return),
    ]));
    parts.push(this._sect("Interaction", [
      this._kv("Pickable", d.interaction.pickable ? "✓" : "✗"),
      this._kv("Usable", d.interaction.usable ? "✓" : "✗"),
      this._kv("Passable", d.interaction.passable ? "✓" : "✗"),
    ]));
    parts.push(this._sect("Observation", [
      this._kv("Vision encoding", d.observation.vision_encoding),
      this._kv("Value source", d.observation.value_source),
    ]));
    return parts.join("");
  }

  _buildRegistry() {
    const host = $("tab-registry");
    const types = this.meta.object_types || {};
    const cards = Object.values(types)
      .sort((a, b) => a.category.localeCompare(b.category) || a.display_name.localeCompare(b.display_name))
      .map((d) => {
        const badges = (d.components || []).map((c) => `<span class="badge">${c}</span>`).join("");
        return `
          <div class="obj-card" data-type="${d.type_id}">
            <div class="obj-head">
              ${iconImg(d.type_id, d.category, d.color)}
              <span class="obj-name">${d.display_name}</span>
              <span class="obj-cat">${d.category}</span>
            </div>
            <div class="obj-detail">
              <div>${badges}</div>
              ${this._objectDetailHTML(d)}
            </div>
          </div>`;
      })
      .join("");
    host.innerHTML = cards || `<div class="muted">No object types registered.</div>`;

    host.querySelectorAll(".obj-head").forEach((head) => {
      head.addEventListener("click", () => head.parentElement.classList.toggle("open"));
    });
  }

  // ------------------------------------------------------------------
  // Spawn tool
  // ------------------------------------------------------------------

  _buildSpawn() {
    const host = $("tab-spawn");
    const types = this.meta.object_types || {};
    const items = Object.values(types)
      .map(
        (d) => `
        <div class="spawn-item" data-type="${d.type_id}">
          ${iconImg(d.type_id, d.category, d.color)}
          <span class="obj-name">${d.display_name}</span>
          <span class="obj-cat">${d.category}</span>
        </div>`
      )
      .join("");
    host.innerHTML = `
      <div class="spawn-hint">Pick an object, then <b>click a tile</b> in the world to place it. Click the selected item again to cancel.</div>
      ${items}`;

    host.querySelectorAll(".spawn-item").forEach((item) => {
      item.addEventListener("click", () => {
        const t = item.dataset.type;
        if (this.spawnType === t) {
          this.spawnType = null;
          item.classList.remove("selected");
        } else {
          host.querySelectorAll(".spawn-item").forEach((el) => el.classList.remove("selected"));
          item.classList.add("selected");
          this.spawnType = t;
        }
        this.cb.onSpawnSelect?.(this.spawnType);
      });
    });
  }

  clearSpawnSelection() {
    this.spawnType = null;
    $("tab-spawn").querySelectorAll(".spawn-item").forEach((el) => el.classList.remove("selected"));
  }

  // ------------------------------------------------------------------
  // Legend
  // ------------------------------------------------------------------

  _buildLegend() {
    const host = $("tab-legend");
    const pal = this.meta.terrain_palette || {};
    const terrainRows = Object.entries(pal)
      .map(
        ([name, c]) =>
          `<div class="legend-item"><span class="swatch" style="background:${rgb(c)}"></span>${name}</div>`
      )
      .join("");
    host.innerHTML = `
      <div class="sect"><div class="sect-title">Terrain</div>${terrainRows}</div>
      <div class="sect"><div class="sect-title">Agents (energy)</div>
        <div class="legend-item"><span class="swatch" style="background:rgb(80,220,80)"></span>High (&gt;70%)</div>
        <div class="legend-item"><span class="swatch" style="background:rgb(255,200,0)"></span>Medium</div>
        <div class="legend-item"><span class="swatch" style="background:rgb(255,100,0)"></span>Low</div>
        <div class="legend-item"><span class="swatch" style="background:rgb(255,50,50)"></span>Critical</div>
      </div>
      <div class="sect"><div class="sect-title">Object icons</div>
        <div class="legend-item">${iconImg("", "food", null, 18)} food</div>
        <div class="legend-item">${iconImg("", "seed", null, 18)} seed</div>
        <div class="legend-item">${iconImg("", "plant", null, 18)} plant</div>
        <div class="legend-item">${iconImg("", "fertilizer", null, 18)} fertilizer</div>
        <div class="legend-item">${iconImg("", "tool", null, 18)} tool</div>
      </div>`;
  }

  // ------------------------------------------------------------------
  // Inspector
  // ------------------------------------------------------------------

  showAgent(d) {
    const ratio = d.energy_pct / 100;
    const col = ratio > 0.5 ? "var(--good)" : ratio > 0.2 ? "var(--warn)" : "var(--bad)";
    const inv = d.inventory.length
      ? d.inventory
          .map(
            (it) =>
              `<span class="inv-chip">${iconImg(it.type_id, it.category, it.color, 16)}${it.name}</span>`
          )
          .join("")
      : `<span class="muted">empty</span>`;
    const traits = Object.entries(d.traits)
      .map(([k, v]) => this._kv(k.replace(/_/g, " "), v))
      .join("");

    $("inspector-title").textContent = `Agent #${d.id}`;
    $("inspector-body").innerHTML = `
      <div class="kv"><span class="k">Energy</span><span class="v">${d.energy} / ${d.max_energy} (${d.energy_pct}%)</span></div>
      <div class="energy-bar"><div class="energy-fill" style="width:${d.energy_pct}%;background:${col}"></div></div>
      ${this._kv("Position", `(${d.x}, ${d.y})`)}
      ${this._kv("Facing", d.facing)}
      ${this._kv("Age", `${d.age} / ${d.max_age}`)}
      ${this._kv("Generation", d.generation)}
      ${this._kv("Fitness", d.fitness)}
      ${this._kv("Metabolism", d.metabolism_rate)}
      ${this._kv("Vision radius", d.vision_radius)}
      ${this._kv("Learning", d.learning_enabled ? "on" : "off")}
      ${this._sect("Traits", [traits])}
      <div class="sect"><div class="sect-title">Inventory (${d.inventory.length}/${d.inventory_size})</div>${inv}</div>
    `;
    this._openInspector();
  }

  showObject(d) {
    const comps = Object.entries(d.components || {})
      .map(([name, vals]) => {
        const rows = Object.entries(vals)
          .map(([k, v]) => this._kv(k.replace(/_/g, " "), typeof v === "boolean" ? (v ? "✓" : "✗") : v))
          .join("");
        return this._sect(name, [rows]);
      })
      .join("");
    $("inspector-title").textContent = d.name;
    $("inspector-body").innerHTML = `
      <div class="kv"><span class="k">Type</span><span class="v">${iconImg(d.type_id, d.category, d.color, 18)} ${d.type_id}</span></div>
      ${this._kv("Category", d.category)}
      ${this._kv("Position", `(${d.x}, ${d.y})`)}
      ${comps || '<div class="muted">No components.</div>'}
    `;
    this._openInspector();
  }

  showTile(d) {
    const objs = d.objects.length
      ? d.objects.map((o) => `<div class="legend-item">${iconImg(o.type_id, o.category, o.color, 18)}${o.name} <span class="obj-cat">${o.category}</span></div>`).join("")
      : '<div class="muted">none</div>';
    const agents = d.agents.length
      ? d.agents.map((a) => this._kv(`Agent #${a.id}`, `E ${a.energy} · gen ${a.generation}`)).join("")
      : '<div class="muted">none</div>';
    $("inspector-title").textContent = `Tile (${d.x}, ${d.y})`;
    $("inspector-body").innerHTML = `
      ${this._kv("Terrain", d.terrain)}
      ${this._kv("Fertility", d.fertility)}
      ${this._kv("Moisture", d.moisture)}
      ${this._kv("Passable", d.passable ? "✓" : "✗")}
      ${this._kv("Plantable", d.plantable ? "✓" : "✗")}
      <div class="sect"><div class="sect-title">Objects (${d.objects.length})</div>${objs}</div>
      <div class="sect"><div class="sect-title">Agents (${d.agents.length})</div>${agents}</div>
    `;
    this._openInspector();
  }

  _openInspector() { $("inspector").classList.remove("hidden"); }
  closeInspector() { $("inspector").classList.add("hidden"); }

  // ------------------------------------------------------------------
  // Population / energy graph
  // ------------------------------------------------------------------

  pushGraph(state) {
    this.history.push({
      pop: state.agent_count,
      energy: state.stats?.avg_energy ?? 0,
    });
    if (this.history.length > 120) this.history.shift();
    this._drawGraph();
  }

  _drawGraph() {
    const cv = $("graph");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (this.history.length < 2) return;

    const maxPop = Math.max(10, ...this.history.map((h) => h.pop));
    const maxEnergy = Math.max(10, ...this.history.map((h) => h.energy));
    const n = this.history.length;

    const line = (key, max, color) => {
      ctx.beginPath();
      this.history.forEach((h, i) => {
        const x = (i / (n - 1)) * (W - 8) + 4;
        const y = H - 6 - (h[key] / max) * (H - 14);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    };
    line("pop", maxPop, "#50dc64");
    line("energy", maxEnergy, "#64c8ff");

    ctx.fillStyle = "#8c8c9b";
    ctx.font = "9px Segoe UI";
    ctx.fillText("● population", 6, 12);
    ctx.fillStyle = "#64c8ff";
    ctx.fillText("● avg energy", 80, 12);
  }

  // ------------------------------------------------------------------
  // Controls wiring
  // ------------------------------------------------------------------

  _wireControls() {
    $("btn-play").addEventListener("click", () => this.cb.onControl({ cmd: "toggle" }));
    $("btn-step").addEventListener("click", () => this.cb.onControl({ cmd: "step", ticks: 1 }));
    $("btn-reset").addEventListener("click", () => {
      if (confirm("Reset the world to a fresh state?")) this.cb.onControl({ cmd: "reset" });
    });
    const speed = $("speed");
    speed.addEventListener("input", () => {
      $("speed-val").textContent = `${parseFloat(speed.value).toFixed(1)}×`;
      this.cb.onControl({ cmd: "set_speed", speed: parseFloat(speed.value) });
    });
    if (!this.meta.reset_available) $("btn-reset").disabled = true;

    // Tabs.
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        $(`tab-${tab.dataset.tab}`).classList.add("active");
      });
    });

    $("inspector-close").addEventListener("click", () => this.closeInspector());
    this._wireToggles();
  }

  _wireToggles() {
    $("tg-grid").addEventListener("change", (e) => this.cb.onToggle?.("grid", e.target.checked));
    $("tg-trails").addEventListener("change", (e) => this.cb.onToggle?.("trails", e.target.checked));
    $("tg-height").addEventListener("change", (e) => this.cb.onToggle?.("height", e.target.checked));
  }

  // ------------------------------------------------------------------
  // Tooltip
  // ------------------------------------------------------------------

  showTooltip(html, x, y) {
    const tt = $("tooltip");
    tt.innerHTML = html;
    tt.style.left = `${x + 14}px`;
    tt.style.top = `${y + 14}px`;
    tt.classList.remove("hidden");
  }

  hideTooltip() { $("tooltip").classList.add("hidden"); }
}
