/*
 * Shared object-icon resolver for the Emergent World-Model Sandbox web UI.
 *
 * Maps a world object's type_id (or, as a fallback, its category) to a real
 * SVG art asset shipped under web/static/assets/. The same resolver is used
 * for the 3D world sprites AND the DOM panels (registry cards, spawn list,
 * inspector, tooltips) so every object has one consistent visual identity.
 *
 * Custom YAML object types automatically get a sensible category icon, tinted
 * by their registry colour so they stay visually distinct.
 *
 * Author: Karan Vasa
 */

export const ICON_BASE = "/static/assets/";

// Exact type_id → asset (built-in objects get bespoke art).
const BY_TYPE = {
  berry: "berry.svg",
  berry_seed: "seed.svg",
  berry_plant: "plant.svg",
  fertilizer: "fertilizer.svg",
  sand: "sand.svg",
};

// Category → fallback asset (covers custom YAML objects).
const BY_CATEGORY = {
  food: "food.svg",
  seed: "seed.svg",
  plant: "plant.svg",
  fertilizer: "fertilizer.svg",
  tool: "tool.svg",
  terrain: "sand.svg",
  object: "object.svg",
};

/**
 * Resolve the icon for an object type.
 * @param {string} typeId
 * @param {string} category
 * @returns {{url: string, specific: boolean}} url and whether it is bespoke
 *          art for this exact type (specific icons are not colour-tinted).
 */
export function iconForType(typeId, category) {
  if (typeId && BY_TYPE[typeId]) {
    return { url: ICON_BASE + BY_TYPE[typeId], specific: true };
  }
  const file = BY_CATEGORY[category] || BY_CATEGORY.object;
  return { url: ICON_BASE + file, specific: false };
}

/** Asset URL for a state-specific variant (planted seed, mature plant). */
export const VARIANT = {
  seed_planted: ICON_BASE + "seed_planted.svg",
  plant_mature: ICON_BASE + "plant_mature.svg",
};

/** Agent + facing-arrow assets. */
export const AGENT_ICON = ICON_BASE + "agent.svg";
export const ARROW_ICON = ICON_BASE + "arrow.svg";
