/*
 * Networking layer for the Emergent World-Model Sandbox web UI.
 *
 * Thin wrapper around fetch() for the JSON API exposed by web_server.py.
 * State is polled (rather than streamed) which keeps the server fully
 * dependency-free and robust across browsers; the client interpolates agent
 * positions between snapshots for smooth motion.
 *
 * Author: Karan Vasa
 */

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

export const Net = {
  /** Fetch static metadata (world size, object registry, palette…). */
  meta: () => getJSON("/api/meta"),

  /** Fetch the current per-frame dynamic state snapshot. */
  state: () => getJSON("/api/state"),

  /** Fetch the flat terrain-type grid. */
  terrain: () => getJSON("/api/terrain"),

  /** Fetch full detail for one agent. */
  inspectAgent: (id) => getJSON(`/api/inspect/agent/${id}`),

  /** Fetch full detail for one world object. */
  inspectObject: (id) => getJSON(`/api/inspect/object/${id}`),

  /** Fetch full detail for one tile. */
  inspectTile: (x, y) => getJSON(`/api/inspect/tile?x=${x}&y=${y}`),

  /**
   * Send a control command to the server.
   * @param {Object} payload - must contain a `cmd` field.
   */
  async control(payload) {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  },
};
