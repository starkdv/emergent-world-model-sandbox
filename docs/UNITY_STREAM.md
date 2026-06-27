# Live Stream Protocol — for a Unity (or any external) frontend

This is the wire contract for building an external UI (Unity, Unreal, a custom
viewer) on top of the running simulation. The Python side already streams the
**real** world read-only; a client only needs to consume this stream and render
it. Nothing here mutates the simulation.

> **Transport note (read first):** the stream is **Server-Sent Events (SSE)
> over plain HTTP**, *not* a WebSocket. SSE is one-directional (server → client),
> which is all a viewer needs. Unity has no built-in `EventSource`, but SSE is
> just an HTTP response you read line-by-line — see the C# client in §6. If you
> specifically want a WebSocket endpoint, see §8 (it's a thin add-on; the
> server is transport-agnostic).

---

## 1. Start the server

```bash
python -m render.server                       # world from config/default.yaml
python -m render.server --host 0.0.0.0 --port 8000 --tps 15
python -m render.server --checkpoint run.pkl  # stream a saved run
python -m render.server --replay run.jsonl    # stream a recording (loops)
```

- `--host 0.0.0.0` to accept connections from another machine (e.g. Unity on
  your desktop talking to the sim in Codespaces — forward the port).
- `--tps` = simulation ticks broadcast per second (the stream cadence).
- CORS is open (`Access-Control-Allow-Origin: *`).

## 2. Endpoints

| Method / path | Returns |
|---|---|
| `GET /api/snapshot` | One **full snapshot** as a single JSON body (no streaming). Good for a quick poll or a one-frame import. |
| `GET /api/stream` | **SSE stream**: first a `snapshot` event, then one `delta` event per tick. The canonical live feed. |
| `GET /` , `GET /<file>` | Static files from `web/` (the reference Three.js client). Ignore for Unity. |

## 3. SSE framing

Each message on `/api/stream` is a standard SSE block:

```
event: snapshot
data: {...one line of JSON...}

event: delta
data: {...one line of JSON...}

: keep-alive
```

Rules a client must follow:
- Blocks are separated by a blank line (`\n\n`). The `data:` payload is a single
  line of JSON (no embedded newlines).
- Lines beginning with `:` are **comments** (keep-alives) — ignore them.
- The **first** event after connecting is always a `snapshot` (full state of the
  *current* tick — each client gets a fresh one). Every event after is a
  `delta`. On reconnect you get a fresh `snapshot` again.
- A `delta` may also arrive with `event: delta` and `data` containing
  `{"type":"error","message":...}` if the sim hiccuped — log and keep reading.

## 4. Message schemas

### `snapshot` (full frame)

```jsonc
{
  "type": "snapshot",
  "version": 1,                       // bridge schema version (reject if unknown)
  "tick": 1234,
  "terrain": {
    "width": 160, "height": 160,
    "terrain_codes": {"soil":0,"rock":1,"water":2,"sand":3},
    "elevation":  {"dtype":"uint8","shape":[160,160],"b64":"<base64>"},
    "terrain":    {"dtype":"uint8","shape":[160,160],"b64":"<base64>"},
    "fertility":  {"dtype":"uint8","shape":[160,160],"b64":"<base64>"},
    "moisture":   {"dtype":"uint8","shape":[160,160],"b64":"<base64>"}
  },
  "objects": [ <object>, ... ],       // non-terrain objects only
  "agents":  [ <agent>, ... ],
  "sky":     <sky>,
  "burning": [ <objectId>, ... ],     // object ids currently on fire (W3)
  "has_pheromones": true,
  "signal_enabled": false,            // W4 SIGNAL active?
  "transfer_enabled": false,          // W5 trade active?
  "brain_output_size": 8,             // 9 = Brain v3.5 (+SIGNAL); MAJORITY brain
  "brain_class": "BrainV3",           // "Brain" (v2) | "BrainV3" (v3/v3.5); majority
  "brain_versions": { "v3": 96, "v2": 4 } // FULL cohort mix (counts per version)
}
```

> In a brain-cohort competition the world runs **more than one architecture at
> once**. `brain_versions` is the authoritative breakdown (counts per version,
> most-common first); `brain_class`/`brain_output_size` describe only the
> *majority* brain, kept for backward compatibility. `brain_versions` is also
> sent on every **delta**, so it stays live as one cohort out-competes the
> other.

**`<object>`** — one world object (berry, tree/plant, seed, fertilizer, thorns…):

```jsonc
{
  "id": 42,
  "x": 73, "y": 12,                   // grid cell (ints)
  "type_id": "berry_plant",           // exact registry id
  "category": "plant",                // food|plant|seed|fertilizer|hazard|...
  "terrain": false,                   // terrain-layer object (e.g. sand)? usually false here
  "value": 0.6,                       // freshness(food)/maturity(plant)/viability(seed), 0..1
  "planted": true                     // planted by an agent?
}
```

**`<agent>`**:

```jsonc
{
  "id": 7,
  "x": 40, "y": 55,
  "dir": [0, -1],                     // facing (dx,dy): N=(0,-1) E=(1,0) S=(0,1) W=(-1,0)
  "energy": 0.62,                     // fraction of max (0..1)
  "age": 0.21,                        // fraction of max age (0..1)
  "alive": true,
  "action": "MOVE_FORWARD",           // last action taken (the brain's decision) or null
  "inv": 1,                           // inventory item count
  "has_food": true, "has_seed": false,
  "lineage": 3,                       // genome lineage id (tint families)
  "generation": 4
}
```

**`<sky>`** (W1 climate):

```jsonc
{
  "time_of_day": 0.25,                // 0..1 around the day
  "light": 0.9,                       // 0..1 (drives brightness)
  "temperature": 0.55,               // 0..1
  "season": 0.3,                      // 0..1
  "raining": false, "drought": false,
  "enabled": true
}
```

### `delta` (per-tick change set)

Apply on top of the last snapshot/delta you hold:

```jsonc
{
  "type": "delta",
  "version": 1,
  "tick": 1235,
  "objects":         [ <object>, ... ],   // upserts (added or moved); replace by id
  "removed_objects": [ <objectId>, ... ], // delete these ids
  "agents":          [ <agent>, ... ],    // upserts; replace by id
  "removed_agents":  [ <agentId>, ... ],  // delete these ids (died/left)
  "signals":         [ [x, y, v], ... ],  // pheromone cells, v in 0..1 (full set each tick)
  "burning":         [ <objectId>, ... ], // full set of burning ids this tick
  "sky":             <sky>,
  "brain_versions":  { "v3": 96, "v2": 4 } // live cohort mix (full map each tick)
}
```

Client update algorithm per `delta`:
1. For each `objects[i]`: upsert by `id` (create if new, else move/update).
2. For each `removed_objects`: destroy that id.
3. Same for `agents` / `removed_agents`.
4. Replace your pheromone overlay with `signals` (it's the full current set).
5. Replace your burning set with `burning`.
6. Update sky/lighting from `sky`.

## 5. Decoding the terrain grids

The four terrain grids are **row-major `uint8` arrays**, base64-encoded, shape
`[height, width]`. Index a cell `(x, y)` as `data[y * width + x]`, value `0..255`:

- `elevation` → normalize `/255` then multiply by your max height for the voxel
  column / surface height.
- `terrain` → the biome code (see `terrain_codes`): 0 soil, 1 rock, 2 water,
  3 sand.
- `fertility`, `moisture` → `/255`, useful for tinting / overlays.

Decode in C#: `System.Convert.FromBase64String(b64)` gives the `byte[]` directly.

## 6. Coordinate & rendering conventions (match the reference client)

The simulation grid is 2-D `(x, y)`; the third axis is the renderer's choice.
The reference Three.js client uses:

- world X = `x + 0.5 - width/2`, world Z = `y + 0.5 - height/2` (centered),
- **up = Y**; column/surface height = `round(elevation/255 * MAX_H)`
  (reference `MAX_H = 10`),
- water drawn as one flat surface at a sea level = mean height of water tiles,
- agents grounded at the surface height of their tile, yaw from `dir`, tweened
  between ticks for smooth motion (the sim moves one cell per tick).

Unity is left-handed / Y-up; map sim `(x, y)` → Unity `(x, elevation, y)` and
pick a tile size. None of this is mandated by the protocol — it's just what
keeps a Unity scene visually consistent with the web client.

## 7. Minimal Unity (C#) SSE client

Unity has no `EventSource`, but SSE is a chunked HTTP body you read line-by-line
with `System.Net.Http`. Run it on a background task and marshal parsed frames to
the main thread (Unity API is main-thread-only). Sketch:

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class SimStreamClient : MonoBehaviour
{
    public string url = "http://127.0.0.1:8000/api/stream";
    readonly System.Collections.Concurrent.ConcurrentQueue<(string ev, string data)> _q = new();
    CancellationTokenSource _cts;

    void Start() { _cts = new CancellationTokenSource(); _ = ReadLoop(_cts.Token); }
    void OnDestroy() { _cts?.Cancel(); }

    async Task ReadLoop(CancellationToken ct)
    {
        using var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        while (!ct.IsCancellationRequested)
        {
            try
            {
                using var resp = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
                using var stream = await resp.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream);
                string ev = "message";
                string line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    if (line.Length == 0) continue;            // end of a block
                    if (line[0] == ':') continue;              // keep-alive comment
                    if (line.StartsWith("event:")) ev = line.Substring(6).Trim();
                    else if (line.StartsWith("data:"))
                        _q.Enqueue((ev, line.Substring(5).Trim()));
                }
            }
            catch (Exception e) { Debug.LogWarning($"stream reconnect: {e.Message}"); }
            await Task.Delay(1000, ct);                        // reconnect backoff
        }
    }

    void Update()
    {
        // drain on the main thread; parse JSON with your serializer of choice
        while (_q.TryDequeue(out var msg))
        {
            if (msg.ev == "snapshot") ApplySnapshot(msg.data);
            else if (msg.ev == "delta") ApplyDelta(msg.data);
        }
    }

    void ApplySnapshot(string json) { /* JsonUtility/Newtonsoft → build scene */ }
    void ApplyDelta(string json)    { /* upsert/remove agents+objects, sky, signals */ }
}
```

JSON tips for Unity:
- `JsonUtility` can't parse the nested/`Dictionary` bits (e.g. `terrain_codes`).
  Use **Newtonsoft Json.NET** (com.unity.nuget.newtonsoft-json) for the snapshot;
  `JsonUtility` is fine for the flat agent/object DTOs if you prefer.
- Decode grids with `Convert.FromBase64String(packed.b64)`.
- Only the `snapshot` carries terrain; cache it and apply `delta`s on top.

A no-streaming alternative for a first prototype: `GET /api/snapshot` on a timer
(e.g. 5–10 Hz) and rebuild — simpler, less efficient, no SSE parsing.

## 8. Optional: a WebSocket endpoint

If a true WebSocket fits Unity better (`System.Net.WebSockets.ClientWebSocket`
or the NativeWebSocket package), it's a small server-side add-on because the
streaming core (`render/sim_session.SimSession` → `snapshot()` / per-tick
`delta`) is transport-agnostic. It would require a WebSocket library
(`websockets`/FastAPI) added to the Python env, then a handler that, per client,
sends the same `snapshot` then `delta` JSON frames documented above. Ask and it
can be added; the message schemas in §4 stay identical.

## 9. Versioning

Every message carries `"version"` (the bridge schema version, currently `1`).
A client should check it and refuse/ warn on an unknown version. Fields are only
**added**, never renamed or reordered, within a version — so a client that
ignores unknown fields stays forward-compatible.

---

Reference implementation of a consumer: `web/main.js` (Three.js) — it does
exactly what §3–§6 describe. Bridge source: `render/state_bridge.py`; server:
`render/server.py`. See also `FRONTEND_3D_PROPOSAL.md` and `web/README.md`.
