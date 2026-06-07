# Changelog

## [Unreleased]

### Added
- **Three.js Web UI** — a full browser-based 3D renderer (`python main.py --web`):
  - `utils/ui/web_server.py`: dependency-free (stdlib) HTTP server with a
    background simulation thread, a JSON API, and control commands
    (pause/resume/step/speed/spawn/reset).
  - `utils/ui/web_serialize.py`: pure world → JSON serialisation (meta, state,
    terrain, and agent/object/tile inspection); unit-testable without a browser.
  - `web/`: the front-end — `index.html`, `static/css/style.css`, and ES modules
    `app.js` (orchestration), `world3d.js` (Three.js scene), `ui.js` (DOM panels),
    `net.js` (API client).
  - Live 3D terrain (height + colour by type), per-type object meshes, agents as
    energy-coloured oriented cones with smooth interpolation, optional trails and
    grid, orbit camera, hover tooltips, click inspectors, a spawn tool, a rolling
    population/energy graph, and an **Object Registry panel with UI for every
    registered object type** (built-in and custom YAML objects alike).
  - `WEB_UI_GUIDE.md`: complete reference for the web renderer.
- `main.py`: new `--web`, `--host`, `--port`, and `--open-browser` flags.
- `agents/agent.py`: Improved agent lifecycle and GRU hidden-state integration.
- `tests/test_evolution.py`: New tests covering mating, selection, and lineage tracking.

### Changed
- `main.py`: world + population creation refactored into a reusable
  `build_world_and_population()` factory (shared by every run mode and used by the
  web UI's "Reset" control); the Pygame renderer import is now lazy so the web UI
  runs without `pygame`/`moderngl` installed.
- `agents/learning.py`: Refactored actor-critic update loop for numerical stability and batch handling.

### Notes
This PR consolidates migration to Brain v2 (GRU + Actor-Critic), reorganizes utilities into `utils/agents/`, and updates tests to match new APIs. It also adds a browser-based Three.js renderer realising the spectator-client aspirations in `SUGGESTIONS.md` (Parts 4, 7, 9).
