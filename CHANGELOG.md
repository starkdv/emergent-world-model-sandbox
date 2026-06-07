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
  - Live 3D terrain (height + colour by type); objects and agents rendered as
    **real textured image sprites** loaded from shipped SVG art assets
    (`web/static/assets/`) rather than primitive geometry — berries, seeds,
    plants, fertilizer, an agent creature (energy-tinted) with a flat facing
    arrow. The same icons are reused across the registry cards, spawn list,
    inspector, tooltips, and inventory chips. Custom YAML objects fall back to a
    category icon tinted by their registry colour.
  - Smooth agent interpolation, optional trails and grid, orbit camera, hover
    tooltips, click inspectors, a spawn tool, a rolling population/energy graph,
    and an **Object Registry panel with UI for every registered object type**
    (built-in and custom YAML objects alike).
  - `web/static/js/icons.js`: shared icon resolver used by both the 3D scene and
    the DOM panels.
  - `WEB_UI_GUIDE.md`: complete reference for the web renderer.
- `main.py`: new `--web`, `--host`, `--port`, and `--open-browser` flags.
- `agents/agent.py`: Improved agent lifecycle and GRU hidden-state integration.
- `tests/test_evolution.py`: New tests covering mating, selection, and lineage tracking.

### Fixed
- numpy 2.x compatibility: `agents/brain.py` value head used `float()` on a
  length-1 array (rejected by numpy ≥ 2). Now uses `.item()` (works on 1.x/2.x).
- Restored the missing `utils/data/agent_logger.py` module. `utils/data/__init__.py`
  imported `AgentLogger` and `WorldModelLogger` from it, but the file was never
  committed — breaking the `--log` CLI flag and 8 tests on import. Reimplemented
  both as synchronous, persistent-handle CSV loggers (`agent_actions_*.csv` /
  `agent_states_*.csv`; `transitions_*.csv` / `episodes_*.csv` /
  `world_states_*.csv`) matching `WORLD_MODEL_LOGGING_FORMAT.md`.
- `.gitignore`: anchored the `data/` pattern to the repo root (`/data/`). The
  unanchored pattern also matched the `utils/data/` source package, which is why
  `agent_logger.py` was silently skipped by `git add` and never committed.

### Changed
- `main.py`: world + population creation refactored into a reusable
  `build_world_and_population()` factory (shared by every run mode and used by the
  web UI's "Reset" control); the Pygame renderer import is now lazy so the web UI
  runs without `pygame`/`moderngl` installed.
- `agents/learning.py`: Refactored actor-critic update loop for numerical stability and batch handling.

### Notes
This PR consolidates migration to Brain v2 (GRU + Actor-Critic), reorganizes utilities into `utils/agents/`, and updates tests to match new APIs. It also adds a browser-based Three.js renderer realising the spectator-client aspirations in `SUGGESTIONS.md` (Parts 4, 7, 9).
