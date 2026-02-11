# Changelog

## [Unreleased]

### Added
- `agents/agent.py`: Improved agent lifecycle and GRU hidden-state integration.
- `tests/test_evolution.py`: New tests covering mating, selection, and lineage tracking.

### Changed
- `agents/learning.py`: Refactored actor-critic update loop for numerical stability and batch handling.

### Notes
This PR consolidates migration to Brain v2 (GRU + Actor-Critic), reorganizes utilities into `utils/agents/`, and updates tests to match new APIs.
