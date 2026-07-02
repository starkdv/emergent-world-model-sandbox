# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue:

- **Preferred:** GitHub's private vulnerability reporting — *Security →
  Report a vulnerability* on this repository.
- **Email:** starkdv123@gmail.com (include steps to reproduce and the commit
  or version affected).

This is a solo-maintained research project; reports are handled on a
best-effort basis. You can expect an acknowledgement within about a week.
Please allow time for a fix before public disclosure.

## Supported versions

Only the latest state of the `main` branch is supported. There are no
maintained release lines; fixes land on `main`.

## Scope and deployment notes

This is a **research simulation sandbox**, not a hardened service. Points
worth knowing before deploying anything from it:

- The live viewer / render server (`render/`, `main.py --serve` modes) is
  designed for **local use** (localhost or a trusted network). It has no
  authentication or TLS — do **not** expose it directly to the public
  internet; put it behind a reverse proxy with auth if remote access is
  needed.
- Saved states, replays, genomes, and config files are loaded with standard
  Python parsers. Load them **only from sources you trust**.
- In scope for reports: anything allowing code execution, file access
  outside expected directories, or remote compromise via the viewer/stream
  server. Out of scope: simulation-quality issues (aberrant agent behaviour,
  fitness exploits inside the ecology) — those are science, not
  vulnerabilities, and are welcome as regular issues.

## Dependencies

Dependency alerts are handled via GitHub's Dependabot on this repository.
