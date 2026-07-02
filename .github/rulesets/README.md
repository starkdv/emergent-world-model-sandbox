# Repository rulesets

Importable GitHub ruleset definitions for this repository. Rulesets are
repository **settings**, so committing these files does not activate them —
import each once via the GitHub UI:

**Settings → Rules → Rulesets → New ruleset ▾ → Import a ruleset → pick the
JSON file → Create.**

## `protect-main.json` — Protect main

Applies to the default branch (`main`):

- **No direct pushes** — changes land via pull request only (squash merge,
  matching this repo's history style).
- **Required status checks** (must pass, and the branch must be up to date):
  `Lint`, `Type Check`, `Test (Python 3.11)`, `Test (Python 3.12)` — the four
  jobs of the CI workflow.
- **No force pushes, no deletion, linear history.**
- Review approvals are set to 0 (solo-maintainer repo — you cannot approve
  your own PR); review threads must still be resolved before merging. Raise
  `required_approving_review_count` to 1 if a second maintainer ever joins.
- **Bypass:** repository admins (`actor_id: 5` = the Repository admin role)
  can bypass in an emergency.

## `protect-tags.json` — Protect release tags

Applies to `v*` tags: once a release tag is published it cannot be deleted,
moved, or force-updated (admin bypass as above). Keeps citations of tagged
versions (e.g. the Zenodo-archived paper) stable.

## Notes

- Working branches (`claude/*`, feature branches) are intentionally
  unrestricted — protection applies where history must stay immutable.
- If CI job names change in `.github/workflows/`, update the
  `required_status_checks` contexts here and re-import (or edit the ruleset
  in the UI).
