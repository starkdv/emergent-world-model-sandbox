# Research paper

`emergent_world_model.pdf` — the research-style write-up of the implemented,
measured work (architecture competition, the learned world model, and the
planning + curiosity ablation). This is the canonical, citable version; a
markdown mirror lives at [`../docs/PAPER.md`](../docs/PAPER.md).

- **Source:** `paper.html` (self-contained, two-column academic layout, no
  external fonts/CDN).
- **Author:** Karan Vasa.

## Rebuild the PDF

The PDF is typeset by Chromium's print engine (no LaTeX needed). With the
pre-installed Playwright Chromium:

```bash
python3 - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    pg = b.new_page()
    pg.goto("file://" + __import__("os").path.abspath("paper/paper.html"),
            wait_until="networkidle")
    pg.pdf(path="paper/emergent_world_model.pdf", format="Letter",
           print_background=True,
           margin={"top":"0in","bottom":"0in","left":"0in","right":"0in"})
    b.close()
PY
```

(If `chromium` isn't on the default Playwright path, pass
`executable_path=...`.) Every number in the paper is sourced from the committed
run artifacts under `../docs/sample_competition/`, `../docs/sample_world_model/`,
and `../docs/sample_planning_curiosity/`.
