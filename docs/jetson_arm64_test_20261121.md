# Jetson ARM64 test evidence — 2026-11-21

## Remote hardware results (GitHub bridge)

- Device: Jetson Nano, Ubuntu 18.04, ARM64 (`aarch64`), Python 3.6.9.
- CPU NMS equivalence: 50 scenarios passed; reference 624.160 ms, optimized 374.295 ms, **1.67×** (local NMS workload, not camera-to-UART FPS).
- Python tests before installing pytest: could not start because `python3 -m pytest` was unavailable.
- Python tests after installation, against the source-only public checkout: **127 passed, 7 skipped, 5 failed**. The failures were not evidence of changed production UART behavior:
  - Four tests required private image or zone fixtures excluded from public GitHub; two attempted to use `None` images, two opened an absent zone JSON fixture.
  - One test assumed a fixed first vertex from OpenCV's `approxPolyDP`; Jetson OpenCV returned the same four vertices with a cyclic shift. The downstream quad geometry is indifferent to which vertex begins a contour, but winding and adjacency must be preserved.

## Corrective revision

- Fixture-dependent tests explicitly `skipTest` **only when their source assets are absent**; when assets exist they still run normally. A related test previously iterated zero images and silently passed; it now explicitly skips if no private images exist.
- The quad-corner regression accepts cyclic rotation of the same winding order, and does not accept reversed or changed polygons. No production geometry, model or UART changes.
- On a separate x86 host with current OpenCV: public source-only suite **119 passed / 12 skipped**; with original private `data/` temporarily mounted (never committed), **124 passed / 7 skipped**, both with 0 failed and 45 subtests passed.
- These local counts are not directly comparable to the Nano's pytest collection/plugin counts. The corrected suite must be rerun on Nano after `git pull --ff-only` to establish its own result.

## Scope and missing verification

- Public GitHub intentionally excludes `data/`, `models/`, and binary libraries; production hardware model and end-to-end FPS are untested in this check.
- Keep the private images and model out of GitHub. To run all data-backed tests on Nano, provide the original approved fixture directory locally under `data/`, with appropriate permission; do not create fake image fixtures.
- To rerun from Nano after the source test revision is pulled:

```sh
cd ~/nano
git pull --ff-only
python3 -m pytest -q tests
```

The GitHub bridge only runs its predetermined actions and does not automatically update the source checkout.