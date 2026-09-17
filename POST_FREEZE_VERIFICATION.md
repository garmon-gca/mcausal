# Post-freeze verification (after `v0.1.0`)

`PUBLIC_RELEASE_CHECKLIST.md` inside tag `v0.1.0` still has two open boxes. Those checks were run **after** the freeze commit and tag. This file records them. **`v0.1.0` was not rewritten.**

Frozen object:

- tag: `v0.1.0`
- commit: `a9d3b56f914568bc32618db0548911cace13744a`

| check | result |
|---|---|
| `git clone --branch v0.1.0 https://github.com/garmon-gca/mcausal.git` | **PASS** |
| tag → exact frozen commit (`v0.1.0^{}` = `a9d3b56`) | **PASS** |
| `pip install .` from that clone | **PASS** |
| tests | **26/26 PASS** |
| `mcausal ref` | **R1/R2/R3 PASS** (live) |
| `mcausal --version` | `0.1.0` |
| JSON + HTML from `examples/minimal_pytorch/run_update_reality.py` | **PASS** |

Python used for this verification: 3.11.9.

Visibility was **not** changed. External usability remains **NOT YET**.
`PUBLIC_VISIBILITY_CHANGE = REQUIRES_EXPLICIT_USER_APPROVAL`
