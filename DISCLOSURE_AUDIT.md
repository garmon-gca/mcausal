# Disclosure audit — mcausal 0.1.0 staging

Scan of the release tree (relative paths only).

Patterns: `C:\AI`, `C:\Users`, `AppData`, `.cursor`, `token`, `secret`, `password`, `api_key`, `Bearer`, `machineId`, `garmon_lbm`, `agent-transcripts`.

Also: binaries, archives, files > 500 KiB.

## Result: PASS

- No `C:\AI` or `C:\Users` in tracked source/docs.
- No `AppData`, `.cursor`, `garmon_lbm`, `agent-transcripts`, `Bearer`, `api_key`, `password`, `machineId`.
- No checkpoints, weights, ZIPs, or files > 500 KiB.
- `__pycache__`, `build/`, `*.egg-info`, pytest cache removed before freeze.

## False positives (kept)

Issue templates contain the word “secret(s)” as **instructions not to paste secrets**. Not credentials.

Author string `garmon-gca` and URL `https://github.com/garmon-gca/mcausal` are intentional public attribution.

## Not in this tree

Research labs, replication-pack run JSON, local corpora, Hugging Face weights, virtualenvs.
