# Research-project — Claude Code context

Primary work: `research/NCMAPSS/` — N-CMAPSS turbofan RUL, two-stage self-supervised chain (MOGP residuals → confidence-gated onset → healthy-range retraining → HI → RUL).
**Start every session by reading `research/NCMAPSS/RESEARCH_LOG.md` (section 0: current state).**

## Environment
- Python: conda env `pt_prac` (`environment.yml`); run scripts from `research/NCMAPSS/experiments/` with `DS=`, `SEEDS_ONLY=`, `STATS_ONLY=`, `NTHREADS=`, `NJOBS=` env vars as documented in each script header.
- Data: N-CMAPSS h5 files at `$NCMAPSS_DATA` (not in git). Caches `cache*.npz` and GP models `*.pt` are rebuilt locally (not in git).
- Documents: `research/NCMAPSS/paper/` (node `docx` — `npm install` once).

## Rules (never relax)
- Selection on dev only. Any contact with a test split is a numbered "opening": pre-register (script header), count, disclose in the ledger — even single-seed screens and re-summaries.
- Sweeps on absolute axes past the minimum until collapse; tie bands, not nominal minima; all seeds must finish; no blank table cells; report results regardless of direction; no post-hoc repair.
- Figure labels in English, no parentheses in labels; no LaTeX in chat (unicode).
- Do not regenerate `research_overview.pptx`; edit its XML only.

## Hardware policy
Vecchia GP fits use ~1.7 cores per process: always run independent (dataset × seed) jobs as concurrent processes, and use `stats_parallel` / `point_stats_parallel` (chunk pools, identical results) for inference. Keep the machine busy; plug in power; prevent sleep.
