# Ethio-ASR — UF HiPerGator Reproduction

An independent reproduction of the *Ethio-ASR* paper (Abdullah et al., 2026)
carried out at the University of Florida on the HiPerGator HPC cluster
(NVIDIA B200 GPUs, SLURM).

This directory contains everything needed to reproduce the paper's
results on a SLURM/HPC environment, plus a few negative-result candidate
experiments and an interactive Gradio demo.

> **For the methodology, results, and discussion**, see
> [`paper/paper.md`](paper/paper.md). The TL;DR is in `paper/paper.md`
> Section 4.3 (Table 3): three of four reproductions match or beat the
> authors' own published checkpoints under identical evaluation.

## Layout

```
reproduction_uf/
├── paper/                # Reproduction paper (paper.md)
├── configs/              # 7 YAML training configs + generator
├── slurm_scripts/        # SLURM jobs for download, training, eval, aggregation
├── webapp/               # Gradio demo (app.py + serve_webapp.sbatch)
└── results/
    ├── paper_published/      # Eval of paper's HF checkpoints (raw)
    ├── paper_published_pp/   #   ... with paper post-processing
    └── ours/                 # Eval of our 6 trained checkpoints
```

## Quick start

```bash
# 1. Get a HiPerGator shell, then:
module load pytorch/2.8.0
source /blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts/setup_env.sh
# (the script lives under the project root, NOT inside the git repo)

# 2. Download the WAXAL Ethiopian dataset (one-time, ~58 GB)
sbatch slurm_scripts/download_waxal.sbatch

# 3. Pre-fetch all pretrained encoders (one-time, ~42 GB)
sbatch slurm_scripts/prefetch_models.sbatch

# 4. Train all 6 CTC models (4 paper baselines + 2 candidates)
./slurm_scripts/launch_all.sh        # ~12-24 h each on a single B200

# 5. Evaluate on test set with paper post-processing
sbatch slurm_scripts/eval_all_postproc.sbatch

# 6. Print the comparison table
python3 slurm_scripts/aggregate_results.py
```

## Where things live (outside the repo)

To keep this git directory clean for `git push`, the following large /
private artefacts live in the project root (`../`) rather than under
`reproduction_uf/`:

* `../../.env`              — HF token, wandb config (file mode 600)
* `../../data/`             — local dataset cache (currently unused —
                              we load straight from `../../hf_cache/hub/`)
* `../../hf_cache/`         — ~162 GB combined dataset + pretrained models
* `../../models/`           — trained checkpoint weights (~50 GB total)
* `../../logs/`             — every SLURM stdout/stderr we generated
* `../../pylibs/`           — pip-installed Python deps
* `../../tmp_cache/`        — numba / librosa / matplotlib scratch
* `../../wandb_cache/`      — wandb run dir

The scripts under `slurm_scripts/` reference these paths absolutely, so
they keep working after `git clone`. The web app caches its own
samples next to `webapp/samples/`.

## Reproducing on a non-HiPerGator cluster

Replace the SLURM headers in `slurm_scripts/*.sbatch` with your
scheduler's equivalents. The Python code in `evaluate_ctc.py`,
`train_whisper.py`, and `webapp/app.py` is cluster-agnostic.
The only HiPerGator-specific assumption is the `pytorch/2.8.0` module
plus the small set of extra pip packages documented in
`paper/paper.md` Section 3.2.

## Live web demo

See [`webapp/README.md`](webapp/README.md) for instructions on
launching the Gradio demo and accessing it via SSH port-forward or
the public `gradio.live` tunnel.
