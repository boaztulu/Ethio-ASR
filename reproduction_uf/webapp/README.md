# Ethio-ASR Demo (Gradio)

Interactive web demo for the reproduced Ethio-ASR models. Loads any
trained CTC checkpoint and exposes three input modes:

1. **Pre-loaded WAXAL validation samples** — one utterance per language
   (Amharic, Tigrinya, Oromo, Sidaama, Wolaytta), shown alongside the
   gold reference for quick visual comparison.
2. **Upload audio** — drag-and-drop or pick a WAV / FLAC / MP3 file.
3. **Microphone** — record directly in the browser.

For every input the app shows:

* the predicted language (from the model's `[LANG]` head token),
* the raw CTC decode,
* the paper post-processed decode (Ge'ez homophone collapse + punctuation
  removal — same code path as in `reproduction_uf/slurm_scripts/evaluate_ctc.py`),
* the audio duration.

## Run on HiPerGator (recommended)

```bash
cd /blue/rcstudents/btulu/Projects/Ethio-ASR
sbatch Ethio-ASR/reproduction_uf/webapp/serve_webapp.sbatch
```

Tail the log to find the assigned node:

```bash
tail -f logs/ethio-demo_<jobid>.out
```

You will see a line like:

```
Running on local URL: http://0.0.0.0:7860
```

On **your laptop**, open an SSH tunnel:

```bash
ssh -L 7860:<node-name>:7860 btulu@hpg.rc.ufl.edu
```

Then open <http://localhost:7860> in your browser.

To use a different trained model:

```bash
sbatch --export=ALL,MODEL_DIR=/blue/.../models/facebook/mms-1b-09062026-203848 \
       Ethio-ASR/reproduction_uf/webapp/serve_webapp.sbatch
```

## Run on a HiPerGator OnDemand desktop

If you launched a Desktop / Open OnDemand session with a GPU, simply
open a terminal in the desktop and run:

```bash
module load pytorch/2.8.0
source /blue/rcstudents/btulu/Projects/Ethio-ASR/slurm_scripts/setup_env.sh
python3 /blue/rcstudents/btulu/Projects/Ethio-ASR/Ethio-ASR/reproduction_uf/webapp/app.py \
    --port 7860
```

then open <http://localhost:7860> in the desktop's Firefox.

## Public share URL

Add `--share` to the `app.py` command line (or pass `share=True` via env)
to get a `gradio.live` tunnel URL that works without SSH forwarding.
Requires outbound internet from the compute node; default HiPerGator
nodes do have it.

## Notes

* First run caches one validation sample per language under
  `webapp/samples/` (~1 MB total). Subsequent runs reuse the cache.
* Default model is `w2v-bert-2.0` (the paper's best). Override with
  `--model_dir` to demo any other checkpoint we trained.
* Audio is internally resampled to 16 kHz mono.
