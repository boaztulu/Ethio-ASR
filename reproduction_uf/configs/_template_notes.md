# Config notes (not loaded by training code)

Paper hyperparams to match (Section 5.1):
- 7 epochs, effective batch 32 -> ~36.8k steps
- AdamW, LR tuned over {3e-5, 7e-5, 3e-4, 7e-4}, linear warmup 10%
- Frozen conv feature extractor
- bfloat16, except AfriHuBERT uses float32
- Eval every 800 steps, select best by 0.5*WER + 0.5*CER

Note: actual training samples in WAXAL ~= 197k. With effective batch 32
that gives ~6.2k steps/epoch. 7 epochs ≈ 43k steps (paper says 36.8k,
so they likely truncated slightly with max_steps or shuffled to ~168k/epoch).
We set max_steps=36800 to match paper.

Per-model LR picks (based on common practice / existing repo defaults):
- AfriHuBERT (94M, HuBERT):  3e-4 (small models like higher LR)
- MMS-300M:                  3e-4 (mms is robust to higher LR)
- MMS-1B:                    7e-5 (1B model needs gentler LR)
- w2v-bert-2.0 (600M):       3e-5 (matches paper's best from existing config)
- XLS-R-1B:                  7e-5 (similar size to MMS-1B)
- w2v-bert-2.0-extended:     3e-5 (same as paper but longer + SpecAug)
- Whisper-large-v3 + LoRA:   1e-4 (LoRA tolerates higher LR than full FT)
