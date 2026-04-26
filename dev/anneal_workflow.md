# Trapezoidal anneal-from-checkpoint workflow

The default LR / weight-decay / Muon-momentum schedules in `base_train.py` are
trapezoidal: flat for the first `(1 - warmdown_ratio)` of training, then
linear decay over the warmdown window. This shape supports two
post-training-launch operations without re-warmup or schedule distortion:

1. **Anneal an earlier checkpoint to convergence** -- take a flat-phase
   checkpoint, run only the warmdown.
2. **Extend a finished run** -- take the last pre-warmdown checkpoint of an
   already-finished run, train further, then anneal again.

Both depend on the schedule being flat at the branch point. A checkpoint
that's already been through warmdown can't be extended cleanly because its
optimizer state has cooled down (low LR / WD / momentum); extending from
there means either re-warming up (weird) or training with cold momentum.
For the same reason, the in-place trapezoidal save logic always emits a
checkpoint at `step == warmdown_start` (the last flat step) regardless of
`--save-every`, so every run produces a clean branch point even if you
forget to lower `save_every`.

## Quick reference

Both CLI flags and env vars work; CLI takes precedence. Env-var form keeps
parity with `runs/speedrun_byte.sh`.

```bash
# Anneal (CLI):
uv run python runs/resume.py --source-tag d24-byte --from-step 4500 --mode anneal

# Anneal (env-var):
SOURCE_TAG=d24-byte FROM_STEP=4500 MODE=anneal uv run python runs/resume.py

# Extend:
uv run python runs/resume.py --source-tag d24-byte --from-step 4500 \
    --mode extend --new-total 10000
```

The script inherits depth, `byte_tokenizer`, `max_seq_len`, `device_batch_size`,
`target_param_data_ratio`, `window_pattern`, and `fp8` from the source run's
saved `meta_<step>.json`, so you don't repeat them.

Defaults that you'll usually leave alone:
- `--warmdown-ratio 0.1` -- matches the default in `runs/speedrun_byte.sh`.
- `--save-every 500` -- intermediate checkpoints during the resumed run.
- `--nproc 8`.

Override the output checkpoint dir with `--output-tag <name>` (or `OUTPUT_TAG=`).
Default is `${source}-anneal-from-${from_step}` or `${source}-ext-${new_total}`.

Pass `--skip-eval` to skip the `base_eval` pass after training.

## Recipe: plan-long, stop-early, anneal-later

1. Plan a long run with the default trapezoidal schedule:

   ```bash
   DEPTH=24 TARGET_DATA_RATIO=32 SAVE_EVERY=500 \
       bash runs/speedrun_byte.sh
   ```

   This produces checkpoints at every 500 steps plus the explicit pre-warmdown
   checkpoint plus the final annealed checkpoint.

2. To anneal an earlier checkpoint (say step 4500) and compare against the
   full-length annealed model:

   ```bash
   uv run python runs/resume.py --source-tag d24-byte --from-step 4500 --mode anneal
   ```

   Output lands at `~/.cache/nanochat/base_checkpoints/d24-byte-anneal-from-4500/`,
   eval runs automatically.

3. If the full run looks promising and you want to push further:

   ```bash
   # --from-step must be the last pre-warmdown checkpoint of the original
   # run. If original num_iterations was N and warmdown_ratio was 0.1, that's
   # step N - round(0.1 * N).
   uv run python runs/resume.py --source-tag d24-byte --from-step 9000 \
       --mode extend --new-total 20000
   ```

   Output lands at `d24-byte-ext-20000/`. The original `d24-byte/` checkpoints
   are untouched.

## Why this works (no code changes needed in base_train at runtime)

The optimizer schedules are pure functions of `step`, `num_iterations`, and
`warmdown_ratio` -- all read from CLI args at startup. So:

- A resumed run sees `step = FROM_STEP` (loaded from checkpoint), and the
  schedule we configure via `--num-iterations` and `--warmdown-ratio` is
  evaluated from there.
- For pure anneal mode the script picks `num_iterations = FROM_STEP + anneal_len`
  and `warmdown_ratio = anneal_len / num_iterations`, so `warmdown_start =
  num_iterations - warmdown_iters = FROM_STEP`. The first iteration of the
  resumed run is `step = FROM_STEP = warmdown_start`, i.e. the start of the
  warmdown ramp. Schedule values at that step:
  - `lr_multiplier(FROM_STEP) = 1.0` (top of ramp)
  - `weight_decay(FROM_STEP) = weight_decay_scaled` (full WD)
  - `muon_momentum(FROM_STEP) = 0.97` (top)
- The WD scaling factor `D_REF / target_tokens` cancels the data:param ratio,
  so changing `--target-param-data-ratio` between source and resumed runs
  doesn't distort regularization. Only depth and batch must match.

The new `--resume-from-tag` arg in `base_train.py` is what lets the resumed
run read from the source checkpoint dir while writing to a fresh output dir.
Without it, the resumed run would have to share a directory with the source.

## Gotchas

- **Branch from a flat-phase checkpoint, not a warmdown one.** Resuming from
  a checkpoint that was saved during warmdown means starting with cooled-down
  LR/WD/momentum; the schedule will reset to top-of-ramp but the optimizer
  state mismatch produces a small training disturbance. The script doesn't
  enforce this -- it'll let you do it, but you shouldn't.
- **`--device-batch-size` and `--max-seq-len` must match the source.** They
  determine total tokens per fwd/bwd, which has to divide `total_batch_size`,
  and the dataloader state assumes the same shape. The script inherits both
  from the source meta to enforce this.
- **`window_pattern` and `byte_tokenizer` and `depth` define the model
  architecture.** Loading a state_dict into a different-shape model fails
  loudly, so this isn't a silent footgun, but it's why those are inherited.
- **`fp8` flag is inherited but not strictly required to match.** FP8 is
  about which matmul codepath runs; the stored params are always
  bf16/fp32. Switching it off mid-resume is safe; switching it on requires
  `nanochat.fp8.convert_to_float8_training` to apply, which the resume
  pathway handles correctly.
- **Multiple anneals from the same source checkpoint** are fine -- each one
  writes to a different output dir if you give them different
  `--output-tag`s. By default `--mode anneal` produces a deterministic name
  based on `--from-step`, so re-running the same anneal will overwrite. Pass
  `--output-tag ...` to keep multiple variants (e.g. for sweeping
  `--warmdown-ratio`).
