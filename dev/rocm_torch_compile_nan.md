# torch.compile NaN bug on RDNA 4 (gfx1201)

## Hardware / Software
- GPU: AMD Radeon AI PRO R9700 (gfx1201, RDNA 4), 32GB VRAM
- ROCm: 7.2.1
- PyTorch: 2.11.0+rocm7.2
- triton-rocm: 3.6.0

## Symptoms
- Training with `torch.compile` produces NaN loss at step 2 (steps 0-1 are fine)
- Without `torch.compile` (`TORCH_COMPILE_DISABLE=1`), training runs correctly
- fp8 is not involved (disabled)
- bf16 matmuls, model forward/backward, and optimizer math all work correctly in eager mode
- The compiled optimizer kernels (`muon_step_fused`, `adamw_step_fused`) also pass in isolation

## Repro (requires nanochat repo)
```bash
# NaN at step 2:
python -m scripts.base_train --depth=6 --window-pattern=L --num-iterations=5 \
    --device-batch-size=4 --total-batch-size=16384 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1

# Works fine:
TORCH_COMPILE_DISABLE=1 python -m scripts.base_train --depth=6 --window-pattern=L --num-iterations=5 \
    --device-batch-size=4 --total-batch-size=16384 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1
```

## What we tested
- bf16 matmul: works
- torch.compile with simple model: works
- model forward/backward (eager): works, no NaN grads
- muon_step_fused compiled kernel (isolated): works
- Full training loop with torch.compile: NaN at step 2

## Next steps to narrow down
- Determine if it's the compiled model forward, compiled optimizer, or their interaction
- Could disable compile on just the model or just the optimizer separately
- Try `torch.compile(model, backend="aot_eager")` to check if it's a Triton codegen issue vs graph capture
- Build a standalone repro (no nanochat dependency) for a pytorch issue

## Workaround
`TORCH_COMPILE_DISABLE=1` — ~2x slower (26k vs 51k tok/sec) but trains correctly.
