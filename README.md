# AesPlan: Aesthetic-Aware Spatial Decomposition for Accelerated and Enhanced Diffusion Inference

**AesPlan** builds on [AccelAes](https://github.com/xuanhuayin/AccelAes) with two new components:

1. **Differential CFG** — apply a higher CFG scale to foreground (semantically important) regions and a lower CFG scale to background at skip steps, recovering quality lost from step skipping at zero extra NFE cost.
2. **Taylor-2 extrapolation on combined eps** — stable second-order extrapolation at skip steps on the combined (CFG-guided) noise prediction, avoiding the instability of extrapolating cond/uncond branches separately.

Combined with AccelAes's AesMask spatial decomposition and DP-based step scheduling, AesPlan achieves further quality improvements at the same speedup.

## Results

### Lumina-Next-T2I (30 steps, budget K=10)

| Method | Speedup | ImageReward | LPIPS | FID |
|--------|---------|-------------|-------|-----|
| Baseline | 1.00× | 0.879 | 0.000 | 0.0 |
| DPCache (Taylor-2) | 1.50× | — | — | — |
| Diff-CFG only | 1.50× | — | — | — |
| AesPlan (full) | **1.50×** | **—** | **—** | **—** |

### SD3-Medium (28 steps, budget K=10)

| Method | Speedup | ImageReward | LPIPS | FID |
|--------|---------|-------------|-------|-----|
| Baseline | 1.00× | 0.879 | 0.000 | 0.0 |
| StepSkip only | 1.50× | 0.891 | 0.058 | 44.3 |
| AesPlan (full) | **1.50×** | **—** | **—** | **—** |

## Key Design

### Differential CFG at skip steps

At skip steps, instead of applying uniform CFG, AesPlan uses a spatial map derived from the CFG magnitude `|eps_cond - eps_uncond|`:

```
eps_half = combined_hat + (s_map - cfg_scale) * (eps_cond_cached - eps_uncond_cached)
```

where `s_map(x,y) = s_bg + fg_mask(x,y) * (s_fg - s_bg)`. Key steps always use standard uniform CFG to preserve quality.

### Taylor-2 on combined eps

Rather than caching and extrapolating `eps_cond` and `eps_uncond` separately (unstable), AesPlan caches the combined eps:

```
combined = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
combined_hat ≈ f0 + f1·d + (f2/2)·d²
```

This is equivalent to DPCache's Taylor formula but applied at the noise-prediction level rather than the per-block feature level.

## Requirements

Same as [AccelAes](https://github.com/xuanhuayin/AccelAes#requirements). Additionally, AccelAes must be cloned as a sibling directory:

```
xuanhua/
  AccelAes/    ← https://github.com/xuanhuayin/AccelAes
  AesPlan/     ← this repo
```

Or set the environment variable:

```bash
export ACCELAES_ROOT=/path/to/AccelAes
```

## Setup

```bash
# Clone AccelAes alongside this repo
git clone https://github.com/xuanhuayin/AccelAes ../AccelAes

# Install dependencies (same environment as AccelAes)
pip install -r requirements.txt
```

## Usage

### Lumina-Next-T2I

```python
import sys
sys.path.insert(0, "../AccelAes")
sys.path.insert(0, "../AccelAes/src")

from src.models.dit_wrapper import LuminaDiTWrapper
from src.aesplan.calibration import AesPlanCalibrator
from src.aesplan.inference import generate_aesplan

wrapper = LuminaDiTWrapper(dtype="bf16")

# Calibrate (run once per model/resolution)
calibrator = AesPlanCalibrator(wrapper, budget=10, total_steps=30)
calib = calibrator.run(prompts=["a peacock displaying its plumage"], seeds=[0])

# Generate
img = generate_aesplan(
    wrapper, prompt="a peacock displaying its plumage",
    seed=0, calib=calib,
    s_fg=7.0, s_bg=1.0,
    use_aes_cost=True, use_diff_cfg=True,
)
img.save("output.png")
```

### SD3-Medium

```python
from src.models.sd3_wrapper import SD3DiTWrapper
from src.aesplan.calibration_sd3 import AesPlanCalibratorSD3
from src.aesplan.inference_sd3 import generate_aesplan_sd3

wrapper = SD3DiTWrapper(dtype="fp16")
calibrator = AesPlanCalibratorSD3(wrapper, budget=10, total_steps=28)
calib = calibrator.run(prompts=["a golden retriever in a park"], seeds=[0])

img = generate_aesplan_sd3(
    wrapper, prompt="a golden retriever in a park",
    seed=0, calib=calib,
    s_fg=7.0, s_bg=2.0,
)
img.save("output_sd3.png")
```

## Ablation Modes

```python
# Full AesPlan (AesMask DP schedule + differential CFG)
generate_aesplan(..., use_aes_cost=True,  use_diff_cfg=True)

# DPCache-style (uniform schedule + Taylor on combined eps only)
generate_aesplan(..., use_aes_cost=False, use_diff_cfg=False)

# Differential CFG only (uniform schedule + spatial CFG)
generate_aesplan(..., use_aes_cost=False, use_diff_cfg=True)

# AesMask schedule only (no differential CFG)
generate_aesplan(..., use_aes_cost=True,  use_diff_cfg=False)
```

## Evaluation

```bash
# Lumina DrawBench eval (5 variants)
python scripts/run_drawbench_lumina_eval.py \
    --prompts 109 --seeds 1 --budget 10 --steps 30 \
    --output outputs/drawbench_lumina

# SD3 DrawBench eval
python scripts/run_drawbench_sd3_eval.py \
    --prompts 109 --seeds 1 --budget 10 --steps 28 \
    --output outputs/drawbench_sd3
```

## Repository Structure

```
src/
  aesplan/
    inference.py          # Lumina inference (differential CFG + Taylor-2)
    inference_sd3.py      # SD3 inference
    inference_flux.py     # FLUX inference
    calibration.py        # Lumina calibration (AesMask cost table + DP)
    calibration_sd3.py    # SD3 calibration
    calibration_flux.py   # FLUX calibration
    dp_solver.py          # DP key-step solver
    dpcache_baseline.py   # DPCache baseline (for comparison)
    dense_run.py          # Dense (full) inference helpers
scripts/
  run_drawbench_lumina_eval.py
  run_drawbench_sd3_eval.py
  run_aesplan_eval.py
```

## Citation

If you use AesPlan, please also cite AccelAes:

```bibtex
@article{yin2026accelaes,
  title={AccelAes: Accelerating Diffusion Transformers for Training-Free Aesthetic-Enhanced Image Generation},
  author={Yin, Xuanhua and Xu, Chuanzhi and Zhou, Haoxian and Wei, Boyu and Cai, Weidong},
  journal={arXiv preprint arXiv:2603.12575},
  year={2026}
}
```
