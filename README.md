# Adaptive fcHMRF-LIS — Final Simulation Implementation

## Project
Adaptive fcHMRF-LIS with Deep Learning-Based ROI Detection for Neuroimaging Analysis

This package is a **simulation-based research prototype**. It does NOT claim clinical validation or implementation on patient/ADNI data.

## What it implements

1. Synthetic 3D neuroimaging data with known ground truth.
2. Voxel-wise two-group z and p statistics.
3. Lightweight 3D CNN-based ROI detection.
4. Adaptive sparse voxel graph.
5. Adaptive spatial + appearance affinities.
6. Mean-field sparse fcHMRF inference.
7. LIS estimation and LIS-based FDR at alpha=0.05.
8. Benjamini-Hochberg, q-value/Storey-style, and LocalFDR baselines.
9. Main comparison, effect-strength, abnormality-size, and noise-robustness experiments.
10. Monte Carlo repetitions.
11. FDP, FNR, TP, FP, Precision, Recall, Dice, discoveries, runtime.
12. 2D maps and interactive 3D HTML visualization.

## Dataset

The input is generated reproducibly. Each experiment contains:
- 3D volumes: 32 x 32 x 16
- 20 control subjects
- 20 abnormal-condition subjects
- 3 spatial abnormal regions
- known ground-truth abnormality mask
- configurable effect size and noise

The generated dataset is saved as `data/main_dataset.npz`.

## Windows / VS Code run instructions

### 1. Open terminal in this folder

```bat
cd adaptive_fchmrf_lis_final
```

### 2. Create virtual environment

```bat
py -m venv .venv
```

### 3. Activate it

```bat
.venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt, or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 4. Install packages

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the complete implementation

```bat
python main.py
```

The program will:
- generate the dataset,
- train/test the 3D ROI detector,
- run all statistical methods,
- run robustness experiments,
- create CSV tables,
- create figures,
- create `interactive_3d.html`.

### 6. Open the results

Open the `results` folder.

Important files:
- `main_results.csv`
- `main_summary.csv`
- `effect_strength_results.csv`
- `size_results.csv`
- `noise_robustness.csv`
- `monte_carlo_results.csv`
- `method_comparison.png`
- `metric_comparison.png`
- `noise_robustness.png`
- `discovery_maps.png`
- `interactive_3d.html`

## Expected runtime

On a normal student laptop:
- dataset generation: seconds
- ROI CNN: usually under a few minutes on CPU
- main experiment: seconds to a few minutes
- robustness experiments: several minutes

If it is slow, edit `CONFIG` in `main.py`:
- `roi_epochs = 8`
- `mc_reps = 5`
- `strength_reps = 3`
- `size_reps = 3`
- `noise_reps = 3`

For final paper results, use the default values.

## Important scientific wording

Use:
> "The framework was evaluated using controlled synthetic 3D neuroimaging datasets with known ground-truth abnormal regions."

Do NOT write:
> "Clinically validated"
> "Diagnoses Alzheimer's disease"
> "Validated on patients"
> "Implemented on ADNI"

The implementation is a simulation-based prototype designed to evaluate the proposed statistical workflow.
