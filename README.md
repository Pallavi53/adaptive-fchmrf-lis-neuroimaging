\# Adaptive fcHMRF-LIS with Deep Learning-Based ROI Detection for Neuroimaging Analysis



\## Overview



This project presents an implementation-oriented framework for brain abnormality detection from simulated 3D neuroimaging data by combining \*\*Deep Learning-based Region of Interest (ROI) detection\*\* with an \*\*Adaptive Sparse Fully Connected Hidden Markov Random Field–Local Index of Significance (fcHMRF-LIS)\*\* framework.



The proposed approach integrates voxel-wise statistical testing, spatial and appearance information, adaptive sparse graph construction, mean-field inference, and LIS-based False Discovery Rate (FDR) control.



The framework is evaluated using controlled synthetic 3D neuroimaging datasets with known abnormal regions.



> \*\*Note:\*\* This is a simulation-based research prototype and is not a clinical diagnostic system or clinical validation study.



\---

\---



\## Experimental Results



The framework was evaluated using controlled synthetic 3D neuroimaging experiments. The experiments examine detection performance, comparison with baseline multiple-testing methods, and robustness under increasing noise.



\### Method Comparison



The proposed Adaptive Sparse fcHMRF-LIS framework is compared with Benjamini-Hochberg (BH), q-value, and LocalFDR methods using detection and false-discovery metrics.



!\[Method Comparison](method\_comparison.png)



\### Abnormality Detection Maps



The detected abnormal regions are visualized against the simulated neuroimaging data to demonstrate spatial localization of the detected abnormalities.



!\[Discovery Maps](discovery\_maps.png)



\### Noise Robustness



The framework is evaluated under different noise levels to study its false-discovery control and detection performance as noise increases.



!\[Noise Robustness](noise\_robustness.png)





\## Objectives



1\. Develop a deep learning-based ROI detection mechanism combined with an Adaptive Sparse fcHMRF-LIS framework for brain abnormality detection.



2\. Analyze voxel-wise neuroimaging data by incorporating spatial information, adaptive model parameters, and False Discovery Rate (FDR) control.



3\. Compare the proposed method with existing multiple-testing approaches and evaluate detection performance, robustness, and statistical stability.



4\. Visualize detected abnormal regions using three-dimensional representations.



\---



\## Proposed Methodology



The overall workflow of the proposed framework is:



```text

Synthetic 3D Neuroimaging Data

&#x20;           ↓

&#x20;     Preprocessing

&#x20;           ↓

&#x20;  Voxel-wise Statistical Testing

&#x20;           ↓

&#x20;Deep Learning-based ROI Detection

&#x20;           ↓

&#x20;Adaptive Sparse Graph Construction

&#x20;           ↓

&#x20;Adaptive Spatial + Appearance Affinity

&#x20;           ↓

&#x20;Sparse fcHMRF Mean-Field Inference

&#x20;           ↓

&#x20;       LIS Estimation

&#x20;           ↓

&#x20;     LIS-based FDR Control

&#x20;           ↓

&#x20;  Abnormality Detection Results

&#x20;           ↓

&#x20;Comparison + Evaluation + 3D Visualization

```



\---



\## Key Components



\### 1. Synthetic 3D Neuroimaging Dataset



The implementation generates controlled 3D neuroimaging data containing:



\* Control subjects

\* Abnormal-condition subjects

\* Brain-shaped regions

\* Known abnormal spherical regions

\* Controlled signal strength and noise



The known abnormal regions provide ground truth for quantitative evaluation.



\### 2. Voxel-wise Statistical Testing



Voxel-wise group comparisons are performed to obtain statistical evidence for abnormality at individual voxels.



The resulting statistical values and probabilities are used as input evidence for the subsequent spatial modeling and multiple-testing procedures.



\### 3. Deep Learning-based ROI Detection



A lightweight 3D convolutional neural network is used to identify candidate regions of interest.



The ROI prediction is incorporated as a soft prior so that the model can use learned regional information without completely suppressing strong voxel-wise statistical evidence.



\### 4. Adaptive Sparse Graph Construction



Instead of constructing a dense fully connected matrix, the implementation uses an adaptive sparse k-nearest-neighbor graph.



The graph incorporates:



\* Spatial proximity

\* Image/feature similarity

\* Adaptive affinity values



This reduces computational and memory requirements compared with a dense graph representation.



\### 5. Sparse fcHMRF Mean-Field Inference



The sparse graph is incorporated into a fully connected HMRF-inspired spatial model.



Mean-field inference is used to estimate the probability of abnormal and null states while incorporating information from neighboring voxels.



\### 6. LIS-based FDR Control



The Local Index of Significance (LIS) is estimated from the model probabilities.



The LIS values are used to control the empirical false discovery proportion at a specified significance level.



The implementation uses an FDR level of:



```text

α = 0.05

```



\### 7. Baseline Methods



The proposed method is compared with:



\* Benjamini-Hochberg (BH)

\* q-value / Storey method

\* LocalFDR



\---



\## Experimental Evaluation



The implementation performs multiple experiments to study the behavior of the proposed framework.



\### Monte Carlo Evaluation



Repeated simulations are performed to evaluate:



\* False Discovery Proportion (FDP)

\* False Negative Rate (FNR)

\* Precision

\* Recall

\* Dice coefficient

\* True Positives (TP)

\* Runtime



\### Effect Strength Experiment



The abnormality effect strength is varied to study how detection performance changes with signal strength.



Tested effect-strength levels include:



```text

0.4

0.6

0.8

1.0

1.2

```



\### Abnormality Size Experiment



The size of abnormal regions is varied to evaluate the sensitivity of the framework to different abnormality sizes.



\### Noise Robustness Experiment



Different noise levels are introduced to evaluate the stability of the detection framework under increasing noise.



\---



\## Evaluation Metrics



The following metrics are calculated:



| Metric    | Purpose                                                  |

| --------- | -------------------------------------------------------- |

| TP        | Correctly detected abnormal voxels                       |

| FP        | Incorrectly detected normal voxels                       |

| FN        | Missed abnormal voxels                                   |

| TN        | Correctly identified normal voxels                       |

| FDP       | Proportion of detected voxels that are false discoveries |

| FNR       | Proportion of abnormal voxels that are missed            |

| Precision | Accuracy of positive detections                          |

| Recall    | Ability to detect abnormal voxels                        |

| Dice      | Overlap between detected and ground-truth regions        |

| Runtime   | Computational execution time                             |



\---



\## Implementation Technologies



\### Programming Language



\* Python



\### Scientific Computing



\* NumPy

\* SciPy

\* pandas



\### Deep Learning



\* PyTorch



\### Neuroimaging / Data Processing



\* NIfTI-compatible workflow components

\* Synthetic 3D neuroimaging data



\### Visualization



\* Matplotlib

\* Plotly



\### Development Environment



\* Python Virtual Environment

\* Windows Command Prompt

\* Git

\* GitHub



\---



\## Project Structure



```text

adaptive\_fchmrf\_lis\_final/

│

├── main.py

├── main\_backup.py

├── requirements.txt

├── README.md

├── run\_final.bat

│

├── data/

│   └── Generated simulation dataset

│

└── results/

&#x20;   ├── CSV evaluation results

&#x20;   ├── Generated figures

&#x20;   └── Interactive 3D visualization

```



Generated datasets and experimental results are excluded from Git using `.gitignore`.



\---



\## Installation



\### 1. Clone the Repository



```bash

git clone https://github.com/YOUR\_USERNAME/adaptive-fchmrf-lis-neuroimaging.git

cd adaptive-fchmrf-lis-neuroimaging

```



Replace `YOUR\_USERNAME` with your GitHub username.



\### 2. Create a Virtual Environment



```bash

py -m venv .venv

```



\### 3. Activate the Environment



Windows:



```bash

.venv\\Scripts\\activate

```



\### 4. Install Dependencies



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\---



\## Running the Project



\### Option 1: Run using Python



```bash

python main.py

```



\### Option 2: Run using the batch file



```bash

run\_final.bat

```



The program generates the simulation dataset, performs the proposed analysis, executes the comparison experiments, and stores the results in the `results/` directory.



\---



\## Generated Results



The implementation generates:



\* Main comparison results

\* Monte Carlo summary

\* Effect-strength experiment results

\* Abnormality-size experiment results

\* Noise-robustness results

\* Performance figures

\* Interactive 3D visualization



The interactive visualization is generated as:



```text

results/interactive\_3d.html

```



\---



\## Results Interpretation



The experiments are designed to evaluate the trade-off between false discoveries and abnormality detection.



The proposed Adaptive Sparse fcHMRF-LIS framework is intended to provide a strong balance between:



\* False-positive control

\* Detection sensitivity

\* Spatial consistency

\* Abnormal-region localization



In the simulation experiments, the proposed framework generally maintains a low empirical FDP while achieving improved detection performance compared with the tested non-spatial baseline methods.



LocalFDR can provide higher recall in some experiments, but with a substantially higher false-discovery proportion. Therefore, the proposed method focuses on achieving a better balance between detection and false-positive control.



\---



\## Reproducibility



The implementation uses controlled random seeds to make the simulation experiments reproducible.



All major experiments are executed from:



```text

main.py

```



The generated datasets and result files can be reproduced by running the project again.



\---



\## Limitations



\* The current evaluation uses simulated 3D neuroimaging data.

\* The implementation is a research prototype rather than a clinical diagnostic tool.

\* Performance on real patient datasets has not been established by this implementation.

\* The sparse graph and mean-field inference are designed as computationally practical approximations.

\* The experimental results should not be interpreted as clinical evidence.



\---



\## Future Work



Future extensions may include:



\* Evaluation on real neuroimaging datasets.

\* Integration with standard NIfTI-based neuroimaging pipelines.

\* More advanced 3D CNN/U-Net architectures for ROI detection.

\* Further optimization of adaptive graph construction.

\* Advanced spatial inference methods.

\* Larger-scale experiments on real-world datasets.

\* Interactive analysis of detected brain abnormalities.



\---



\## Research Focus



The project focuses on the integration of:



```text

Deep Learning

&#x20;     +

Neuroimaging Analysis

&#x20;     +

Spatial Statistical Modeling

&#x20;     +

Multiple Testing

&#x20;     +

False Discovery Rate Control

&#x20;     +

3D Abnormality Visualization

```



\---



\## Disclaimer



This project is developed for academic and research purposes. The generated results are based on controlled simulation experiments and should not be used for medical diagnosis, treatment decisions, or clinical interpretation.



