# Feature-based Knowledge Distillation for Monocular Depth Estimation

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: G22
- **Project ID**: 29

## 📝 Project Description
This project implements a Feature-based Knowledge Distillation framework to compress high-capacity Monocular Depth Estimation models into lightweight, edge-compatible architectures. The training pipeline pairs standard ground-truth supervision with advanced feature alignment, transferring rich geometric and spatial representations from a heavy Teacher (e.g., ResNet50) to a tiny Student (e.g., MobileNetV3 or Custom CNN), enabling accurate, real-time depth prediction on resource-constrained devices.

> 📖 **Official Report**: For all theoretical details, performance analysis, the architecture used, and group contributions, please refer to the formal paper: **[REPORT.md](docs/REPORT.md)**.

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

**Prerequisites:**
Ensure you have Anaconda/Miniconda installed. Set up the environment using the following commands:

```bash
git clone https://github.com/PMETGAIC/Feature-based-Knowledge-Distillation-for-Monocular-Depth-Estimation.git
cd your-repo
conda env create -f environment.yml
conda activate dl-project
```

**Dataset:**
The NYU Depth V2 dataset is automatically downloaded and cached via the Hugging Face datasets library upon the first execution. It will reside locally in the data/ directory, requiring no manual intervention.

### 2. Network Training

**Baseline Training:**
```bash
# first run
python -m src.training.train --task baseline --mode train --epoch "#epochs"
# it will be saved in directory './models/checkpoints/baseline/logs/version_x
# to resume
python -m src.training.train --task baseline --mode resume --ckpt "[/directory/*.ckpt]" --epoch "#epochs"
```

**Improved Model Training:**
```bash
python -m src.training.train --task kd --mode train --t_ckpt "[/teacher_directory/*.ckpt]" --epoch "#epochs"
# it will be saved in directory './models/checkpoints/kd/logs/version_x
# to resume
python -m src.training.train --task kd --mode train --ckpt "[/directory/*.ckpt]" --t_ckpt "[/teacher_directory/*.ckpt]" --epoch "#epochs"
```
the epochs number is the "target_epoch", so if you make a resume from the 30th epoch and you want another 30 you have to write 60 
By default, the training script initializes a standard Medium model (ResNet18). You can change the architecture by appending these flags:
 -   (No flag): Medium model (ResNet18).
 -   --teacher: Large model (ResNet50).
 -   --mini: Small compact model (MobileNetV3-Small).
 -   --use_custom_model: Uses the custom CNN trained from scratch for the new model.

**Optional Finetune**
The training pipeline employs a cosine annealing learning rate scheduler that systematically decays the LR over epochs. If you wish to verify that a converged model maintains significant learning capacity, or if you need to adapt it further, you can use the `finetune` mode. This mode resumes training from a checkpoint but resets the learning rate to its initial value (e.g., 1e-4/1e-5), allowing for broader weight updates:

```bash
python -m src.training.train --task baseline --mode finetune --ckpt experiments/checkpoints/baseline/best.ckpt --epoch "#epochs"
```
# Unlike the `resume` mode, in `finetune` mode the trainer starts a new session from epoch 0. Therefore, the `--epoch` argument behaves **additively**. For example, setting `--epoch 10` will train the loaded model for exactly 10 new epochs.


### 3. Evaluation
To reproduce the visual comparisons (`comparison.png`, `pixels_error.png`) and the MAE metrics across all models, run the evaluation script by providing the paths to your saved checkpoints:
# this script plot the comparison between the inference of every model and the ground truth
# Every argument is optional. You can omit any checkpoint if you only want to evaluate a specific subset of models. The command below represents a complete example evaluating all 7 models at once:
```bash
python -m src.evaluation.Visual_comparison \
  --t_grande experiments/checkpoints/t_grande/best.ckpt \
  --med_base experiments/checkpoints/med_base/best.ckpt \
  --med_kd experiments/checkpoints/med_kd/best.ckpt \
  --mini_base1 experiments/checkpoints/mini_base1/best.ckpt \
  --mini_kd1 experiments/checkpoints/mini_kd1/best.ckpt \
  --mini_base2 experiments/checkpoints/mini_base2/best.ckpt \  # this two is for custom architecture
  --mini_kd2 experiments/checkpoints/mini_kd2/best.ckpt        # with backbone trained from scratch
```
# this script simply load the three different architecture (Big/Mid/Tiny) to calculate the velocity (ms/FPS) 
```bash
python -m src.evaluation.Speed_comparison
```




*For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`.*
