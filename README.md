# Feature Fusion and Knowledge-Distilled Multi-Modal Multi-Target Detection
**Ngoc Tuyen Do and Tri Nhu Do**

- [tuyen.dn242305m@sis.hust.edu.vn](https://scholar.google.com/citations?hl=en&user=o8w6IZ4AAAAJ)
- [tri-nhu.do@polymtl.ca](https://scholar.google.com/citations?hl=en&user=cwdP-oYAAAAJ)

## Abstract
In the surveillance and defense domain, multi-target detection and classification (MTD) is considered essential yet challenging due to heterogeneous inputs from diverse data sources and the computational complexity of algorithms designed for resource-constrained embedded devices, particularly for AI-based solutions. To address these challenges, we propose a knowledge-distilled fusion framework for multi-modal MTD that leverages data fusion to enhance accuracy and employs knowledge distillation for improved domain adaptation. Specifically, our approach utilizes both RGB and thermal image inputs within a novel fusion-based multi-modal model, coupled with a distillation training pipeline. We formulate the problem as a posterior probability optimization task, which is solved through a multi-stage training pipeline supported by a composite loss function. This loss function effectively transfers knowledge from a teacher model to a student model. Experimental results demonstrate that our student model achieves approximately 95% of the teacher model’s mean Average Precision while reducing inference time by approximately 50%, underscoring its suitability for practical MTD deployment scenarios.

## Code

## Run Locally

Clone the project

```bash
  git clone https://github.com/tnd-lab/Feature-Fusion-Knowledge-Distilled-Multi-Modal-Multi-Target-Detection.git
```

Go to the project directory

```bash
  cd Feature-Fusion-Knowledge-Distilled-Multi-Modal-Multi-Target-Detection
```

Install requirements

```bash
  # create virtual environment
  python3 -m venv .env
  # active venv
  source .env/bin/activate
  # install packages with pip
  pip3 instal -r requirements.txt
```
## Dataset FLIR Aligned
Download from [my drive](https://drive.google.com/file/d/1i6iWs2OUVbbKEUOEnORfBJpoI525Guwy/view?usp=sharing)
Unzip and put it in folder dataset
```
  unzip FLIR_Aligned.zip
  cd /path/to/project/
  mkdir dataset
  cp /path/to/dataset/FLIR_Aligned /path/to/project/dataset/FLIR_Aligned
```