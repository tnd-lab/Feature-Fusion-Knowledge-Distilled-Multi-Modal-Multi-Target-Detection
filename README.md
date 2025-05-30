# Feature Fusion and Knowledge-Distilled Multi-Modal Multi-Target Detection
**Ngoc Tuyen Do and Tri Nhu Do**

## Abstract
In the surveillance and defense domain, multi-target detection and classification (MTD) is considered essential yet challenging due to heterogeneous inputs from diverse data sources and the computational complexity of algorithms designed for resource-constrained embedded devices, particularly for AI-based solutions. To address these challenges, we propose a knowledge-distilled fusion framework for multi-modal MTD that leverages data fusion to enhance accuracy and employs knowledge distillation for improved domain adaptation. Specifically, our approach utilizes both RGB and thermal image inputs within a novel fusion-based multi-modal model, coupled with a distillation training pipeline. We formulate the problem as a posterior probability optimization task, which is solved through a multi-stage training pipeline supported by a composite loss function. This loss function effectively transfers knowledge from a teacher model to a student model. Experimental results demonstrate that our student model achieves approximately 95% of the teacher model’s mean Average Precision while reducing inference time by approximately 50%, underscoring its suitability for practical MTD deployment scenarios.

## Code
