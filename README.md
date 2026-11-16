# CompassNet: Orientation-Aware Network for Solar Filament Segmentation

This repository contains a solution submitted to the [Solar Filament Segmentation Challenge 2026](https://kaggle.com/competitions/filament-segmentation-2026), a Kaggle competition organized by the Earth-Space AI Research (ESAIR) Lab. The challenge consists of the automatic segmentation of solar filaments in GONG H-alpha observations and uses the MAGFiLO dataset.

The submitted model is CompassNet. The repository includes the source code, the trained weights, a notebook with the entire pipeline and the list of packages (`requirements.txt`). The pipeline can also be found publicly on Kaggle: [Orientation-Aware Net for Filament Segmentation](https://www.kaggle.com/code/hermengardo/orientation-aware-net-for-filament-segmentation). The solution is described in the report: [`report_compass.pdf`](report_compass.pdf).

## Repository guide

```
.
├── training.ipynb            # the whole pipeline
├── self-evaluation-notebook.ipynb  # evaluation notebook, used for the validation results
├── LastSubmited.csv          # submitted predictions (submission format)
├── models/
│   ├── cnet@1024x567.pt      # final weights, used for the submission
│   ├── DistilSSL.pt          # backbone pretrained with distillation (LightlyTrain)
│   └── MaskedSSL.pt          # backbone pretrained with masked image completion
├── src/
│   ├── CompassNet.py         # the model architecture
│   ├── Config.py             # configuration parameters and directory paths
│   ├── Loss.py               # loss functions (segmentation and spine)
│   ├── Pretraining.py        # self-supervised pretraining (masked image completion)
│   ├── PretrainingLightlyTrain.py # self-supervised pretraining via distillation (LightlyTrain)
│   ├── SunDataset.py         # dataset and soft-target masking
│   ├── training.py           # standalone script for training CompassNet
│   └── utils.py              # image processing and evaluation helper functions
├── data/
│   ├── SILSO.csv             # sunspot numbers, used for the solar cycle-stratified split
│   ├── images_table.csv      # metadata table of the GONG images used in pretraining
│   ├── ssl_post_prep.py      # preprocessing script for pretraining images
│   └── README.md             # detailed documentation of the dataset
├── latex/                    # LaTeX source code and assets for the technical report
├── report_compass.pdf        # compiled technical report of the solution
└── requirements.txt          # Python dependencies and exact package versions
```

## Reproducing the submission

The final weights (`models/cnet@1024x567.pt`) are stored in the repository and loaded by the notebook, so **no re-training is needed**.

1. `pip install -r requirements.txt` (Python 3.13)
2. Download the competition data and set the paths in the `Config` cell of `training.ipynb`.

The training cell in the notebook runs only 1 epoch, as a demo. The real training (100 epochs) produced the stored weights.

## Pretraining

Before the supervised training, the ResNet-34 backbone was evaluated with two self-supervised learning (SSL) strategies on 5,000 GONG images outside the MAGFiLO period (avoiding data leakage):

### 1. Masked Image Completion (`models/MaskedSSL.pt`)
- **Method:** Reconstructs randomly masked image regions with an auxiliary loss predicting the multi-scale orientation field extracted by Kirsch filters.
- **Training:** Up to 50 epochs starting from ImageNet weights.
- **Code:** `src/Pretraining.py`

### 2. Self-Supervised Distillation (`models/DistilSSL.pt`)
- **Method:** Distillation-based SSL using the [LightlyTrain](https://github.com/lightly-ai/lightly-train) framework.
- **Training:** 300 epochs starting from ImageNet weights.
- **Code:** `src/PretrainingLightlyTrain.py`

## Validation results

Evaluated with `self-evaluation-notebook.ipynb`.

| Metric | CompassNet |
| --- | --- |
| Panoptic Quality (PQ) | 0.3537 |
| Mean IoU | 0.498 |
| Mean Dice | 0.627 |
