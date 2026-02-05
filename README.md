# FaceScore

Deep learning models for clinical prediction from facial images, combining computer vision with survival analysis and clinical data.

## Overview

FaceScore trains neural networks to predict clinical outcomes (age, mortality risk) from facial images. The framework supports:

- **Face-based models**: Image classification/regression using CNNs (EfficientNet, Xception)
- **Survival analysis**: Cox proportional hazards models for time-to-event prediction
- **Unified models**: Integration of image predictions with clinical data (vitals, blood tests)
- **Cross-validation**: Built-in support for k-fold cross-validation
- **Ensemble learning**: Multiple model training for robust predictions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/indra-icmm/facescore.git
cd facescore

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set environment variables for data paths:

```bash
export DATA_DIR=/path/to/your/data
export MODEL_DIR=/path/to/your/models
export OUT_DIR=/path/to/your/output
```

### Training

Train a face-based mortality prediction model:

```bash
CUDA_VISIBLE_DEVICES=0 python code/train_faces.py
```

Generate predictions:

```bash
python code/predict_faces.py
```

Train unified model combining face predictions with clinical data:

```bash
python code/train_unified.py
```

## Project Structure

```
facescore/
├── code/
│   ├── data_faces.py        # Image dataset preparation and transforms
│   ├── model_faces.py       # Neural network architectures
│   ├── train_faces.py       # Training script for face models
│   ├── predict_faces.py     # Inference script
│   ├── train_unified.py     # Training for unified models
│   ├── sampler.py           # Data sampling utilities
│   ├── utils.py             # Helper functions and custom datasets
│   ├── data_triage.py       # Clinical data interface (stub)
│   ├── data_blood.py        # Blood test data interface (stub)
│   └── data_vitals.py       # Vital signs data interface (stub)
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── SETUP.md               # Detailed setup instructions
```

## Data Requirements

### Image Data

Images should be preprocessed and stored as pickle files with the format:

```python
(sample_keys, sample_xs, sample_ys)
```

- `sample_keys`: List of unique identifiers (filenames)
- `sample_xs`: NumPy arrays of images (H x W x 3, uint8)
- `sample_ys`: Labels/outcomes (not used if loading from clinical data)

### Clinical Data

You must implement the stub files to load your clinical data:

- **data_triage.py**: Patient demographics, outcomes, survival data
- **data_blood.py**: Laboratory test results
- **data_vitals.py**: Vital signs measurements

See [SETUP.md](SETUP.md) for detailed data format specifications.

## Key Features

### Supported Models

- **EfficientNet** (B0-B7): Efficient convolutional networks
- **Xception**: Deep separable convolutions
- Any model from [timm](https://github.com/huggingface/pytorch-image-models)

### Training Options

- **Labels**: Age regression, mortality prediction (survival analysis)
- **Loss functions**: MSE (regression), Cox proportional hazards (survival)
- **Data augmentation**: Rotation, color jitter, random crops, flips
- **Class balancing**: ImbalancedDatasetSampler support
- **Ensemble training**: Multiple models with different initializations

### Evaluation Metrics

- **Regression**: MAE, RMSE
- **Survival**: Concordance index, time-dependent AUC-ROC

## Configuration

Edit configuration variables at the top of training scripts:

### train_faces.py

```python
IMG_SET = "cv3"                    # Dataset identifier
CROSS_VAL_FOLDS = 3               # K-fold cross-validation
LABEL = ModelLabel.MORTALITY      # AGE or MORTALITY
MODEL = 'efficientnet_b3'         # Model architecture
SIZE = 299                        # Input image size
BATCH_SIZE = 32                   # Training batch size
NUM_EPOCHS = 50                   # Training epochs
LEARNING_RATE = 1e-3             # Learning rate
ENSEMBLE = 10                     # Number of ensemble models
```

### train_unified.py

```python
USE_AGE_SEX = 1      # Include demographics
USE_FACE = 1         # Include face predictions
USE_BLOOD = 1        # Include blood tests
USE_VITALS = 1       # Include vital signs
USE_PRED_AGE = 1     # Include predicted age
```

## Output

Models save:
- **Trained weights**: `{MODEL_DIR}/model_weights-{KEY}.pth`
- **Predictions**: `{OUT_DIR}/{IMG_SET}/out-{KEY}.csv`
- **TensorBoard logs**: `./runs/` (unified model only)

Prediction CSV format:
```csv
key,true,pred
patient001.jpg,[0 100],0.234
patient002.jpg,[1 45],-0.156
```

For survival models, `true` contains `[censored, duration]`.

## GPU Support

The code automatically uses CUDA if available. Specify GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 python code/train_faces.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{facescore2026,
  title={FaceScore: Clinical Prediction from Facial Images},
  author={Indra Heckenbach},
  year={2026},
  url={https://github.com/indra-icmm/facescore}
}
```

## License


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues, please open a GitHub issue or contact [indra@sund.ku.dk].
