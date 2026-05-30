# FaceRisk

Deep learning models for clinical prediction from facial images, combining computer vision with survival analysis and clinical data.

## Overview

FaceRisk trains neural networks to predict clinical outcomes (age, mortality risk) from facial images. The framework supports:

- **Face-based models**: Image classification/regression using CNNs (EfficientNet, Xception)
- **Survival analysis**: Cox proportional hazards models for time-to-event prediction
- **Unified models**: Integration of image predictions with clinical data (vitals, blood tests)
- **Cross-validation**: Built-in support for k-fold cross-validation
- **Ensemble learning**: Multiple model training for robust predictions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/scheibye-knudsen-lab/facerisk.git
cd facerisk

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (typical install time: ~5 minutes)
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

### Demo

Run prediction on the included sample data (100 images, requires trained model weights):

```bash
python code/predict_faces.py sample/faces_100.pickle
```

## Project Structure

```
facerisk/
├── code/
│   ├── data_faces.py        # Image dataset preparation and transforms
│   ├── model_faces.py       # Neural network architectures
│   ├── train_faces.py       # Training script for face models
│   ├── predict_faces.py     # Inference script
│   ├── sampler.py           # Data sampling utilities
│   ├── utils.py             # Helper functions and custom datasets
│   └── data_triage.py       # Clinical data interface (stub)
├── sample/
│   └── faces_100.pickle     # Sample data for testing
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
IMG_SET = "wiki"                   # Dataset identifier
LABEL = ModelLabel.MORTALITY      # AGE or MORTALITY
MODEL = 'efficientnet_b3'         # Model architecture
SIZE = 299                        # Input image size
BATCH_SIZE = 32                   # Training batch size
NUM_EPOCHS = 50                   # Training epochs
LEARNING_RATE = 1e-3             # Learning rate
ENSEMBLE = 10                     # Number of ensemble models
```

## Output

Models save:
- **Trained weights**: `{MODEL_DIR}/model_weights-{KEY}.pth`
- **Predictions**: `{OUT_DIR}/{IMG_SET}/out-{KEY}.csv`

Prediction CSV format:
```csv
label,pred,key
[0. 0.],-0.968469,/euler/indrah/facerisk/wiki_imdb/wiki_crop/23/1507423_1969-01-02_2004.jpg
[0. 0.],-1.1902257,/euler/indrah/facerisk/wiki_imdb/wiki_crop/59/330859_1965-05-24_2009.jpg
[0. 0.],-0.15366133,/euler/indrah/facerisk/wiki_imdb/wiki_crop/35/32935935_1935-06-10_2005.jpg
[0. 0.],-2.6203792,/euler/indrah/facerisk/wiki_imdb/wiki_crop/56/1628256_1983-12-06_2008.jpg
[0. 0.],-0.7129433,/euler/indrah/facerisk/wiki_imdb/wiki_crop/18/6901318_1987-11-30_2012.jpg
```

For survival models, `label` contains `[censored, duration]`. The `pred` column is the log hazard ratio from the risk model; more negative values indicate lower predicted mortality risk.

## GPU Support

The code automatically uses CUDA if available. Specify GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 python code/train_faces.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{facerisk2026,
  title={FaceRisk: Clinical Prediction from Facial Images},
  author={Indra Heckenbach},
  year={2026},
  url={https://github.com/scheibye-knudsen-lab/facerisk}
}
```

## License

MIT License — Copyright (c) 2026 Indra Heckenbach

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues, please open a GitHub issue or contact [indra@sund.ku.dk].
