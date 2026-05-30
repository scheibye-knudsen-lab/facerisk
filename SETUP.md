# Setup Instructions

## System Requirements

- **Python**: 3.12.7
- **PyTorch**: 2.4.1
- **CUDA**: 12.1 (optional, for GPU acceleration)
- **Tested on**: Ubuntu 22.04.5 LTS

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure paths**:
   ```bash
   export DATA_DIR=/path/to/your/data
   export MODEL_DIR=/path/to/your/models
   export OUT_DIR=/path/to/your/output
   ```

3. **Implement data loaders** (see section below)

4. **Prepare your data** (see Data Format section)

5. **Run training**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python code/train_faces.py
   ```

## Environment Variables

Before running the training scripts, configure the following environment variables:

```bash
export DATA_DIR=/path/to/your/data
export MODEL_DIR=/path/to/your/models
export OUT_DIR=/path/to/your/output
```

Or create a `.env` file:
```
DATA_DIR=/path/to/your/data
MODEL_DIR=/path/to/your/models
OUT_DIR=/path/to/your/output
```

## GPU Configuration

To specify which GPU to use, set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
CUDA_VISIBLE_DEVICES=0 python code/train_faces.py
```

## Dependencies

### Core Dependencies
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install lifelines
pip install tqdm
pip install tensorboard
pip install Pillow
```

### Optional Dependencies

For ImbalancedDatasetSampler:
```bash
pip install torchsampler
```

For EVE optimizer (alternative to Adam):
```bash
pip install eve-optimizer
```

## Data Module Implementation

The repository includes **stub implementations** for clinical data modules. You must implement these with your actual data loading logic:

### data_triage.py (Provided as stub)

Implements the `TriageData` class and helper functions:

```python
class TriageData:
    def include_in_training(self, key) -> bool:
        """Check if sample should be included in training"""
        
    def get_survival(self, key) -> tuple:
        """Return (censored, duration) for survival analysis"""
        
    def get_age(self, key) -> float:
        """Return patient age in years"""
        
    def get_readmission(self, key) -> tuple:
        """Return readmission outcome"""

def prep_data_facts(df):
    """Add demographic columns (id, age, sex) to dataframe"""
    
def prep_data_survival(df) -> pd.DataFrame:
    """Return DataFrame with 'event' and 'duration' columns"""
```

**Implementation notes:**
- Load data from your database/files in `__init__`
- Map patient keys to outcomes
- Handle missing data appropriately
- See stub file for detailed docstrings and examples

## Missing Module Files

**IMPORTANT**: `data_triage.py` is provided as a **stub only**. You must implement it with your actual data sources.

## Data Format

### Sample Data (pickle format)
The training scripts expect pickled data in the format:
```python
(sample_keys, sample_xs, sample_ys)
```

Where:
- `sample_keys`: List of unique identifiers
- `sample_xs`: List of image arrays
- `sample_ys`: List of labels/outcomes

### CSV Output Format
Prediction outputs are saved as CSV with columns:
- `key`: Sample identifier
- `true`: True label/outcome
- `pred`: Predicted value

## Usage Examples

### Train Face Model
```bash
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
export OUT_DIR=/path/to/output

CUDA_VISIBLE_DEVICES=0 python code/train_faces.py
```

### Generate Predictions
```bash
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
export OUT_DIR=/path/to/output

python code/predict_faces.py
```

## Configuration

Edit the configuration variables at the top of each training script:

### train_faces.py
- `IMG_SET`: Dataset identifier (e.g., 'cv3')
- `LABEL`: Target variable (AGE or MORTALITY)
- `MODEL`: Model architecture ('xception', 'efficientnet_b3')
- `SIZE`: Input image size (299)
- `BATCH_SIZE`: Batch size for training (32)
- `NUM_EPOCHS`: Number of training epochs (50)
- `LEARNING_RATE`: Learning rate (1e-3)
