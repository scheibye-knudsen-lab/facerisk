import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import os

from sampler import SampleManager

import model_faces
import data_faces
from data_faces import ModelLabel
import utils

from lifelines.utils import concordance_index


# Configuration
DEV_MODE = 0

# Dataset configuration
IMG_SET = "wiki"

# Model configuration
LABEL = ModelLabel.MORTALITY
MODEL = 'efficientnet_b3'

KEY = 'v1'
SIZE = 299  # Image size for xception, efficientnet
ENSEMBLE = 10

# Training hyperparameters
BATCH_SIZE = 32
TRAIN_SUBSET = 0.4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BALANCE_CLASS_WEIGHT = 0
BALANCE_SAMPLES = 0

DROPOUT_RATE = 0.3

# Data source
IMAGE_SRC = 'faces5'

# Paths - configure via environment variables
DATA_DIR = os.environ.get('DATA_DIR', './data')
MODEL_DIR = os.environ.get('MODEL_DIR', './models')
OUT_DIR = os.environ.get('OUT_DIR', './output')

MODEL_WEIGHTS_PATH = f'{MODEL_DIR}/model_weights-KEY.pth'

# Label-specific settings
REGRESSION = 0
SURVIVAL = 0
OUT_BINS = 1

if LABEL == ModelLabel.AGE:
    REGRESSION = 1
elif LABEL == ModelLabel.MORTALITY:
    SURVIVAL = 1

train_img_path = f'{DATA_DIR}/{IMAGE_SRC}-train-{IMG_SET}.pickle'
val_img_path = f'{DATA_DIR}/{IMAGE_SRC}-val-{IMG_SET}.pickle'


#
# Training functions
#

def get_key():
    return f"{str(LABEL).replace('ModelLabel.','')}-{MODEL}-{IMG_SET}-{KEY}"

def train_phase(model, train_dataset, device):
    if REGRESSION:
        criterion = torch.nn.MSELoss()
    elif SURVIVAL:
        criterion = model_faces.CoxPHLoss()

    running_loss = 0.0

    y_true, y_pred, y_key = [], [], []

    subset_size = int(TRAIN_SUBSET * len(train_dataset))
    print("Train subset:", subset_size)
    subset_sampler = utils.get_training_sampler(train_dataset, subset_size, BALANCE_SAMPLES, False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=subset_sampler, num_workers=4, pin_memory=True, persistent_workers=False) # shuffle=True, sampler does shuffle

    # Using Adam optimizer. For EVE optimizer, install: pip install eve-optimizer
    # from eve import eve
    # optimizer = eve.EVE(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    progress_bar = tqdm(train_loader, desc=f'Train', unit='batch')
    model.train()

    for batch in progress_bar:
        optimizer.zero_grad()

        images = batch['image']
        keys = batch['key']

        images = images.to(device)

        if SURVIVAL:
            durations_tensor = batch['duration']
            censored_tensor = batch['censored']
            age_tensor = batch.get('age', None)

        else:
            labels_tensor = batch['label']


        outputs = model(images)
        
        
        if SURVIVAL:
            loss = criterion(outputs, censored_tensor.to(device), durations_tensor.to(device))
        else:
            labels_tensor = labels_tensor.unsqueeze(1)
            loss = criterion(outputs, labels_tensor.to(device))
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        outputs = outputs.detach()
        preds = outputs.cpu().numpy()

        if REGRESSION:
            mae = mean_absolute_error(labels_tensor.numpy(), preds)
            progress_bar.set_postfix(loss=loss.item(), mae=mae)

        elif SURVIVAL:
            progress_bar.set_postfix(loss=loss.item())

        if SURVIVAL:
            y_true.extend(np.stack([censored_tensor.numpy(), durations_tensor.numpy()], axis=1)) 
        else:
            y_true.extend(labels_tensor.numpy()) 

        y_pred.extend(preds) 
        y_key.extend(keys) 

    train_loss = running_loss / subset_size
    
    return train_loss, y_true, y_pred, y_key

#
# validation phase
# 
def validate_phase(model, val_dataset, device):
    if REGRESSION:
        criterion = torch.nn.MSELoss() 

    elif SURVIVAL:
        criterion = model_faces.CoxPHLoss()

    model.eval()
    val_loss = 0.0
    
    y_true, y_pred, y_key = [], [], []

    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=False)

    with torch.no_grad():

        progress_bar = tqdm(val_loader, desc=f'Validation', unit='batch')

        for batch in progress_bar:
            images = batch['image']
            keys = batch['key']

            images = images.to(device)

            if SURVIVAL:
                durations_tensor = batch['duration']
                censored_tensor = batch['censored']
                age_tensor = batch.get('age', None)
            else:
                labels_tensor = batch['label']

            outputs = model(images)
            

            if SURVIVAL:
                loss = criterion(outputs, censored_tensor.to(device), durations_tensor.to(device))
            elif REGRESSION:
                labels_tensor = labels_tensor.unsqueeze(1)
                loss = criterion(outputs, labels_tensor.to(device))
                        
            val_loss += loss.item() * images.size(0)
    
            outputs = outputs.detach()
            preds = outputs.cpu().numpy()

            if REGRESSION:
                mae = mean_absolute_error(labels_tensor.numpy(), preds)
                progress_bar.set_postfix(loss=loss.item(), mae=mae)
                y_true.extend(labels_tensor.numpy()) 

            elif SURVIVAL:
                progress_bar.set_postfix(loss=loss.item())
                y_true.extend(np.stack([censored_tensor.numpy(), durations_tensor.numpy()], axis=1)) 
                
            y_pred.extend(preds) 
            y_key.extend(keys)


    val_loss /= len(val_loader.dataset)

    return val_loss, y_true, y_pred, y_key



def do_epoch(epoch, ens_idx, train_dataset, val_dataset, model, device, last_save_metric=0, val_fold=None):
    train_loss, train_true, train_pred, train_key = train_phase(model, train_dataset, device)

    val_loss, val_true, val_pred, val_key = validate_phase(model, val_dataset, device)

    key = get_key()

    print(f'KEY: {key}')

    ensinfo = f"Ensemble {ens_idx} | " if ENSEMBLE > 1 else ""

    print(f'\033[1m*** {ensinfo} Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f} [{key}]\033[0m')

    if REGRESSION:
        val_pred = [k[0] for k in val_pred]

        mae = mean_absolute_error(val_true, val_pred)
        rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'... Val MAE: {mae:.4f}, RMSE: {rmse:.4f}')

        save_metric = 1/mae

    elif SURVIVAL:
        train_pred = [k[0] for k in train_pred]
        val_pred = [k[0] for k in val_pred]

        event = [ 1-label[0] for label in train_true ]
        train_true_durations = [ label[1] for label in train_true ]

        ci = concordance_index(train_true_durations, -np.array(train_pred), event)

        val_true_event = [ 1-label[0] for label in val_true ]
        val_true_durations = [ label[1] for label in val_true ]
        vci = concordance_index(val_true_durations, -np.array(val_pred), val_true_event)

        print(f"{KEY}: [{epoch+1}/{NUM_EPOCHS}], Concordance Index: {ci}, Val CI: {vci}")
        
        save_metric = vci

    # Track best for monitoring but don't use for model selection to avoid leakage
    if save_metric > last_save_metric:
        last_save_metric = save_metric
        print(f"*** New best validation metric: {save_metric}")

    # Save model at last epoch (to avoid leakage in CV predictions)
    if epoch == NUM_EPOCHS - 1:
        print("*** SAVING FINAL MODEL (last epoch)")
        enspath = f"-e{ens_idx}" if ENSEMBLE > 1 else ""
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH.replace("KEY", f'{key}{enspath}'))

    print()

    return last_save_metric


def process_samples(train_sampler, val_sampler, val_fold=None):
    """Train models across ensemble."""
    for ens_idx in range(0, ENSEMBLE):
        train_dataset = data_faces.prep_dataset(train_sampler, data_faces.train_transforms(SIZE), LABEL, DEV_MODE, train_mode=True)
        val_dataset = data_faces.prep_dataset(val_sampler, data_faces.val_transforms(SIZE), LABEL, DEV_MODE, train_mode=True)

        model = model_faces.get_model(MODEL, OUT_BINS, DROPOUT_RATE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        last_save_metric = 0

        for epoch in range(NUM_EPOCHS):
            save_metric = do_epoch(epoch, ens_idx, train_dataset, val_dataset, model, device, last_save_metric, val_fold)

            if save_metric > last_save_metric:
                last_save_metric = save_metric


# Main execution
if __name__ == "__main__":
    train_sampler = SampleManager(train_img_path)
    train_sampler.load_samples()

    val_sampler = SampleManager(val_img_path)
    val_sampler.load_samples()

    process_samples(train_sampler, val_sampler)
    