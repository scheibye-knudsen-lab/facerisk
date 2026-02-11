import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os

import data_triage
import data_vitals
import data_blood
import utils

from lifelines.utils import concordance_index
from torch.utils.tensorboard import SummaryWriter



# Configuration
IMG_SET = "cv3"
CROSS_VAL_FOLDS = 3
KEY = 'v1'

# Paths - configure via environment variables
OUT_DIR = os.environ.get('OUT_DIR', './output')
MODEL_DIR = os.environ.get('MODEL_DIR', './models')

OUT_BASE = f'{OUT_DIR}/{IMG_SET}/'

FACE_TRAIN_PATH = f'{OUT_BASE}/out-MORTALITY-efficientnet_b3-{IMG_SET}-{KEY}-FOLD-mean.csv'
FACE_VAL_PATH = f'{OUT_BASE}/out-MORTALITY-efficientnet_b3-{IMG_SET}-{KEY}-FOLD-mean.csv'

AGE_TRAIN_PATH = f'{OUT_BASE}/out-AGE-xception-{IMG_SET}-{KEY}-FOLD-mean.csv'
AGE_VAL_PATH = f'{OUT_BASE}/out-AGE-xception-{IMG_SET}-{KEY}-FOLD-mean.csv'

MODEL_WEIGHTS_PATH = f'{MODEL_DIR}/model_weights-KEY.pth'

# Feature flags
USE_AGE_SEX = 1
USE_FACE = 1
USE_BLOOD = 1
USE_VITALS = 1
USE_PRED_AGE = 1

LABEL = "HAZARD"

KEY = f"uni-{IMG_SET}-{KEY}-{LABEL}-as{USE_AGE_SEX}_face{USE_FACE}_blood{USE_BLOOD}_vitals{USE_VITALS}_page{USE_PRED_AGE}"

# Training hyperparameters
ENSEMBLE = 10
CROSS_VAL_FOLDS_LIST = list(range(CROSS_VAL_FOLDS))
NUM_EPOCHS = 50
BATCH_SIZE = 8
WEIGHT_DECAY = 0
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
CENSORED_WEIGHTS = 0


#
# Model definitions
#

class MixedInputSurvivalModel(nn.Module):
    """Neural network for survival prediction from mixed input features."""
    
    def __init__(self, input_size):
        super(MixedInputSurvivalModel, self).__init__()
        print(f"Model input size: {input_size}")
        self.fc1 = nn.Linear(input_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        return self.fc3(x)
    
class MixedInputAgeModel(nn.Module):
    """Neural network for age prediction from mixed input features."""
    
    def __init__(self, input_size):
        super(MixedInputAgeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def cox_ph_loss(predictions, event, duration):
    """Cox proportional hazards loss function."""
    # Sort by duration in descending order
    sorted_idx = torch.argsort(duration, descending=True)
    predictions = predictions[sorted_idx]
    event = event[sorted_idx]
    censored = 1 - event

    log_hazard_ratio = predictions
    cumulative_hazard = torch.logcumsumexp(log_hazard_ratio, dim=0)
    log_likelihood = log_hazard_ratio - cumulative_hazard

    weights = event + CENSORED_WEIGHTS * censored

    return -torch.sum(log_likelihood * weights)


def train_phase(model, train_loader, device):
    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    running_loss = 0.0

    y_true, y_pred, y_key = [], [], []

    progress_bar = tqdm(train_loader, desc=f'Train', unit='batch', disable=True)

    for batch in progress_bar:
        optimizer.zero_grad()

        keys = batch['key']
        features = batch['features']
        features = features.to(device)

        preds = model(features)
     
        durations = batch['duration']
        events = batch['event']
        
        durations_dev = durations.to(device)
        events_dev = events.to(device)

        loss = cox_ph_loss(preds, events_dev, durations_dev)
       
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        preds = preds.detach()
        preds = preds.cpu()

        progress_bar.set_postfix(loss=loss.item())
        y_true.extend(np.stack([events.numpy(), durations.numpy()], axis=1)) 

        y_pred.extend(preds) 
        y_key.extend(keys) 

    train_loss = running_loss / len(train_loader.dataset)
    
    return train_loss, y_true, y_pred, y_key

#
# validation phase
# 
def validate_phase(model, val_loader, device):
    model.eval()

    val_loss = 0.0
    
    y_true, y_pred, y_key = [], [], []

    with torch.no_grad():

        progress_bar = tqdm(val_loader, desc=f'Validation', unit='batch', disable=True)

        for batch in progress_bar:
            keys = batch['key']
            features = batch['features']
            features = features.to(device)

            preds = model(features)
        
            durations = batch['duration']
            events = batch['event']

            durations_dev = durations.to(device)
            events_dev = events.to(device)
            loss = cox_ph_loss(preds, events_dev, durations_dev)
                     
            val_loss += loss.item() * features.size(0)
    
            preds = preds.cpu().numpy()

            progress_bar.set_postfix(loss=loss.item())
            y_true.extend(np.stack([events.numpy(), durations.numpy()], axis=1)) 
                
            y_pred.extend(preds) 
            y_key.extend(keys)


    val_loss /= len(val_loader.dataset)

    return val_loss, y_true, y_pred, y_key


class SurvivalDataset(Dataset):
    """Dataset for survival analysis with features, events, and durations."""
    
    def __init__(self, x, y_events, y_durations, keys):
        self.keys = keys
        self.features = torch.tensor(x, dtype=torch.float32)
        self.events = torch.tensor(y_events, dtype=torch.float32)
        self.durations = torch.tensor(y_durations, dtype=torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'key': self.keys[idx],
            'duration': self.durations[idx],
            'event': self.events[idx]
        }


def prep_dataset(df_x, scaler=None, fit_scaler=True):
    """Prepare dataset by loading and preprocessing all features.
    
    Args:
        df_x: Input dataframe
        scaler: Pre-fitted StandardScaler (for val/test sets)
        fit_scaler: If True, fit the scaler on this data (for training set only)
    
    Returns:
        ds: Dataset object
        df_x: Processed feature dataframe
        scaler: Fitted scaler (to use for val/test sets)
    """
    if not USE_FACE:
        df_x = df_x[['key']]

    data_triage.prep_data_facts(df_x)
    df_y = data_triage.prep_data_survival(df_x)

    if USE_BLOOD:
        data_blood.prep_data_blood(df_x)

    if USE_VITALS:
        data_vitals.prep_data_vitals(df_x)

    df_keys = df_x['key']
    df_x = df_x.drop(columns=["key", "id"])

    if not USE_AGE_SEX:
        df_x = df_x.drop(columns=['age', 'sex'])

    # Standardize features - CRITICAL: fit only on training data!
    if fit_scaler:
        scaler = StandardScaler()
        x = scaler.fit_transform(df_x)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        x = scaler.transform(df_x)
    
    x = np.nan_to_num(x, nan=-1)

    ds = SurvivalDataset(x, df_y['event'].values, df_y['duration'].values, df_keys.values)

    return ds, df_x, scaler




def load_datasets(val_fold):
    df_train_x, df_test_x = None, None

    if CROSS_VAL_FOLDS > 1:
        for train_fold in CROSS_VAL_FOLDS_LIST:
            if val_fold == train_fold:
                continue

            train_path = FACE_TRAIN_PATH.replace("FOLD", f"fold{train_fold}")

            df = pd.read_csv(train_path)
            df_train_x = pd.concat([df_train_x, df], ignore_index=True) if df_train_x is not None else df
        
            if USE_PRED_AGE:
                train_path = AGE_TRAIN_PATH.replace("FOLD", f"fold{train_fold}")
                age_df_train_x = pd.read_csv(train_path)
                df_train_x['pred_age'] = df_train_x['key'].map(age_df_train_x.set_index('key')['pred_mean'])

        val_path = FACE_VAL_PATH.replace("FOLD", f"fold{val_fold}")
        df_test_x = pd.read_csv(val_path)

        if USE_PRED_AGE:
            val_path = AGE_VAL_PATH.replace("FOLD", f"fold{val_fold}")
            age_df_test_x = pd.read_csv(val_path)
            df_test_x['pred_age'] = df_test_x['key'].map(age_df_test_x.set_index('key')['pred_mean'])


    else:
        train_path = FACE_TRAIN_PATH.replace("-FOLD", "")
        df_train_x = pd.read_csv(train_path)

        val_path = FACE_VAL_PATH.replace("-FOLD", "")
        df_test_x = pd.read_csv(val_path)

        if USE_PRED_AGE:
            train_path = AGE_TRAIN_PATH.replace("-FOLD", "")
            age_df_train_x = pd.read_csv(train_path)
            df_train_x['pred_age'] = df_train_x['key'].map(age_df_train_x.set_index('key')['pred_mean'])

            val_path = AGE_VAL_PATH.replace("-FOLD", "")
            age_df_test_x = pd.read_csv(val_path)
            df_test_x['pred_age'] = df_test_x['key'].map(age_df_test_x.set_index('key')['pred_mean'])

    # Shuffle and split training data into train and internal validation
    df_train_x = df_train_x.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val_x = df_train_x.sample(n=min(1000, int(0.15 * len(df_train_x))), random_state=42)
    df_train_x = df_train_x.drop(df_val_x.index).reset_index(drop=True)

    # Prepare datasets - FIT scaler on training data only!
    train_dataset, df_train_x, scaler = prep_dataset(df_train_x, scaler=None, fit_scaler=True)
    
    # Use training scaler for validation and test (no fitting!)
    val_dataset, df_val_x, _ = prep_dataset(df_val_x, scaler=scaler, fit_scaler=False)
    test_dataset, df_test_x, _ = prep_dataset(df_test_x, scaler=scaler, fit_scaler=False)

    return train_dataset, val_dataset, test_dataset, df_train_x, df_val_x, df_test_x


def do_epoch(ens_idx, epoch, model, device, train_loader, val_loader, test_loader, writer, last_save_metric, val_fold):
    train_loss, train_true, train_pred, train_key = train_phase(model, train_loader, device)

    val_loss, val_true, val_pred, val_key = validate_phase(model, val_loader, device)
   
    ensinfo = f"Ensemble {ens_idx} | " if ENSEMBLE > 1 else ""

    train_true_event = [ label[0] for label in train_true ]
    train_true_durations = [ label[1] for label in train_true ]
    ci = concordance_index(train_true_durations, -np.array(train_pred), train_true_event)

    val_true_event = [ label[0] for label in val_true ]
    val_true_durations = [ label[1] for label in val_true ]
    vci = concordance_index(val_true_durations, -np.array(val_pred), val_true_event)

    duration_event30 = np.where(np.array(val_true_durations)<30, 1, 0)
    duration_event90 = np.where(np.array(val_true_durations)<90, 1, 0)
    duration_event180 = np.where(np.array(val_true_durations)<180, 1, 0)

    auc_ev = roc_auc_score(val_true_event, val_pred)
    auc_30 = roc_auc_score(duration_event30, val_pred)
    auc_90 = roc_auc_score(duration_event90, val_pred)
    auc_180 = roc_auc_score(duration_event180, val_pred)

    auc_avg = (auc_ev+auc_30+auc_90+auc_180)/4

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Concordance Index/train', ci, epoch)
    writer.add_scalar('Concordance Index/val', vci, epoch)
    writer.add_scalar('AUC-ROC avg/val', auc_avg, epoch)
    
    print(f"{KEY}: {ensinfo} [{epoch+1}/{NUM_EPOCHS}], AUC_AVG={auc_avg:0.3f}, Concordance Index: {ci}, Val CI: {vci}, AUC_event: {auc_ev}, AUC_30: {auc_30:0.3f}, AUC_90: {auc_90:0.3f}, AUC_180: {auc_180:0.3f}")

    save_metric = auc_avg

    # Track best for monitoring but don't use for model selection to avoid leakage
    if save_metric > last_save_metric:
        last_save_metric = save_metric
        print(f"*** New best validation metric: {save_metric}")

    # Save model at last epoch (to avoid leakage in CV predictions)
    if epoch == NUM_EPOCHS - 1:
        print("*** SAVING FINAL MODEL (last epoch)")
        cvpath = f"-fold{val_fold}" if CROSS_VAL_FOLDS > 1 else ""
        enspath = f"-e{ens_idx}" if ENSEMBLE > 1 else ""
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH.replace("KEY", f'{KEY}{cvpath}{enspath}'))

        output_file = f'{OUT_BASE}/KEY-val_out.csv'
        output_file = output_file.replace("KEY", f'{KEY}{cvpath}{enspath}')

        test_loss, test_true, test_pred, test_key = validate_phase(model, test_loader, device)
        utils.dump_predictions(output_file, test_true, test_pred, test_key)

    return last_save_metric

def process_samples(val_fold):
    """Train ensemble models for a given validation fold."""
    for e_index in range(ENSEMBLE):

        train_dataset, val_dataset, test_dataset, df_train_x, df_val_x, df_test_x = load_datasets(val_fold)

        model = MixedInputSurvivalModel(input_size=train_dataset.features.shape[1])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        writer = SummaryWriter(log_dir='./runs/experiment_1')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=False)

        last_save_metric = 0
        for epoch in range(NUM_EPOCHS):
            save_metric = do_epoch(e_index, epoch, model, device, train_loader, val_loader,
                                  test_loader, writer, last_save_metric, val_fold)
            if save_metric > last_save_metric:
                last_save_metric = save_metric

        writer.close()


# Main execution
if __name__ == "__main__":
    for val_fold in CROSS_VAL_FOLDS_LIST:
        process_samples(val_fold)
