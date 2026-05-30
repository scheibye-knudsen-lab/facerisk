import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import sys

from sampler import SampleManager

import model_faces
import data_faces
import utils
from data_faces import ModelLabel


# Configuration
DEV_MODE = 0
ENSEMBLE = 1
EXTRACT_FEATURES = 0

# Dataset configuration
IMG_SET = "faces7"
KEY = ""

# Model configuration
MODEL = 'efficientnet_b3'
LABEL = ModelLabel.MORTALITY

# Paths - configure via environment variables
DATA_DIR = os.environ.get('DATA_DIR', './data')
MODEL_DIR = os.environ.get('MODEL_DIR', './models')
OUT_DIR = os.environ.get('OUT_DIR', './output')

OUT_BASE = f'{OUT_DIR}/{IMG_SET}/'
IMAGE_SRC = 'faces5'

MODEL_WEIGHTS_PATH = f'{MODEL_DIR}/model_weights-KEY.pth'

# Model settings
SIZE = 299  # Image size for xception, efficientnet
BATCH_SIZE = 64
DROPOUT_RATE = 0.3

# Optional mask/metadata paths
MASK_BASE_PATH = None
IMG_META_PATH = None
EXTRA_OUT_CODE = ""

# Optional color shift augmentation
COLOR_OFFSETS = None

# Label-specific settings
REGRESSION = 0
OUT_BINS = 1
SURVIVAL = 0

if LABEL == ModelLabel.AGE:
    REGRESSION = 1
elif LABEL == ModelLabel.MORTALITY:
    SURVIVAL = 1


#
# Prediction functions
#


def predict(model, dataset, device):
    model.eval()
    
    y_true, y_pred, y_key = [], [], []

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False) # shuffle=True, sampler does shuffle

    with torch.no_grad():

        progress_bar = tqdm(data_loader, desc=f'Prediction', unit='batch')

        for batch in progress_bar:
            images = batch['image']
            keys = batch['key']

            if SURVIVAL:
                durations_tensor = batch['duration']
                censored_tensor = batch['censored']
            else:
                labels_tensor = batch['label']
                    
            images = images.to(device)
            
            outputs = model(images)
    
            outputs = outputs.detach()
            preds = outputs.cpu().numpy()

            if SURVIVAL:
                y_true.extend(np.stack([censored_tensor.numpy(), durations_tensor.numpy()], axis=1)) 
            else:
                y_true.extend(labels_tensor.numpy()) 
                
            y_pred.extend(preds) 
            y_key.extend(keys)

    return y_true, y_pred, y_key


def predict_data(dataset, model, device, output_file):

    if EXTRACT_FEATURES:
        model.classifier = nn.Identity()  # efficient net
        model.fc = torch.nn.Identity()  # xception

    y_true, y_pred, y_key = predict(model, dataset, device)


    if EXTRACT_FEATURES:
        num_preds = np.array(y_pred).shape[1]
        # excluding true values, force lookup later, because inclusion means unstack of arbitrary values and column match error
        columns = ['key', ] + [f'pred_{i}' for i in range(num_preds)]
        
        df = pd.DataFrame(
            np.column_stack([y_key, y_pred]),
            columns=columns
        )

        df.to_csv(output_file, index=False)

    else:
        utils.dump_predictions(output_file, y_true, y_pred, y_key)


def process_samples(sampler, val_fold=None):
    """Process samples for prediction across ensemble models."""
    key_suffix = f"-{KEY}" if KEY else ""
    key = f"{str(LABEL).replace('ModelLabel.','')}-{MODEL}-{IMG_SET}{key_suffix}"
    print("***********", key)
    
    for ens_idx in range(ENSEMBLE):
        model = model_faces.get_model(MODEL, OUT_BINS, DROPOUT_RATE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enspath = f"-e{ens_idx}" if ENSEMBLE > 1 else ""
        mwpath = MODEL_WEIGHTS_PATH.replace("KEY", f'{key}{enspath}')
        print("Loading model weights from", mwpath)
        model.load_state_dict(torch.load(mwpath, map_location=device, weights_only=True))
        model.to(device)

        if COLOR_OFFSETS:
            for red_shift in COLOR_OFFSETS:
                for green_shift in COLOR_OFFSETS:
                    for blue_shift in COLOR_OFFSETS:
                        rgb_shift = (red_shift, green_shift, blue_shift)
                        
                        dataset = data_faces.prep_dataset(
                            sampler, 
                            data_faces.val_transforms(SIZE, rgb_shift), 
                            LABEL, 
                            DEV_MODE, 
                            train_mode=False
                        )

                        extra_code = f"{EXTRA_OUT_CODE}+" + "+".join(map(str, rgb_shift))
                        output_file = f'{OUT_BASE}/out-{key}-{EXTRACT_FEATURES}{extra_code}.csv'
                        os.makedirs(OUT_BASE, exist_ok=True)
                        predict_data(dataset, model, device, output_file)

        else:
            dataset = data_faces.prep_dataset(
                sampler, 
                data_faces.val_transforms(SIZE), 
                LABEL, 
                DEV_MODE, 
                mask_base_path=MASK_BASE_PATH, 
                image_meta_path=IMG_META_PATH, 
                train_mode=False
            )

            output_file = f'{OUT_BASE}/out-{key}-{EXTRACT_FEATURES}{EXTRA_OUT_CODE}{enspath}.csv'
            os.makedirs(OUT_BASE, exist_ok=True)
            print("Generating:", output_file)
            predict_data(dataset, model, device, output_file)


# Main execution
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else f'{DATA_DIR}/{IMAGE_SRC}-val-{IMG_SET}.pickle'
    sampler = SampleManager(img_path)
    sampler.load_samples()
    process_samples(sampler)


