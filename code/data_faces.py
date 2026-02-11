from torchvision import transforms
import imageio
from skimage.transform import resize

import pandas as pd
import numpy as np

import utils
import data_triage

from enum import Enum

class ModelLabel(Enum):
    AGE = "AGE"
    MORTALITY = "MORTALITY"


def prep_dataset(sampler, transformer, label: ModelLabel, dev_mode, mask_base_path = None, image_meta_path = None, train_mode=False):
    xs, ys, keys = sampler.get()

    labels = [0] * len(xs)

    if train_mode:
        fdata = data_triage.TriageData()

        inclusions = [ fdata.include_in_training(key) for key in keys ]
        
        filter_idxs = np.array(inclusions)
        xs = np.array(xs)[filter_idxs]
        keys = np.array(keys)[filter_idxs]
        print(f"Train cutoff: {len(ys)} to {len(xs)}")
    

        if label == ModelLabel.MORTALITY:
            labels = [ fdata.get_survival(key) for key in keys ]
        elif label == ModelLabel.AGE:
            labels = [ fdata.get_age(key) for key in keys ]

        # cut ones with None, filtered out
        filter_idxs = np.array([ lab != None for lab in labels])
        labels = np.array(labels)[filter_idxs]
        xs = np.array(xs)[filter_idxs]
        keys = np.array(keys)[filter_idxs]
        print(f"Reduced: {len(ys)} to {len(xs)}")

        # show distribution
        if label == ModelLabel.MORTALITY:
            count_0 = sum(1 for lbl in labels if lbl[0] == 0)
            count_1 = sum(1 for lbl in labels if lbl[0] == 1)
            print(f"Number of samples by censoring: {count_0} vs {count_1}")


    if dev_mode:
        print("DEV MODE")
        xs = xs[0:100]
        labels = labels[0:100]
        keys = keys[0:100]
 
    if mask_base_path:
        im_df = pd.read_csv(image_meta_path)
        im_map = im_df.set_index('filename').to_dict('index')

        new_xs, new_labels, new_keys = [], [], []
        for idx,x in enumerate(xs):
            fn = keys[idx]
            bfn = fn.replace(".JPG", "")
            mask_path = f'{mask_base_path}/{bfn}.tif'
            try:
                mask = imageio.imread(mask_path)
            except FileNotFoundError:
                print(f"Missing mask for {fn}")
                continue

            imeta = im_map[fn]

            scale_factor, x_offset, y_offset = float(imeta['scale_factor']), int(imeta['x_offset']), int(imeta['y_offset'])
            cutout_minx, cutout_miny, cutout_maxx, cutout_maxy = int(imeta['cutout_minx']), int(imeta['cutout_miny']), int(imeta['cutout_maxx']), int(imeta['cutout_maxy'])
    
            mask = mask[cutout_miny:cutout_maxy, cutout_minx:cutout_maxx]

            new_w = mask.shape[1] * scale_factor 
            new_h = mask.shape[0] * scale_factor 
            mask = resize(mask, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False)

            overlay = np.zeros_like(x)
            overlay[y_offset:y_offset+mask.shape[0], x_offset:x_offset+mask.shape[1]] = mask
        
            x = np.where(overlay == 0, 0, x)
            
            new_xs.append(x)
            new_labels.append(labels[idx])
            new_keys.append(keys[idx])

        xs, labels, keys = new_xs, new_labels, new_keys

    if label == ModelLabel.MORTALITY:
        if train_mode:
            censored = [label[0] for label in labels]
            durations = [label[1] for label in labels]
            ages = [fdata.get_age(key) for key in keys] if label == ModelLabel.MORTALITY else None
        else:
            ages = None 
            censored, durations = [0]*len(xs), [0]*len(xs)
        return utils.SurvivalImageDataset(xs, censored, durations, keys, transform=transformer, ages=ages)
    else:
        return utils.CustomImageDataset(xs, labels, keys, transform=transformer)


def val_transforms(size, rgb_shift=(0,0,0)):
    def shift_channels(img):
        if rgb_shift == (0,0,0):
            return img
        else:
            img = np.array(img)
            img[:, :, 0] = np.clip(img[:, :, 0] + rgb_shift[0], 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] + rgb_shift[1], 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] + rgb_shift[2], 0, 255)
            return transforms.ToPILImage()(img)

    return transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.CenterCrop(size),
        transforms.Lambda(shift_channels),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_transforms(size):
    return transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
