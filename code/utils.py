from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import SubsetRandomSampler #, WeightedRandomSampler
import torch

from lifelines import CoxPHFitter

from PIL import Image

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, keys, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.keys = keys
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert the image (numpy array) to PIL image for transformations
        #image = Image.fromarray((image * 255).astype(np.uint8))  # Assuming image is in range [0, 1]
        image = Image.fromarray(image)  # Assuming image is 8b

        if self.transform:
            image = self.transform(image)  # Apply the transform (augmentations)

        return { 'image': image, 'key':self.keys[idx], 'label':self.labels[idx] }
    
    def get_labels(self):
        return self.labels
    
class SurvivalImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, censored, durations, keys, transform=None, ages=None):
        self.images = images
        self.keys = keys
        self.transform = transform
        self.durations = torch.tensor(durations, dtype=torch.float32)
        self.censored = torch.tensor(censored, dtype=torch.float32)
        self.ages = torch.tensor(ages, dtype=torch.float32) if ages is not None else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert the image (numpy array) to PIL image for transformations
        #image = Image.fromarray((image * 255).astype(np.uint8))  # Assuming image is in range [0, 1]
        image = Image.fromarray(image)  # Assuming image is 8b

        if self.transform:
            image = self.transform(image)  # Apply the transform (augmentations)

        item = { 'image': image, 'key':self.keys[idx], 'duration':self.durations[idx], 'censored':self.censored[idx] }
        if self.ages is not None:
            item['age'] = self.ages[idx]
        return item

    def get_labels(self):
        return self.durations+self.events*1000
        

def get_training_sampler(dataset, subset_size, balance_samples, one_hot_out):

    #return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    if balance_samples:
        if one_hot_out:
            # TODO: Implement proper balancing for one-hot encoded outputs
            indices = np.random.choice(len(dataset), size=subset_size, replace=False)
            return SubsetRandomSampler(indices)
        
        else:
            # balance on first state only -- EXPAND LATER
            labs = [ np.sum(v) for v in dataset.get_labels().numpy() ]
            return ImbalancedDatasetSampler(dataset, labels=labs, num_samples=subset_size)
    else:
        indices = np.random.choice(len(dataset), size=subset_size, replace=False)
        return SubsetRandomSampler(indices)





def cm_breakdown(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    correct_class_0 = cm[0, 0]
    correct_class_1 = cm[1, 1]

    total_class_0 = cm[0, 0] + cm[0, 1]
    total_class_1 = cm[1, 0] + cm[1, 1]

    percent_correct_class_0 = (correct_class_0 / total_class_0) * 100 if total_class_0 > 0 else 0
    percent_correct_class_1 = (correct_class_1 / total_class_1) * 100 if total_class_1 > 0 else 0

    print(f'0: {correct_class_0} of {total_class_0} ({percent_correct_class_0:.2f}%)')
    print(f'1: {correct_class_1} of {total_class_1} ({percent_correct_class_1:.2f}%)')


def cm_breakdown_multi(y_true, y_pred):
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    num_classes = len(y_true[0])
    
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(num_classes))
    
    for class_idx in range(num_classes):
        # TP = cm[class_idx, class_idx]  # True Positives
        # total_samples = cm[class_idx, :].sum()  # Total true instances of this class        
        # accuracy = 100.0 * TP / total_samples if total_samples > 0 else 0        
        # print(f"{class_idx}: {TP} of {total_samples} ({accuracy:.2f}%)")

        total_samples = cm[class_idx, :].sum()
        
        s = f"Class {class_idx} ({total_samples})> "
        
        for pred_class_idx in range(num_classes):
            pred_count = cm[class_idx, pred_class_idx]
            pred_percentage = 100.0 * pred_count / total_samples if total_samples > 0 else 0
            s += f"  {pred_class_idx}: {pred_count} )"
        for pred_class_idx in range(num_classes):
            pred_count = cm[class_idx, pred_class_idx]
            pred_percentage = 100.0 * pred_count / total_samples if total_samples > 0 else 0
            s += f"  {pred_class_idx}: {pred_percentage:.2f}%"
        print(s)

def dump_predictions(output_file, ys, y_preds, keys):
    if isinstance(y_preds[0], np.ndarray) or isinstance(y_preds[0], list):
        if len(y_preds[0]) == 1:
            y_preds = [k[0] for k in y_preds]

    df = pd.DataFrame({
        'true': ys,
        'pred': y_preds,
        'key': keys
    })

    df.to_csv(output_file, index=False)



def cox_diff(df, fields, formula, factors):
    cph_full = CoxPHFitter()

    cph_full.fit(df[fields], duration_col='time', event_col='event', formula=formula)

    full_log_likelihood = cph_full.log_likelihood_

    partial_likelihood_ratios = {}

    for factor in factors:
        rbf = factors.copy()
        rbf.remove(factor)
        formula = "+".join(rbf)

        reduced_df = df.drop(columns=[factor])

        cph_reduced = CoxPHFitter()
        cph_reduced.fit(reduced_df, duration_col='time', event_col='event', formula=formula)
        reduced_log_likelihood = cph_reduced.log_likelihood_
        
        partial_likelihood_ratio = 2 * (full_log_likelihood - reduced_log_likelihood)
        partial_likelihood_ratios[factor] = partial_likelihood_ratio

    partial_likelihood_ratios_df = pd.DataFrame.from_dict(partial_likelihood_ratios, orient='index', columns=['Partial Likelihood Ratio'])
    total_likelihood_ratio = partial_likelihood_ratios_df['Partial Likelihood Ratio'].sum()
    partial_likelihood_ratios_df['Percent Contribution'] = (partial_likelihood_ratios_df['Partial Likelihood Ratio'] / total_likelihood_ratio) * 100

    partial_likelihood_ratios_df = partial_likelihood_ratios_df.sort_values(by='Percent Contribution', ascending=False)

    return partial_likelihood_ratios_df
