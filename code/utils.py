from torch.utils.data import SubsetRandomSampler
import torch

from PIL import Image

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
        return self.durations+self.censored*1000
        

def get_training_sampler(dataset, subset_size, balance_samples, one_hot_out):

    #return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    if balance_samples:
        if one_hot_out:
            # TODO: Implement proper balancing for one-hot encoded outputs
            indices = np.random.choice(len(dataset), size=subset_size, replace=False)
            return SubsetRandomSampler(indices)
        
        else:
            from torchsampler import ImbalancedDatasetSampler
            # balance on first state only -- EXPAND LATER
            labs = [ np.sum(v) for v in dataset.get_labels().numpy() ]
            return ImbalancedDatasetSampler(dataset, labels=labs, num_samples=subset_size)
    else:
        indices = np.random.choice(len(dataset), size=subset_size, replace=False)
        return SubsetRandomSampler(indices)






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


