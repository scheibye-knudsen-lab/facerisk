import torch
from torch import nn
import timm


class CoxPHLoss(nn.Module):
    def __init__(self, l2_lambda=1e-3):  # 1e-4
        super(CoxPHLoss, self).__init__()

        self.l2_lambda = l2_lambda

        self.censored_weight = 0
        # self.censored_weight = 0.1  # to stabalize weights for calibration between ensemble


    def forward(self, preds, censored, time_to_event):
        """
        log_hazard_ratio: predicted log hazard ratios from the model (shape: batch_size)
        time_to_event: observed time to event (shape: batch_size)
        events: binary event indicator (1 if event occurred, 0 if censored) (shape: batch_size)
        """

        sorted_indices = torch.argsort(time_to_event, descending=True)
        preds = preds[sorted_indices].reshape(-1)
        censored = censored[sorted_indices].float().reshape(-1)

        log_hazard_ratio = preds

        cumulative_hazard = torch.logcumsumexp(log_hazard_ratio, dim=0)
        log_likelihood = log_hazard_ratio - cumulative_hazard
          
        events = 1 - censored
        weights = events + self.censored_weight * censored

        base_loss = -torch.sum(log_likelihood * weights)
        
        return base_loss

def get_model(model, out_bins, dropout_rate):
    if model == 'xception':
        model = timm.create_model('legacy_xception', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, out_bins)

    else:

        model = timm.create_model(model, pretrained=True,  drop_rate=dropout_rate)

        def get_head(n_features):
            return nn.Linear(n_features, out_bins)
        
        if hasattr(model, 'head'):
            model.head = get_head(model.head.in_features)
        elif hasattr(model, 'classifier'):
            model.classifier = get_head(model.classifier.in_features)
        elif hasattr(model, 'fc'):
            model.fc = get_head(model.fc.in_features)

    return model

