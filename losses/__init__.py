import torch.nn as nn

def get_criterion(loss_name, config):
    if loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Invalid Criterion name: {loss_name}")