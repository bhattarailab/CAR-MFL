import sys

import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)

class EncoderResNet(nn.Module):
    def __init__(self, embed_dim, cnn_type):
        super(EncoderResNet, self).__init__()
        
        # Backbone CNN
        self.cnn = getattr(models, cnn_type)(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_dim)

        self.cnn.fc = nn.Sequential()
        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = True


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(pooled)
        output = {}
        out = l2_normalize(out)
        output['embedding'] = out
        return output