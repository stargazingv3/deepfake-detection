# utils/models.py

import torch
import torch.nn as nn
from timm import create_model
from config import Config

class DeepFakeModel(nn.Module):
    def __init__(self, model_name=None):
        super(DeepFakeModel, self).__init__()
        model_name = model_name or Config.MODEL_NAME
        self.model = create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.model.num_features
        
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, Config.NUM_CLASSES)
        )

    def forward(self, x):
        features = self.model(x)
        features = self.dropout(features)
        return self.head(features)

def load_model(model_path):
    model = DeepFakeModel()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model
