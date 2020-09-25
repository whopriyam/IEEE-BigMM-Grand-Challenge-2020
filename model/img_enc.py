import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
from pytorch_pretrained_bert.modeling import BertModel
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tqdm
import datetime
import random
from flask import Flask,render_template
from flask import request
import glob

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]

        self.model = nn.Sequential(*modules)
        # if(torch.cuda.is_available()):
        #     self.model = self.model.cuda()

    def forward(self, x):
        out = (self.model(x))

        out = nn.AdaptiveAvgPool2d((7, 1))(out)

        out = torch.flatten(out, start_dim=2)

        out = out.transpose(1, 2).contiguous()
        
        return out