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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tqdm
import datetime
import random
from flask import Flask,render_template
from flask import request
import glob

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        
    def forward(self, y_pred, y_true):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y_true.squeeze(-1), reduce = None)#this automatically  takes sigmoid of logits
        else:
            BCE_loss = F.binary_cross_entropy(y_pred, y_true, reduce = None)
            
        pt = torch.exp(-BCE_loss)

        F_loss =self.alpha * ((1-pt)**self.gamma) * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        return F_loss