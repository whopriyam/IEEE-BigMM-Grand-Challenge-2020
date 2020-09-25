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

from model.mmbt_enc import MultiModalBertEncoder

class MultiModalBertClf(nn.Module):
    def __init__(self, no_of_classes, tokenizer):
        super(MultiModalBertClf, self).__init__()
        self.no_of_classes = no_of_classes
        self.enc = MultiModalBertEncoder(self.no_of_classes, tokenizer)
        self.batch_norm = nn.BatchNorm1d(768)
        self.clf = nn.Linear(768, self.no_of_classes)
    
    def forward(self, text, text_attention_mask, text_segment, image):
        # if(torch.cuda.is_available()):
        #     text = text.cuda()
        #     text_attention_mask=text_attention_mask.cuda()
        #     text_segment=text_segment.cuda()
        #     image = image.cuda()
        #     self.clf = self.clf.cuda()
        #     self.batch_norm = self.batch_norm.cuda()
        x = self.enc(text, text_attention_mask, text_segment, image)

        x = self.batch_norm(x)
        x = self.clf(x)

        return x 