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

class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi={}
            self.itos=[]
            self.vocab_size=0
        else:
            self.stoi={
                w:i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_size = len(self.itos)
    
    def add(self, words):
        counter = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w]=counter
            counter+=1
            self.itos.append(w)
        self.vocab_size = len(self.itos)