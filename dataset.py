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

from config import UPLOAD_FOLDER

class SubmissionDataSet(Dataset):
    def __init__(self,data,image_location,transforms,tokenizer,vocab):
        self.data = data
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_len_sent = 128 - 7 -2
        self.vocab = vocab
        self.image_location = image_location

    def __getitem__(self, index):
        text = self.data
        text = str(text)
        text = self.tokenizer.tokenize(text)[:self.max_len_sent]
        text = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(text))

        image = None

        if self.image_location is not None:
            image = Image.open(self.image_location).convert('RGB')
        else:
            image = Image.fromarray(128*np.ones((256, 256, 3), dtype=np.uint8))

        image = self.transforms(image)
        

        # try:
        #     for f in glob.iglob(UPLOAD_FOLDER + "\\*"):
        #         print(f)
        #         image = Image.open(f)

        #     image = self.transforms(image)
        
        # except:
        #     image = Image.fromarray(128*np.ones((256, 256, 3), dtype=np.uint8))
        #     image = self.transforms(image)

        return text,image

    def __len__(self):
        return len(self.data)