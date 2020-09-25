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

def model_forward_predict(i_epoch, model, batch):
    txt, segment, mask, img = batch
    
    # if(torch.cuda.is_available()):
    # 	txt, img = txt.cuda(), img.cuda()
    # 	mask, segment = mask.cuda(), segment.cuda()

    out = model(txt,mask,segment,img)

    return out

def model_predict(dataloader, model, no_of_classes, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in dataloader:
            out = model_forward_predict(1, model, batch)
            if no_of_classes==1:
                pred = torch.sigmoid(out).cpu().detach().numpy()

            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()


            preds.append(pred)

    preds = [l for sl in preds for l in sl]

    return preds