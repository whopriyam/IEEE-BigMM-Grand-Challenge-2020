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

class ImageEmbeddingsForBert(nn.Module):
    def __init__(self, embeddings, vocabObject):
        super(ImageEmbeddingsForBert, self).__init__()
        self.vocab = vocabObject

        self.img_embeddings = nn.Linear(2048, 768)

        
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = embeddings.dropout
        
    def forward(self, batch_input_imgs, token_type_ids):
        batch_size = batch_input_imgs.size(0)
        seq_length = 7 + 2

        cls_id = torch.LongTensor([101])
        # if torch.cuda.is_available():
        #     cls_id = cls_id.cuda()
        #     self.word_embeddings = self.word_embeddings.cuda()
        cls_id = cls_id.unsqueeze(0).expand(batch_size, 1)
        
        # if torch.cuda.is_available():
        #     cls_id = cls_id.cuda()
        cls_token_embeddings = self.word_embeddings(cls_id)
        
        sep_id = torch.LongTensor([102])
        # if torch.cuda.is_available():
        #     sep_id = sep_id.cuda()
        #     self.img_embeddings = self.img_embeddings.cuda()
        sep_id = sep_id.unsqueeze(0).expand(batch_size, 1)
        sep_token_embeddings = self.word_embeddings(sep_id)
        
        batch_image_embeddings_768 = self.img_embeddings(batch_input_imgs)
        
        token_embeddings = torch.cat(
        [cls_token_embeddings, batch_image_embeddings_768, sep_token_embeddings], dim=1)
        
        position_ids = torch.arange(seq_length, dtype=torch.long)
        # if torch.cuda.is_available():
        #     position_ids = position_ids.cuda()
        #     self.position_embeddings = self.position_embeddings.cuda()
        #     self.token_type_embeddings= self.token_type_embeddings.cuda()
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        position_embeddings = self.position_embeddings(position_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeddings+position_embeddings+token_type_embeddings
        # if torch.cuda.is_available():
        #     embeddings = embeddings.cuda()
        #     self.LayerNorm=self.LayerNorm.cuda()
        #     self.dropout=self.dropout.cuda()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings