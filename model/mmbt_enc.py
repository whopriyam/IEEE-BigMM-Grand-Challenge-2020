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

from model.vocab import Vocab
from model.img_emb import ImageEmbeddingsForBert
from model.img_enc import ImageEncoder

class MultiModalBertEncoder(nn.Module):
    def __init__(self, no_of_classes, tokenizer):
        super(MultiModalBertEncoder, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        self.embeddings = bert.embeddings
        self.vocab=Vocab()
        self.image_embeddings = ImageEmbeddingsForBert(self.embeddings, self.vocab)
        self.image_encoder = ImageEncoder()
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.clf = nn.Linear(768, no_of_classes)
        
    def forward(self, input_text, text_attention_mask, text_segment, input_image):
        batch_size = input_text.size(0)

        temp = torch.ones(batch_size, 7+2).long()
        # if torch.cuda.is_available():
        #     temp = temp.cuda()
        #     self.encoder = self.encoder.cuda()
        #     self.pooler = self.pooler.cuda()
        attention_mask = torch.cat(
            [
                temp, text_attention_mask
            ],
            dim=1
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        
        image_token_type_ids = torch.LongTensor(batch_size, 7+2).fill_(0)
        # if(torch.cuda.is_available()):
        #     image_token_type_ids= image_token_type_ids.cuda()
        
        image = self.image_encoder(input_image)
        image_embedding_out = self.image_embeddings(image, image_token_type_ids)
        
        text_embedding_out = self.embeddings(input_text, text_segment)
        
        encoder_input = torch.cat([image_embedding_out, text_embedding_out], dim=1)

    
        encoded_layers = self.encoder(encoder_input, extended_attention_mask, output_all_encoded_layers=False)

        final = self.pooler(encoded_layers[-1])

        return final