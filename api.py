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

from loss import FocalLoss
from dataset import SubmissionDataSet
from engine import model_predict, model_forward_predict
from model.vocab import Vocab
from config import UPLOAD_FOLDER
from model.mmbt_clf import MultiModalBertClf


app = Flask(__name__, template_folder='templates')

# bert = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
bert_tokenizer.convert_tokens_to_ids(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

no_of_classes = 1

img_transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )

device = torch.device("cpu")

# if torch.cuda.is_available():
# 	device = torch.device("cuda")



chosen_criteria = FocalLoss()

def collate_function_for_submission(batch, task_type='singlelabel'):
    lengths = [len(row[0]) for row in batch]
    batch_size = len(batch)
    max_sent_len = max(lengths)
    if(max_sent_len>128-7-2):
        max_sent_len=128-7-2
    text_tensors = torch.zeros(batch_size, max_sent_len).long()
    text_attention_mask = torch.zeros(batch_size, max_sent_len).long()
    text_segment = torch.zeros(batch_size, max_sent_len).long()
    batch_image_tensors = torch.stack([row[1] for row in batch])
    
    for i, (row, length) in enumerate(zip(batch, lengths)):
        text_tokens = row[0]
        if(length>128-7-2):
            length = 128-7-2
        text_tensors[i, :length] = text_tokens
        text_segment[i, :length] = 1
        text_attention_mask[i, :length]=1
    
    return text_tensors, text_segment, text_attention_mask, batch_image_tensors

model_list = []

@app.route("/",methods = ["GET","POST"])
def upload_predict():

    if request.method == "POST":
        image_file = request.files["image"]
        txt = request.form["Tweet"]
        if txt == "":
            txt = "-"
        

        if image_file or txt:
            try:
                image_location = os.path.join(UPLOAD_FOLDER,image_file.filename)
                image_file.save(image_location)
            except:
                image_location = None

            for m in model_list:
                m.eval()

            print("txt => ", txt)
            print("image_location => ", image_location)

            test_submission_dataset = SubmissionDataSet(txt,image_location,img_transformations,bert_tokenizer,vocab)
            test_submission_dataloader = torch.utils.data.DataLoader(test_submission_dataset,batch_size=4,collate_fn=collate_function_for_submission)
            
            sarcasm_predictions = model_predict(test_submission_dataloader,model_list[0],no_of_classes,1)
            # del model_list[0]

            allegation_predictions = model_predict(test_submission_dataloader,model_list[1],no_of_classes,1)
            # del model_list[0]

            text_only_preds = model_predict(test_submission_dataloader,model_list[2],no_of_classes,1)
            # del model_list[0]

            filelist = glob.glob(os.path.join(UPLOAD_FOLDER,"*.jpg"))
            filelist2 = glob.glob(os.path.join(UPLOAD_FOLDER,"*.png"))
            for f in filelist:
                os.remove(f)
            for f in filelist2:
                os.remove(f)

            return render_template("index.html",sarcasm_prediction = sarcasm_predictions[0], allegation_prediction = allegation_predictions[0], text_only_pred = text_only_preds[0])

    return render_template("index.html",sarcasm_prediction=0, allegation_prediction=0, text_only_pred=0)


if __name__ == "__main__":
    model = MultiModalBertClf(no_of_classes,bert_tokenizer)
    vocab = Vocab()
    try:
	    model.load_state_dict(torch.load('pretrained/sarcasm.pth'))
	    print('Sarcasm Model Loaded Successfully')  
    except:
	    print('Sarcasm Model load was Unsucessful')

    model_list.append(model)

    del model
    
    model = MultiModalBertClf(no_of_classes,bert_tokenizer)

    try:
        model.load_state_dict(torch.load('pretrained/allegation.pth'))
        print('Allegation Model loaded Successfully')
    except:
        print('Allegation Model load was Unsucessful')

    model_list.append(model)

    del model

    model = MultiModalBertClf(no_of_classes,bert_tokenizer)

    try:
        model.load_state_dict(torch.load('pretrained/text_only_informative.pth'))
        print('Text_Only Model loaded Successfully')
    except:
        print('Text_Only Model load was unsuccessful')
    
    model_list.append(model)

    del model

    print(len(model_list))

    app.run(port = 12000, debug = True)