import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import os
from pytorch_pretrained_bert.modeling import BertModel
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tqdm
import datetime
import random
from flask import Flask,render_template
from flask import request
import glob


app = Flask(__name__)
UPLOAD_FOLDER = "C:\\Users\\josep\\OneDrive\\Desktop\\Sarcasm\\img_upload"

bert = BertModel.from_pretrained('bert-base-uncased')
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


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]

        self.model = nn.Sequential(*modules)
        if(torch.cuda.is_available()):
            self.model = self.model.cuda()

    def forward(self, x):
        out = (self.model(x))

        out = nn.AdaptiveAvgPool2d((7, 1))(out)

        out = torch.flatten(out, start_dim=2)

        out = out.transpose(1, 2).contiguous()
        
        return out


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
        if torch.cuda.is_available():
            cls_id = cls_id.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
        cls_id = cls_id.unsqueeze(0).expand(batch_size, 1)
        
        if torch.cuda.is_available():
            cls_id = cls_id.cuda()
        cls_token_embeddings = self.word_embeddings(cls_id)
        
        sep_id = torch.LongTensor([102])
        if torch.cuda.is_available():
            sep_id = sep_id.cuda()
            self.img_embeddings = self.img_embeddings.cuda()
        sep_id = sep_id.unsqueeze(0).expand(batch_size, 1)
        sep_token_embeddings = self.word_embeddings(sep_id)
        
        batch_image_embeddings_768 = self.img_embeddings(batch_input_imgs)
        
        token_embeddings = torch.cat(
        [cls_token_embeddings, batch_image_embeddings_768, sep_token_embeddings], dim=1)
        
        position_ids = torch.arange(seq_length, dtype=torch.long)
        if torch.cuda.is_available():
            position_ids = position_ids.cuda()
            self.position_embeddings = self.position_embeddings.cuda()
            self.token_type_embeddings= self.token_type_embeddings.cuda()
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        position_embeddings = self.position_embeddings(position_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeddings+position_embeddings+token_type_embeddings
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
            self.LayerNorm=self.LayerNorm.cuda()
            self.dropout=self.dropout.cuda()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

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
        if torch.cuda.is_available():
            temp = temp.cuda()
            self.encoder = self.encoder.cuda()
            self.pooler = self.pooler.cuda()
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
        if(torch.cuda.is_available()):
            image_token_type_ids= image_token_type_ids.cuda()
        
        image = self.image_encoder(input_image)
        image_embedding_out = self.image_embeddings(image, image_token_type_ids)
        
        text_embedding_out = self.embeddings(input_text, text_segment)
        
        encoder_input = torch.cat([image_embedding_out, text_embedding_out], dim=1)

    
        encoded_layers = self.encoder(encoder_input, extended_attention_mask, output_all_encoded_layers=False)

        final = self.pooler(encoded_layers[-1])

        return final

class MultiModalBertClf(nn.Module):
    def __init__(self, no_of_classes, tokenizer):
        super(MultiModalBertClf, self).__init__()
        self.no_of_classes = no_of_classes
        self.enc = MultiModalBertEncoder(self.no_of_classes, tokenizer)
        self.batch_norm = nn.BatchNorm1d(768)
        self.clf = nn.Linear(768, self.no_of_classes)
    
    def forward(self, text, text_attention_mask, text_segment, image):
        if(torch.cuda.is_available()):
            text = text.cuda()
            text_attention_mask=text_attention_mask.cuda()
            text_segment=text_segment.cuda()
            image = image.cuda()
            self.clf = self.clf.cuda()
            self.batch_norm = self.batch_norm.cuda()
        x = self.enc(text, text_attention_mask, text_segment, image)

        x = self.batch_norm(x)
        x = self.clf(x)

        return x 

device = torch.device("cpu")

if torch.cuda.is_available():
	device = torch.device("cuda")

class SubmissionDataSet(Dataset):
    def __init__(self,data,transforms,tokenizer,vocab):
        self.data = data
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_len_sent = 128 - 7 -2
        self.vocab = vocab

    def __getitem__(self, index):
        text = self.data
        text = str(text)
        text = self.tokenizer.tokenize(text)[:self.max_len_sent]
        text = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(text))

        image = None

        try:
            for f in glob.iglob(UPLOAD_FOLDER + "\\*"):
                image = Image.open(f)

            image = self.transforms(image)
        
        except:
            image = Image.fromarray(128*np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        return text,image

    def __len__(self):
        return len(self.data)

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


def model_forward_predict(i_epoch, model, batch):
    txt, segment, mask, img = batch
    
    if(torch.cuda.is_available()):
    	txt, img = txt.cuda(), img.cuda()
    	mask, segment = mask.cuda(), segment.cuda()

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

@app.route("/",methods = ["GET","POST"])

def upload_predict():

    if request.method == "POST":
        image_file = request.files["image"]
        txt = request.form["Tweet"]

        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)

            model.eval()

            test_submission_dataset = SubmissionDataSet(txt,img_transformations,bert_tokenizer,vocab)
            test_submission_dataloader = torch.utils.data.DataLoader(test_submission_dataset,batch_size=4,collate_fn=collate_function_for_submission)
            predictions = model_predict(test_submission_dataloader,model,no_of_classes,1)

            filelist = glob.glob(os.path.join(UPLOAD_FOLDER,"*.jpg"))
            for f in filelist:
                os.remove(f)

            print(predictions[0])

            return render_template("index.html",prediction = predictions[0],txt = txt)

    return render_template("index.html",prediction = 1)


if __name__ == "__main__":
    model = MultiModalBertClf(no_of_classes,bert_tokenizer)
    vocab = Vocab()
    try:
	    model.load_state_dict(torch.load('sarcasm.pth'))
	    print('Model Loaded Successfully')  
    except:
	    print('Model load was Unsucessful')

    app.run(port = 12000, debug = True)