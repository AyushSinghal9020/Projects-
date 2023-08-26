import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
from transformers import AutoTokenizer
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset , DataLoader
import torch
from transformers import AutoModel
import torch.nn as nn
import wandb

train = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv").merge(
    pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv") , on = "prompt_id")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

length = np.empty(shape = train.shape[0] , dtype = int)

tokenizer.model_max_length = 8100

tokens = np.empty(shape = train.shape[0] , dtype = np.ndarray)

for index in tqdm.tqdm(range(train.shape[0]) , total = train.shape[0]):
                       
    stri = ""
    
    for columns in ["prompt_question" , "prompt_title" , "prompt_text" , "text"]:
        
        stri += "\n\n" + str(train[columns][index])
        
    tokens[index] = tokenizer(stri , return_tensors = 'np')["input_ids"]

class DataSet(Dataset):
    
    def __init__(self , target = "content"):
        
        train = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv").merge(
            pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv") , on = "prompt_id")
        
        self.embeds = np.load("/kaggle/working/Sample Tokens.npy" , allow_pickle = True).tolist()
        
        self.content = train["content"]
        self.wording = train["wording"]
        
        self.target = target
        
    def __len__(self): return self.content.shape[0]
    
    def __getitem__(self , index):
        
        r_embeds = torch.tensor(self.embeds[index] , dtype = torch.long)
        
        if self.target == "content": r_targets = torch.tensor(self.content[index] , dtype = torch.float32)
        if self.target == "wording": r_targets = torch.tensor(self.wording[index] , dtype = torch.float32)
            
        return r_embeds , r_targets

train = DataSet(target = "content")

train_d = DataLoader(train , shuffle = True , batch_size = 1)

class model(nn.Module):
    
    def __init__(self):
        super(model, self).__init__()
    
        self.r_model = AutoModel.from_pretrained("roberta-base")

        self.ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 1)
    
    def forward(self, inputs):
    
        emb = self.r_model(inputs)[0]
        emb = torch.mean(emb, axis=1)
        
        output = self.ln(emb)
        output = self.out(output)
        
        return output

sample_model = model().to("cuda")
loss_func = nn.MSELoss()
optim = torch.optim.Adam(sample_model.parameters() , lr = 1e-6 , weight_decay = 0.01)

# wandb.watch(ro , loss_func)

for x , y in tqdm.tqdm(train_d , total = len(train_d)):

    torch.cuda.empty_cache()
    x = x[0]
    if x.shape[1] > 512: x = x[: , :512]

    x = x.to("cuda")
    y = y.to("cuda")

    pred = sample_model(x)

    loss = loss_func(pred , y)
    wandb.log({"loss": loss})

    loss.backward

    optim.step()
