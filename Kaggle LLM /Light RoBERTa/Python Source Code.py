import pandas as pd
import os 
import numpy as np 
import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer , AutoModel , AutoConfig
import torch 
from torch.utils.data import Dataset
import torch.nn as nn

train = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

conf = AutoConfig.from_pretrained("roberta-base")

conf.update({"output_hidden_states":True,
               "hidden_dropout_prob": 0.0,
               "layer_norm_eps": 1e-7})

model = AutoModel.from_pretrained("roberta-base" , config = conf)

np.load("/kaggle/input/kaggle-llm-sample-embeddings/Robert A/Train Embeds.npy")

class PT_DataSet(Dataset):
    
    def __init__(self):
        
        self.embeds = np.load("/kaggle/input/kaggle-llm-sample-embeddings/Robert A/Train Embeds.npy")
        
    def __len__(self):return self.embeds.shape[0]
    
    def __getitem__(self , index):
        
        embeds = torch.tensor(self.embeds[index] , dtype = torch.float32)
                
        return embeds

train_e = PT_DataSet()
train_e = torch.utils.data.DataLoader(train_e , shuffle = True , batch_size = 1)

class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.linear1 = torch.nn.Linear(768, 512)
        self.activation1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(512, 256)
        self.activation2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(256, 64)
        self.activation3 = torch.nn.ReLU()

        self.linear4 = torch.nn.Linear(64, 5)
        # self.activation4 = torch.nn.Softmax(dim = 1)

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.activation2(x)
        # print(x.shape)

        x = self.linear3(x)
        x = self.activation3(x)

        x = self.linear4(x)
        # x = self.activation4(x)

        return x
        
model = Model()
loss = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters() , lr = 0.1)

t = pd.get_dummies(train["answer"]).to_numpy()
losss = []
for x , y in tqdm.tqdm(zip(train_e , t), total = len(train_e)):

    x = torch.tensor(x , dtype = torch.float32)
    y = torch.tensor(y , dtype = torch.float32)

    output = model(x).squeeze()

    losses = loss(output , y)

    losses.backward()

    optim.step()

    losss.append(losses)
