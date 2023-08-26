import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np

from transformers import AutoTokenizer
import os 
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule as LDM
from transformers import AutoModelForSequenceClassification
from pytorch_lightning import LightningModule as LM
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
from kaggle_secrets import UserSecretsClient
import wandb
import torch
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("API LOGIN KEY")

wandb.login(key = api_key)



train = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")

def save_tokens(tokenizer_path):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    os.makedirs("/kaggle/working/Pseudo Dir/Embeds")
    
    tokens = [
    tokenizer(train['text'][index])['input_ids']
    for index 
    in tqdm.tqdm(range(train.shape[0]) , total = train.shape[0] , desc = 'Tokenizing Input --->')
    ]
    
    np.save('/kaggle/working/Pseudo Dir/Embeds/Hui Hui' , np.array(tokens))
    
    print('Tokens Saved')

save_tokens('roberta-base')
# save_tokens('albert-base-v2')
# save_tokens('microsoft/deberta-base')
# save_tokens('google/electra-small-discriminator')

class data(Dataset):
    
    def __init__(self):
        
        super().__init__()
        
        self.data = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv')
        self.tokens = np.load('/kaggle/working/Pseudo Dir/Embeds/Hui Hui.npy' , allow_pickle = True)
        
        self.wordings = self.data['wording']
        
    def __len__(self) : return self.data.shape[0]
    
    def __getitem__(self , index):
        
        r_tokens = torch.tensor(self.tokens[index] , dtype = torch.long)
        r_wordings = torch.tensor(self.wordings[index] , dtype = torch.float32)
        
        return r_tokens , r_wordings

class Data(LDM):
    
    def __init__(self , batch_size = 1):
        
        super().__init__()
        
        self.batch_size = batch_size
        
    def setup(self , stage = None):self.train = data()
       
    def train_dataloader(self) : return DataLoader(self.train , batch_size = self.batch_size)

data_module = Data()

class lightning(LM):
    
    def __init__(self , model_path):
        
        super().__init__()
        
        self.model_path = model_path
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path , num_labels = 1)
        self.loss_func = nn.MSELoss()
        
    def forward(self , inps): 
        
        if inps.shape[1] > 512 : inps = inps[: , :512]
            
        return self.model(inps).logits
    
    def training_step(self , batch , batch_idx):
        
        inputs , labels = batch
        outputs = self(inputs)
        
        loss = self.loss_func(outputs , labels)
        self.log('train_loss' , loss)
        
        return loss
    
    def configure_optimizers(self): return torch.optim.Adam(self.parameters())

model = lightning(model_path = 'roberta-base')
# model = lightning(model_path = 'albert-base-v2')
# model = lightning(model_path = 'mircosoft/deberta-base')
# model = lightning(model_path = 'google/electra-small-discriminator')

trainer = Trainer(
    max_epochs = 1 , 
    logger = WandbLogger(
        name = "PL | Roberta_Base | SC | ComonLit") , 
    accelerator = 'gpu' , devices = 2
)
trainer.fit(model , data_module)
wandb.finish()
