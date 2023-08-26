import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self , vocab_size , num_heads):
        
        super(MultiHeadSelfAttention , self).__init__()
    
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        self.queries = nn.Linear(self.vocab_size , self.vocab_size)
        self.keys = nn.Linear(self.vocab_size , self.vocab_size)
        self.values = nn.Linear(self.vocab_size , self.vocab_size)
        
        self.softmax = torch.nn.Softmax()
        
    def split_heads(self , gate):
        
#         batch_size = gate.shape[0]
        
        split_gates = torch.reshape(gate, (1 , # -----> batch_size 
                                        self.num_heads , 
                                        int(self.vocab_size / self.num_heads) , 
                                        self.num_heads)).permute(2 , 1 , 0 , 3)
        
        return split_gates
    
    def forward(self , key , query , value , mask = None):
        
        query_output = self.queries(query)
        key_output = self.keys(key)
        value_output = self.values(value)

        query_output = self.split_heads(query_output)
        key_output = self.split_heads(key_output)
        value_output = self.split_heads(value_output)

        attention = (query_output * key_output) / (key_output.shape[-1] ** (1/2)) 
        
        if mask : attention = tf.where(mask == 0 , float('-inf') , attention)

        weights = self.softmax(attention)
        weights = torch.reshape(weights , (weights.shape[2] , int(self.vocab_size / self.num_heads) , 
                                           self.num_heads , self.num_heads))
        value_output = torch.reshape(value_output , (value_output.shape[2] , int(self.vocab_size / self.num_heads) , 
                                                     self.num_heads , self.num_heads))
        output = torch.matmul(weights , value_output)
        
        output = torch.reshape(output , (self.num_heads , self.vocab_size))

        return output , weights

class Encoder(nn.Module):
    
    def __init__(self , vocab_size , num_heads):
        
        super(Encoder , self).__init__()
        
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        
        self.mhsa = MultiHeadSelfAttention(self.vocab_size , self.num_heads)
        
        self.layer_norm_1 = nn.LayerNorm(self.vocab_size)
        self.layer_norm_2 = nn.LayerNorm(self.vocab_size)
        
        self.dropout_1 = nn.Dropout(0.05)
        self.dropout_2 = nn.Dropout(0.05)
        
        self.linear_1 = nn.Linear(self.vocab_size , self.vocab_size)
        
    def forward(self , inps):
        
        attention , weights = self.mhsa(inps , inps , inps , mask = None)

        attention = self.dropout_1(attention)
        attention = self.layer_norm_1(inps + attention)
        
        linear_attention = self.linear_1(attention)
        
        linear_attention = self.layer_norm_2(linear_attention)
        attention = self.dropout_2(linear_attention + attention)
        
        return attention , weights 
