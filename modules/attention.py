import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    https://arxiv.org/pdf/1506.07503.pdf
    """
    def __init__(self, dec_hidden_dim, enc_hidden_dim, attn_dim):
        super(Attention, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim#256
        self.attn_dim = attn_dim#256
        self.enc_hidden_dim = enc_hidden_dim#128

        self.W = nn.Linear(self.dec_hidden_dim, self.attn_dim, bias=False)#64 347 256
        self.V = nn.Linear(self.enc_hidden_dim, self.attn_dim, bias=False)#64 347 256

        self.fc = nn.Linear(self.attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.rand(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)



    def forward(self,Si_1,Hj):
        score =  self.fc(self.tanh(
         self.W(Si_1) + self.V(Hj)  + self.b
        )).squeeze(dim=-1)#64 347 1
        #print(score.shape)64 347
        attn_weight = self.softmax(score) 
        #print(attn_weight.shape) 64 347
        context = torch.bmm(attn_weight.unsqueeze(dim=1), Hj)#hj 64 347 128 为encoder的全部中间状态结果 si-l为decoder输出结果
        #print(context.shape) 64 1 128

        return context, attn_weight