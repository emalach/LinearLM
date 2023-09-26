import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, num_tokens, T=128, d=256, activation='none'):
        super().__init__()
        self.d = d
        self.T = T
        self.in_embedding = nn.Embedding(num_tokens, d)
        self.linear1 = nn.Linear(T*d, T*d)
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(d,d)
        self.out_embedding = nn.Linear(d, num_tokens)
        self.mask = nn.Parameter(self._generate_square_subsequent_mask(T), requires_grad=False)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask
    
    def _masked_linear(self, linear, x, d_in, d_out):
        T = self.T
        W = linear.weight
        W = W.view(T, d_out, T, d_in)
        mask = self.mask.view((T, 1, T,1))
        masked_W = W*mask
        masked_W = masked_W.view((T*d_out,T*d_in))
        return F.linear(x, masked_W)
        
    def forward(self, x):
        if x.shape[-1] < self.T:
            x = F.pad(x, (0, self.T-x.shape[-1]))
        if x.shape[-1] > self.T:
            x = x[:,-self.T:]
        x = self.in_embedding(x)
        x = x.view((-1, self.d*self.T))
        x = self._masked_linear(self.linear1, x, self.d, self.d)
        x = x.view((-1, self.T, self.d))
        if self.activation is not None:
            x = self.activation(x)
            x = self.linear2(x)
            x = self.activation(x)
        x = self.out_embedding(x)
        return x
    
    def cuda(self):
        super().cuda()
        self.mask = self.mask.to('cuda')