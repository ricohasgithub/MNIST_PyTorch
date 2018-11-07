import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np

torch.manual_seed(0)

class Net_Xavier(nn.Module):
    def __init__(self,Layers):
        super(Net_Xavier,self).__init__()
        self.hidden = nn.ModuleList()

        for input_size,output_size in zip(Layers,Layers[1:]):
            linear=nn.Linear(input_size,output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(nn.Linear(input_size,output_size))
            
    def forward(self,x):
        L=len(self.hidden)
        for (l,linear_transform)  in zip(range(L),self.hidden):
            if l<L-1:
                x =F.tanh(linear_transform (x))
           
            else:
                x =linear_transform (x)
        
        return x

