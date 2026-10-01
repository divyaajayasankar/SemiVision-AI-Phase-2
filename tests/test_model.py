import torch
from src.models_convae import ConvAE

def test_convae_forward_shape():
    model=ConvAE(8); x=torch.rand(2,1,64,64); y=model(x); assert y.shape==x.shape; assert torch.all((y>=0)&(y<=1))
