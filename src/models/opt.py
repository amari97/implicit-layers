from torch import nn
import torch
import cvxpy as cp 
from cvxpylayers.torch import CvxpyLayer 

class Autoencoder(nn.Module):

    def __init__(self, enc_channel=[6,16,32],dec_channel=[32,16,6], projection_layer=False,n_constraint=256):
        super(Autoencoder,self).__init__()
        assert enc_channel[-1]==dec_channel[0]
        self.enc_channel=enc_channel
        self.projection_layer=projection_layer
        if self.projection_layer:
            self.projection_layer_in=self.build_proj(n_constraint)
 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, enc_channel[0], 5, stride=2), # 13x13
            nn.BatchNorm2d( enc_channel[0]),
            nn.ReLU(),
            nn.Conv2d(enc_channel[0],enc_channel[1],5, stride=2),# 7x7
            nn.BatchNorm2d( enc_channel[1]),
            nn.ReLU(),
            nn.Conv2d(enc_channel[1],enc_channel[2],4),# 1x1
            nn.BatchNorm2d(enc_channel[2]),
            nn.ReLU())
        
        self.classifier=nn.Sequential(nn.Linear(enc_channel[-1],20),nn.ReLU(),nn.Linear(20,10))
        
        self.decoder = nn.Sequential( 
            nn.ReLU(), 
            nn.ConvTranspose2d(dec_channel[0],dec_channel[1],4),
            nn.BatchNorm2d( dec_channel[1]),
            nn.ReLU(),           
            nn.ConvTranspose2d(dec_channel[1],dec_channel[2],5,stride=2, output_padding=1),
            nn.BatchNorm2d( dec_channel[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(dec_channel[2],1,5,stride=2, output_padding=1),
            nn.Sigmoid())
    
    def build_proj(self,n_constraint):
        m=self.enc_channel[-1]
        X = cp.Variable((m))
        H = cp.Parameter((n_constraint,m))
        g = cp.Parameter((n_constraint))
        p = cp.Parameter((m))
        objective_fn = cp.norm2(X - p)
        constraints = [H@X <= g,g>=0]
            
        problem = cp.Problem(cp.Minimize(objective_fn), constraints) 
        proj1 =CvxpyLayer(problem, parameters=[H,g,p], variables=[X])
        # Register parameter to module
        self.H = nn.Parameter(torch.randn(n_constraint,m, requires_grad=True))
        self.g = nn.Parameter(torch.ones(n_constraint, requires_grad=True))
        return proj1


    def forward(self,x):
        x_latent = self.encoder(x)
        x_proj= x_latent
        if self.projection_layer:
            shape=x_latent.shape
            x_latent=x_latent.view(-1,self.enc_channel[-1])
            x_proj=self.projection_layer_in(self.H,self.g,x_latent)[0]
            x_proj=x_proj.view(*shape)
        y=self.classifier(x_proj.view(x_proj.shape[0],-1))
        x = self.decoder(x_proj)
        return x,y,x_proj,x_latent
    