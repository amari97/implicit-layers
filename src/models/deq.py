import os
import numpy as np
from torch import nn
import torch
from torch import autograd
import torch.nn.functional as F
import sys

sys.path.append("deq-master")

from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
class DEQLayer(nn.Module):
    def __init__(self, layer, solver=broyden,f_thres=10,b_thres=10,device=device):
        super().__init__()
        self.f = layer
        self.solver = solver
        self.f_thres=f_thres
        self.b_thres=b_thres
    
    def forward(self, x):
        z0 = torch.zeros_like(x).to(device)

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x), z0, threshold=self.f_thres)['result']   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            new_z_star = self.f(z_star.requires_grad_(), x)
            
            # Jacobian-related computations, see additional step above. For instance:
            jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    #torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = self.solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad, \
                                       torch.zeros_like(grad), threshold=self.b_thres)['result']
                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)
        return new_z_star
    
class Convex_f(nn.Module):
    def __init__(self,x_size,device=device) -> None:
        super().__init__()
        self.D=torch.Tensor(-np.diag(np.ones(x_size-1),-1)+2*np.diag(np.ones(x_size))-np.diag(np.ones(x_size-1),1)).view(1,x_size,x_size).to(device)
        self.mid=1/2*torch.Tensor(np.diag(np.ones(x_size-1),-1)+np.diag(np.ones(x_size-1),1)).view(1,x_size,x_size).to(device)
        self.D[:,0,:]=0
        self.D[:,x_size-1,:]=0
     
    def forward(self,x,param):
        x=x+param
        dx=(torch.matmul(self.D,x)>0)*(x-torch.matmul(self.mid,x))
        return (x-dx-param)

class Convex(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        def solve(f,z0,*args,**kwargs):
            z=z0
            for i in range(100):
                z=f(z)
            return {"result":z}
        self.model=DEQLayer(layer,solver=solve)
        
    def forward(self,x):
        return x+self.model(x)
    
class EvalConvex(nn.Module):
    def __init__(self, max_range):
        super().__init__()
        self.max_range=max_range
        
    def forward(self,x,param):
        # assume x in [0,1]
        x=x*(self.max_range-1)
        x=torch.round(x).type(torch.int64).view(-1,1,1)
        return torch.gather(param,2,x)
        
    
class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        

    def forward(self, input):
        return input.view(-1,*self.shape)
    
class FullModel(nn.Module):
    def __init__(self, n_inp,n_f,n_hidden=20) -> None:
        super().__init__()
        self.n_inp=n_inp
        self.lin1=nn.Linear(n_inp,n_hidden)
        self.lin2=nn.Linear(n_hidden,n_f)
        self.conv=nn.Sequential(View(n_f,1),Convex(Convex_f(n_f)),View(1,n_f))
        self.eval_f=EvalConvex(n_f)
        
    def forward(self,x):
        x_=self.lin2(F.relu(self.lin1(x[...,:self.n_inp])))
        param=self.conv(x_)
        return self.eval_f(x[...,self.n_inp],param)
    
class FullModel2(nn.Module):
    def __init__(self, n_inp,n_f,n_hidden=20) -> None:
        super().__init__()
        self.n_inp=n_inp
        self.lin1=nn.Linear(n_inp,n_hidden)
        self.lin2=nn.Sequential(nn.Linear(n_hidden,n_f),View(1,n_f))
        self.eval_f=EvalConvex(n_f)
        
    def forward(self,x):
        x_=self.lin2(F.relu(self.lin1(x[...,:self.n_inp])))
        return self.eval_f(x[...,self.n_inp],x_)