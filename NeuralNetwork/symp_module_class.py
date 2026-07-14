"""
Neural network class for PyTorch
"""
import torch
from torch import nn
import numpy as np
torch.set_default_dtype(torch.float64)
    
# Symplctic gradient (sigmoid) Module 
class SympGradModule(nn.Module):
    def __init__(self, d, n, L, sigma, k):
        super().__init__()
        """
        Weights: W, w and bias vector b
        """
        self.d = d
        self.D = 2*d
        self.L = L
        self.k = k
        self.Wp = torch.nn.Parameter(sigma*torch.randn((n, d), dtype = torch.float64))
        self.wp = torch.nn.Parameter(sigma*torch.randn((n, 1), dtype = torch.float64))
        self.bp = torch.nn.Parameter(sigma*torch.zeros((1, n), dtype = torch.float64))
        self.Wq = torch.nn.Parameter(sigma*torch.randn((n, d), dtype = torch.float64))
        self.wq = torch.nn.Parameter(sigma*torch.randn((n, 1), dtype = torch.float64))
        self.bq = torch.nn.Parameter(sigma*torch.zeros((1, n), dtype = torch.float64))

    def forward(self, x, tau):
        """
        Forward function
        """
        sigma = nn.Sigmoid()
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
 
        # Symplectic Euler step
        Q = q + torch.matmul(tau**self.k, torch.matmul(sigma(torch.matmul(
                p, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        P = p - torch.matmul(tau**self.k, torch.matmul(sigma(torch.matmul(
                Q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        
        
        return torch.cat((Q, P), 2), tau

    def back(self, x, tau):
        """
        Backward function
        """
        sigma = nn.Sigmoid()
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
 
        # Symplectic Euler step
        P = p + torch.matmul(tau**self.k, torch.matmul(sigma(torch.matmul(
                q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        Q = q - torch.matmul(tau**self.k, torch.matmul(sigma(torch.matmul(
                P, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        
        
        
        return torch.cat((Q, P), 2), tau