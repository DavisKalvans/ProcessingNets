import numpy as np
from numpy import linalg as LA
import torch
### Classes with problems that contain all necessary functions.
### .problem for solve_ivp, .H for calculating Hamiltonian energy

class Pendulum():

    def problem(t, z): # Problem for odesolvers
        q, p = z
        f = np.zeros(2)
        f[0] = p
        f[1] = -np.sin(q)
        return f 
    
    def problem_odeint(z, t): # Problem for odesolvers
        q, p = z
        f = np.zeros(2)
        f[0] = p
        f[1] = -np.sin(q)
        return f 
    
    def H(z): # Hamiltonian
        q = z[:, 0]
        p = z[:, 1]
        H = p**2/2 - np.cos(q)
        return H
    
    def H_torch(z, extraParams = None): # Hamiltonian with tensors
        q = z[:, 0, 0]
        p = z[:, 0, 1]
        H = torch.pow(p, 2)/2 -torch.cos(q)

        return H
   

class HenonHeiles():

    def problem(t, z):
        q1, q2, p1, p2 = z
        f = np.zeros(4) # f's are partial derivatives of Hamiltonian function
        f[0] = p1
        f[1] = p2
        f[2] = -q1 -2*q1*q2
        f[3] = -q2 -q1**2 +q2**2
        return f
    
    def problem_odeint(z, t):
        q1, q2, p1, p2 = z
        f = np.zeros(4) # f's are partial derivatives of Hamiltonian function
        f[0] = p1
        f[1] = p2
        f[2] = -q1 -2*q1*q2
        f[3] = -q2 -q1**2 +q2**2
        return f
    
    def H(z):
        q1 = z[:, 0]
        q2 = z[:, 1]
        p1 = z[:, 2]
        p2 = z[:, 3]
        H = (p1**2 +p2**2)/2 +(q1**2 +q2**2)/2 +q1**2*q2 -q2**3/3
        return H
    
    def U(z): # Only potential energy
        q1 = z[:, 0]
        q2 = z[:, 1]
        p1 = z[:, 2]
        p2 = z[:, 3]
        U = (q1**2 +q2**2)/2 +q1**2*q2 -q2**3/3
        return U
    
    def H_torch(z, extraParams = None):
        q1 = z[:, 0, 0]
        q2 = z[:, 0, 1]
        p1 = z[:, 0, 2]
        p2 = z[:, 0, 3]
        H = (torch.pow(p1, 2) +torch.pow(p2, 2))/2 +(torch.pow(q1, 2) +torch.pow(q2, 2))/2 +torch.pow(q1, 2)*q2 -torch.pow(q2, 3)/3
        return H

class FrenkelKontorova():
    def problem(z, t, N, g):
        u = z[:N]
        p = z[N:]
        du = p
        dp = g * (np.roll(u, -1) - 2*u + np.roll(u, 1)) - np.sin(u)
        return np.concatenate([du, dp])
    
    def problem_odeint(z, t, N, g):
        u = z[:N]
        p = z[N:]
        du = p
        dp = g * (np.roll(u, -1) - 2*u + np.roll(u, 1)) - np.sin(u)
        return np.concatenate([du, dp])

    def H(z, N, g):
        u = z[:, :N]
        p = z[:, N:]
        H = p**2 / 2 + (1 - np.cos(u)) + g / 2 * (np.roll(u, -1, axis=1) - u)**2
        return np.sum(H, axis=1)

    def H_torch(z, N, g):
        u = z[:, 0, :N]
        p = z[:, 0, N:]
        H = torch.pow(p, 2) / 2 + (1 - torch.cos(u)) + g / 2 * torch.pow(torch.roll(u, -1, dims=1) - u, 2)
        return torch.sum(H, dim=1)