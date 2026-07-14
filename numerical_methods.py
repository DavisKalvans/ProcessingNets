import torch
import numpy as np

def eulerStep_Pendulum(x, tau):
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)

    p = x[:, 0, 1] -tau_vec*torch.sin(x[:, 0, 0])
    q = x[:, 0, 0] +tau_vec*p

    XX[:, 0, 0] = q
    XX[:, 0, 1] = p

    return XX

# def eulerStepNumpyVec_Pendulum(x, tau, d, D):
#     p = x[:, :, d:D] -tau*np.sin(x[:, :, 0:d])
#     q = x[:, :, 0:d] + tau*p

#     return np.concatenate((q, p), 2)

def eulerStep_HenonHeiles(x, tau):
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)

    p1 = x[:, 0, 2] +tau_vec*(-x[:, 0, 0] -2*x[:, 0, 0]*x[:, 0, 1])
    p2 = x[:, 0, 3] +tau_vec*(-x[:, 0, 1] -x[:, 0, 0]**2 +x[:, 0, 1]**2)

    q1 = x[:, 0, 0] +tau_vec*p1
    q2 = x[:, 0, 1] +tau_vec*p2

    XX[:, 0, 0] = q1
    XX[:, 0, 1] = q2
    XX[:, 0, 2] = p1
    XX[:, 0, 3] = p2

    return XX


### Verlet step functions with tensors for the training and testing loops
def verletStep_Pendulum(x, tau):
    tau_vec = tau.reshape(tau.size(0))
    tau_vec_half = tau_vec/2
    XX = torch.zeros_like(x)

    p_half = x[:, 0, 1] -tau_vec_half*torch.sin(x[:, 0, 0])
    q = x[:, 0, 0] +tau_vec*p_half
    p = p_half -tau_vec_half*torch.sin(q)

    XX[:, 0, 0] = q
    XX[:, 0, 1] = p

    return XX


def verletStep_HenonHeiles(x, tau):
    tau_vec = tau.reshape(tau.size(0))
    tau_vec_half = tau_vec/2
    XX = torch.zeros_like(x)

    p1_half = x[:, 0, 2] +tau_vec_half*(-x[:, 0, 0] -2*x[:, 0, 0]*x[:, 0, 1])
    p2_half = x[:, 0, 3] +tau_vec_half*(-x[:, 0, 1] -x[:, 0, 0]**2 +x[:, 0, 1]**2)

    q1 = x[:, 0, 0] +tau_vec*p1_half
    q2 = x[:, 0, 1] +tau_vec*p2_half
    p1 = p1_half +tau_vec_half*(-q1 -2*q1*q2)
    p2 = p2_half +tau_vec_half*(-q2 -q1**2 +q2**2)

    XX[:, 0, 0] = q1
    XX[:, 0, 1] = q2
    XX[:, 0, 2] = p1
    XX[:, 0, 3] = p2
    
    return XX


def verletStep_FrenkelKontorova(x, tau, g):
    D = x.shape[2]
    d = D // 2
    tau_vec = tau.reshape(tau.size(0), 1)
    tau_vec_half = tau_vec / 2
    XX = torch.zeros_like(x)
    p_half = x[:, 0, d:] + tau_vec_half * (g * (torch.roll(x[:, 0, :d], -1, dims=1) - 2*x[:, 0, :d] + torch.roll(x[:, 0, :d], 1, dims=1)) - torch.sin(x[:, 0, :d]))
    q = x[:, 0, :d] + tau_vec * p_half
    p = p_half + tau_vec_half * (g * (torch.roll(q, -1, dims=1) - 2*q + torch.roll(q, 1, dims=1)) - torch.sin(q))
    XX[:, 0, :d] = q
    XX[:, 0, d:] = p
    return XX

### Euler step functions as numpy arrays
def eulerStepNumpy_Pendulum(x, tau):
    p = x[1] -tau*np.sin(x[0])
    q = x[0] +tau*p

    return (q, p)

def eulerStepNumpy_HenonHeiles(x, tau):
    p1 = x[2] +tau*(-x[0] -2*x[0]*x[1])
    p2 = x[3] +tau*(-x[1] -x[0]**2 +x[1]**2)

    q1 = x[0] +tau*p1
    q2 = x[1] +tau*p2

    return (q1, q2, p1, p2)

### Verlet step functions as numpy arrays
def verletStepNumpy_Pendulum(x, tau):
    tau_half = tau/2

    p_half = x[1] -tau_half*np.sin(x[0])
    q = x[0] +tau*p_half
    p = p_half -tau_half*np.sin(q)

    return (q, p)

def verletStepNumpy_HenonHeiles(x, tau):
    tau_half = tau/2

    p1_half = x[2] +tau_half*(-x[0] -2*x[0]*x[1])
    p2_half = x[3] +tau_half*(-x[1] -x[0]**2 +x[1]**2)

    q1 = x[0] +tau*p1_half
    q2 = x[1] +tau*p2_half
    p1 = p1_half +tau_half*(-q1 -2*q1*q2)
    p2 = p2_half +tau_half*(-q2 -q1**2 +q2**2)

    return (q1, q2, p1, p2)

def verletStepNumpy_FrenkelKontorova(x, tau, g):
    tau_half = tau / 2
    D = len(x)
    d = D // 2
    p_half = x[d:] + tau_half * (g * (np.roll(x[:d], -1) - 2*x[:d] + np.roll(x[:d], 1)) - np.sin(x[:d]))
    q = x[:d] + tau * p_half
    p = p_half + tau_half * (g * (np.roll(q, -1) - 2*q + np.roll(q, 1)) - np.sin(q))
    return np.append(q, p)