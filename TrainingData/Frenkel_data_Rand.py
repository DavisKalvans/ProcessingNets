import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from TrainingData.general_problems import FrenkelKontorova

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

def build_argparser():
  p = argparse.ArgumentParser(description="Generate training and testing data for FK (const tau).")

  p.add_argument("--N", type = int, default = 5000)
  p.add_argument("--M", type = int, default = 1000)
  p.add_argument("--tau", type = float, default = 0.1)
  p.add_argument("--Nx", type = int, default = 4)
  p.add_argument("--g", type = float, default = 1)

  return p

def main(argv = None):
  args = build_argparser().parse_args(argv)

  # dimension of the problem D=2*N_x, number of space discretization points
  N_x = args.Nx
  D = int(2*N_x)
  g = args.g # Parameter of FK
  g_txt = str(g).replace('.', '_')

  # constant time step
  Tau = args.tau
  Tau_txt = str(Tau).replace('.', '')

  # Getting testing and training data
  seed = 3 # Seed for generating data
  np.random.seed(seed)

  N = args.N # Ammount of training data
  train_X = np.zeros((N, 1, D))
  train_Y = np.zeros((N, 1, D))
  train_Tau = Tau*np.ones((N, 1, 1))
  train_x0 = np.zeros((N, D))
  train_x0[:, :N_x] = 0.2*np.random.rand(N, N_x) -0.1 # Set q
  train_x0[:, N_x:] = 3*np.pi/2*np.random.rand(N, N_x) -3*np.pi/4 # Set p

  M = args.M # Ammount of testing data data
  test_X = np.zeros((M, 1, D))
  test_Y = np.zeros((M, 1, D))
  test_Tau = Tau*np.ones((M, 1, 1))
  test_x0 = np.zeros((M, D))
  test_x0[:, :N_x] = 0.2*np.random.rand(M, N_x) -0.1 # Set q
  test_x0[:, N_x:] = 3*np.pi/2*np.random.rand(M, N_x) -3*np.pi/4 # Set p


  # Compute training data
  for n in range(N):
      train_X[n, 0, :] = train_x0[n, :] 
      tau = train_Tau[n, 0, 0]
      # solve pendulum equations with RK45
      sol = odeint(FrenkelKontorova.problem, train_x0[n, :], [0, tau], args=(N_x, g), 
                      rtol = 1e-12, atol = 1e-12) 
      # save data
      train_Y[n, 0, :] = sol[1]

  # Compute testing data
  for m in range(M):
      test_X[m, 0, :] = test_x0[m, :]  
      tau = test_Tau[m, 0, 0]
      # solve pendulum equations with RK45
      sol = odeint(FrenkelKontorova.problem, test_x0[m, :], [0, tau], args=(N_x, g), 
                      rtol = 1e-12, atol = 1e-12) 
      # save data
      test_Y[m, 0, :] = sol[1]

  # Saving data
  np.savez(f'TrainingData/SavedTrainingData/Frenkel/Frenkel{D}d_g{g_txt}_RandN{N}M{M}ConstTau{Tau_txt}', 
            train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, 
            test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)
