import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from TrainingData.general_problems import HenonHeiles

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

plt.rcParams.update({
  "text.usetex": False,
  "font.family": "serif"
})

def build_argparser():
    p = argparse.ArgumentParser(description="Generate training and testing data for HenonHeiles (const tau).")

    p.add_argument("--N", type = int, default = 1000)
    p.add_argument("--M", type = int, default = 250)
    p.add_argument("--tau", type = float, default = 0.1)

    return p

def main(argv = None):
    args = build_argparser().parse_args(argv)

    # dimension of the problem D=2d
    d = 2
    D = 4

    Tau = args.tau # Constant time step

    # Getting testing and training data
    seed = 3 # Seed for generating data
    np.random.seed(seed)

    q1_max, q2_max = 0.9, 0.9
    p1_max, p2_max = 1, 1

    N = args.N # Ammount of training data
    train_X = np.zeros((N, 1, D))
    train_Y = np.zeros((N, 1, D))
    train_Tau = Tau*np.ones((N, 1, 1))

    train_done = 0
    train_q1 = []
    train_q2 = []
    train_p1 = []
    train_p2 = []
    while train_done != N: # Generate starting points for training
        q1 = 2*q1_max*np.random.rand()-q1_max
        q2 = 2*q2_max*np.random.rand()-q2_max
        p1 = 2*p1_max*np.random.rand()-p1_max
        p2 = 2*p2_max*np.random.rand()-p2_max
        point = np.array([[q1, q2, p1, p2]])

        if HenonHeiles.H(point) < 1/6 and HenonHeiles.H(point) >=0 and HenonHeiles.U(point) <= 1/6 and HenonHeiles.U(point) >=0:
            train_q1.append(q1)
            train_q2.append(q2)
            train_p1.append(p1)
            train_p2.append(p2)
            train_done += 1


    M = args.M # Ammount of testing data data
    test_X = np.zeros((M, 1, D))
    test_Y = np.zeros((M, 1, D))
    test_Tau = Tau*np.ones((M, 1, 1))

    test_done = 0
    test_q1 = []
    test_q2 = []
    test_p1 = []
    test_p2 = []
    while test_done != M: # Generate starting points for training
        q1 = 2*q1_max*np.random.rand()-q1_max
        q2 = 2*q2_max*np.random.rand()-q2_max
        p1 = 2*p1_max*np.random.rand()-p1_max
        p2 = 2*p2_max*np.random.rand()-p2_max
        point = np.array([[q1, q2, p1, p2]])

        if HenonHeiles.H(point) < 1/6 and HenonHeiles.H(point) >=0 and HenonHeiles.U(point) <= 1/6 and HenonHeiles.U(point) >=0:
            test_q1.append(q1)
            test_q2.append(q2)
            test_p1.append(p1)
            test_p2.append(p2)
            test_done += 1

    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")         
    ax.set_title("Phase portrait of Training data")  
    ax.grid(True)


    # Compute training data
    for n in range(N):
        train_X[n, 0, 0] = train_q1[n]
        train_X[n, 0, 1] = train_q2[n]
        train_X[n, 0, 2] = train_p1[n] 
        train_X[n, 0, 3] = train_p2[n] 
        x0 = train_X[n, 0, :]
        tau = train_Tau[n]
        # solve HenonHeiles equations with RK45
        sol = solve_ivp(HenonHeiles.problem, [0, tau], x0, method='RK45', 
                        rtol = 1e-12, atol = 1e-12) 
        # save data
        train_Y[n, 0, :] = sol.y[:, -1]

        ax.plot(np.stack((train_X[n, 0, 0], train_Y[n, 0, 0])),
                np.stack((train_X[n, 0, 1], train_Y[n, 0, 1])), 
                ls='-', color='k', linewidth='1.5')
        ax.plot(train_X[n, 0, 0], train_X[n, 0, 1], marker='o', ms=4, mfc='b', mec='b')
        ax.plot(train_Y[n, 0, 0], train_Y[n, 0, 1], marker='o', ms=4, mfc='r', mec='r', color='k')

    # compute testing data
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")         
    ax.set_title("Phase portrait of Testing data")  
    ax.grid(True)

    for m in range(M):
        test_X[m, 0, 0] = test_q1[m]
        test_X[m, 0, 1] = test_q2[m]
        test_X[m, 0, 2] = test_p1[m]  
        test_X[m, 0, 3] = test_p2[m]
        x0 = test_X[m, 0, :]
        tau = test_Tau[m]
        # solve HenonHeiles equations with RK45
        sol = solve_ivp(HenonHeiles.problem, [0, tau], x0, method='RK45', 
                        rtol = 1e-12, atol = 1e-12) 
        # save data
        test_Y[m, 0, :] = sol.y[:, -1]
        
        #==========================================================================
        # plot numerical result in phase plane
        #==============================================================================   
        ax.plot(np.stack((test_X[m, 0, 0], test_Y[m, 0, 0])),
                np.stack((test_X[m, 0, 1], test_Y[m, 0, 1])), 
                ls='-', color='k', linewidth='1.5')
        ax.plot(test_X[m, 0, 0], test_X[m, 0, 1], marker='o', ms=4, mfc='b', mec='b')
        ax.plot(test_Y[m, 0, 0], test_Y[m, 0, 1], marker='o', ms=4, mfc='r', mec='r', color='k')

    # Saving data
    Tau_txt = str(Tau).replace('.', '')
    np.savez(f'TrainingData/SavedTrainingData/HenonHeiles/HenonHeilesRandN{N}M{M}ConstTau{Tau_txt}', 
            train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, 
            test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)
