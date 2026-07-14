import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from TrainingData.general_problems import Pendulum

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

plt.rcParams.update({
  "text.usetex": False,
  "font.family": "serif"
})

def build_argparser():
    p = argparse.ArgumentParser(description="Generate training and testing data for Pendulum (const tau).")

    p.add_argument("--N", type = int, default = 320)
    p.add_argument("--M", type = int, default = 80)
    p.add_argument("--tau", type = float, default = 0.1)

    return p

def main(argv = None):
    args = build_argparser().parse_args(argv)

    # dimension of the problem D=2d
    d = 1
    D = 2

    Tau = args.tau # Constant time step
    Tau_txt = str(Tau).replace('.', '')

    # Getting testing and training data
    seed = 3 # Seed for generating data
    np.random.seed(seed)

    q_max = np.pi
    p_max = 2

    N = args.N # Ammount of training data
    train_X = np.zeros((N, 1, D))
    train_Y = np.zeros((N, 1, D))
    train_Tau = Tau*np.ones((N, 1, 1))

    train_done = 0
    train_q = []
    train_p = []
    while train_done != N: # Generate starting points for training
        q = 2*q_max*np.random.rand()-q_max 
        p = 2*p_max*np.random.rand()-p_max 
        point = np.array([[q, p]])

        if np.abs(Pendulum.H(point)) < 1: # Only take periodic trajectories, i.e. satisfying -1<H(q,p)<1
            train_q.append(q)
            train_p.append(p)
            train_done += 1

    M = args.M # Ammount of testing data data
    test_X = np.zeros((M, 1, D))
    test_Y = np.zeros((M, 1, D))
    test_Tau = Tau*np.ones((M, 1, 1))

    test_done = 0
    test_q = []
    test_p = []
    while test_done != M: # Generate starting points for training
        q = 2*q_max*np.random.rand()-q_max 
        p = 2*p_max*np.random.rand()-p_max 
        point = np.array([[q, p]])

        if np.abs(Pendulum.H(point)) < 1: # Only take periodic trajectories, i.e. satisfying -1<H(q,p)<1
            test_q.append(q)
            test_p.append(p)
            test_done += 1


    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$")         
    ax.set_title("Phase portrait of Training data")  
    ax.grid(True)
    ax.axis([-np.pi-0.1, np.pi+0.1, -2.1, 2.1])

    # Plot some trajectories
    qs = (-3.14, -2.5, -2, -1.5, -1, -0.5)
    truth = []
    Tend = 40
    for q in qs:
        sol = solve_ivp(Pendulum.problem, [0, Tend], (q, 0), method='RK45',
                        rtol = 1e-12, atol = 1e-12)
        sol = sol.y.T
        ax.plot(sol[:, 0], sol[:, 1], linewidth='1', ls='-', color='tab:gray')


    # Compute training data
    for n in range(N):
        train_X[n, 0, 0] = train_q[n]
        train_X[n, 0, 1] = train_p[n] 
        x0 = train_X[n, 0, :]
        tau = train_Tau[n]
        # solve pendulum equations with RK45
        sol = solve_ivp(Pendulum.problem, [0, tau], x0, method='RK45', 
                        rtol = 1e-12, atol = 1e-12) 
        # save data
        train_Y[n, 0, :] = sol.y[:, -1]

        ax.plot(np.stack((train_X[n, 0, 0], train_Y[n, 0, 0])),
                np.stack((train_X[n, 0, 1], train_Y[n, 0, 1])), 
                ls='-', color='k', linewidth='1.5')
        ax.plot(train_X[n, 0, 0], train_X[n, 0, 1], marker='o', ms=4, mfc='b', mec='b')
        ax.plot(train_Y[n, 0, 0], train_Y[n, 0, 1], marker='o', ms=4, mfc='r', mec='r', color='k')

    # plt.savefig(f'PendulumRandN{N}M{M}ConstTau{Tau_txt}_training', dpi=300, bbox_inches='tight')
    plt.show()

    # compute testing data
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$")         
    ax.set_title("Phase portrait of Testing data")  
    ax.grid(True)
    ax.axis([-np.pi-0.1, np.pi+0.1, -2.1, 2.1])


    # Plot some trajectories
    qs = (-3.14, -2.5, -2, -1.5, -1, -0.5)
    truth = []
    Tend = 40
    for q in qs:
        sol = solve_ivp(Pendulum.problem, [0, Tend], (q, 0), method='RK45',
                        rtol = 1e-12, atol = 1e-12)
        sol = sol.y.T
        ax.plot(sol[:, 0], sol[:, 1], linewidth='1', ls='-', color='tab:gray')

    for m in range(M):
        test_X[m, 0, 0] = test_q[m]
        test_X[m, 0, 1] = test_p[m]  
        x0 = test_X[m, 0, :]
        tau = test_Tau[m]
        # solve pendulum equations with RK45
        sol = solve_ivp(Pendulum.problem, [0, tau], x0, method='RK45', 
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

    # plt.savefig(f'PendulumRandN{N}M{M}ConstTau{Tau_txt}_testing', dpi=300, bbox_inches='tight')
    plt.show()
    # Saving data

    np.savez(f'TrainingData/SavedTrainingData/Pendulum/PendulumRandN{N}M{M}ConstTau{Tau_txt}', 
            train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, 
            test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)
