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

# dimension of the problem D=2d
d = 1
D = 2

def build_argparser():
    p = argparse.ArgumentParser(description="Generate training and testing data for Pendulum (varied tau).")

    p.add_argument("--N", type = int, default = 640)
    p.add_argument("--M", type = int, default = 160)
    p.add_argument("--T1", type = float, default = 0.05)
    p.add_argument("--T2", type = float, default = 0.2)
    p.add_argument("--data_num", type = int, required = True)

    return p
def main(argv = None):
    args = build_argparser().parse_args(argv)

    # for UniformTau123 timestep is from U[0.05, 0.2]
    T1 = args.T1
    T2 = args.T2
    data_num = args.data_num

    # Intervals for starting values of trajectories
    q_max = np.pi
    p_max = 2

    # Getting testing and training data
    seed = 2024 # Seed for generating data
    np.random.seed(seed)

    N = args.N # Ammount of training data
    train_X = np.zeros((N, 1, D))
    train_Y = np.zeros((N, 1, D))

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

    # Generate the timesteps
    h = (T2-T1)/10

    train_Tau = []
    train_Tau.append(np.random.uniform(T1+0*h, T1+1*h, size=N//10))
    for i in range(1, 10):
        train_Tau.append(np.random.uniform(T1+ i*h, T1 +(i+1)*h, size = N//10))

    train_Tau = np.concatenate(train_Tau)
    rng = np.random.default_rng() 
    rng.shuffle(train_Tau) # Shuffle them, so they aren't clumped up together
    train_Tau = train_Tau.reshape(N, 1, 1)


    M = args.M # Ammount of testing data data
    test_X = np.zeros((M, 1, D))
    test_Y = np.zeros((M, 1, D))
    test_Tau = np.random.uniform(T1, T2, size=M).reshape(M, 1, 1)

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
                ls='-', color='k', linewidth='1')
        ax.plot(train_X[n, 0, 0], train_X[n, 0, 1], marker='o', ms=4, mfc='tab:blue', mec='tab:blue')
        ax.plot(train_Y[n, 0, 0], train_Y[n, 0, 1], marker='o', ms=4, mfc='tab:red', mec='tab:red')

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
        
        # ==========================================================================
        # plot numerical result in phase plane
        # ==========================================================================
        ax.plot(np.stack((test_X[m, 0, 0], test_Y[m, 0, 0])),
                np.stack((test_X[m, 0, 1], test_Y[m, 0, 1])), 
                ls='-', color='k', linewidth='1.5')
        ax.plot(test_X[m, 0, 0], test_X[m, 0, 1], marker='o', ms=4, mfc='tab:blue', mec='tab:blue')
        ax.plot(test_Y[m, 0, 0], test_Y[m, 0, 1], marker='o', ms=4, mfc='tab:red', mec='tab:red')

    # Saving data
    np.savez(f'TrainingData/SavedTrainingData/Pendulum/Pendulum_TauN{N}M{M}UniformTau{data_num}', 
            train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, 
            test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)

    plt.tight_layout()
    plt.show()

