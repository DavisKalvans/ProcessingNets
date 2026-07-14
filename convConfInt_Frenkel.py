import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from numerical_methods import verletStepNumpy_FrenkelKontorova
from scipy.integrate import odeint 
from TrainingData.general_problems import FrenkelKontorova
import scipy.stats as st

# Plotting setup 
import matplotlib
matplotlib.rc('font', size=24) # 24
matplotlib.rc('axes', titlesize=20) #20

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif"
})

def build_argparser():
    p = argparse.ArgumentParser(description="Convergence graph for Pendulum (const tau).")

    # PyTorch args
    p.add_argument("--threads", type = int, default = 1)
    p.add_argument("--device", type = str, default = "cpu")

    # Training/testing data
    p.add_argument("--N", type = int, default = 1000)
    p.add_argument("--M", type = int, default = 250)
    p.add_argument("--tau", type = float, default = 0.1)
    p.add_argument("--Nx", type = int, default = 4)
    p.add_argument("--g", type = float, default = 1)

    # Model params
    p.add_argument("--k", type = int, required = True)
    p.add_argument("--kernel", type = str, required = True)
    p.add_argument("--epochs", type = int, default = 100000)
    p.add_argument("--learning_rate", type = float, default = 1e-3)
    p.add_argument("--sch", type = bool, default = True)
    p.add_argument("--eta1", type = float, default = 1e-1)
    p.add_argument("--eta2", type = float, default = 1e-3)

    p.add_argument("--nL", type = int, required = True)
    p.add_argument("--nN", type = int, required = True)
    p.add_argument("--nM", type = int, required = True)

    # For convergence graphs
    p.add_argument("--Tend", type = float, default = 10)
    p.add_argument("--nr_trajects", type = int, default = 100)
    p.add_argument("--conf", type = float, default = 0.9)

    return p
def main(argv = None):
    args = build_argparser().parse_args(argv)

    # PyTorch setup 
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    device = args.device

    # Model parameters
    N, M, tau = args.N, args.M, args.tau # Training data, testing data, timestep
    tau_txt = str(tau).replace('.', '')

    N_x = args.Nx
    D = 2*N_x # Total dimensions of problem (2*d)
    g = args.g # FK parameter
    g_txt = str(g).replace('.', '_')

    epochs = args.epochs
    epochs_th = str(epochs/1000).replace('.0', '')

    # Parameters regarding scheduling
    learning_rate = args.learning_rate # Not used if sch = True
    sch = args.sch # Use scheduling for learning rate
    eta1 = args.eta1 # For scheduling (starting learning rate)
    eta2 = args.eta2 # For scheduling (ending learning rate)
    gamma = np.exp(np.log(eta2/eta1)/epochs)

    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    problem = "Frenkel"
    kernel = args.kernel
    if kernel == "Verlet":
        numeric_stepNumpy = verletStepNumpy_FrenkelKontorova

    k = args.k # tau exponent, natural number

    # Confidence level # How many trajectories to make predictions for
    conf = args.conf

    # Same as area from training data
    nr_trajects = args.nr_trajects
    np.random.seed(11)
    x0s = np.zeros((nr_trajects, D))
    x0s[:, :N_x] = 0.2*np.random.rand(nr_trajects, N_x) -0.1 # Set q
    x0s[:, N_x:] = 3*np.pi/2*np.random.rand(nr_trajects, N_x) -3*np.pi/4 # Set p

    taus = np.array([1, 10/12, 10/14, 10/16, 0.5, 0.4, 10/35, 10/39, 10/42, 0.2, 10/55, 10/60, 10/65, 10/70, 10/75, 10/80, 10/85, 10/90, 10/95,  0.1, 10/110, 10/120, 10/130, 10/140, 10/150, 10/160, 10/170, 0.05, 10/230, 10/270, 10/300, 10/350, 10/400, 10/450, 0.01, 0.005, 0.001], dtype = np.float64)

    Tend = args.Tend # How long to make predictions for

    # Get initial "analytical" solutions with a solver, to compute pred error at endpoint Tend later
    exacts = []
    H0s = []

    for i in range(nr_trajects):
        exact = odeint(FrenkelKontorova.problem, x0s[i], [0, Tend], args=(N_x, g), rtol = 1e-12, atol = 1e-12)
        exact = exact[-1]
        # exact = torch.tensor(exact, dtype = torch.float64, device = device).reshape((D))
        exacts.append(exact)

        H0 = FrenkelKontorova.H(np.array([x0s[i]]).reshape((1, D)), N_x, g)
        H0s.append(H0)

    # Load selected model
    nL, nN, nM = args.nL, args.nN, args.nM
    f = str(nL) + "L" + str(nN) + "n" + str(nM) + "m"
    if sch:
        model_name = f'TrainedModels/{kernel +f"_k{k}"}/{problem}{D}d/sch{problem}Rand_g{g_txt}_N{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_' +f
    else:
        model_name = f'TrainedModels/{kernel +f"_k{k}"}/{problem}{D}d/{problem}Rand_g{g_txt}_N{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_' +f

    model, *_ = torch.load(model_name, weights_only=False)


    # To save all the values
    errors = np.zeros(len(taus))
    errors_numeric = np.zeros(len(taus))
    errors_energy = np.zeros(len(taus))
    errors_energy_numeric = np.zeros(len(taus))

    # To actually save all values for the confidence interval calculation
    errors_all = [[] for _ in range(len(taus))]
    errors_numeric_all = [[] for _ in range(len(taus))]
    errors_energy_all = [[] for _ in range(len(taus))]
    errors_energy_numeric_all = [[] for _ in range(len(taus))]

    for tr in range(nr_trajects):
        # Save everything in lists, for different tau values
        predictions = []
        predictions_numeric = []
        energies_pred = []
        energies_numeric = []


        ### Calculate everything with differing tau values
        for tau in taus:
            MM = int(Tend/tau) # Time steps
            tm = np.linspace(0, Tend, MM+1)

            # Get model predictions
            pred_preprocess = np.zeros([MM+1, D])
            Z = torch.tensor(x0s[tr], dtype=torch.float64, device=device).reshape((1, 1, D))
            Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)

            with torch.no_grad():
                preprocess, _ = model(Z, Tau) # Pass trough model

            pred_preprocess[0] = preprocess.reshape((1, D)).numpy()

            for i in range(MM): # Do the kernel method
                pred_preprocess[i+1] = numeric_stepNumpy(pred_preprocess[i], tau, g)
                
            pred_preprocess = torch.from_numpy(np.float64(pred_preprocess)).reshape((MM+1, 1, D))
            with torch.no_grad():
                pred, _ = model.back(pred_preprocess, Tau) # Pass trough inverse model

            pred = pred.reshape(MM+1, D)
            predictions.append(pred[-1].numpy()) 
            energies_pred.append(FrenkelKontorova.H(pred.numpy(), N_x, g))

            # Get numerical method predictions
            pred_numeric = np.zeros([MM+1, D])
            pred_numeric[0, :] = x0s[tr]

            for i in range(MM): 
                pred_numeric[i+1] = numeric_stepNumpy(pred_numeric[i], tau, g)
            
            predictions_numeric.append(pred_numeric[-1])
            energies_numeric.append(FrenkelKontorova.H(pred_numeric, N_x, g))

        ### Calculate errors for our predictions at endpoint
        # and energy/angular errors as max deviation in the whole interval [0, Tend]
        for i in range(len(taus)):
            errors[i] += np.sqrt(np.sum((predictions[i] - exacts[tr])**2, 0)) /np.sqrt(np.sum((exacts[tr])**2, 0))
            errors_all[i].append(np.sqrt(np.sum((predictions[i] - exacts[tr])**2, 0)) /np.sqrt(np.sum((exacts[tr])**2, 0)))

            errors_numeric[i] += np.sqrt(np.sum((predictions_numeric[i] -exacts[tr])**2, 0)) /np.sqrt(np.sum((exacts[tr])**2, 0))
            errors_numeric_all[i].append(np.sqrt(np.sum((predictions_numeric[i] -exacts[tr])**2, 0)) /np.sqrt(np.sum((exacts[tr])**2, 0)))

            errors_energy[i] += max(abs((energies_pred[i] -H0s[tr])/H0s[tr]))
            errors_energy_all[i].append(max(abs((energies_pred[i] -H0s[tr])/H0s[tr])))

            errors_energy_numeric[i] += max(abs((energies_numeric[i] -H0s[tr])/H0s[tr]))
            errors_energy_numeric_all[i].append(max(abs((energies_numeric[i] -H0s[tr])/H0s[tr])))

    # Plot name
    plot_name = model_name.replace(f'{problem}{D}d/', f'{problem}{D}d/ConvGraphs/')

    ### Lines with specific order, to compare numeric method to new method
    line1 = np.array(taus)**1
    line2 = np.array(taus)**2

    ### Plot absolute errors
    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.loglog(np.array(taus), errors/nr_trajects, color = 'tab:red', ls='--', marker='s', linewidth = '1.5', label = f'Proc. {kernel}')
    ax.loglog(np.array(taus), errors_numeric/nr_trajects, color = 'tab:green', ls='-', marker='o', linewidth = '1.5', label = f'{kernel}')
    #ax.loglog(np.array(taus), line1, color = 'k', label = "line of slope 1")
    #ax.loglog(np.array(taus), line2, color = 'k', label = "line of slope 2")

    # Confidence intervals
    error_ci = np.zeros((len(taus), 2))
    error_numeric_ci = np.zeros((len(taus), 2))
    error_energy_ci = np.zeros((len(taus), 2))
    error_energy_numeric_ci = np.zeros((len(taus), 2))

    for i in range(len(taus)):
        error_ci[i] = st.t.interval(conf, len(errors_all[i])-1, loc = np.mean(errors_all[i]), scale = st.sem(errors_all[i]))
        error_numeric_ci[i] = st.t.interval(conf, len(errors_numeric_all[i])-1, loc = np.mean(errors_numeric_all[i]), scale = st.sem(errors_numeric_all[i]))
        error_energy_ci[i] = st.t.interval(conf, len(errors_energy_all[i])-1, loc = np.mean(errors_energy_all[i]), scale = st.sem(errors_energy_all[i]))
        error_energy_numeric_ci[i] = st.t.interval(conf, len(errors_energy_numeric_all[i])-1, loc = np.mean(errors_energy_numeric_all[i]), scale = st.sem(errors_energy_numeric_all[i]))

    ax.fill_between(np.array(taus), error_ci[:, 0], error_ci[:, 1], color = 'r', alpha=.1)
    ax.fill_between(np.array(taus), error_numeric_ci[:, 0], error_numeric_ci[:, 1], color = 'g', alpha=.1)
    ax.axis([10**(-3), 0.5, 10**(-7), 10**(0)])
    plt.xticks([10**(-3), 10**(-2), 10**(-1), 0.2, 0.5])
    plt.yticks([10**(-7), 10**(-6), 10**(-5), 10**(-4), 
                10**(-3), 10**(-2), 10**(-1), 10**(0)])

    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Solution error')
    ax.set_title(f"{kernel} Frenkel {D}d: k={k}, L={nL}, N={nN}, M={nM}")
    plt.tight_layout()
    plt.savefig(plot_name + '_mse', dpi=300, bbox_inches='tight')
    plt.show()

    ### Plots hamiltonian errors
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.loglog(np.array(taus), errors_energy/nr_trajects, color = 'tab:red', ls='--', marker='s', linewidth = '1.5', label = f'Proc. {kernel}')
    ax.loglog(np.array(taus), errors_energy_numeric/nr_trajects, color = 'tab:green', ls='-', marker='o', linewidth = '1.5', label = f'{kernel}')
    #ax.loglog(np.array(taus), line1, color = 'k', label = "line of slope 1")
    #ax.loglog(np.array(taus), line2, color = 'k', label = "line of slope 2")
    ax.fill_between(np.array(taus), error_energy_ci[:, 0], error_energy_ci[:, 1], color = 'r', alpha=.1)
    ax.fill_between(np.array(taus), error_energy_numeric_ci[:, 0], error_energy_numeric_ci[:, 1], color = 'g', alpha=.1)

    ax.axis([10**(-3), 0.5, 10**(-7), 10**(0)])
    plt.xticks([10**(-3), 10**(-2), 10**(-1), 0.2, 0.5])
    plt.yticks([10**(-7), 10**(-6), 10**(-5), 10**(-4), 
                10**(-3), 10**(-2), 10**(-1), 10**(0)])

    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Hamiltonian error')
    ax.set_title(f"{kernel} Frenkel {D}d: k={k}, L={nL}, N={nN}, M={nM}")
    plt.tight_layout()
    plt.savefig(plot_name + '_ham', dpi=300, bbox_inches='tight')
    plt.show()

