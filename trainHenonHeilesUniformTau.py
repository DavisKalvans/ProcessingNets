import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from numerical_methods import eulerStep_HenonHeiles, verletStep_HenonHeiles
from NeuralNetwork.symp_module_class import SympGradModule
from NeuralNetwork.mySequential import mySequential
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.training_class import train_loopGeneral, test_loopGeneral

# Plotting setup
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

def build_argparser():
    p = argparse.ArgumentParser(description="Train HenonHeiles (varied tau).")

    # PyTorch args
    p.add_argument("--threads", type = int, default = 1)
    p.add_argument("--device", type = str, default = "cpu")


    # Training/testing data
    p.add_argument("--N", type = int, default = 1000)
    p.add_argument("--M", type = int, default = 250)
    p.add_argument("--T1", type = float, default = 0.1)
    p.add_argument("--T2", type = float, default = 0.25)
    p.add_argument("--data_num", type = int, required = True)

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

    return p

def main(argv = None):
    args = build_argparser().parse_args(argv)

    # PyTorch setup 
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    device = args.device

    # Load training data
    N, M, data_num = args.N, args.M, args.data_num # Training data, testing data, data_num = '4' - timesteps from U[0.1, 0.25]

    npz_file = np.load(f'TrainingData/SavedTrainingData/HenonHeiles/HenonHeiles_TauN{N}M{M}UniformTau{data_num}.npz')
    problem = "HenonHeilesUniTau"

    x_train = torch.from_numpy(np.float64(npz_file['train_X'])).to(device)  
    y_train = torch.from_numpy(np.float64(npz_file['train_Y'])).to(device) 
    tau_train = torch.from_numpy(np.float64(npz_file['train_Tau'])).to(device)
    x_test = torch.from_numpy(np.float64(npz_file['test_X'])).to(device)
    y_test = torch.from_numpy(np.float64(npz_file['test_Y'])).to(device)
    tau_test = torch.from_numpy(np.float64(npz_file['test_Tau'])).to(device)

    # Data loader for PyTorch
    training_data = CustomDataset(x_train, y_train, tau_train)
    testing_data = CustomDataset(x_test, y_test, tau_test)

    train_dataloader = DataLoader(training_data, batch_size=N)
    test_dataloader = DataLoader(testing_data, batch_size=M)


    kernel = args.kernel
    if kernel == "Euler":
        numeric_step = eulerStep_HenonHeiles
    elif kernel == "Verlet":
        numeric_step = verletStep_HenonHeiles

    k = args.k # tau exponent, natural number
    kernel += "_k" +str(k)

    D = x_train.shape[2]
    d = int(D/2)

    # Model parameters
    epochs = args.epochs
    epochs_th = str(epochs/1000).replace('.0', '')
    sigma = 1 # Standartdeviation for model weight initialization

    # Scheduling stuff
    learning_rate = args.learning_rate # Not used, if sch = True
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


    # Check if such model isn't already trained, skip training if so
    nL, nN, nM = args.nL, args.nN, args.nM
    f = str(nL) + "L" + str(nN) + "n" + str(nM) + "m"
    if sch:
        model_name = f'TrainedModels/{kernel}/{problem}/sch{problem}Rand_N{N}M{M}UniformTau{data_num}_{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_' +f
    else:
        model_name = f'TrainedModels/{kernel}/{problem}/{problem}RandN{N}M{M}UniformTau{data_num}_{epochs_th}TH_{eta1_txt}eta1_' +f

    if os.path.exists(model_name):
        print('Model is already trained')
    else:

        # Create model
        torch.manual_seed(nM)
        layers = []
        for n in  range(nL):
            layers.append(SympGradModule(d, nN, nL, sigma, k))

        model = mySequential(*layers).to(device)
                            
        # Initialize the loss function
        loss_fn = nn.MSELoss()
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                        
        # start time
        start = time.time()
        
        # Training
        loss = np.zeros(epochs)
        acc = np.zeros(epochs)

        for t in range(epochs):
            loss[t] = train_loopGeneral(numeric_step, train_dataloader, model, loss_fn, optimizer, scheduler, sch)
            acc[t] = test_loopGeneral(numeric_step, test_dataloader, model, loss_fn)

            if t % 1000 == 0:
                print('Epoch %d / loss: %.5e / acc: %.5e' % (t+1, loss[t], acc[t]))


        # end time
        end = time.time()
        # total time taken
        print(f"Runtime of the program was {(end - start)/60:.4f} min.")
        
        # Save models
        model = model.to('cpu') # Transfer to cpu, otherwise there can be problems loading on cpu if was trained and saved with gpu
        torch.save([model, loss, acc, start, end], model_name)

        ### Plot MSE
        fig, ax = plt.subplots(figsize=(9, 6.5))
        v = np.linspace(1, epochs, epochs)
        ax.loglog(v, loss, ls='-', color='tab:red', linewidth='1.5', label='Loss')
        ax.loglog(v, acc, ls='--', color='tab:blue', linewidth='1.5', label='Accuracy')
        ax.set_xlabel("epochs")
        ax.set_ylabel("error")
        ax.grid(True)
        ax.legend(loc=3, shadow=True, prop={'size': 20}) 
        ax.axis([1, epochs, 10**(-9), 10^3])
        plt.yticks([10**(-11), 10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10**1])

        ax.set_title(f'{problem}, L={nL}, N={nN}, m={nM}')
        # Save figure
        plt.savefig(model_name.replace(f'/{problem}/', f'/{problem}/MSE/') +".png", dpi=300, bbox_inches='tight')
        #plt.show()

        del model