from TrainingData.Frenkel_data_Tau_uniform import main as data_gen
from trainFrenkelRandUniformTau import main as train_model
from convConfInt_FrenkelUniformTau import main as conv_graphs


N = 5000 # Training points
M = 1000 # Testing points
data_num = "1"
if data_num == "1": # Time steps for training and testing data are sampled from (T1, T2)
    T1 = 0.01
    T2 = 0.2

N_x = 64 # dimensions D = 2*N_x
g = 1 # Parameter forn Frenkel
kernel = "Verlet" # "Euler" or "Verlet"
k = 1 # Tau exponent
nL = 2 # Layers
nN = 32 # Width of Layer
nM = 0 # Seed for model parameter initialization
epochs = 100000 # Number of epochs for training

data_gen([
    "--N", str(N),
    "--M", str(M),
    "--Nx", str(N_x),
    "--g", str(g),
    "--data_num", str(1),
])

train_model([
    "--N", str(N),
    "--M", str(M),
    "--Nx", str(N_x),
    "--g", str(g),
    "--data_num", str(1),
    "--kernel", kernel,
    "--epochs", str(epochs),
    "--k", str(k),
    "--nL", str(nL),
    "--nN", str(nN),
    "--nM", str(nM),
])

conv_graphs([
    "--N", str(N),
    "--M", str(M),
    "--Nx", str(N_x),
    "--g", str(g),
    "--data_num", str(1),
    "--kernel", kernel,
    "--epochs", str(epochs),
    "--k", str(k),
    "--nL", str(nL),
    "--nN", str(nN),
    "--nM", str(nM),
])
