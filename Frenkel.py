from TrainingData.Frenkel_data_Rand import main as data_gen
from trainFrenkelRand import main as train_model
from convConfInt_Frenkel import main as conv_graphs


N = 5000 # Training points
M = 1000 # Testing points
tau = 0.1 # Timestep for training data
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
    "--tau", str(tau),
    "--Nx", str(N_x),
    "--g", str(g),
])

train_model([
    "--N", str(N),
    "--M", str(M),
    "--tau", str(tau),
    "--Nx", str(N_x),
    "--g", str(g),
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
    "--tau", str(tau),
    "--Nx", str(N_x),
    "--g", str(g),
    "--kernel", kernel,
    "--epochs", str(epochs),
    "--k", str(k),
    "--nL", str(nL),
    "--nN", str(nN),
    "--nM", str(nM),
])