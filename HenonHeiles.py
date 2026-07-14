from TrainingData.HenonHeiles_data_Rand import main as data_gen
from trainHenonHeilesRand import main as train_model
from convConfInt_HenonHeiles import main as conv_graphs


N = 1000 # Training points
M = 250 # Testing points
tau = 0.1 # Timestep for training data
kernel = "Verlet" # "Euler" or "Verlet"
k = 2 # Tau exponent
nL = 8 # Layers
nN = 32 # Width of Layer
nM = 9 # Seed for model parameter initialization
epochs = 100000 # Number of epochs for training

data_gen([
    "--N", str(N),
    "--M", str(M),
    "--tau", str(tau),
])

train_model([
    "--N", str(N),
    "--M", str(M),
    "--tau", str(tau),
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
    "--kernel", kernel,
    "--epochs", str(epochs),
    "--k", str(k),
    "--nL", str(nL),
    "--nN", str(nN),
    "--nM", str(nM),
])