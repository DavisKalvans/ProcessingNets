from TrainingData.pendulum_data_Rand import main as data_gen
from trainPendulumRand import main as train_model
from convConfInt_Pendulum import main as conv_graphs


N = 320 # Training points
M = 80 # Testing points
tau = 0.1 # Timestep for training data
kernel = "Verlet" # "Euler" or "Verlet"
k = 2 # Tau exponent
nL = 4 # Layers
nN = 32 # Width of Layer
nM = 1 # Seed for model parameter initialization
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