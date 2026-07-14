from TrainingData.pendulum_data_Tau_uniform import main as data_gen
from trainPendulumUniformTau import main as train_model
from convConfInt_PendulumUniformTau import main as conv_graphs

N = 640 # Training points
M = 160 # Testing points
data_num = "123"
if data_num == "123": # Time steps for training and testing data are sampled from (T1, T2)
    T1 = 0.05
    T2 = 0.2

kernel = "Verlet" # "Euler" or "Verlet"
k = 2 # Tau exponent
nL = 8 # Layers
nN = 32 # Width of Layer
nM = 0 # Seed for model parameter initialization
epochs = 100000 # Number of epochs for training

data_gen([
    "--N", str(N),
    "--M", str(M),
    "--T1", str(T1),
    "--T2", str(T2),
    "--data_num", data_num,
])

train_model([
    "--N", str(N),
    "--M", str(M),
    "--T1", str(T1),
    "--T2", str(T2),
    "--data_num", data_num,
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
    "--T1", str(T1),
    "--T2", str(T2),
    "--data_num", data_num,
    "--kernel", kernel,
    "--epochs", str(epochs),
    "--k", str(k),
    "--nL", str(nL),
    "--nN", str(nN),
    "--nM", str(nM),
])