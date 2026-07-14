from TrainingData.HenonHeiles_data_Tau_uniform import main as data_gen
from trainHenonHeilesUniformTau import main as train_model
from convConfInt_HenonHeilesUniformTau import main as conv_graphs

N = 1000 # Training points
M = 250 # Testing points
data_num = "4"
if data_num == "4": # Time steps for training and testing data are sampled from (T1, T2)
    T1 = 0.1
    T2 = 0.25

kernel = "Verlet" # "Euler" or "Verlet"
k = 1 # Tau exponent
nL = 4 # Layers
nN = 32 # Width of Layer
nM = 4 # Seed for model parameter initialization
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