import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sigmoid activation
def activation(Z):
    return 1 / (1 + np.exp(-Z))

# loss function: -(y*log(y_hat) + (1-y)*log(1-y_hat))
# cost function: average of loss
def cost(A, Y):
    epsilon = 0.00000001 # prevent log 0
    loss =  -(Y * np.log(A+epsilon) + (1-Y) * np.log(1-A+epsilon))
    size = loss.shape[1]
    return loss.sum(axis = 1)/size

# forward propagtion with cost function calculated
def forward(X, W1, W2, b1, b2):
    #print(b1, b2)
    Z1 = np.dot(W1, X) + b1
    A1 = activation(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = activation(Z2)

    return A1, A2

# backward propagation for derivatives
def backward(X, Y, W1, W2, A1, A2):
    size = X.shape[1]
    dZ2 = np.array(A2 - Y)
    dW2 = np.dot(dZ2, A1.transpose())
    db2 = np.sum(dZ2, axis=1, keepdims=True)/size
    dZ1 = np.dot(W2.transpose(), dZ2) * (A1-(1-A1))
    dW1 = np.dot(dZ1, X.transpose())
    db1 = np.sum(dZ1, axis=1, keepdims=True)/size
    return dW1, dW2, db1, db2

# training to get parameters of 2 hidden layers
def training(wine_training, wine_y, W1, W2, b1, b2, costs, a, m_x):
    for i in range(m_x):
        # forward propagation
        X = wine_training.iloc[:,[i]]
        Y = wine_y.iloc[:,[i]]
        A1, A2 = forward(X, W1, W2, b1, b2)

        # cost function
        wine_cost = cost(A2, Y)
        
        # record cost for ploting
        costs[i] = wine_cost

        # back propagation backward(X, Y, W1, W2, A1, A2):
        dW1, dW2, db1, db2 = backward(X, Y, W1, W2, A1, A2)
        W1 = W1 - a * dW1
        W2 = W2 - a * dW2
        b1 = b1 - a * db1
        b2 = b2 - a * db2
    return W1, W2

def test(wine_test, w_y, W1, W2, b1, b2):
    A1, prediction = forward(wine_test, W1, W2, b1, b2)
    print(prediction)
    return np.where(prediction<0.58, 0, 1)

# plot cost function progress
def plot(costs, m_x):
    plt.scatter(range(m_x), costs, marker = ".")
    plt.show()

# build dataframe of dataset
wine = pd.read_csv("wine.csv")

# hyperparameters
m_x = 1120 # number of training examples (80% of total dataset)
n_x = 11 # number of features (all 11 columns)
n1 = 36 # number of neurons in 1st layer
n2 = 1 # number of neurons in 2nd layer
a = 0.001 # learning rate

# spliting training data into input and output
wine_y = (wine["quality"]=="good").astype(int) # turning quality good to value 1, bad to value 0
wine_y = pd.DataFrame(wine_y).transpose()
wine_training = wine.iloc[:, :-1].transpose()
wine_test = wine_training.iloc[:, m_x:]
wine_y_test = wine_y.iloc[:, m_x:]

# mean and std and normalisation
std = wine_training.std(axis = 1)
mean = wine_training.mean(axis= 1)
#wine_training = (wine_training.sub(mean, axis = 0)).div(std, axis = 0)

# random initialise for neural network
np.random.seed(0)
W1 = pd.DataFrame(np.random.rand(n1,n_x))
W2 = pd.DataFrame(np.random.rand(n2,n1))
b1 = np.random.rand(n1, 1)
b2 = np.random.rand(n2, 1)

costs = np.zeros(m_x)

W1, W2 = training(wine_training, wine_y, W1, W2, b1, b2, costs, a ,m_x)
prediction = test(wine_test, wine_y_test, W1, W2, b1, b2)
print(np.average(prediction))

plot(costs, m_x)