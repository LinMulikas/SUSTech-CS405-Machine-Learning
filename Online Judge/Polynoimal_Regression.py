from cmath import sqrt
import numpy as np

n, m = input().split(" ")
N = int(n)
M = int(m)

X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(N):
    x, y = input().split(" ")
    X_train.append(float(x))
    Y_train.append(float(y))
    

for i in range(M):
    x, y = input().split(" ")
    X_test.append(float(x))
    Y_test.append(float(y))
    
    
Y_test = np.array(Y_test)
    
    
lst_RMSE = []
lst_SD = []

def calVar(X_train:np.ndarray, y_train:np.ndarray):
    w = np.linalg.pinv(X_train) @ y_train
    var = np.mean(np.square(x @ w - y_train))
    return var


def RMSE(y_pred:np.ndarray, y_test:np.ndarray):
    n = len(y_pred)
    return np.sqrt(np.sum(np.power(y_pred - y_test, 2)/n))
    

def SD(y_pred:np.ndarray, y_test:np.ndarray):
    return np.std(y_pred - y_test)


for deg in range(11):
    poly = np.polyfit(X_train, Y_train, deg)
    Y_pred = np.polyval(poly, X_test)
    lst_RMSE.append(RMSE(Y_pred, Y_test))
    lst_SD.append(SD(np.polyval(poly, X_train), Y_train))
    

best_index = np.argmin(lst_RMSE)
print(best_index)
print(round(lst_SD[best_index], 6))