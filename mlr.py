#multiple linear regression

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import csv
import pandas as pd


path = "C:/Users/katagi/OneDrive/デスクトップ/Test/sakanoue_plantar1"    #被験者によってpathを変更

N = 5           #試行回数
D = 48          #センサ位置
t = 480         #計測時間[ms]
combination = N
num_of_header = 5
num_of_col = 2      #2 or 3
RMSE_li = []
corrcoef_li = []


def Analytical_Solution(x,t):
    x_T = x.T
    x_T_x_inv = np.linalg.pinv(np.dot(x_T,x))

    return np.dot(np.dot(x_T_x_inv,x_T),t)

def Calculate_RMSE(y,y_hat,t):
    L = 0.0
    for i in range(t):
        L += (y[i] - y_hat[i])**2
    RMSE = np.sqrt(L/t)
    return RMSE


#----------- main --------------
def main():
    for cbn in range(combination):
        # W
        W = np.zeros((t,D+1))
        for i in range(t):
            X = np.zeros((N-1,D+1))
            for n in range(N-1):
                if not n == cbn:
                    with open(path + "/s" + str(n+1) + ".csv") as f:
                        file = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
                        tmp = []
                        for row in file:
                            tmp.append(row)
                        
                        for d in range(D):
                            X[n][d] = tmp[i*2][d]
                        X[n][D] = 1
            #print(X)

            theta = np.zeros((N-1,1))
            for n in range(N-1):
                file = pd.read_csv(path + "/g" + str(n+1) + ".csv", header=num_of_header,
                    usecols=[num_of_col], names=['deg'])
                theta[n][0] = file.loc[i]
            #print(theta)

            w = Analytical_Solution(X,theta)
            for d in range(D):
                W[i][d] = w[d][0]
        #print(W)
        #print(W.shape)


        # test and RMSE
        Theta_hat = []
        for i in range(t):
            X = np.zeros((D+1,1))
            with open(path + "/s" + str(cbn+1) + ".csv") as f:
                file = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
                tmp = []
                for row in file:
                    tmp.append(row)            
                for d in range(D):
                    X[d][0] = tmp[i*2][d]
                X[D][0] = 1
            #print(X)
            theta_hat = np.dot(W[i].T,X)
            Theta_hat.append(theta_hat[0])
        #print(Theta_hat)
            
        Theta_test = []
        file = pd.read_csv(path + "/g" + str(cbn+1) + ".csv", header=num_of_header,
            usecols=[num_of_col], names=['deg'])
        for v in file['deg']:
            Theta_test.append(v)
        #print(Theta_test)

        print(f'結果{cbn+1}:')
        RMSE = Calculate_RMSE(Theta_test,Theta_hat,t)
        print(f'RMSE = {RMSE}')
        RMSE_li.append(RMSE)


        # graph
        x = np.arange(t)
        y_1 = np.array(Theta_hat)
        y_2 = np.array(Theta_test)
        plt.figure(figsize=(15,4))

        plt.title('Wrist angle')
        plt.xlabel('Time [ms]')
        plt.ylabel('Wrist angle')
        plt.plot(x,y_1,color='blue',linewidth=3,label='Estimated angle')
        plt.plot(x,y_2,color='red',linewidth=3,label='Measured angle')
        plt.ylim(-60,30)
        plt.xlim(0,t)
        plt.legend(loc='upper left',fontsize=10)
        plt.grid(True)

        plt.figure(figsize=(4,4))
        plt.title('scatter plot')
        plt.scatter(y_1,y_2)
        plt.show()
        
        corrcoef = np.corrcoef(y_1,y_2)
        corrcoef = corrcoef[0][1]
        print(f'相関係数 = {corrcoef}\n\n')
        corrcoef_li.append(corrcoef)


    with open(path+'/result.csv','w') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(['','RMSE','相関係数'])
        for n in range(N):
            writer.writerow([n+1,RMSE_li[n],corrcoef_li[n]])

if __name__ == '__main__':
    main()