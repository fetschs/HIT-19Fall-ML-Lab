import numpy as np
import matplotlib.pyplot as plt
import math


def getMatrix(filename,cols):
    with open(filename,'r') as f:
        lines = f.readlines()
        ret_matrix = np.matrix(np.zeros((len(lines),cols)))
        line_ind = 0
        for line in lines:
            line = line.strip()
            elements = line.split(" ")
            elements = [i for i in elements if i != '']
            ret_matrix[line_ind] = elements
            line_ind+=1
        #print(ret_matrix)
        return ret_matrix
def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * (-x)))
def drawPoints(X_matrix,Y_matrix,W_matrixes):
    X_matrix = X_matrix[:,1:]
    X1_array = np.array(X_matrix[:,0])
    X2_array = np.array(X_matrix[:,1])
    for ind in range(len(X1_array)):
        if Y_matrix[ind,0] == 1:
            plt.scatter(X1_array[ind],X2_array[ind],c='r',marker='^')
        else:
            plt.scatter(X1_array[ind],X2_array[ind],c='b',marker='.')
    sample_num = 1000
    sample_X_array = np.random.uniform(low=np.min(X1_array),high=np.max(X1_array),size=sample_num)
    # ax + by + c = 0
    # y = -c/b -a/bx
    
    for W_matrix in W_matrixes:
        c = W_matrix[0,0]
        a = W_matrix[1,0]
        b = W_matrix[2,0]
        k = -a/b
        b = -c/b
        sample_Y_array = sample_X_array*k+b
        plt.plot(sample_X_array,sample_Y_array)         
    plt.show()
    
if __name__ == "__main__":
    W_matrixes = []
    W_matrixes.append(getMatrix(r"./cmake-build-debug/W_matrixByBGD.txt",1))
    X_matrix = getMatrix(r"./cmake-build-debug/X_matrix.txt",3)
    Y_matrix = getMatrix(r"./cmake-build-debug/Y_matrix.txt",1)
    drawPoints(X_matrix,Y_matrix,W_matrixes)
    lines = []
    # with open("data_banknote_authentication.txt","r") as f:
    #     lines = f.readlines()
    #     for ind in range(len(lines)):
    #         lines[ind] = lines[ind].strip()
    #         lines[ind] = lines[ind].split(",")
    # with open("data_banknote_authentication.txt","w") as f:
    #     for line in lines:
    #         for element in line:
    #             f.write(element)
    #             if element != line[-1]:
    #                 f.write(' ')
    #         f.write("\n")