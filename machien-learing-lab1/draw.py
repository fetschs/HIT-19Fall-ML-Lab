import numpy as np
import matplotlib.pyplot as plt
import math


def plot(W_matrixs, dataNum=50):
    col_vector = np.linspace(-math.pi, math.pi, dataNum)
    col_vector = np.matrix(col_vector)
    col_vector = col_vector.T
    order = len(W_matrixs[0])
    X_matrix = np.matrix(np.ones((dataNum, order), dtype='float'))
    # print(type(X_matrix[:, 1]))
    # print(type(col_vector))
    # print (X_matrix[:, 1].shape)
    # print(col_vector.shape)
    # print(np.multiply(X_matrix[:, 1], col_vector))
    for ind in range(1, order):
        X_matrix[:, ind] = np.multiply(X_matrix[:, (ind - 1)], col_vector)
        # print(np.multiply(X_matrix[:, ind],col_vector))
        # print("213123")
    X_plot = list(np.array(X_matrix[:, 1].T)[0])
    T_Y_plot = np.sin(X_plot)
    ind = 0
    labels = ['sinx', 'analysis', 'BGD', 'CG']
    colors = ['r', 'g', 'c', 'b', 'm', 'y', 'k', 'w']
    plt.plot(X_plot, T_Y_plot, label=labels[ind], color='r')
    for W_matrix in W_matrixs:
        ind = ind + 1
        my_Y_matrix = X_matrix * W_matrix
        Y_plot = list(np.array(my_Y_matrix.T)[0])
        temp = list(zip(X_plot, Y_plot))
        temp.sort()
        X_plot = [tup[0] for tup in temp]
        Y_plot = [tup[1] for tup in temp]
        plt.plot(X_plot, Y_plot, color=colors[ind], label=labels[ind])
    plt.legend()
    # plt.legend(line, ['10Sinx,Analysis,BGD,ConjugateGradient'], loc='lower right',markerscale=0.5)
    plt.show()


def getMatrix(filename, W_matrixs):
    with open(filename) as f:
        lines = f.read()
        lines = lines.split('\n')
        W_matrix = []
        for i in range(len(lines)):
            W_matrix.append(float(lines[i]))
        W_matrix = np.matrix(W_matrix)
        W_matrix = W_matrix.T
        W_matrixs.append(W_matrix)
    return W_matrixs


if __name__ == "__main__":
    W_matrixs = []
    W_matrixs = getMatrix("cmake-build-debug/W_matrixByAnalysis.txt", W_matrixs)
    W_matrixs = getMatrix("cmake-build-debug/W_matrixByAnalysis.txt", W_matrixs)
    W_matrixs = getMatrix("cmake-build-debug/W_matrixByAnalysis.txt", W_matrixs)
    plot(W_matrixs)
