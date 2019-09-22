import numpy as np
import matplotlib.pyplot as plt
import math

#draw the solutions figures.
def plot(W_matrixes, dataNum=1000):
    col_vector = np.linspace(-math.pi, math.pi, dataNum)
    col_vector = np.matrix(col_vector)
    col_vector = col_vector.T
    order = len(W_matrixes[0])
    X_matrix = np.matrix(np.ones((dataNum, order), dtype='float'))
    for ind in range(1, order):
        X_matrix[:, ind] = np.multiply(X_matrix[:, (ind - 1)], col_vector)
    X_plot = list(np.array(X_matrix[:, 1].T)[0])
    T_Y_plot = np.sin(X_plot)
    ind = 0
    labels = ['sinx', 'analysis', 'CG', 'BGD']
    colors = ['r', 'g', 'm', 'k', 'c', 'y', 'b', 'w']
    plt.plot(X_plot, T_Y_plot, label=labels[ind], color='r')
    #first create sin x as standard.
    for W_matrix in W_matrixes:
        ind = ind + 1
        my_Y_matrix = X_matrix * W_matrix
        Y_plot = list(np.array(my_Y_matrix.T)[0])
        temp = list(zip(X_plot, Y_plot))
        temp.sort()
        X_plot = [tup[0] for tup in temp]
        Y_plot = [tup[1] for tup in temp]
        plt.plot(X_plot, Y_plot, color=colors[ind], label=labels[ind])
        #create these solutions.
    plt.legend()# add legend for instruction.
    plt.show()# show the figure.


def getMatrix(filename):
    with open(filename) as f:
        lines = f.read()
        lines = lines.split('\n')
        W_matrix = []
        for i in range(len(lines)):
            W_matrix.append(float(lines[i]))
        W_matrix = np.matrix(W_matrix)
        W_matrix = W_matrix.T
    return W_matrix


if __name__ == "__main__":
    W_matrixes = []
    W_matrixes.append(getMatrix(r"./cmake-build-debug/W_matrixByAnalysis.txt"))
    W_matrixes.append(getMatrix(r"./cmake-build-debug/W_matrixByConjugateGradient.txt"))
    W_matrixes.append(getMatrix(r"./cmake-build-debug/W_matrixByBGD.txt"))
    #read solutions.
    plot(W_matrixes)
