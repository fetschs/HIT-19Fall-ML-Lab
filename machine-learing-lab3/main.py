import numpy as np
import random
import GetData
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import itertools


def multi_gaussian_pdf(x, mean, cov, threshold=1e-8):
    if np.linalg.det(cov) == 0:
        row, col = np.diag_indices(cov.shape[0])
        cov[row, col] += threshold
    ret_val = multivariate_normal.pdf(x, mean, cov)
    return ret_val


class KMeans:

    def __init__(self, data, k, epochs=1000):
        self.data = data
        self.k = k
        self.epochs = epochs
        self.data_num, self.dimension = data.shape
        self.centers = list(np.random.choice(self.data_num, size=k, replace=False))
        for ind in range(k):
            self.centers[ind] = list(data[self.centers[ind], :])
        self.centers = np.array(self.centers)

    @staticmethod
    def __get_distance(x1, x2):
        return np.linalg.norm(x1 - x2, ord=2)

    def cal(self):
        epoch = 0
        label = {}
        group = {}
        while True:
            epoch += 1
            if epoch % (self.epochs / 10) == 0:
                print("KMeans epoch: ", epoch)
            for group_ind in range(self.k):
                group[group_ind] = []
            for data_ind in range(self.data_num):
                data_point = self.data[data_ind, :]
                distance = np.zeros(self.k)
                for center_ind in range(self.k):
                    center = self.centers[center_ind]
                    distance[center_ind] = KMeans.__get_distance(center, data_point)
                new_center_ind = int(np.argmin(distance))
                label[data_ind] = new_center_ind
                group[new_center_ind].append(data_point)
            for center_ind in range(self.k):
                if len(group[center_ind]) == 0:
                    continue
                new_center = np.zeros(self.centers[center_ind].shape)
                for data_point in group[center_ind]:
                    new_center += data_point
                self.centers[center_ind] = new_center / len(group[center_ind])
            if epoch > self.epochs:
                break
        return label


class GMM:
    def __init__(self, data, k, epochs=1000, threshold=1e-8):
        self.data = data
        self.k = k
        self.epochs = epochs
        self.threshold = threshold
        self.data_num, self.dimension = data.shape
        self.mean = np.array(random.sample(list(self.data), self.k))
        self.cov = np.zeros((self.k, self.dimension, self.dimension))
        for k_ind in range(self.k):
            self.cov[k_ind] = np.multiply(np.eye(N=self.dimension, M=self.dimension, dtype=np.float),
                                          np.random.rand(self.dimension, self.dimension))
        self.pi = np.random.rand(self.k)
        self.pi = self.pi / self.pi.sum()
        self.pi = self.pi.reshape(self.k, 1)
        self.n_k_prob = np.zeros((self.data_num, self.k))

    def __e_step(self):
        n_k_prob = np.zeros(shape=(self.data_num, self.k))
        label = {}
        for data_ind in range(self.data_num):
            pi_pdf_sum = 0
            pi_pdf = [0] * self.k
            for k_ind in range(self.k):
                pi_pdf[k_ind] = self.pi[k_ind] * multi_gaussian_pdf(self.data[data_ind], self.mean[k_ind],
                                                                    self.cov[k_ind])
                pi_pdf_sum += pi_pdf[k_ind]
            for k_ind in range(self.k):
                n_k_prob[data_ind][k_ind] = pi_pdf[k_ind] / pi_pdf_sum
            label[data_ind] = int(np.argmax(n_k_prob[data_ind]))

        self.n_k_prob = n_k_prob
        return label

    def __m_step(self):
        new_mean = np.zeros(self.mean.shape)
        new_cov = np.zeros(self.cov.shape)
        new_pi = np.zeros(self.pi.shape)
        for k_ind in range(self.k):
            N_k = np.sum(self.n_k_prob[:, k_ind])
            gamma = self.n_k_prob[:, k_ind]
            gamma = gamma.reshape(self.data_num, 1)
            new_mean[k_ind, :] = np.dot(gamma.T, self.data) / N_k
            new_pi[k_ind, :] = N_k / self.data_num
            new_cov[k_ind] = np.dot((self.data - self.mean[k_ind]).T,
                                    np.multiply((self.data - self.mean[k_ind]), gamma)) / N_k
        end_flag = self.__convergence_judge(new_mean, new_cov, new_pi)
        self.mean = new_mean
        self.cov = new_cov
        self.pi = new_pi
        return end_flag

    def __log_likelihood(self):
        P = np.zeros([self.data_num, self.k])
        for k_ind in range(self.k):
            for data_ind in range(self.data_num):
                P[data_ind, k_ind] = multi_gaussian_pdf(self.data[data_ind], self.mean[k_ind], self.cov[k_ind])
        return np.sum(np.log(P.dot(self.pi)))
        #return np.sum((logsumexp(self.n_k_prob, axis=1)))

    def __convergence_judge(self, new_mean, new_cov, new_pi):
        diff_val = np.linalg.norm(self.mean - new_mean) + np.linalg.norm(self.cov - new_cov) + np.linalg.norm(
            self.pi - new_pi)
        if diff_val < self.threshold:
            return True
        return False

    def cal(self):
        epoch = 0
        while True:
            epoch += 1
            if epoch % (self.epochs / 10) == 0:
                print("GMM epoch: ", epoch, " likelihood: ", self.__log_likelihood())
            self.__e_step()
            if self.__m_step() or epoch > self.epochs:
                print("GMM epoch: ", epoch, " likelihood: ", self.__log_likelihood())
                print("GMM end")
                return self.__e_step()


def read_iris_data(filename):
    with open(filename) as f:
        file_lines = f.readlines()
        ret_data = np.zeros(shape=(len(file_lines), 4))
        ret_label = {}
        for file_ind in range(len(file_lines)):
            file_line = file_lines[file_ind]
            file_line = file_line.strip()
            data_line = file_line.split(',')
            ret_data[file_ind] = data_line[0:4]
            ret_label[file_ind] = data_line[4]
    return ret_data, ret_label


if __name__ == '__main__':
    K = 3
    # mean_list = []
    # cov_list = []
    # for i in range(K):
    #     mean_list.append(np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10)]))
    #     cov_list.append(np.array([[np.random.uniform(0.1, 5), 0], [0, np.random.uniform(0.1, 5)]]))
    # random_data = GetData.generate_random_data(mean_list, cov_list, size=50)
    # KMeans_instance = KMeans(random_data, K, epochs=1000)
    # KMeans_label = KMeans_instance.cal()
    # GMM_instance = GMM(random_data, K, epochs=300, threshold=1e-8)
    # GMM_label = GMM_instance.cal()
    # colors = ['r', 'g', 'b']
    # for data_item in KMeans_label.items():
    #     draw_point = KMeans_instance.data[data_item[0]]
    #     plt.scatter(draw_point[0], draw_point[1], c=colors[data_item[1]])
    # plt.show()
    # for data_item in GMM_label.items():
    #     draw_point = GMM_instance.data[data_item[0]]
    #     plt.scatter(draw_point[0], draw_point[1], c=colors[data_item[1]])
    # plt.show()
    iris_data, iris_label = read_iris_data(r"Iris.txt")
    GMM_instance = GMM(iris_data, K, epochs=300, threshold=1e-8)
    GMM_label = GMM_instance.cal()
    assert (len(GMM_label) == len(iris_label))
    FLOWER_CLASS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    FLOWER_CLASS_perms = list(itertools.permutations(FLOWER_CLASS, 3))
    ans = np.zeros(shape=(len(FLOWER_CLASS_perms), 1))
    for FLOWER_CLASS_perm in FLOWER_CLASS_perms:
        same_label = 0
        for label_ind in range(len(GMM_label)):
            if iris_label[label_ind] == FLOWER_CLASS_perm[GMM_label[label_ind]]:
                same_label += 1
        ans[FLOWER_CLASS_perms.index(FLOWER_CLASS_perm)] = same_label
    print(np.max(ans))
