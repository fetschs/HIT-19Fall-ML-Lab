import numpy as np
import matplotlib.pyplot as plt


def generate_random_data(mean_list, cov_list, size=10):
    assert (len(mean_list) == len(cov_list))
    ret_data = []
    colors = ['r', 'g', 'b']
    for ind in range(len(mean_list)):
        new_data = np.random.multivariate_normal(mean=mean_list[ind], cov=cov_list[ind], size=size)
        ret_data.append(new_data)
        for data_point in new_data:
            plt.scatter(data_point[0], data_point[1], c=colors[ind])
    ret_data = np.concatenate(ret_data, axis=0)
    plt.show()
    return ret_data
