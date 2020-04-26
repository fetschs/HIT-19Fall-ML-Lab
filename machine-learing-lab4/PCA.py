import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# 读取目录下所有的jpg图片
def load_image(image_path):
    file_names = glob(image_path + "/*jpg")
    sample = []
    for file_name in file_names:
        pic = Image.open(file_name)
        pic = pic.convert('L')

        # pic.save(r"C:\Ccode\HIT-ML-Lab\machine-learing-lab4\testt.jpg")
        # exit(0)
        pic = np.asarray(pic)
        sample.append(pic)
    return np.array(sample)


def generate_random_data(means, covs, size=10):
    assert (len(means) == len(covs))
    ret_data = []
    for ind in range(len(means)):
        new_data = np.random.multivariate_normal(mean=means[ind], cov=covs[ind], size=size)
        ret_data.append(new_data)
    ret_data = np.concatenate(ret_data, axis=0)
    return ret_data


def pca_by_eig(data_mat, top_n_feat=2):
    mean = np.mean(data_mat)
    data_mat_removed = data_mat - mean
    cov = np.cov(data_mat_removed, rowvar=False)
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov))
    eig_val_ind = np.argsort(eig_values)
    eig_val_ind = eig_val_ind[:-(top_n_feat + 1):-1]
    red_eig_vectors = eig_vectors[:, eig_val_ind]
    low_data_mat = np.array(np.matmul(data_mat_removed, red_eig_vectors))
    ret_mat = np.array(np.matmul(low_data_mat, red_eig_vectors.T) + mean)
    return low_data_mat, ret_mat


def pca_by_svd(data_mat, top_n_feat=20):
    mean = np.mean(data_mat)
    data_mat_removed = data_mat - mean
    u_mat, sigma_values, v_mat = np.linalg.svd(data_mat_removed)
    sigma_ind = np.argsort(sigma_values)
    sigma_ind = sigma_ind[:-(top_n_feat + 1):-1]
    sigma_mat = np.diag(sigma_values)
    new_sigma_mat = sigma_mat[sigma_ind, :]
    new_sigma_mat = new_sigma_mat[:, sigma_ind]
    new_u_mat = u_mat[:, sigma_ind]
    new_v_mat = v_mat[sigma_ind, :]
    low_data_mat = np.array(np.matmul(new_u_mat, new_sigma_mat))
    ret_mat = np.array(np.matmul(low_data_mat, new_v_mat) + mean)
    return low_data_mat, ret_mat


def cal_psnr(img1, img2):
    pixel_max = 255.
    mse = np.mean((img1 / pixel_max - img2 / pixel_max) ** 2)
    if mse < 1.0e-10:
        return 100
    pixel_max = pixel_max / pixel_max
    return 20 * np.log10(pixel_max / math.sqrt(mse))


def create_rotation_matrix(theta_max=30., dimension=2):
    # print(np.random.random()*theta_max)
    random_theta = np.radians(np.random.random() * theta_max)
    cos_val, sin_val = np.cos(random_theta), np.sin(random_theta)
    ret_matrix = None
    if dimension == 2:
        ret_matrix = np.array(((cos_val, -sin_val), (sin_val, cos_val)))
    if dimension == 3:
        ret_matrix = np.array(((1, 0, 0), (0, cos_val, -sin_val), (0, sin_val, cos_val)))
    return ret_matrix


if __name__ == '__main__':
    face_data = load_image(r"asian")
    for face in face_data:
        plt.imshow(face)
        plt.show()
        low_face, new_face = pca_by_svd(data_mat=face, top_n_feat=20)
        print(cal_psnr(face, new_face))
        plt.imshow(new_face)
        plt.show()
        exit()
    mean_list = []
    cov_list = []
    # mean_list.append(np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10)]))
    # cov_list.append(np.array([[np.random.uniform(1, 10), 0], [0, np.random.uniform(0.01, 0.10)]]))
    # DIMENSION = 2
    mean_list.append(np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)]))
    cov_list.append(np.array(
        [[np.random.uniform(1, 10), 0, 0], [0, np.random.uniform(1, 10), 0], [0, 0, np.random.uniform(0.1, 1.0)]]))
    DIMENSION = 3
    random_data = generate_random_data(mean_list, cov_list, size=50)
    random_data = np.dot(random_data, create_rotation_matrix(dimension=DIMENSION))

    low_random_data, new_random_data = pca_by_eig(random_data, top_n_feat=2)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(random_data[:, 0], random_data[:, 1], random_data[:, 2])
    # ax.scatter(new_random_data[:, 0], new_random_data[:, 1], new_random_data[:, 2])
    # plt.savefig('fig.png', bbox_inches='tight')  # 替换 plt.show()
