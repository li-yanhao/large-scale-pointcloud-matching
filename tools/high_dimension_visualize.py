import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris

import numpy as np
from sklearn import manifold, datasets

def pca_visualize_tutorial():
    data = load_iris()
    y = data.target
    x = data.data
    pca = PCA(n_components=3)  # 加载PCA算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(x)  # 对样本进行降维

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []

    for i in range(len(reduced_x)):
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])

        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])

        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])

    # 可视化
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()


def tsne_visualize_tutorial():

    digits = datasets.load_digits(n_class=6)
    X, y = digits.data, digits.target
    n_samples, n_features = X.shape

    '''显示原始数据'''
    n = 20  # 每行20个数字，每列20个数字
    img = np.zeros((10 * n, 10 * n))
    for i in range(n):
        ix = 10 * i + 1
        for j in range(n):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    # pca_visualize_tutorial()
    tsne_visualize_tutorial()
