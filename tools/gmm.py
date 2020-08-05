#!/usr/bin/env python3

# Gaussian Mixed Model tutorial

import math, random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy.linalg import cholesky

def generate_data(mu, sigma, num_sample):
    R = cholesky(sigma)
    return np.dot(np.random.randn(num_sample, 2), R) + mu

def probability(x, pis, mus, sigmas):
    sigmas_inv = np.linalg.inv(sigmas)
    joint_probas = 0.5/np.pi/ np.abs(np.linalg.det(sigmas)) \
        * np.exp(-0.5*np.sum(((x - mus) @ sigmas_inv) * (x - mus), axis=2)).T
    return np.divide(joint_probas.T, np.sum(joint_probas, axis=1)).T

def update_parameters(pis, mus, sigmas, samples, proba):
    mus_new = []
    k = pis.shape[0]
    n = samples.shape[0]
    for i in range(k):
        mu_new = np.sum(proba[:,i].reshape(-1,1) * samples, axis=0)
        mu_new = mu_new / np.sum(proba[:,i].reshape(-1,1), axis=0)
        mus_new.append(mu_new)
    sigmas_new = []
    for i in range(k):
        diff = samples - mus_new[i]
        sigma = np.zeros((samples.shape[1], samples.shape[1]))
        for j in range(n):
            sigma += diff[j:j+1, :].T @ diff[j:j+1, :] * proba[j, i]
        sigma = sigma / np.sum(proba[:,i].reshape(-1,1), axis=0)
        sigmas_new.append(sigma)
    pis_new = np.mean(proba, axis=0)
    mus_new = np.array(mus_new).reshape(k, 1, samples.shape[1])
    sigmas_new = np.array(sigmas_new)
    return pis_new, mus_new, sigmas_new

def GMM(samples):
    pass

# mu_1 = np.array([[1, 5]])
# sigma_1 = np.array([[1, 0.5], [1.5, 3]])
# samples_1 = generate_data(mu_1, sigma_1, 1000)

# mu_2 = np.array([[10, 0]])
# sigma_2 = np.array([[4, 0], [0, 2]])
# samples_2 = generate_data(mu_2, sigma_2, 1000)

# samples = np.vstack([samples_1, samples_2])

# # plt.subplot(141)
# plt.plot(samples_1[:,0], samples_1[:,1],'b+')
# plt.plot(samples_2[:,0], samples_2[:,1],'ro')
# # plt.subplot(144)
# # plt.plot(samples_2[:,0], samples_2[:,1],'+')

# plt.show()

# k = 2
# pis = []
# mus = []
# sigmas = []
# pis.append(1)
# pis.append(1)
# mus.append(np.array([[0,0]]))
# mus.append(np.array([[3,3]]))
# sigmas.append(np.array([[1,0],[0,1]]))
# sigmas.append(np.array([[1,0],[0,1]]))
# pis = np.array(pis)
# mus = np.array(mus)
# sigmas = np.array(sigmas)

n = 10000
mu_1 = np.array([[1, 5]])
sigma_1 = np.array([[1, 0.5], [0.5, 3]])
samples_1 = generate_data(mu_1, sigma_1, n)

mu_2 = np.array([[4, 0]])
sigma_2 = np.array([[5, 1], [1, 3]])
samples_2 = generate_data(mu_2, sigma_2, n)

samples = np.vstack([samples_1, samples_2])

# Initialization
k = 2
pis = []
mus = []
sigmas = []
pis.append(1)
pis.append(1)
mus.append(np.array([[7,0]]))
mus.append(np.array([[21,4]]))
sigmas.append(np.array([[1,0],[0,1]]))
sigmas.append(np.array([[1,0],[0,1]]))
pis = np.array(pis)
mus = np.array(mus)
sigmas = np.array(sigmas)

# E-step
# post_prob = probability(samples, pis, mus, sigmas)
# print(post_prob)
for i in range(20):
    # E-step
    post_prob = probability(samples, pis, mus, sigmas)
    # print(post_prob)

    # M-step
    pis, mus, sigmas = update_parameters(pis, mus, sigmas, samples, post_prob)
    # print("pis: \n", pis)
    print("mus: \n", mus)
    # print("sigma: \n", sigmas)


