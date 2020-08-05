#!/usr/bin/env python3

## Markov Chain Monte Carlo (MCMC) sampling ##

import math, random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


## M-H sampling, from https://zhuanlan.zhihu.com/p/30003899 ##
def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y

# 在例子里，我们的目标平稳分布是一个均值3，标准差2的正态分布，而选择的马尔可夫链状态转移矩阵Q(i,j)的条件转移概率是以i为均值,方差1的正态分布在位置j的值。

T = 20000
pi = [i for i in range(T)] # initial sampling
sigma = 1
t = 0
while t < T-1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None) # generate samplings on distribution N(sigma, 1) 
    alpha = min(1, norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t-1]))
    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t-1]

plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 100
plt.hist(pi, num_bins, density=True, facecolor='red', alpha=0.6)
plt.show()

#################################################################

## MCMC sampling ##

def Q(i, j):
    return norm.pdf(j, loc=i, scale=1)

T = 20000
pi = [i for i in range(T)] # initial sampling
sigma = 1
t = 0
while t < T-1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None) # generate samplings on distribution N(sigma, 1) 
    # alpha = min(1, norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t-1]))
    alpha = norm_dist_prob(pi_star[0]) * Q(pi_star[0], pi[t-1])
    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t-1]

plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 50
plt.hist(pi, num_bins, density=True, facecolor='red', alpha=0.6)
plt.show()

##################################################################

## naive MCMC sampling

def Q(i, j):
    return norm.pdf(j, loc=i, scale=1)

def MCMC_sampling(Q, target_pdf):
    T = 200000
    pi = [i for i in range(T)] # initial sampling
    sigma = 1
    t = 0
    while t < T-1:
        t = t + 1
        pi_star = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None) # generate samplings on distribution N(sigma, 1) 
        alpha = norm_dist_prob(pi_star[0]) * Q(pi_star[0], pi[t-1])
        u = random.uniform(0, 1)
        if u < alpha:
            pi[t] = pi_star[0]
        else:
            pi[t] = pi[t-1]
    return pi

## Metropolis-Hastings sampling
def MH_sampling(target_pdf):
    T = 20000
    pi = [i for i in range(T)] # initial sampling
    sigma = 1
    t = 0
    while t < T-1:
        t = t + 1
        pi_star = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None) # generate samplings on distribution N(sigma, 1) 
        alpha = min(1, norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t-1]))
        u = random.uniform(0, 1)
        if u < alpha:
            pi[t] = pi_star[0]
        else:
            pi[t] = pi[t-1]
    return pi

def Gibbs_sampling(target_pdf):
    pass

def roulette_resampling(Xt, Wt):
    X = []
    a = np.sum(Wt)
    M = Xt.shape[0]
    for m in range(M):
        r = random.uniform(0, a)
        c = Wt[0]
        i = 0
        while r > c:
            i = i + 1
            c = c + Wt[i]
        X.append(Xt[i])
    return np.array(X)

def visualize_sampling(samples, target_pdf):
    plt.scatter(samples, target_pdf(samples))
    num_bins = 50
    plt.hist(samples, num_bins, density=True, facecolor='red', alpha=0.6)
    plt.show()

if False:
    samples = MH_sampling(norm_dist_prob)
    visualize_sampling(samples, norm_dist_prob)

if False:
    samples = MCMC_sampling(Q, norm_dist_prob)
    visualize_sampling(samples, norm_dist_prob)
