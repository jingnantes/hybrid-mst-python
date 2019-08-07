#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Wed Aug  7 10:25:07 2019

@author: jingli, jing.li.univ@gmail.com

This is the python version for the Hybrid-MST pair sampling strategy published in NIPS2018.

"Hybrid-MST: A Hybrid Active Sampling Strategy for Pairwise Preference Aggregation"

The time cost has been reduced significantly.

"""

import numpy as np
import sys
from scipy import linalg
import numpy.polynomial.hermite as herm
import math
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

__copyright__ = "Copyright 2019, Jing LI"
__license__ = "Apache, Version 2.0"


def run_modeling_Bradley_Terry(alpha):
    """this code is from zhi li, sureal package"""

        # alpha = np.array(
        #     [[0, 3, 2, 7],
        #      [1, 0, 6, 3],
        #      [4, 3, 0, 0],
        #      [1, 2, 5, 0]]
        #     )

    M, M_ = alpha.shape
    assert M == M_

    iteration = 0
    p = 1.0 / M * np.ones(M)
    change = sys.float_info.max

    DELTA_THR = 1e-8

    while change > DELTA_THR:
        iteration += 1
        p_prev = p
        n = alpha + alpha.T
        pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
        p = np.sum(alpha, axis=1) / np.sum(n / pp, axis=1)
 
        p = p / np.sum(p)

        change = linalg.norm(p - p_prev)

    n = alpha + alpha.T
    pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
    lbda_ii = np.sum(-alpha / np.tile(p, (M, 1))**2 + n / pp**2, axis=1)
   
    lbda_ij = n / pp*2
    lbda = lbda_ij + np.diag(lbda_ii)
    cova = np.linalg.pinv(np.vstack([np.hstack([-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])]))
    vari = np.diagonal(cova)[:-1]
    stdv = np.sqrt(vari)

    scores = np.log(p)
    scores_std = stdv / p # y = log(x) -> dy = 1/x * dx

    
    return scores,cova[:-1,:-1],scores_std

def EIG_GaussianHermitte_matrix_Hybrid_MST(mu_mtx,sigma_mtx):
    """ this is the matrix implementation version"""
    """mu is the matrix of difference of two means (si-sj), sigma is the matrix of sigma of si-sj"""
    epsilon = 1e-9
    M,M_ = np.shape(mu_mtx)
    
    
    mu = np.reshape(mu_mtx, (1,-1))
    sigma = np.reshape(sigma_mtx, (1,-1))
    
  
    fs1 = lambda x: (1./(1+np.exp(-np.sqrt(2)*sigma*x-mu)))*(-np.log(1+np.exp(-np.sqrt(2)*sigma*x-mu)))/np.sqrt(math.pi);
    fs2 = lambda x: (1-1./(1+np.exp(-np.sqrt(2)*sigma*x-mu)))*(np.log(np.exp(-np.sqrt(2)*sigma*x-mu)/(1+np.exp(-np.sqrt(2)*sigma*x-mu))))/np.sqrt(math.pi);
    fs3 = lambda x: 1./(1+np.exp(-np.sqrt(2)*sigma*x-mu))/np.sqrt(math.pi);
    fs4 = lambda x: (1-1./(1+np.exp(-np.sqrt(2)*sigma*x-mu)))/np.sqrt(math.pi);
        
    x,w = herm.hermgauss(30)
    x = np.reshape(x,(-1,1))
    w = np.reshape(w,(-1,1))
    
    es1 = np.sum(w*fs1(x),0)
    es2 = np.sum(w*fs2(x),0)
    es3 = np.sum(w*fs3(x),0)
    es3 = es3*np.log(es3+epsilon)
    es4 = np.sum(w*fs4(x),0)
    es4 = es4*np.log(es4+epsilon)
    
    ret = es1 + es2 - es3 + es4
    ret = np.reshape(ret,(M,M_))
    ret = -np.triu(ret,1)
    return ret+ret.T
    
def ActiveLearningPair_matrix_Hybrid_MST(mu,mu_cova):
    pvs_num = len(mu)
                 
    eig = np.zeros((pvs_num,pvs_num))
    
    
    mu_1 = np.tile(mu, (pvs_num, 1))
    
    sigma = np.diag(mu_cova)
    sigma_1 = np.tile(sigma,(pvs_num,1))
    
    mu_diff = mu_1.T-mu_1
    sigma_diff = np.sqrt(sigma_1.T+sigma_1-2*mu_cova)

    eig = EIG_GaussianHermitte_matrix_Hybrid_MST(mu_diff,sigma_diff)
    return eig

def Initial_learning(pcm):
    """this is used for the case that at the begining,
    there is no comparison results, so the initial results
    is that every one is compared once and selected once"""
    pvs_num,pvs_num_ = np.shape(pcm) 
    pcm = np.ones((pvs_num,pvs_num_))
    pcm[range(pvs_num),range(pvs_num)]=0
    return pcm


def GeneratePlaylist_Hybrid_MST(pre_pcm):
    """ given a pre_pcm, 
    select the optimal MST pairs for next observers based on nips paper
    """
    
    
    if np.sum(pre_pcm)==0:
        pre_pcm = Initial_learning(pre_pcm)
    
    [mu,mu_cova,stdv] = run_modeling_Bradley_Terry(pre_pcm)
    
    
    
    EIG_mtx = ActiveLearningPair_matrix_Hybrid_MST(mu, mu_cova)
        
    tcsr = minimum_spanning_tree(-np.triu(EIG_mtx))
    tcsr_tmp = tcsr.toarray()
    result = np.where(tcsr_tmp<0)
    pairMST = np.array(result).T
    
    return pairMST


if __name__ == '__main__':
    
    """set the number of observers in Pair Comparison experiment"""
    #budget_observer4PC = 50
    
    # supposing we already have some pair comparison results, pcm
    pcm = pd.read_csv('./data/pcm_48stimuli.csv',header=None)
    pcm = pcm.values
    pvs_num = len(pcm)
    
    # accoding to the current comparison results, select the optimal pairs based on MST
    """please note that in the original paper, if the comparison number is smaller than one
    standard number, then we should SELECT the top one pair which has the maximum EIG. Please
    change the code accordingly, here I ONLY CONSIDER the MST case."""
    
    pair_list = GeneratePlaylist_Hybrid_MST(pcm)
    