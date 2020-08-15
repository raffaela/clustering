# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 01:50:22 2020

@author: Raffaela
"""

from clustering import generate_data,cluster_SA
# data generation parameters configuration

# number of clusters to be generated
NC = 8
# dimensionality of each data vector
M = 3
# number of data vectors per cluster
P = 50
# standard deviation of clusters generator
SC = 5
# standard deviation of data generator
SD = 1
# standard deviation of cluster generator (zero level)
S = 7
# flag for plotting generated data
flg_plot = 1
# Simulated Annealing
data_vectors, cl_centers = generate_data(NC,M,P,SC,SD,S,flg_plot)
# number of loops for each temperature
N = 1000
# factor for reducing perturbation
epsilon = 0.1
# number of temperatures to pass through
K = 8
# initial temperature
T0 = 1

Xmin,Jmin = cluster_SA(data_vectors,N,K,epsilon,T0,M,NC,S,flg_plot)