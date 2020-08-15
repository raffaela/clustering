# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:41:09 2020

@author: Raffaela
"""
import numpy as np
import matplotlib.pyplot as plt 

# cost function
def Jxy(X,Y):
    # X: data vectors
    # Y: codebook
    # Returns: total cost 
    J=0; M,N=np.shape(X); M,K=np.shape(Y)
    p=np.zeros([K,1])
    for n in range(N):
        d=np.sum(np.power(X[:,n].reshape([M,1])-Y,2),axis=0)
        k=np.where(d==np.min(d))
        p[k[0]]+=1
        J+=np.min(d)
    J=J/N
    if np.min(p)==0: J=100
    return J


# parameters configuration

# number of clusters to be generated
NC = 8
# dimensionality of each data vector
M = 3
# number of data vectors per cluster
P = 10
# standard deviation of clusters generator
SC = 5
# standard deviation of data generator
SD = 1


# data generation
cl_centers = np.random.normal(0,SC, [M,NC])
data_vectors = np.repeat(cl_centers, P, axis=1) + np.random.normal(0,SD,[M,NC*P])
# plot generated data
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(cl_centers[0,:], cl_centers[1,:], cl_centers[2,:])
ax.scatter(data_vectors[0,:], data_vectors[1,:], data_vectors[2,:])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Simulated Annealing
# number of loops for each temperature
N = 1000
# factor for reducing perturbation
epsilon = 0.1
# number of temperatures to pass through
K = 8
# initial temperature
T0 = 1

X = np.random.normal(0,SC,[M,NC])
Xmin = X
Jmin = Jxy(data_vectors,X)
J = Jxy(data_vectors,X)
stop = 0; T = T0; k = 0
hist_J=np.array([]); hist_T=np.array([])
while not(stop):
    for n in range(N):
        Xhat = X + epsilon*np.random.normal(0,SC,[M,NC])
        Jhat = Jxy(data_vectors,Xhat)
        r = np.random.uniform(0,1)
        if r < np.exp((J-Jhat)/T):
            X = Xhat
            J = Jhat
            if J < Jmin:
                Jmin = J
                Xmin = X
        hist_J = np.append(hist_J,J)                
        hist_T = np.append(hist_T,T)
            
    T=T0/np.log2(2+k)
    k+=1
    if k==K:
        stop = 1 
        
fig = plt.figure()
ax = fig.gca()
ax.plot(range(K*N),hist_J)
plt.show()

# plot resulting clusters with data vectors
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(Xmin[0,:], Xmin[1,:], Xmin[2,:])
ax.scatter(data_vectors[0,:], data_vectors[1,:], data_vectors[2,:])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(cl_centers[0,:], cl_centers[1,:], cl_centers[2,:])
