# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:33:41 2020

@author: Raffaela
"""
import numpy as np

def Dxy(x,y):
    return np.sum(np.power(x-y,2))


def clustering_DA(data_vectors,NC,params,flg_plot):
    M = data_vectors.shape[0]
    N = data_vectors.shape[1]
    K = int(params.K); T0 = params.T0; epsilon = params.eps
    k = 0; T = T0; i = 0; stop = 0
    Y = np.random.normal(0,1,[M,K]); 
    J = np.zeros(K)
    D = np.zeros(K)
    LocalT = np.zeros(K) 
    alpha = 0.9; delta = 1e-3
    while not(stop):
        dxy = np.zeros([K,N])
        # calculating p of y given x
        for n in range(N):
              for k in range(K):
                  dxy[k,n]= Dxy(data_vectors[:,n],Y[:,k])
        py_x = np.exp(-dxy/T)
        Zx = np.sum(py_x,axis=0) 
        py_x= py_x/Zx
        
        # updating Y (cluster centers)
        Y = np.zeros([M,K])
        for k in range(K):
            for n in range(N):
                Y[:,k] += data_vectors[:,n]*py_x[k,n]
        Y = Y/np.sum(py_x, axis=1)            
    
        # Cost Function and Loop Control
        J[k]=-T/N*np.sum(np.log(Zx))
        D[k]=np.mean(np.sum(py_x*dxy,axis=0))
        LocalT[k]=T
        if (k>0):
            if abs(J[i]-J[i-1])/abs(J[i-1])<delta:
                T=alpha*T
                Y=Y+epsilon*np.random.normal(0,1,np.shape(Y))
        #print([i,J[i],D[i],LocalT[i]])   
        k+=1
        # if (T<Tmin)or(i==I): fim=1
        if k==K:
            stop = 1 