# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:41:09 2020

@author: Raffaela
"""
import numpy as np
import matplotlib.pyplot as plt 




# cost function
def Jxy(X,Y, ignore_empty=True):
    # X: data vectors
    # Y: codebook
    # Returns: total cost  
    high_cost = 1e2
    J=0; M,N=np.shape(X); M,K=np.shape(Y)
    d = np.zeros([K,N])
    #p=np.zeros([K,1])
    for n in range(N):
        d[:,n] =np.sum(np.power(X[:,n].reshape([M,1])-Y,2),axis=0)
    row_min = np.argmin(d, axis = 0)
    #aux = np.copy(d)
    for k in range(K):
        if k not in row_min:
            if ignore_empty==True:
                return high_cost
            # ind_min = np.argmin(aux[k,:])
            # row_min[ind_min] = k
            # aux[:,ind_min]=1e6 
    for ind,m in enumerate(row_min):
        J+= d[m,ind]  
    J=J/N
    return J

def generate_data(params, flg_plot):
    # data generation
    NC = params.NC; cl = params.cl; M = params.M; P = params.P; S = params.S
    SC = S+2
    if cl==1:
        #SD = 1
        centers = np.random.normal(0,SC, [M,int(NC/4)])
        cl_centers = np.repeat(centers, 4, axis=1) + np.random.normal(0,S,[M,NC])
    else: 
        cl_centers = np.random.normal(0,S,[M,NC])
    data_vectors = np.repeat(cl_centers, P, axis=1) + np.random.normal(0,1,[M,NC*P])
    
    if flg_plot==1:
        # plot generated data
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if cl==1:
            ax.scatter(centers[0,:], centers[1,:], centers[2,:])
        ax.scatter(cl_centers[0,:], cl_centers[1,:], cl_centers[2,:])
        ax.scatter(data_vectors[0,:], data_vectors[1,:], data_vectors[2,:])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    return (data_vectors, cl_centers)


def cluster_SA(data_vectors,NC,params,S):
    # loop preparation
    np.random.seed(1)
    M = data_vectors.shape[0]
    alg = int(params.alg); N = int(params.N); epsilon = params.eps; 
    K = int(params.K); T0 = params.T0
    start = 0
    ctrl_loop =0
    while start==0:
        X = np.random.normal(0,S,[M,NC])
        J = Jxy(data_vectors,X, ignore_empty=True)
        ctrl_loop+=1
        print(ctrl_loop)
        if J<100:
            start=1
    Jmin = J
    Xmin = X
    stop = 0; T = T0; k = 0
    hist_J=np.array([]); hist_T=np.array([])
    while not(stop):
        for n in range(N):
            if alg==0:
                R = np.random.normal(0,S,[M,NC])
            else:  R = np.random.standard_cauchy([M,NC])
            Xhat = X + epsilon*R
            Jhat = Jxy(data_vectors,Xhat)
            r = np.random.uniform(0,1)
            exp_J = (J-Jhat)/T
            if exp_J<=709.78:
                if r < np.exp(exp_J):
                    X = Xhat; J = Jhat
                    if J < Jmin:
                        Jmin = J; Xmin = X
            else: 
                 X = Xhat; J = Jhat
                 if J < Jmin:
                        Jmin = J; Xmin = X
            hist_J = np.append(hist_J,J)                
            hist_T = np.append(hist_T,T)
                
        if alg==0:
            T=T/np.log2(2+k)
        else: 
            T=T/(1+k)
        k+=1
        if k==K:
            stop = 1 
    
    return Xmin,Jmin, hist_J


def Dxy(x,y):
    return np.sum(np.power(x-y,2))


def cluster_DA(data_vectors,NC,params,S):
    M = data_vectors.shape[0]
    N = data_vectors.shape[1]
    #K = NC
    T0 = params.T0; Tmin = params.Tmin 
    alpha = params.alpha; delta = params.delta
    T = T0; i = 0; fim = 0
    Y = np.random.normal(0,S,[M,NC])
    I = 200; epsilon = 1e-6
    J = np.zeros(I); D = np.zeros(I)
    dxy=np.zeros([NC,N])
    py_x=np.zeros([NC,N])
    while (fim==0):
        dxy = np.zeros([NC,N])
        # calculating p of y given x
        for n in range(N):
              for k in range(NC):
                  dxy[k,n]= Dxy(data_vectors[:,n],Y[:,k])
        py_x = np.exp(-dxy/T)
        Zx = np.sum(py_x,axis=0) 
        py_x= py_x/Zx
        
        # updating Y (cluster centers)
        Y = np.zeros([M,NC])
        for k in range(NC):
            for n in range(N):
                Y[:,k] += data_vectors[:,n]*py_x[k,n]
        Y = Y/np.sum(py_x, axis=1)  
        
    
        # Cost Function and Loop Control
        J[i]=-T/N*np.sum(np.log(Zx))
        D[i]=np.mean(np.sum(py_x*dxy,axis=0))
        if (i>0):
            if abs(J[i]-J[i-1])/abs(J[i-1])<delta:
                T=alpha*T
                Y=Y+epsilon*np.random.normal(0,1,np.shape(Y))
        print([i,J[i],D[i]])   
        i+=1
        print(i)
        if (T<Tmin)or(i==I): 
            fim=1
    J = J[:i]   
    return Y, np.min(J), J