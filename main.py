# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 01:50:22 2020

@author: Raffaela
"""

from clustering import generate_data,cluster_SA
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import datetime
from time import perf_counter
import matplotlib.pyplot as plt
import os
from functools import partial


results_dir = os.path.join(
        os.getcwd(), 
        datetime.now().strftime('%Y%m%d_%H%M%S'))
os.mkdir(results_dir)
outfile = os.path.join(results_dir,"results_{}.npz")
histfile = os.path.join(results_dir,"histJ_{}.png")
clusterfile = os.path.join(results_dir,"clustering_{}.png")
csvfile = os.path.join(results_dir,"results.png")

def plot_figures(hist_J,data_vectors,Xmin,index):
        histfile_name = histfile.format(index)
        clusterfile_name = clusterfile.format(index)
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(range(hist_J.shape[0]),hist_J)
        #plt.show()
        plt.savefig(histfile_name)
        # plot resulting clusters with data vectors
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(Xmin[0,:], Xmin[1,:], Xmin[2,:])
        ax.scatter(data_vectors[0,:], data_vectors[1,:], data_vectors[2,:])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(clusterfile_name)
        return (histfile_name,clusterfile_name)
        #ax.scatter(cl_centers[0,:], cl_centers[1,:], cl_centers[2,:])

# parameters configuration
NC_opts = np.array([16])
M_opts = np.array([3])
P_opts = np.array([100])
S_opts = np.array([7])
cl_opts = np.array([1])
N_opts = np.array([1000,10000])
eps_opts = np.array([0.1,0.5])
K_opts = np.array([8,16])
T0_opts = np.array([0.1,1,5])
alg_opts = np.array([0,1]) # 0 for SA, 1 for FSA, 2 for DA

data_params_cols = ["NC","cl","M","P","S"]
clustering_params_cols = ["alg","N","eps","K","T0"]
results_cols = ["Jmin","time","output data","hist plot","cluster plot"]
data_params = np.array(np.meshgrid(NC_opts,cl_opts,M_opts,P_opts,S_opts))
clustering_params = np.array(np.meshgrid(alg_opts,N_opts,eps_opts,K_opts,\
                                         T0_opts))
df_params_data = pd.DataFrame(data_params.T.reshape(-1, data_params.shape[0]),\
                              columns=data_params_cols)
df_params_clustering = pd.DataFrame(\
                clustering_params.T.reshape(-1, \
                    clustering_params.shape[0]),columns=clustering_params_cols)
cols = np.append(data_params_cols,clustering_params_cols)
cols = np.append(cols,results_cols)
df_results = pd.DataFrame([], columns = cols)
flg_plot = 1 
params_data = df_params_data.iloc[0,:]
data_vectors, cl_centers = generate_data(params_data, flg_plot)
NC = cl_centers.shape[1]
S = params_data.S
n=0
#for row in df_params_clustering.iterrows():

# def run_clustering(n):
#     params_clustering = df_params_clustering.iloc[n,:]
#     start = perf_counter()
#     Xmin,Jmin, hist_J = cluster_SA(data_vectors,NC,params_clustering,S,\
#                                     flg_plot)
#     end = perf_counter()
#     exec_time = end-start
#     outfile_name = outfile.format(n)
#     np.savez(data_vectors, Xmin, hist_J)
#     if flg_plot==1:
#         histfile_name,clusterfile_name = plot_figures(hist_J,data_vectors,\
#                                                       Xmin,n)
#         results = pd.Series(np.array([Jmin,exec_time, outfile_name, \
#                                       histfile_name, clusterfile_name]), \
#                             index = results_cols)
#     else: results = pd.Series(np.array([Jmin,exec_time, outfile_name, "", ""])\
#                               , index = results_cols)
#     params =  pd.concat([params_data,params_clustering, results])
#     df_results = df_results.append(params, ignore_index = True)    
#     n+=1
    

for n,row in df_params_clustering.iterrows():
    params_clustering = df_params_clustering.iloc[n,:]
    start = perf_counter()
    Xmin,Jmin, hist_J = cluster_SA(data_vectors,NC,params_clustering,S,\
                                    flg_plot)
    end = perf_counter()
    exec_time = end-start
    outfile_name = outfile.format(n)
    np.savez(outfile_name,data_vectors, Xmin, hist_J)
    if flg_plot==1:
        histfile_name,clusterfile_name = plot_figures(hist_J,data_vectors,\
                                                      Xmin,n)
        results = pd.Series(np.array([Jmin,exec_time, outfile_name, \
                                      histfile_name, clusterfile_name]), \
                            index = results_cols)
    else: results = pd.Series(np.array([Jmin,exec_time, outfile_name, "", ""])\
                              , index = results_cols)
    params =  pd.concat([params_data,params_clustering, results])
    df_results = df_results.append(params, ignore_index = True)    
df_results.to_csv(csvfile)







    