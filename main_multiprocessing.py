# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 01:50:22 2020

@author: Raffaela
"""
import clustering
from clustering import generate_data,cluster_SA, cluster_DA, Jxy
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import datetime
from time import perf_counter
import matplotlib.pyplot as plt
import os
from functools import partial


def plot_figures(hist_J,data_vectors,Xmin,index, results_dir):
        histfile_name = os.path.join(results_dir,"hist_{}.png").format(index)
        clusterfile_name = os.path.join(results_dir,"clusters_{}.png").format(index)
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


def run_clustering(df_params,func,alg_type, params_data, data_vectors, NC ,S, flg_plot, \
                   results_dir, results_cols):
    params_clustering = df_params[1]
    n = df_params[0]
    start = perf_counter()
    Xmin,Jmin, hist_J = func(data_vectors,NC,params_clustering,S)
    end = perf_counter()
    exec_time = end-start
    outfile_name = os.path.join(results_dir,"results_{}.npz").format(n)
    np.savez(outfile_name, Xmin, hist_J)
    if flg_plot==1:
        histfile_name,clusterfile_name = plot_figures(hist_J,data_vectors,\
                                                      Xmin,n, results_dir)
        results = pd.Series(np.array([Jmin,exec_time, outfile_name, \
                                      histfile_name, clusterfile_name]), \
                            index = results_cols)
    else: results = pd.Series(np.array([Jmin, exec_time, outfile_name, "", ""])\
                              , index = results_cols)
    results = pd.concat([params_data,params_clustering, results])
    return results


def main(alg_type, mp, results_dir):

    csvfile = os.path.join(results_dir,"results.png")
            # data parameters configuration
    NC_opts = np.array([32])
    M_opts = np.array([3])
    P_opts = np.array([200])
    S_opts = np.array([3])
    cl_opts = np.array([1])
    data_params_cols = ["NC","cl","M","P","S"]
    results_cols = ["Jmin","time","output data","hist plot","cluster plot"]
    data_params = np.array(np.meshgrid(NC_opts,cl_opts,M_opts,P_opts,S_opts))
        
    if alg_type == "SA":
        func = cluster_SA
        # SA parameters configuration
        N_opts = np.array([1000,10000])
        eps_opts = np.array([0.05,0.1])
        K_opts = np.array([8,16])
        T0_opts = np.array([0.1,1,5])
        alg_opts = np.array([0,1]) # 0 for SA, 1 for FSA
        clustering_params_cols = ["alg","N","eps","K","T0"]
        clustering_params = np.array(np.meshgrid(alg_opts,N_opts,eps_opts,K_opts,T0_opts))
    else:    
        # DA parameters configuration
        func = cluster_DA
        delta_opts = np.array([1e-3])
        alpha_opts = np.array([0.5])
        Tmin_opts = np.array([5e-3])
        T0_opts = np.array([1])
        alg_opts = np.array([0]) # 0 for DA, 1 for GLA
        clustering_params_cols = ["alg","alpha","delta","T0","Tmin"]
        clustering_params = np.array(np.meshgrid(alg_opts,alpha_opts,delta_opts,T0_opts,Tmin_opts))


    df_params_data = pd.DataFrame(data_params.T.reshape(-1, data_params.shape[0]),\
                                  columns=data_params_cols)
    df_params_clustering = pd.DataFrame(\
                    clustering_params.T.reshape(-1, \
                        clustering_params.shape[0]),columns=clustering_params_cols)
    #df_params_clustering = df_params_clustering.iloc[24:,:]
    print(df_params_clustering)
    cols = np.append(data_params_cols,clustering_params_cols)
    cols = np.append(cols,results_cols)
    df_results = pd.DataFrame([], columns = cols)
    flg_plot = 1 
    params_data = df_params_data.iloc[0,:]
    data_vectors, cl_centers = generate_data(params_data, flg_plot)
    np.savez(os.path.join(results_dir,"input_data.npz"), data_vectors,cl_centers)
    
    NC = cl_centers.shape[1]
    S = params_data.S
    
    if mp==True:
        with Pool(8) as pool:
            res = pool.map(partial(run_clustering, func=func, alg_type=alg_type,
                                    params_data = params_data,\
                                    data_vectors=data_vectors,NC =NC,S=S,\
                                    flg_plot=flg_plot, results_dir = results_dir,\
                                    results_cols = results_cols),\
                            df_params_clustering.iterrows())
            df_results = pd.concat(res,axis=1).T

    else:
        
        for n,row in df_params_clustering.iterrows():
            params_clustering = df_params_clustering.iloc[n,:]
            print("test")
            print(params_clustering.alg)
            df_results = run_clustering(df_params = (n,params_clustering), func=func, \
                           alg_type=alg_type,params_data = params_data,\
                                    data_vectors=data_vectors,NC =NC,S=S,\
                                    flg_plot=flg_plot, results_dir = results_dir,\
                                    results_cols = results_cols)
    
    df_results.to_csv(csvfile)
    return data_vectors,cl_centers

if __name__ == '__main__':
    results_dir = os.path.join(
            os.getcwd(), 
            datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(results_dir)
    alg_type = "SA"
    mp = True
    X,Y = main(alg_type,mp, results_dir)
    J_global = Jxy(X,Y)
    print(J_global)
    summary_file = open(os.path.join(results_dir,"summary.txt"),"w")
    summary_file.write(str(J_global))
    summary_file.write("Com subclustering, função custo em sua versão original.")
    summary_file.close()
    



    