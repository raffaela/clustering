# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 06:38:29 2020

@author: Raffaela
"""
import numpy as np
import matplotlib.pyplot as plt 
import bisect
import os
import bisect
import pandas as pd
from multiprocessing import Pool
from datetime import datetime


def plot_figure(data_vectors, cl_centers, max_x_vec):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(cl_centers[0,:], cl_centers[1,:],cl_centers[2,:], marker = "^", color = "b", label = "cl_centers")
    ax.scatter(max_x_vec[0,:], max_x_vec[1,:],max_x_vec[2,:], marker = "v", color = "r", label = "Xmin")
    ax.scatter(data_vectors[0,:], data_vectors[1,:],data_vectors[2,:], marker = ".", color = "y", alpha=.4, label = "data_vectors")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_zlim(-7,7)
    ax.legend()

    
    
if __name__ == '__main__':

        results_dir = "resultados-parte1\\20200924_133154"
        inputfile_name = os.path.join(results_dir,"input.npz")
        #np.savez(inputfile_name, data_vectors, cl_centers, global_max, alg)
        
        input_data = np.load(inputfile_name)
        data_vectors = input_data['arr_0']
        cl_centers = input_data['arr_1']
        global_max = input_data['arr_2']
        alg = input_data['arr_3']
        
        outfile_name = os.path.join(results_dir,"results.npz")
        
        data = np.load(outfile_name, allow_pickle=True)
        result = data['arr_0']
        ini = 0
        for r in result:
            if ini == 0:
                max_fitness_vec = r[0]
                max_x_vec = r[1]
                max_x_vec = max_x_vec.reshape([max_x_vec.shape[0],max_x_vec.shape[1],1])
                num_aes_vec = r[2]
                exec_time_vec = r[3]
            else:
                max_fitness_vec = np.append(max_fitness_vec,r[0]) 
                max_x_vec = np.append(max_x_vec,r[1].reshape([r[1].shape[0],r[1].shape[1],1]), axis=2)
                num_aes_vec = np.append(num_aes_vec, r[2])
                exec_time_vec = np.append(exec_time_vec, r[3])
            ini+=1
            plot_figure(data_vectors, cl_centers, max_x_vec[:,:,-1])
            
        success_thr = 5e-2
        loops = max_fitness_vec.shape[0]
        # calc SR
        num_success = np.where((abs(max_fitness_vec-float(global_max))<success_thr) | (max_fitness_vec>float(global_max)))
        SR = num_success[0].shape[0]/loops
        print("SR:")
        print(SR)
        #calc MBF
        MBF = np.mean(max_fitness_vec)
        print("MBF:")
        print(MBF)
        #calc AES
        pos_success = np.where(num_aes_vec!=-1)
        AES = np.mean(num_aes_vec[pos_success])
        print("AES:")
        print(AES)
        mtime = np.mean(exec_time_vec)
        print("Mean execution time:")
        print(mtime)
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(max_fitness_vec)
        ax.set_title(str(alg))
        ax.set_xlabel('Aptidão Máxima')