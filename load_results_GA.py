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
    ax.scatter(data_vectors[0,:], data_vectors[1,:],data_vectors[2,:], marker = ".", color = "y", alpha=.3, label = "data_vectors")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_zlim(-7,7)
    ax.legend()

    
if __name__ == '__main__':

        results_dir = "20201003_184629"
        #results_dir = "20201003_223547"
        #results_dir = "resultados-parte1\\20200929_010828"
        #results_dir = "resultados-parte1\\20200930_125009"
        inputfile_name = os.path.join(results_dir,"input.npz")
        
        input_data = np.load(inputfile_name)
        data_vectors = input_data['arr_0']
        cl_centers = input_data['arr_1']
        global_max = input_data['arr_2']
        alg = input_data['arr_3']
        
        outfile_name = os.path.join(results_dir,"results.npz")
        
        data = np.load(outfile_name, allow_pickle=True)
        result = data['arr_0']
        ini = 0
        max_fitness_hist = np.zeros([300,100])
        for ind_r, r in enumerate(result):
            if ini == 0:
                max_fitness_vec = r[0]
                max_x_vec = r[1].reshape([r[1].shape[0],r[1].shape[1],1])
                max_fitness_hist[:r[2].shape[0],ind_r] = r[2]
                num_aes_vec = r[3]
                exec_time_vec = r[4]
            else:
                max_fitness_vec = np.append(max_fitness_vec,r[0]) 
                max_x_vec = np.append(max_x_vec,r[1].reshape([r[1].shape[0],r[1].shape[1],1]), axis=2)
                max_fitness_hist[:r[2].shape[0],ind_r] = r[2]
                num_aes_vec = np.append(num_aes_vec, r[3])
                exec_time_vec = np.append(exec_time_vec, r[4])
            ini+=1
            plot_figure(data_vectors, cl_centers, max_x_vec[:,:,-1])
            
        success_thr = 5e-2
        loops = max_fitness_vec.shape[0]
        # calc SR
        pos_success = np.where((abs(max_fitness_vec-global_max)<success_thr) | (max_fitness_vec>global_max))
        SR = pos_success[0].shape[0]/loops
        print("SR:")
        print(SR)
        #calc MBF
        MBF = np.mean(max_fitness_vec)
        print("MBF:")
        print(MBF)
        #calc AES
        #pos_success = np.where(num_aes_vec!=num_gen*lambd+pop_size)
        AES = np.mean(num_aes_vec[pos_success])
        print("AES:")
        print(AES)
        mtime = np.mean(exec_time_vec[pos_success])
        print("Mean execution time:")
        print(mtime)
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(max_fitness_vec)
        ax.set_title(str(alg))
        ax.set_xlabel('Aptidão Máxima')
        fig = plt.figure()