# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 01:26:01 2020

@author: Raffaela
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from clustering import  Jxy

#dir_name = "20200826_174250"
#dir_name = "20200828_161039"
#dir_name = "20200830_031527"

def get_test_data(dir_name):
    
    data_file_name = "input_data.npz"
    data_file_path = os.path.join(os.getcwd(), dir_name, data_file_name)
    data = np.load(data_file_path)
    data_vectors = data["arr_0"]
    cl_centers = data["arr_1"]
    
    return data_vectors, cl_centers

# dir_name = "20200825_212604"  #data for NC = 16, P =100
# dir_name = "20200826_010446"  #data for NC = 32, P = 200
# dir_name = "20200826_174250"  #data for NC = 24, P = 160
# dir_name = "20200828_161039"  #data for NC = 40, P = 250


if __name__ == '__main__':
    dir_name = "20200830_232928"
    
    data_vectors,cl_centers = get_test_data(dir_name)
    res_file_name = "results_0.npz"
    res_file_path = os.path.join(os.getcwd(), dir_name, res_file_name)
    res = np.load(res_file_path)
    Xmin = res["arr_0"]
    hist_J = res["arr_1"]
    
    J_res = Jxy(Xmin,cl_centers)
    Jmin = np.min(J_res)
    print("Minimum J")
    print(Jmin)
    # plot resulting clusters with data vectors
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(cl_centers[0,:], cl_centers[1,:], cl_centers[2,:], marker = "^", color = "b", label = "cl_centers")
    ax.scatter(Xmin[0,:], Xmin[1,:], Xmin[2,:], marker = "v", color = "r", label = "Xmin")
    ax.scatter(data_vectors[0,:], data_vectors[1,:], data_vectors[2,:], marker = ".", color = "y", alpha=.2, label = "data_vectors")
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(range(hist_J.shape[0]),hist_J)
    print("Jmin=")
    print(np.min(hist_J))
    ax.set_xlabel('#iterações')
    ax.set_ylabel('J')
    
    
    J_global = Jxy(data_vectors,cl_centers)
    print("Jglobal=")
    print(J_global)
