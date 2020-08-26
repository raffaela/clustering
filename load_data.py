# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 01:26:01 2020

@author: Raffaela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from clustering import  Jxy

#dir_name = "20200824_224759"
#dir_name = "20200825_005331"
#dir_name = "20200823_195840"
dir_name = "dados-melhor-caso"
data_file_name = "input_data.npz"
res_file_name = "results_27.npz"
res_file_path = os.path.join(os.getcwd(), dir_name, res_file_name)
data_file_path = os.path.join(os.getcwd(), dir_name, data_file_name)
data = np.load(data_file_path)
res = np.load(res_file_path)

data_vectors = data["arr_0"]
cl_centers = data["arr_1"]
data_vectors = data["arr_0"]
Xmin = res["arr_0"]
hist_J = res["arr_1"]
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
