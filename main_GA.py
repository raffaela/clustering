# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:56:15 2020

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
from functools import partial
from time import perf_counter


def plot_figure(data_vectors, cl_centers, max_x_vec):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(cl_centers[0,:], cl_centers[1,:],cl_centers[2,:], marker = "^", color = "b", label = "cl_centers")
    ax.scatter(max_x_vec[0,:], max_x_vec[1,:],max_x_vec[2,:], marker = "v", color = "r", label = "Xmin")
    ax.scatter(data_vectors[0,:], data_vectors[1,:],data_vectors[2,:], marker = ".", color = "y", alpha=.5, label = "data_vectors")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    
# cost function
def calc_fitness(X,Y, ignore_empty=False):
    # X: data vectors
    # Y: codebook
    # Returns: total cost  
    high_cost = -1e2
    J=0; M,N=np.shape(X); M,K=np.shape(Y)
    d = np.zeros([K,N])
    #p=np.zeros([K,1])
    for n in range(N):
        d[:,n] =np.sum(np.power(X[:,n].reshape([M,1])-Y,2),axis=0)
    row_min = np.argmin(d, axis = 0)
    #aux = np.copy(d)
    if ignore_empty==True:
        for k in range(K):
            if k not in row_min:
                    return high_cost
    for ind,m in enumerate(row_min):
        J+= d[m,ind]  
    J=J/N
    return -J

def generate_data(NC,M,P,cl):
    flg_plot=1
    # data generation
    cl_centers = np.random.normal(0,3,[M,NC])
    data_vectors = np.repeat(cl_centers, P, axis=1) + np.random.normal(0,1,[M,NC*P])

    if flg_plot==1:
        # plot generated data
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(cl_centers[0,:], cl_centers[1,:], cl_centers[2,:])
        ax.scatter(data_vectors[0,:], data_vectors[1,:],data_vectors[2,:])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    return (data_vectors, cl_centers)


def run_SGA(data_vectors,df_params):
    start = perf_counter()
    params = df_params[1]
    num_gen = int(params.num_gen)
    pop_size = int(params.pop_size)
    num_bits = int(params.num_bits)
    lambd = int(params.lambd)
    num_parents = params.num_parents
    NC = int(params.num_clusters)
    global_max = params.global_max
    success_thr = params.success_thr
    num_gen_tolerance = 20
    epsilon = 0.1
    num_aes = 0
    prob_mutation = 1
    max_fitness = -100
    x = np.random.uniform(-5,5,size=[num_bits,NC,pop_size])
    fitness = np.zeros([pop_size])
    time_no_improve = 0

    for k in range(pop_size):
        fitness[k] = calc_fitness(data_vectors,x[:,:,k])
        num_aes+=1
    max_fitness_hist = np.array([np.max(fitness)])
    args_sort  = np.argsort(fitness,axis=0)
    fitness = fitness[args_sort]    
    max_fitness_cur = fitness[-1]
    x = x[:,:,args_sort]

    for n in range(num_gen):
        # parents selection
        min_fitness = fitness[0]
        if max_fitness_cur != min_fitness:
            probs = (fitness-min_fitness)/(max_fitness_cur - min_fitness)
            probs = probs/np.sum(probs)
        else: probs = np.ones([pop_size,1])/pop_size
        cum_probs = np.cumsum(probs[:-1])
        # recombination 
        offspring = np.zeros([num_bits,NC,lambd])
        for child in range(0,int(lambd/2),2):
            parents_ind = np.random.choice(range(pop_size), size =2, p = probs,\
                                              replace=False)
            p0_ind = parents_ind[0]
            p1_ind= parents_ind[1]
            p1 = x[:,:,int(p0_ind)]
            p2 = x[:,:,int(p1_ind)]
            split_loc = np.random.randint(num_bits)
            offspring[:split_loc,:,child] = p1[:split_loc,:]
            offspring[split_loc:,:,child] = p2[split_loc:,:]
            offspring[:split_loc,:,child+1] = p2[:split_loc,:]
            offspring[split_loc:,:,child+1] = p1[split_loc:,:]
            
            #mutation
            if np.random.rand()<prob_mutation:
                R = np.random.randn(num_bits,NC)
                offspring[:,:,child] = offspring[:,:,child]+epsilon*R
                R = np.random.randn(num_bits,NC)
                offspring[:,:,child+1] = offspring[:,:,child+1]+epsilon*R
                
        fitness = np.zeros(offspring.shape[2])
        for p in range(offspring.shape[2]):
            fitness[p] = calc_fitness(data_vectors,offspring[:,:,p])
            num_aes+=1
        # sort 
        args_sort  = np.argsort(fitness,axis=0)
        fitness_sort = fitness[args_sort]
        x_sort = offspring[:,:,args_sort]
        #remove weaker parents
        x = x_sort[:,:,-pop_size:]
        fitness = fitness_sort[-pop_size:]
        max_fitness_cur = fitness[-1]
        max_fitness_hist = np.append(max_fitness_hist,max_fitness_cur)
        time_no_improve+=1
        if max_fitness_cur>max_fitness:
            max_fitness = max_fitness_cur
            max_x = x[:,:,-1]
            time_no_improve = 0
        if (max_fitness>=global_max or time_no_improve>=num_gen_tolerance): 
            #stores only first occurrence of reaching global maximum
            break

    end = perf_counter()
    exec_time = end-start
    return max_fitness,max_x,max_fitness_hist,num_aes, exec_time

def run_ES(data_vectors,df_params):
    start = perf_counter()
    params = df_params[1]
    num_gen = int(params.num_gen)
    pop_size = int(params.pop_size)
    num_bits = int(params.num_bits)
    lambd = int(params.lambd)
    NC = int(params.num_clusters)
    global_max = params.global_max
    success_thr = params.success_thr
    epsilon = 5e-4 # for perturbation mutation
    max_fitness = -100
    prob_mutation = 1
    tau1 = 1/np.sqrt(2*num_bits)
    tau2 = 1/np.sqrt(2*np.sqrt(num_bits))
    x = np.random.uniform(-5,5,size=[num_bits,NC,pop_size])
    sigma_x = np.random.uniform(1e-3,1e-1,size=[num_bits,NC,pop_size])
    num_gen_tolerance = 20
    num_aes = 0
    fitness = np.zeros([pop_size])
    for k in range(pop_size):
        fitness[k] = calc_fitness(data_vectors,x[:,:,k])
        num_aes+=1
    max_fitness_hist = np.array([np.max(fitness)])
    args_sort  = np.argsort(fitness,axis=0)
    fitness = fitness[args_sort]    
    max_fitness_cur = fitness[-1]
    x = x[:,:,args_sort]
    time_no_improve = 0
    for n in range(num_gen):
        # parents selection
        min_fitness = fitness[0]
        if max_fitness_cur != min_fitness:
            probs = (fitness-min_fitness)/(max_fitness_cur - min_fitness)
            probs = probs/np.sum(probs)
        else: probs = np.ones([pop_size,1])/pop_size
        cum_probs = np.cumsum(probs[:-1])
        x_offspring = np.zeros([num_bits, NC,lambd])
        sigmax_offspring = np.zeros([num_bits,NC,lambd])
        for child in range(lambd):
            # parents selection
            parents_ind = np.random.choice(range(pop_size), size =2, p = probs,\
                                              replace=False)
            parents = x[:,:,parents_ind]
            sigma_parents = sigma_x[:,:,parents_ind]
            #recombination
            p1 = parents[:,:,0]
            p2 = parents[:,:,1]
            sigma_p1 = sigma_parents[:,:,0]
            sigma_p2 = sigma_parents[:,:,1]
            choice  = np.random.randint(2,size=[num_bits,NC])
            x_offspring[:,:,child] = p1*(1-choice)+p2*choice
            sigmax_offspring[:,:,child] = (sigma_p1+sigma_p2)/2
            #mutation
            if np.random.uniform(0,1)<prob_mutation:
                Rt1 = np.random.randn()
                Rt2= np.random.randn(num_bits,NC)
                sigmax_offspring[:,:,child] = sigmax_offspring[:,:,child]\
                    *np.exp(tau1*Rt1)*np.exp(tau2*Rt2)
                if np.any(sigmax_offspring[:,:,child]<epsilon):
                    sigmax_child = sigmax_offspring[:,:,child]
                    positions = np.where(sigmax_child<epsilon)
                    sigmax_child[positions] = epsilon
                    sigmax_offspring[:,:,child] = sigmax_child
                R = np.random.randn(num_bits,NC)
                x_offspring[:,:,child] = x_offspring[:,:,child]+ sigmax_offspring[:,:,child]*R
 
        x = x_offspring
        sigma_x = sigmax_offspring
        fitness = np.zeros([x.shape[2]])
        for p in range(x.shape[2]):
            fitness[p] = calc_fitness(data_vectors,x[:,:,p])
            num_aes+=1
        args_sort  = np.argsort(fitness,axis=0)
        fitness_sort = fitness[args_sort]
        x_sort = x[:,:,args_sort]
        x = x_sort[:,:,lambd-pop_size:]
        sigmax_sort = sigma_x[:,:,args_sort]
        sigma_x = sigmax_sort[:,:,lambd-pop_size:]
        fitness = fitness_sort[lambd-pop_size:]
        max_fitness_cur = fitness[-1]
        max_fitness_hist = np.append(max_fitness_hist,max_fitness_cur)
        time_no_improve+=1
        if max_fitness_cur > max_fitness:
            max_fitness = max_fitness_cur
            max_x = x[:,:,-1]
            time_no_improve = 0
        if (max_fitness>=global_max or time_no_improve>=num_gen_tolerance): 
            #stores only first occurrence of reaching global maximum
            break
    end = perf_counter()
    exec_time = end-start
    return max_fitness,max_x,max_fitness_hist,num_aes, exec_time

def run_EP(data_vectors,df_params): #not working yet
    start = perf_counter()
    params = df_params[1]
    num_gen = int(params.num_gen)
    pop_size = int(params.pop_size)
    num_bits = int(params.num_bits)
    lambd = int(params.lambd)
    NC = int(params.num_clusters)
    global_max = params.global_max
    success_thr = params.success_thr
    epsilon = 5e-4 # for perturbation mutation
    max_fitness = -100
    prob_mutation = 1
    tau1 = 1/np.sqrt(2*num_bits)
    tau2 = 1/np.sqrt(2*np.sqrt(num_bits))
    x = np.random.uniform(-5,5,size=[num_bits,NC,pop_size])
    sigma_x = np.random.uniform(1e-3,1e-1,size=[num_bits,NC,pop_size])
    num_aes = 0
    fitness = np.zeros([pop_size])
    for k in range(pop_size):
        fitness[k] = calc_fitness(data_vectors,x[:,:,k])
        num_aes+=1
    max_fitness_hist = np.array([np.max(fitness)])
    args_sort  = np.argsort(fitness,axis=0)
    fitness = fitness[args_sort]    
    max_fitness_cur = fitness[-1]
    x = x[:,:,args_sort]
    time_no_improve = 0
    for n in range(num_gen):
        # parents selection
        x_offspring = np.zeros([num_bits, NC,lambd])
        sigmax_offspring = np.zeros([num_bits,NC,lambd])
        for  ind_par in range(x.shape[2]):
        
            parent = x[:,:,ind_par]
            sigma_parent = sigma_x[:,:,ind_par]
            #recombination
            x_offspring[:,:,] = parent
            sigmax_offspring[:,:,child] = sigma_parent
            #mutation
            if np.random.uniform(0,1)<prob_mutation:
                Rt1 = np.random.randn()
                Rt2= np.random.randn(num_bits,NC)
                sigmax_offspring[:,:,child] = sigmax_offspring[:,:,child]\
                    *np.exp(tau1*Rt1)*np.exp(tau2*Rt2)
                if np.any(sigmax_offspring[:,:,child]<epsilon):
                    sigmax_child = sigmax_offspring[:,:,child]
                    positions = np.where(sigmax_child<epsilon)
                    sigmax_child[positions] = epsilon
                    sigmax_offspring[:,:,child] = sigmax_child
                R = np.random.randn(num_bits,NC)
                x_offspring[:,:,child] = x_offspring[:,:,child]+ sigmax_offspring[:,:,child]*R
 
        x = x_offspring
        sigma_x = sigmax_offspring
        fitness = np.zeros([x.shape[2]])
        for p in range(x.shape[2]):
            fitness[p] = calc_fitness(data_vectors,x[:,:,p])
            num_aes+=1
        args_sort  = np.argsort(fitness,axis=0)
        fitness_sort = fitness[args_sort]
        x_sort = x[:,:,args_sort]
        x = x_sort[:,:,lambd-pop_size:]
        sigmax_sort = sigma_x[:,:,args_sort]
        sigma_x = sigmax_sort[:,:,lambd-pop_size:]
        fitness = fitness_sort[lambd-pop_size:]
        max_fitness_cur = fitness[-1]
        if max_fitness_cur > max_fitness:
            max_fitness = max_fitness_cur
            max_x = x[:,:,-1]
            max_fitness_hist = np.append(max_fitness_hist,max_fitness_cur)
            time_no_improve = 0
        if (max_fitness>=global_max or time_no_improve>=num_gen_tolerance): 
                #stores only first occurrence of reaching global maximum
            break
    end = perf_counter()
    exec_time = end-start
    return max_fitness,max_x,max_fitness_hist,num_aes, exec_time


if __name__ == '__main__':
    results_dir = os.path.join(
            os.getcwd(), 
            datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(results_dir)
    csvfile = os.path.join(results_dir,"params.csv")
    input_exist = 1
    num_bits = 3
    NC = 16
    alg = "EP"
    if input_exist==0:
        inputfile_name = os.path.join(results_dir,"input.npz")
        max_fitness = -100
        num_bits = 3
        P = 100
        cl = 0
        data_vectors, cl_centers = generate_data(NC,num_bits,P,cl)
        global_max = calc_fitness(data_vectors,cl_centers)
        np.savez(inputfile_name, data_vectors, cl_centers, global_max, alg)
    else:
        input_dir = "resultados-parte1\\20200924_012545"
        #input_dir = "20200929_221616"
        inputfile_name = os.path.join(input_dir,"input.npz")
        new_inputfile_name = os.path.join(results_dir,"input.npz")
        input_data = np.load(inputfile_name)
        data_vectors = input_data['arr_0']
        cl_centers = input_data['arr_1']
        global_max = input_data['arr_2']     
        np.savez(new_inputfile_name, data_vectors, cl_centers, float(global_max), str(alg))

    if alg == "SGA":
        func = run_SGA
    elif alg =="ES":
        func = run_ES
    elif alg =="EP": 
        func =run_EP
    mp = 0
    loops = 1
    num_gen = 300
    #pop_size = 10
    pop_size = 50
    lambd = 300 #must be even
    #lambd = 60
    num_parents = 50
    success_thr = 5e-2
    
    df = pd.DataFrame([[num_bits,num_gen,pop_size,lambd,num_parents,NC,global_max,success_thr, loops]], \
                      columns=["num_bits","num_gen","pop_size","lambd","num_parents","num_clusters","global_max","success_thr","loops"])
    df.to_csv(csvfile)
    df_params  = df.loc[df.index.repeat(df.loops)].reset_index(drop=True)
    
    df_results = pd.DataFrame([], columns = ["max_fitness","max_x"])
    if mp==1:
        with Pool(8) as pool:
              res = pool.map(partial(func,data_vectors), df_params.iterrows())
             
        outfile_name = os.path.join(results_dir,"results.npz")
        
        np.savez(outfile_name, res)
        
        data = np.load(outfile_name, allow_pickle=True)
        result = data['arr_0']
        ini = 0
        max_fitness_hist = np.zeros([num_gen,loops])
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
            
    else:
        max_fitness_vec = np.zeros(loops)
        max_x_vec = np.zeros([num_bits,NC,loops])
        #max_fitness_hist = np.zeros([loops])
        num_aes_vec = np.zeros(loops)
        exec_time_vec = np.zeros(loops)
        for l in range(loops):
            max_fitness,max_x, mas_fitness_hist, num_aes, exec_time = func(data_vectors,(l,df.loc[0]))
            max_fitness_vec[l] = max_fitness
            max_x_vec[:,:,l] = max_x
            num_aes_vec[l] = num_aes
            exec_time_vec[l] = exec_time
            plot_figure(data_vectors, cl_centers, max_x_vec[:,:,l])

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
    AES = np.mean(num_aes_vec[pos_success])
    print("AES:")
    print(AES)
    mtime = np.mean(exec_time_vec[pos_success])
    print("Mean execution time:")
    print(mtime)