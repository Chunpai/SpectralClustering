import numpy as np
from numpy import linalg as LA
import math
import random
import matplotlib.pyplot as plt

def initialize(data, k):
    '''
    initilize the cluster k representations
    input data is n*d numpy ndarray
    '''
    n,p = data.shape
    cluster_reps = []
    initial_indices = []
    for i in range(k):
        cluster = []
        rep = random.randint(0,n)
        while rep in initial_indices:
            rep = random.randint(0,n)
        initial_indices.append(rep)
        cluster_reps.append(data[rep])
    #print cluster_reps
    return cluster_reps 


def computeMean(data_list):
    '''
    '''
    data_list = np.array(data_list)
    mean = [] 
    dims = len(data_list[0])
    for i in range(dims):
        m = np.mean(data_list[:,i])
        mean.append(m)
    return mean


def find_optimal_reps(data, cluster_reps):
    clusters = {}
    k = len(cluster_reps)
    for i in range(k):
        clusters[i] = []
        clusters[i].append(cluster_reps[i])
    
    line = 0
    for e in data:
        line += 1
        #print line
        minimum = np.inf 
        min_index = 0
        index = -1
        for rep in cluster_reps:
            index += 1
            #print e, rep
            #print e - rep
            l1 = LA.norm(e - rep,1)
            if l1 < minimum:
                minimum = l1
                min_index = index
        clusters[min_index].append(e)
        new_rep = computeMean(clusters[min_index])
        #print "old",cluster_reps[min_index]
        #print "new",new_rep
        cluster_reps[min_index] = new_rep
        #print clusters[0]
    return cluster_reps, clusters 
        

def k_means(data, k):
    cluster_reps = initialize(data, k)
    print "initial", cluster_reps
    new_cluster_reps, clusters = find_optimal_reps(data, cluster_reps)
    print "1st", new_cluster_reps
    print "optional reassignment"
    new_cluster_reps, clusters = find_optimal_reps(data, cluster_reps)
    print "2nd", new_cluster_reps
    new_cluster_reps, clusters = find_optimal_reps(data, cluster_reps)
    print '3rd', new_cluster_reps
    return new_cluster_reps, clusters


def sum_l1_distance(cluster_reps, data): 
    sum = 0
    for e in data:
        minimum = 100000
        for rep in cluster_reps:
            l1 = LA.norm(e - rep, 1)
            if l1 < minimum:
                minimum = l1
        sum += minimum
    print "(2-mean) sum of total l1 distance", sum



def plot(clusters,k):
    plt.xlabel('x')
    plt.axis([-1,1,-1,1])
    color_list = {0:'red',1:'green',2:'blue',3:'yellow'}
    for i in range(k):
        c = np.array(clusters[i])
        x = c[:,0]
        y = c[:,1]
        plt.plot(list(x), list(y), marker=',', color=color_list[i])
    plt.savefig("sc.png")    
