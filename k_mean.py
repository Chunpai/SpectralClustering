import numpy as np
from numpy import linalg as LA
import math
import random


def initialize(data):
    '''
    initilize the cluster k representations
    input data is n*d numpy ndarray
    '''
    n,k = data.shape
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



def compute_mean(data_list):
    data_list = np.array(data_list)
    mean = [] 
    dims = len(data_list[0])
    for d in range(dims):
        #print d, data_list[:,d]
        m = np.mean(data_list[:,d])
        mean.append(m)
    return mean


def find_optimal_reps(cluster_reps,data):
    clusters = {}
    dims = len(cluster_reps)
    for i in range(dims):
        clusters[i] = []
        clusters[i].append(cluster_reps[i])
    
    line = 0
    for e in data:
        line += 1
        #print line
        minimum = 1000000
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
        new_rep = compute_mean(clusters[min_index])
        #print "old",cluster_reps[min_index]
        #print "new",new_rep
        cluster_reps[min_index] = new_rep
        #print clusters[0]
    return cluster_reps 
        

def k_mean(k,data):
    cluster_reps = initialize(k,data)
    print "initial", cluster_reps
    new_cluster_reps = find_optimal_reps(cluster_reps, data)
    print "1st", new_cluster_reps
    #new_cluster_reps = find_optimal_reps(cluster_reps, data)
    #print "2nd", new_cluster_reps
    return new_cluster_reps


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


