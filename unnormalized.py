import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy as sp
import scipy.sparse.linalg as sLA
import pickle



def readNetwork(data_dir, filename):
    '''
    read the network, and construct the Laplacian matrix
    '''
    G = nx.Graph()
    infile = open(data_dir+filename,'r')
    for line in infile:
        edge = line.strip().split(' ')
        source = edge[0]
        target = edge[1]
        G.add_edge(source,target)
    infile.close()
    #print 'number of nodes', G.number_of_nodes()
    #print 'number of edges', G.number_of_edges()
    L = (-1.0)*np.array(nx.to_numpy_matrix(G)) 
    degree_list = nx.degree(G)
    #print degree_list
    for i in range(G.number_of_nodes()):
        L[i][i] = degree_list[str(i)]
    
    eig_vals , eig_vecs = LA.eig(L)
    pickle.dump(L, open(data_dir+'Laplacian.pkl','wb'))
    pickle.dump((eig_vals,eig_vecs), open(data_dir+'eigenpairs.pkl','wb'))


def serieizationRead(data_dir, laplapcian_pickle, eigenpair_pickle):
    L = pickle.load(open(data_dir+laplapcian_pickle,'rb'))
    eigenpairs = pickle.load(open(data_dir+eigenpair_pickle,'rd'))
    return L, eigenpairs

def spectralClustering(L, k):
    '''
    input the unnormalized laplacian matrix L (numpy ndarray), and cluster number k (int) 
    
    return the clusters
    '''
    #print L.shape
    #A = sp.sparse.csr_matrix(L) #store laplacian into scipy sparse matrix,which can compute the eigenvalues fast
    #e_vals, e_vecs = sLA.eigs(A,L.shape[0] -2)
    #e_vals, e_vecs = LA.eig(L)  

    return 


if __name__ == '__main__':
    data_dir = 'datasets/facebook/'
    network = 'facebook_combined.txt'
    laplapcian_pickle = 'Laplacian.pkl'
    eigenpair_pickle = 'eigenpairs.pkl'
    #readNetwork(data_dir, network)
    L, eigenpairs = serieizationRead(data_dir,laplapcian_pickle, eigenpair_pickle)
    k = 2
    spectralClustering(L, eigenpairs, k)


