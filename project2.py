#Importing required packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.optimize import curve_fit
import networkx as nx
import random

#Declare path where we want to save the data and figures to
path = "/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/"

#Loading in the data.
def loadData(network):
    if (network == "Git"):
        G = nx.read_edgelist(path+'dataGit/musae_git_edges.csv', delimiter=',', nodetype=int)
        return G
    elif(network == "Roads"):
        G = nx.read_edgelist(path+'dataRoads/roadNet-PA.txt', nodetype=int)
        return G
    elif (network == "Arxiv"):
        G = nx.read_edgelist(path+'dataArxiv/ca-GrQc.txt', nodetype=int)
        return G
    elif (network == "Enron"):
        G = nx.read_edgelist(path+'dataEnron/email-Enron.txt', nodetype=int)
        return G
    else: return

#Function that returns a powerlaw
def powerLaw(k, c, gamma):
    return c*k**(-gamma)

#Function that fits the above powerlaw to the data
def fitter(unique, tail):
    #We want to fit the power law only to the tail
    uniqueTail = unique[unique >= tail]
    fractionTail = fraction[unique >= tail]
    popt, pcov = curve_fit(powerLaw, uniqueTail, fractionTail)
    return popt, uniqueTail, fractionTail

#Plotting the degree distribution
def plotter(G, fit, scale, network):
    counts = np.array(nx.degree_histogram(G))
    unique = np.arange(0,len(counts))
    fraction = counts/len(list(G.nodes))
    plt.title('Degree distribution plot of '+str(network))
    plt.scatter(unique, fraction, marker='.')
    plt.xlabel('$k$')
    plt.ylabel('$p_k$')
    plt.grid()

    if fit:
        popt, uniqueTail, fractionTail = fitter(unique, 10)
        plt.plot(uniqueTail, powerLaw(uniqueTail, *popt), c='r')

    if scale == "log":
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(path+'plots/loglog'+str(network)+'.pdf')
    else: plt.savefig(path+'plots/'+str(network)+'.pdf')
    plt.clf()

#Computing the clustering coefficient for node j
def C(i,j):
    k = G.degree(j)
    if k==0 or k==1: return 0
    else: return A3_diag[i]/(k*(k-1))

#Computing the average clustering coefficient
def C_av():
    l = len(nx.degree(G))
    x = list(G.nodes)    
    x.sort()
    clustering_sum = 0.0
    for i, j in enumerate(x):
        clustering_sum += C(i,j)
    return clustering_sum/l

#Computing the cube of the matrix and saving its diagonal
def findDiagonal(G, network):
    x = list(G.nodes)
    x.sort()
    A = nx.adjacency_matrix(G, nodelist=x)
    A2 = sparse.csr_matrix.dot(A, A)
    A3 = sparse.csr_matrix.dot(A, A2)
    A3_diag = A3.diagonal()
    np.savetxt(path+'data'+str(network)+'/A3_diag.txt', np.array(A3_diag))
    print('finished '+str(network))

#Function for computing A^3 and saving the diagonal for multiple networks
def diagGen(networks):
    for network in networks:
        G = loadData(network)
        findDiagonal(G, network)


#Declare networks we want to use
networks = ["Git", "Roads", "Arxiv", "Enron"]

#Generate the diagonal matrix of A^3
#diagGen(networks)

#Loop over the networks
for network in networks:
    G = loadData(network) #load data
    plotter(G, False, 'linear', network) #plot data
    A3_diag = np.loadtxt(path+'data'+str(network)+'/A3_diag.txt') #load diag(A^3)

    x = C_av() #Compute average clustering coefficient using our function
    print("average clustering coeffcient of network "+str(network)+" using our function = " + str(x))

    x = nx.average_clustering(G) #Compute average clustering coefficient using build in function
    print("average clustering coeffcient of network "+str(network)+" using build-in function= " + str(x))

#Calculating the degree and average neighbour degree for 5 random nodes of the git network
G = loadData("Git")

#Computing average neighbour degree for five random nodes
for i in range(5):
    random_int = random.randint(0, len(nx.degree(G))-1)
    degree = nx.degree(G, random_int)
    average_deg_random = nx.average_neighbor_degree(G, nodes=[random_int])[random_int]
    print(str(random_int)+" & "+str(degree)+" & "+str(average_deg_random)+" \\ \hline")

#Computing the average degree of the network
degrees = [tup[1] for tup in list(nx.degree(G))]
average_deg = np.average(degrees)
print("average_degree = "+str(average_deg))

#Computing the average neighbour degree of the network
neighbor_degrees = list(nx.average_neighbor_degree(G).values())
print(np.average(neighbor_degrees))

# -------------------- MAAS THIS IS THE DATA FOR THE PLOTS ---------------------------
"""
counts = np.array(nx.degree_histogram(G))
unique = np.arange(0,len(counts))
fraction = counts/len(list(G.nodes))
"""
