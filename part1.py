#Importing required packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import sparse, stats
from scipy.optimize import curve_fit
import networkx as nx
import random
import scipy.special

#Loading in the data.
def loadData(network):
    if (network == "git"):
        G = nx.read_edgelist('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/dataGit/musae_git_edges.csv', delimiter=',', nodetype=int)
        return G
    elif(network == "roads"):
        G = nx.read_edgelist('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/dataRoads/roadNet-PA.txt', nodetype=int)
        return G
    else: return

def powerLaw(k, c, gamma):
    return c*k**(-gamma)

def poisson(k, mu):
    return np.exp(-mu) * np.power(mu, k)/scipy.special.factorial(k)

def fitter(unique, tail):
    #We want to fit the power law only to the tail
    uniqueTail = unique[unique >= tail]
    fractionTail = fraction[unique >= tail]
    popt, pcov = curve_fit(powerLaw, uniqueTail, fractionTail)
    return popt, uniqueTail, fractionTail

def plotter(G, fit, scale, network):
    counts = np.array(nx.degree_histogram(G))
    unique = np.arange(0,len(counts))
    fraction = counts/len(list(G.nodes))
    plt.title('Degree distribution plot')
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
        #plt.savefig('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/plots/loglog'+str(network)+'.pdf')
    else: plt.savefig('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/plots/'+str(network)+'.pdf')
    #plt.show()

network = "git"
G = loadData(network)
#plotter(G, False, 'log', network)

#Create a sparse representation of the adjacency matrix
A = nx.adjacency_matrix(G)
A3 = A**3


#Now compute the clustering coefficient
#Ci = 2*(#triangles with vertex in i)/(ki(ki-1))
#Where (#triangles with vertex in i) = A^3[i,i]/2
#So Ci = A^3[i,i]/(ki(ki-1))

def C(i):
    k = np.array(nx.degree(G))[i][1]
    if k==0 or k==1: return 0
    else: return A3[i,i]/(k*(k-1))


def C_av():
    clustering_sum = 0.0
    for i in range(len(nx.degree(G))-1):
        clustering_sum += C(i)
        if i%1000==0: print(i)
    return clustering_sum/len(nx.degree(G))

x = nx.clustering(G)
print("average clustering using build in function = " + str(np.mean(list(x.values()))))
x = C_av()
print("average clustering using our function = " + str(x))


random_int = random.randint(0, len(nx.degree(G))-1)
average_deg_random = nx.average_neighbor_degree(G, nodes=[random_int])[random_int]

degrees = np.array(nx.degree(G)).flatten()[1::2]
average_deg = np.average(degrees)

print("average_degree_random = "+str(average_deg_random))
print("average_degree = "+str(average_deg))


# -------------------- MAAS THIS IS THE DATA FOR THE PLOTS ---------------------------

counts = np.array(nx.degree_histogram(G))
unique = np.arange(0,len(counts))
fraction = counts/len(list(G.nodes))













