#Importing required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import curve_fit
import networkx as nx

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

    if scale == "log":
        plt.xscale('log')
        plt.yscale('log')

    if fit:
        popt, uniqueTail, fractionTail = fitter(unique, 10)
        plt.plot(uniqueTail, powerLaw(uniqueTail, *popt), c='r')

    if scale == "log":
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/plots/loglog'+str(network)+'.pdf')
    else: plt.savefig('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/plots/'+str(network)+'.pdf')
    plt.show()

network = "roads"
G = loadData(network)
#plotter(G, False, 'log', network)


#Create a sparse representation of the adjacency matrix
A = nx.adjacency_matrix(G)
A3 = A**3
print("done")

#Now compute the clustering coefficient
#Ci = 2*(#triangles with vertex in i)/(ki(ki-1))
#Where (#triangles with vertex in i) = A^3[i,i]/2
#So Ci = A^3[i,i]/(ki(ki-1))