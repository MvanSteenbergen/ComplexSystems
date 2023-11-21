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

G = loadData("git")

counts = np.array(nx.degree_histogram(G))
unique = np.arange(0,len(counts))
#Note that the network is undirected, but if we have edge (a,b) we do not have edge (b,a) in the data.

#Converting the number of degrees to frequency
#unique, counts = np.unique(degrees, return_counts=True)

fraction = counts/len(list(G.nodes))


"""
#Fitting a power law
def powerLaw(k, c, gamma):
    return c*k**(-gamma)

#We want to fit the power law only to the tail
tail = 10
uniqueTail = unique[unique >= tail]
fractionTail = fraction[unique >= tail]
popt, pcov = curve_fit(powerLaw, uniqueTail, fractionTail)
"""

#Plotting the degree distribution and the fit
plt.title('Degree distribution plot')
plt.scatter(unique, fraction, marker='.')
#plt.plot(uniqueTail, powerLaw(uniqueTail, *popt), c='r')
plt.xscale('log')
plt.xlabel('$k$')
plt.yscale('log')
plt.ylabel('$p_k$')
plt.grid()
plt.savefig('/home/melle/OneDrive/Master/Year1/ComplexSystems/Project2/plots/loglogroads.pdf')
plt.show()