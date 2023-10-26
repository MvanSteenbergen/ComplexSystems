
#Import packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#Function to compute the evolution
#Here we use numba to speed up the code
@njit
def evolution(l, t, n, rule):
    #Create a array to save all the data
    data = np.zeros((t,n))

    #Initial conditions and rule
    initial = np.random.binomial(1, 0.5, n)
    if rule == 6: living = [[0,1,0], [0,0,1]] #rule 6
    elif rule == 50: living = [[1,0,1], [1,0,0], [0,0,1]] #rule 50
    else: living = [[1,1,1], [1,0,1], [1,0,0], [0,0,1]] #rule 178

    #Set configs to the initial array, this makes numba happy
    config = initial
    config_prev = initial

    #Loop over all timesteps
    for i in range(t):
        #Save configuration to the data array
        data[i] = config

        #If we are not in the first timestep, the previous config is the config of one timestep in the past
        if i > 0: config_prev = config

        #Copy previous configuration to current configuration    
        config = np.copy(config_prev)

        #Loop over all cells
        for j in range(n):
            #Make list of neightbours
            check = [config_prev[(j-1)%n], config_prev[j], config_prev[(j+1)%n]]

            #Generate number between 0 and 1 with probability l
            #If equal to one we apply f2
            if np.random.binomial(1, l, 1) == 1:
                #If the local rule says that it lives, it lives 
                if check in living:
                    config[j] = 1
                #Else it dies
                else: config[j] = 0
            #Else we do not change
    
    #Return density
    return data

#Define amount of cells and timesteps
t = 5 #Timestes
n = 10 #Number of cells

alp = 0.25


data = evolution(alp, t, n, 6)

plt.rcParams['image.cmap'] = 'binary'


fig, ax = plt.subplots(figsize=(16, 9))
ax.matshow(data)
ax.axis(False)
fig.show()


