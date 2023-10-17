import numpy as np
import matplotlib.pyplot as plt

n = 1000
t = 500

initial = np.random.binomial(1, 0.5, n)
living_58 = [[0,0,1], [0,1,1], [1,0,0], [1,0,1]]
living_146 = [[0,0,1], [1,0,0], [1,1,1]]

def evolution(l, t, n, living = living_58):
    for i in range(t):
        if i == 0:
            config_prev = initial
        else: config_prev = config
            
        config = np.copy(config_prev)
    
        for j in range(n):
            check = [config_prev[(j-1)%n], config_prev[j], config_prev[(j+1)%n]]
    
            if np.random.binomial(1, l, 1) == 1:         
                if check in living:
                    config[j] = 1
                else: config[j] = 0
            else:
                config[j] = 0   

    return np.sum(config)/n

density_list = []
lambdas1 = np.linspace(0,0.5,10)
lambdas2 = np.linspace(0.5,1,30)
lambdas = np.concatenate((lambdas1, lambdas2))
print(lambdas)

for l in lambdas:
    d = evolution(l, t ,n)
    density_list.append(d)
    print(l)

plt.scatter(lambdas, density_list, marker='o')
plt.grid()
plt.xlabel("lambda")
plt.ylabel("density")
plt.title("bla")
plt.show()
