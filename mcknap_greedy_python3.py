import numpy as np
import itertools

def mcknap_greedy(capacity, profits, weights, groups, items):
    
    if sum(weights[items == min(items[items > 0])]) > capacity:
        print("No solution possible!")
        return None
    if sum(weights[items == max(items)]) <= capacity:
        print("trivial solution!")
        return None
    
    J = len(np.unique(items))
    
    dominated = np.zeros(len(profits), dtype = int)
    lightest = np.empty(len(np.unique(groups)), dtype = int)
    
    for i in np.unique(groups):
        group = np.array(range(i*J, (i+1)*J))
        group = group[weights[group].argsort()]
        for p in itertools.combinations(group, 2):
            r = p[0]
            s = p[1]
            if weights[r] <= weights[s] and profits[r] >= profits[s]:
                #dominated_indices.append(p[1])
                dominated[p[1]] = 1
        for p in itertools.combinations(group, 3):
            r = p[0]
            s = p[1]
            t = p[2]
            det = (weights[s] - weights[r])*(profits[t] - profits[r]) - (profits[s] - profits[r])*(weights[t] - weights[r])
            if det >= 0:
                dominated[s] = 1        
        lightest[i] = int(group[dominated[group] == 0][0]) 
    
    taken = np.zeros(len(profits), dtype = int)
    taken[lightest] = 1
    
    W = sum(weights[taken==1])
    P = sum(profits[taken==1])
    
    # now we need the gradients (lambdas)
    lambdas = np.array([-np.inf]*len(profits))

    for i in np.unique(groups):
        group = np.array(range(i*J, (i+1)*J))
        group = group[weights[group].argsort()]
        group_undominated = group[dominated[group]==0]
        if len(group_undominated) > 1:
            for k in range(1, len(group_undominated)):
                j = group_undominated[k]
                j_minus_one = group_undominated[k-1]
                lambdas[j] = (profits[j] - profits[j_minus_one])/(weights[j] - weights[j_minus_one])

    lambda_order = lambdas.argsort()[::-1]
    L = len(profits) - sum(np.isneginf(lambdas)) 
    
    for i in range(0, L):
        j = lambda_order[i]
        g = groups[j]
        group = np.array(range(g*J, (g+1)*J))
        group = group[weights[group].argsort()]
        group_undominated = group[dominated[group]==0]
        k = list(group_undominated).index(j)
        j_minus_one = group_undominated[k-1]
        w_j = weights[j]
        w_j_minus_one = weights[j_minus_one]
        p_j = profits[j]
        p_j_minus_one = profits[j_minus_one]
        if (W + w_j - w_j_minus_one) > capacity:
            break
        taken[j] = 1
        taken[j_minus_one] = 0
        W = W + w_j - w_j_minus_one
        P = P + p_j - p_j_minus_one 
    
    if W == capacity:
        print("We have an optimal solution!")
        
    return taken
        
capacity = 5e7
N = int(5e6)
offers = [0,100,250,500,1000]
J = len(offers)
propensities = np.random.uniform(low = 0.0, high = 0.2, size = N)
uplifts = propensities - np.random.uniform(low = 0.0, high = 0.15, size = N)
uplifts[::J] = 0
uplifts[uplifts < 0] = 0
bonuses = np.tile(offers,int(N/J))

profits = uplifts
weights = propensities*bonuses
groups = np.repeat(range(int(N/J)), J)
items = bonuses

taken = mcknap_greedy(capacity, profits, weights, groups, items)

print(sum(profits[taken==1]))
print(sum(weights[taken==1]))
