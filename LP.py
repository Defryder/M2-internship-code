# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:57:08 2021

@author: Luc
"""

from mip import Model, xsum, maximize, minimize, BINARY, CONTINUOUS
import numpy as np

# N = 7;
# M = 3;
# O = 2;

# c =    [[0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0]];

# p =    [[3, 1, 1, 1, 1, 1, 1],
#         [3, 1, 1, 1, 1, 1, 1],
#         [3, 1, 1, 1, 1, 1, 1]];

# d =    [[1, 2],
#         [2, 1]];

# f =    [[1, 1],
#         [2, 2]];

# env =  [1, 1, 2, 2, 2];

# Cmax = 0
# Tmax = 3


#LP without environment constraints
def LPS(Cmax, Tmax, M, N, K, c, p, d, b, env):
    
    m = Model("LP")
    
    x = [[m.add_var(var_type=CONTINUOUS) for j in range(N)] for i in range(M)]
    
    m.objective = minimize(xsum(xsum(c[i][j]*x[i][j] for j in range(N)) for i in range(M)))
    
    # Add constraints
    m += xsum(xsum(c[i][j]*x[i][j] for j in range(N)) for i in range(M)) <= Cmax
    
    for j in range(0, N):
        m += xsum(x[i][j] for i in range(M)) == 1
    
    for i in range(0, M):
        m += xsum(p[i][j]*x[i][j] for j in range(N)) <= Tmax
    
    for i in range(0, M):
        for j in range(0, N):
            if p[i][j] > Tmax:
                m += x[i][j] == 0
    
    status = m.optimize()
    
    if(status.value == 0):
        out = np.array([[float(x[i][j].x) for j in range(N)] for i in range(M)])
    else:
        out = 0
    # print(out)
    return status.value, out

#LP with environment constraints
def LP(Cmax, Tmax, M, N, K, c, p, d, b, env):
    
    m = Model("LP")
    
    x = [[m.add_var(var_type=CONTINUOUS) for j in range(N)] for i in range(M)]
    
    e = [[m.add_var(var_type=CONTINUOUS) for k in range(K)] for i in range(M)]
    
    m.objective = minimize(xsum(xsum(c[i][j]*x[i][j] for j in range(N)) for i in range(M)) + xsum(xsum(d[i][k]*e[i][k] for k in range(K)) for i in range(M)))
    
    # Add constraints
    m += xsum(xsum(c[i][j]*x[i][j] for j in range(N)) for i in range(M)) + xsum(xsum(d[i][k]*e[i][k] for k in range(K)) for i in range(M)) <= Cmax
    
    for j in range(0, N):
        m += xsum(x[i][j] for i in range(M)) == 1
    
    for i in range(0, M):
        m += xsum(p[i][j]*x[i][j] for j in range(N)) + xsum(b[i][k]*e[i][k] for k in range(K)) <= Tmax
    
    for i in range(0, M):
        for j in range(0, N):
            m += x[i][j] <= e[i][env[j]]
    
    for k in range(0, K):
        for j in range(0, N):
            m += e[i][k] <= 1
    
    for i in range(0, M):
        for j in range(0, N):
            if p[i][j] + b[i][env[j]] > Tmax:
                m += x[i][j] == 0
    
    status = m.optimize()
    
    if(status.value == 0):
        out = np.array([[float(x[i][j].x) for j in range(N)] for i in range(M)])
    else:
        out = 0
    
    if(status.value == 0):
        out_e = np.array([[float(e[i][k].x) for k in range(K)] for i in range(M)])
    else:
        out_e = 0
    # print(out)
    return status.value, out, out_e