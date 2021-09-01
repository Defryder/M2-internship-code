# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:51:57 2021

@author: Luc
"""

import time
from datetime import datetime, date
from heuristic import *
from ILP import *
from LP import *
from approx import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import csv

def get_cost_s(x, e, M, N, K, c, p, d, b, env):
    tcost = np.sum(x*c)
    
    return tcost

def get_time_s(x, e, M, N, K, c, p, d, b, env):
    ttime =  max(sum((x*p)))
    return ttime

def get_cost(x, e, M, N, K, c, p, d, b, env):
    tcost = np.sum(x*c)
    ecost = np.sum(e*d)
#    ecost = 0
#    em = np.zeros((M, K))
#    for i in range(M):
#        for j in range(N):
#            if x[i][j] == 1 and em[i][env[j]] == 0:
#                ecost = ecost + d[i][env[j]]*e[i][env[j]]
#                em[i][env[j]] = 1
    
    return tcost + ecost

def get_cost_bad(x, e, M, N, K, c, p, d, b, env):
    tcost = np.sum(x*c)
    for i in range(M):
        for j in range(N):
            e[i][env[j]] = e[i][env[j]] + x[i][j]
    ecost = np.sum(e*d)
#    ecost = 0
#    em = np.zeros((M, K))
#    for i in range(M):
#        for j in range(N):
#            if x[i][j] == 1 and em[i][env[j]] == 0:
#                ecost = ecost + d[i][env[j]]*e[i][env[j]]
#                em[i][env[j]] = 1
    
    return tcost + ecost

def get_time(x, e, M, N, K, c, p, d, b, env):
    ttime =  max(sum((x*p).T))
    etime = 0
    etimemax = 0
    em = np.zeros((M, K))
    for i in range(M):
        for j in range(N):
            if x[i][j] == 1 and em[i][env[j]] == 0:
                etime = etime + b[i][env[j]]*e[i][env[j]]
                em[i][env[j]] = 1
        if etime < etimemax:
            etimemax = etime
    return ttime + etimemax

def generate(M, N, K):
    c = np.random.randint(1, 10, (M, N))
    p = np.random.randint(1, 10, (M, N))
    d = np.random.randint(1, 10, (M, K))
    b = np.random.randint(1, 10, (M, K))
    env = np.random.randint(0, K, (N))
    
    return c, p, d, b, env

def is_valid(x, M, N):
    if(np.array_equal(sum(x), np.ones((N)))):
        return True
    else:
        return False

def search(M, N, c, p):
    Cmax = N*10
    Tmax = N*10
    
    a = greedy(Cmax, Tmax, M, N, c, p)
    
    while(is_valid(a, M, N)):
        Cmax -= 1
        a = greedy(Cmax, Tmax, M, N, c, p)
    
    Cmax += 1
    
    a = greedy(Cmax, Tmax, M, N, c, p)
    
    while(is_valid(a, M, N)):
        Tmax -= 1
        a = greedy(Cmax, Tmax, M, N, c, p)
    
    Tmax += 1
    
    return Cmax, Tmax

def compare(Cmax, Tmax, M, N, c, p):
    a = greedy(Cmax, Tmax, M, N, c, p)
    b = ILP(Cmax, Tmax, M, N, c, p)
    
    a_cost = get_cost(a, M, N, c, p)
    b_cost = get_cost(b, M, N, c, p)
    
    a_time = get_time(a, M, N, c, p)
    b_time = get_time(b, M, N, c, p)
    
    print("Same cost? "+ str(a_cost == b_cost))
    print("Same time? "+ str(a_time == b_time))

def optimize(M, N, c, p):
    Cmax = 0 #useless
    Tmax = 0
    c, p = generate(M, N)
    x = LP(Cmax, Tmax, M, N, c, p)
    
    c_save = np.zeros((N*10))
    
    for Tmax in range(N*10):
        status, x = LP(Cmax, Tmax, M, N, c, p)
        if(status == 0):
            c_save[Tmax] = get_cost(x, M, N, c, p)
        else:
            c_save[Tmax] = 0

def pareto_s(M, N, K, c, p, d, b, env):
    
    algo = LP
    
    Cmax = np.inf
    Tmax = N*10
    
    par_c = []
    par_t = []
    par_c_a = []
    par_t_a = []
    par_c_m = []
    par_t_m = []
    par_c_i = []
    par_t_i = []
    save_xa = []
    save_x = []
    save_xi = []
    fail = []
    
    C = 0
    T = 0
    
    e = 0
    e_a = 0
    
    print("start pareto")

    C = Cmax
    while T < Tmax:
        print("T = "+str(T))
        status, x = LPS(Cmax, T, M, N, K, c, p, d, b, env)
        # print(status)
        if status == 0:
            
            C = np.ceil(get_cost(x, e, M, N, K, c, p, d, b, env))
            # L = 0
            # R = Cmax - 1
            # while L <= R:
            #     C = (L + R) // 2
                
            #     status, x, e = LP(C, T, M, N, K, c, p, d, b, env)
                
            #     if status == 1 :
            #         L = C + 1
            #     elif status == 0 :
            #         R = C - 1
            
            # if status != 0:
            #     C = C + 1
            #     status, x, e = LP(C, T, M, N, K, c, p, d, b, env)
            
            x_a = generate_biparite_s(x, M, N, K, c, p, d, b, env)
            if not is_valid(x_a, M, N):
                fail.append([C, T])
                print("FAILED")
            par_t_a.append(get_time_s(x_a, e_a, M, N, K, c, p, d, b, env))
            par_c_a.append(get_cost_s(x_a, e_a, M, N, K, c, p, d, b, env))
            save_xa.append(x_a)
            
            par_t_m.append(get_time_s(x, e, M, N, K, c, p, d, b, env))
            par_c_m.append(get_cost_s(x, e, M, N, K, c, p, d, b, env))
            save_x.append(x)
            
            par_t.append(T)
            par_c.append(C)
            
        status, x = ILPS(Cmax, T, M, N, K, c, p, d, b, env)
        if status == 0:
            par_t_i.append(T)
            par_c_i.append(get_cost_s(x, e, M, N, K, c, p, d, b, env))
            save_xi.append(x)
            
        T = T+1

#    plt.plot(par_t, par_c)
#    plt.plot(par_t, par_c_a)
#    plt.plot(par_t, par_c_m)
#    plt.plot([x*2 for x in par_t], par_c_m)
#    plt.xlabel("Time")
#    plt.ylabel("Cost")
    
    return par_t_a, par_c_a, par_t_m, par_c_m, par_t, par_c, par_t_i, par_c_i, save_x, save_xa, save_xi

def pareto(M, N, K, c, p, d, b, env):
    
    algo = LP
    
    Cmax = np.inf
    Tmax = N*10
    
    par_c = []
    par_t = []
    par_c_a = []
    par_t_a = []
    par_c_m = []
    par_c_mb = []
    par_t_m = []
    par_c_i = []
    par_t_i = []
    par_c_s = []
    par_t_s = []
    par_c_sa = []
    par_t_sa = []
    save_xsa = []
    save_xs = []
    save_xa = []
    save_x = []
    save_xi = []
    times = []
    fail = []
    
    C = 0
    T = 0
    
    print("start pareto")

    C = Cmax
    while T < Tmax:
        print("T = "+str(T))
        tstart = time.time()
        tl = []
        status, x, e = LP(Cmax, T, M, N, K, c, p, d, b, env)
        tl.append(time.time() - tstart)
        # print(status)
        if status == 0:
            
            C = np.ceil(get_cost(x, e, M, N, K, c, p, d, b, env))
            # L = 0
            # R = Cmax - 1
            # while L <= R:
            #     C = (L + R) // 2
                
            #     status, x, e = LP(C, T, M, N, K, c, p, d, b, env)
                
            #     if status == 1 :
            #         L = C + 1
            #     elif status == 0 :
            #         R = C - 1
            
            # if status != 0:
            #     C = C + 1
            #     status, x, e = LP(C, T, M, N, K, c, p, d, b, env)
            
            x_a, e_a = generate_biparite(x, M, N, K, c, p, d, b, env)
            tl.append(time.time() - tstart)
            if not is_valid(x_a, M, N):
                fail.append([C, T])
                print("FAILED")
            par_t_a.append(get_time(x_a, e_a, M, N, K, c, p, d, b, env))
            par_c_a.append(get_cost(x_a, e_a, M, N, K, c, p, d, b, env))
            save_xa.append(x_a)
            
            par_t_m.append(get_time(x, e, M, N, K, c, p, d, b, env))
            par_c_m.append(get_cost(x, e, M, N, K, c, p, d, b, env))
            par_c_mb.append(get_cost_bad(x, e, M, N, K, c, p, d, b, env))
            save_x.append(x)
            
            par_t.append(T)
            par_c.append(C)
        
        tstart = time.time()
        status, x, e = ILP(Cmax, T, M, N, K, c, p, d, b, env)
        tl.append(time.time() - tstart)
        if status == 0:
            par_t_i.append(T)
            par_c_i.append(get_cost(x, e, M, N, K, c, p, d, b, env))
            save_xi.append(x)
        
        
        pb = np.zeros((M, N))
        cd = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                pb[i][j] = p[i][j] + b[i][env[j]]
                cd[i][j] = c[i][j] + d[i][env[j]]
        
        tstart = time.time()
        status, x = LPS(Cmax, T, M, N, K, cd, pb, d, b, env)
        tl.append(time.time() - tstart)
        if status == 0:
            x_a, e_a = generate_biparite(x, M, N, K, c, p, d, b, env)
            tl.append(time.time() - tstart)
            par_t_sa.append(T)
            par_c_sa.append(get_cost_s(x_a, e_a, M, N, K, cd, pb, d, b, env))
            save_xsa.append(x_a)
            par_t_s.append(T)
            par_c_s.append(get_cost(x_a, e_a, M, N, K, c, p, d, b, env))
            save_xs.append(x)
        
        T = T+1
        times.append(tl)

#    plt.plot(par_t, par_c)
#    plt.plot(par_t, par_c_a)
#    plt.plot(par_t, par_c_m)
#    plt.plot([x*2 for x in par_t], par_c_m)
#    plt.xlabel("Time")
#    plt.ylabel("Cost")
    
    return par_t_a, par_c_a, par_t_m, par_c_m, par_c_mb, par_t, par_c, par_t_i, par_c_i, par_t_s, par_c_s, par_t_sa, par_c_sa, save_x, save_xa, save_xi, save_xs, save_xsa

def timer(M, N, K):
    
    c, p, d, b, env = generate(M, N, K)
    
    Cmax = np.inf
    Tmax = N*10
    
    tl = []
    
    C = 0
    T = 0
    
    brk = 0
    C = Cmax
    while T < Tmax and brk < 3:
        brk = 0 
        
        tl = []
        
        pb = np.zeros((M, N))
        cd = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                pb[i][j] = p[i][j] + b[i][env[j]]
                cd[i][j] = c[i][j] + d[i][env[j]]
        
        tstart = time.time()
        status, x = LPS(Cmax, T, M, N, K, cd, pb, d, b, env)
        if status == 0:
            tl.append(time.time() - tstart)
            x_a, e_a = generate_biparite(x, M, N, K, c, p, d, b, env)
            tl.append(time.time() - tstart)
            brk = brk + 1
        
        tstart = time.time()
        status, x, e = LP(Cmax, T, M, N, K, c, p, d, b, env)
        if status == 0:
            tl.append(time.time() - tstart)
            x_a, e_a = generate_biparite(x, M, N, K, c, p, d, b, env)
            tl.append(time.time() - tstart)
            brk = brk + 1
        
        tstart = time.time()
        status, x, e = ILP(Cmax, T, M, N, K, c, p, d, b, env)
        if status == 0:
            tl.append(time.time() - tstart)
            brk = brk + 1
        
        T = T+1
    
    return tl

M = 3
N = 7
K = 2

c =    [[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]];

p =    [[3, 1, 1, 1, 1, 1, 2],
        [3, 1, 1, 1, 1, 1, 2],
        [3, 1, 1, 1, 1, 1, 2]];

d =    [[1, 2],
        [2, 1]];

b =    [[1, 1],
        [2, 2]];

env =  [0, 0, 1, 1, 1];

Cmax = 0
Tmax = 3


M = 10
N = 21
K = 4

M = 2
N = 4
K = 2

save = []

for i in range(20):
    c, p, d, b, env = generate(M, N, K)
    par_t_a, par_c_a, par_t_m, par_c_m, par_c_mb, par_t, par_c, par_t_i, par_c_i, par_t_s, par_c_s, par_t_sa, par_c_sa, save_x, save_xa, save_xi, save_xs, save_xsa = pareto(M, N, K, c, p, d, b, env)
    save.append((par_t_a, par_c_a, par_t_m, par_c_m, par_c_mb, par_t, par_c, par_t_i, par_c_i, par_t_s, par_c_s, save_x, save_xa, save_xi, save_xs))
    
for i in range(len(save)):
    par_t_a, par_c_a, par_t_m, par_c_m, par_c_mb, par_t, par_c, par_t_i, par_c_i, par_t_s, par_c_s, save_x, save_xa, save_xi, save_xs = save[i]
    
    with open("save"+str(i)+".csv", mode="w") as csvfile:
    
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["par_t_a", "par_c_a", "par_t_m", "par_c_m", "par_c_mb", "par_t", "par_c", "par_t_i", "par_c_i", "par_t_s", "par_c_s"])
        for t in range(len(save[i][0])):
            l = []
            for j in range(11):
                if len(save[i][j]) <= t:
                    l.append("Nan")
                else:
                    l.append(save[i][j][t])
            writer.writerow(l)

times = []
machines = [[2, 4, 2],[3, 7, 2],[5, 11, 3],[10, 21, 4]]
for M, N, K in machines:
    for i in range(30):
        times.append([timer(M, N, K), M])

names = ["LP Shmoys", "LP Shmoys approx", "LP", "LP approx", "ILP"]
better_times = []     
for i in range(len(times)):
    for j in range(5):
        better_times.append([times[i][0][j], names[j], times[i][1]])

out = []
for i in range(len(machines)):
    for j in range(5):
        out.append([sum([item[0] for item in better_times[i*150 + j:(i+1)*150 + j:5]])/30, names[j], machines[i][0]])

with open("times.csv", mode="w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["time", "Algo", "Machine"])
    writer.writerows(out)
        
#    now = datetime.now()
#    current_time = now.strftime("%H-%M-%S")
#    today = date.today()
#    
#    plt.figure()
#    plt.plot(par_t, par_c_a, '-', label='Approximation')
#    plt.plot(par_t, par_c_m, '1', label='LP')
#    plt.plot(par_t, par_c_mb, '2', label='LP with startup cost')
#    plt.plot(par_t_i, par_c_i, ':', label='ILP')
#    plt.plot(par_t_s, par_c_s, '--', label='Schmoys/Tardos')
##    plt.plot(par_t, par_c_a, 'blue')
##    plt.plot(par_t_sa, par_c_sa)
#    plt.legend()
#    plt.xlabel("Maximum makespan (T)")
#    plt.ylabel("Resulting cost")
#    plt.savefig('fig_pareto_'+str(M)+str(N)+str(K)+'_'+str(i)+'_'+str(current_time)+'.png')
#    plt.close()
    # # draw(par_t_m, par_c_m, par_t_a, par_c_a)
    
    # # plt.subplot(131)
    
#    now = datetime.now()
#    current_time = now.strftime("%H-%M-%S")
#    today = date.today()
    
    # plt.plot(par_c_m, par_c_a)
    # plt.plot(par_c_m, par_c_m)
    # plt.plot(par_c_m, par_c_mb)
    # plt.xlabel("Cost of LP solution")
    # plt.ylabel("Cost of approximation")
    # plt.savefig('fig_COST_M'+str(M)+'N'+str(N)+'K'+str(K)+'_'+str(today)+'_'+str(current_time)+'.png')
    # plt.close()
    
    # # plt.subplot(132)
    
    # plt.plot(par_t, par_t_a)
    # plt.plot(par_t, par_t)
    # plt.xlabel("Time of LP solution")
    # plt.ylabel("Time of approximation")
    # plt.savefig('fig_TIME_M'+str(M)+'N'+str(N)+'K'+str(K)+'_'+str(today)+'_'+str(current_time)+'.png')
    # plt.close()
    
    # # plt.subplot(133)
    
    # plt.plot(par_t, par_c_a)
    # plt.plot(par_t, par_c_m)
    # plt.plot(par_t, par_c_mb)
    # plt.plot(par_t_i, par_c_i)
    # plt.plot(par_t_s, par_c_s)
    # # plt.plot([x*3 for x in par_t], par_c_m)
    # plt.xlabel("Time")
    # plt.ylabel("Cost")
    # plt.savefig('fig_PARETO_M'+str(M)+'N'+str(N)+'K'+str(K)+'_'+str(today)+'_'+str(current_time)+'.png')
    # plt.close()
    
    # ax = plt.axes(projection='3d')
    
    # xline = par_t
    # yline = par_t_m
    # ax.plot3D(xline, yline, par_c_a, 'red')
    # ax.plot3D(xline, yline, par_c_m, 'blue')
    # ax.plot3D(xline, yline, par_c_mb, 'green')
    # plt.xlabel("Time budget")
    # plt.ylabel("Time used")
    # plt.zlabel("Cost")
#     ax.plot3D(xline, par_t_i, par_c_i, 'red')
    
    # now = datetime.now()
    # current_time = now.strftime("%H-%M")
    # today = date.today()
    # plt.savefig('fig_M'+str(M)+'N'+str(N)+'K'+str(K)+'_'+str(today)+'_'+str(current_time)+'.png')
    # plt.close()



# Cmax, Tmax = search(M, N, c, p)
# compare(Cmax, Tmax, M, N, c, p)