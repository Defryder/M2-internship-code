# M2-internship-code
The implementation for my M2 internship on modeling scheduling policies for serverless computing.


The python scripts contain implementations of different algorithms presented in my report.
LP.py      -> Two implementations of the LP, with and without taking into account the environments
ILP.py     -> Two implementations of the ILP corresponding to the LP, with and without taking into account the environments
approx.py  -> The approximation algorithm described in the report, the bipartite graph creation and the matching
compare.py -> The generation of pareto fronts and other analysis for the algorithm


To use:

1.  Define variables M, N, K (nb of machines, tasks, and environments), 
                     p, c (processing time and cost of tasks on machines), 
                     b, d (boot time and cost of environments on machines), 
                     env (link between task and environment),
                     Cmax, Tmax (Target cost and makespan)

2.  Run function LP (or LPS if no environment) from LP.py.
    This outputs an array (M, N) of float

3.  Run function generate_bipartite (or generate_bipartite_s if no environment) from approx.py with the output of step 2
    This outputs an array (M, N) of boolean, representing an assignment of tasks to machines

