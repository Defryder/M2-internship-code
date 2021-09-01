# M2-internship-code
The implementation for my M2 internship on modeling scheduling policies for serverless computing.


The python scripts contain implementations of different algorithms presented in my report.<br>
LP.py      -> Two implementations of the LP, with and without taking into account the environments<br>
ILP.py     -> Two implementations of the ILP corresponding to the LP, with and without taking into account the environments<br>
approx.py  -> The approximation algorithm described in the report, the bipartite graph creation and the matching<br>
compare.py -> The generation of pareto fronts and other analysis for the algorithm<br>


To use:

1.  Define variables :<br>
                     M, N, K (nb of machines, tasks, and environments), <br>
                     p, c (processing time and cost of tasks on machines), <br>
                     b, d (boot time and cost of environments on machines), <br>
                     env (link between task and environment), <br>
                     Cmax, Tmax (Target cost and makespan)<br>

2.  Run function LP (or LPS if no environment) from LP.py.<br>
    This outputs an array (M, N) of float<br>

3.  Run function generate_bipartite (or generate_bipartite_s if no environment) from approx.py with the output of step 2<br>
    This outputs an array (M, N) of boolean, representing an assignment of tasks to machines<br>

