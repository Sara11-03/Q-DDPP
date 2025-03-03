# Quantum and Simulated Annealers on the Drone Delivery Packing Problem

This repository implements benchmark generation for the combinatorial optimization problem known as the Drone Delivery Packing Problem. It compares the performance of the Quantum Annealing algorithm against its classical variant, the Simulated
Annealing algorithm, and other commercial branch-and-bound tools (particularly Gurobi solver).

**Paper:** [Testing Quantum and Simulated Annealers on the Drone Delivery Packing Problem](https://arxiv.org/abs/2406.08430)

## QPU access and fundings

The access to the quantum hardware was possible thanks to the ISCRA-C project by the High-Performance Computing Center CINECA. This research is funded by the Quantum Computing Research Laboratories of Leonardo S.p.A. LABS located in Genova, Italy.

## Scripts

- `benchmark_generation_for_testing_DDPP_with_QA&SA.py`: Generates benchmark datasets for the problem. Sampled problem sets. For each delivery instance:
  - Available drones are fixed;
  - Deliveries are randomly picked in a chosen set;
  - Battery budget of drones is randomly picked in a chosen set; 
  - A distribution is randomly chosen between Uniform and Gaussian, and according to this, lists of delivery energy costs and delivery time intervals are generated;
  - Time windows are generated ranging from 8 a.m. to 8 p.m. and are at least one and at most three hours long.
- `testing_quantum_and_simulated_annealers_on_DDPP_code.py`: Implements the aforementioned solving methods. It outputs a .csv file that summarizes experimental outcomes, including metrics for evaluating the QPU capabilities, for example:
   - Time to solution, hardware-specific timings, like QPU access time;
   - Average of the objective function value on multiple samples, and "best" (smallest) value among all samples;
   - Average on multiple samples of the number of prescribed drones for completing the desired deliveries, and "best" (smallest) number of drones (computed on feasible solutions);
   - Average on multiple samples feasibility of solutions (simultaneous satisfaction of constraints), and individual percentages of satisfaction of single constraints;
   - Logical and physical (after the Embedding in the Machine) variables, for both the QUBO models presented in the paper. The first model results from standard procedures, while the second proposed model
     enhances the variable count, e.g. the performance. 

