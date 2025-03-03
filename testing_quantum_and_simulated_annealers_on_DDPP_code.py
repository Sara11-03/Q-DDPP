import numpy as np
from pyqubo import Array
import math 
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
import time
import csv
qpu = DWaveSampler('Advantage_system5.4') 

def find_intersection(N, intervals, j):
    """
    Identify intersections between a given interval and all other intervals.

    Parameters:
    N (int): Total number of intervals.
    intervals (list of tuples): List of intervals represented as (start, end).
    j (int): Index of the interval to check intersections for.

    Returns:
    list: A binary list where 1 indicates an intersection with another interval.
    """
    N_t = [0] * N  # Initialize list with zeros
    interval_j = intervals[j]  # Get the target interval

    for i in range(N):
        sample = intervals[i]  # Current interval to compare
        if sample[0] < interval_j[1] and interval_j[0] < sample[1]:  # Check overlap
            N_t[i] = 1
    
    return N_t 


def check_battery(en_bud, N, m, sol, c):
    """
    Verify if all drones stay within the battery budget.

    Parameters:
    en_bud (float): Battery budget.
    N (int): Number of deliveries.
    m (int): Number of drones.
    sol (list): Solution list mapping deliveries to drones.
    c (list): Energy cost per delivery.

    Returns:
    int: 1 if battery constraint is satisfied for all drones, otherwise 0.
    """
    for i in range(m):
        total_energy = sum(c[j] * sol[j + i * N] for j in range(N))
        if total_energy > en_bud:
            print(f"Drone {i+1} ran out of battery.")
            return 0
    return 1 


def check_each_delivery(N, m, sol):
    """
    Ensure each delivery is assigned to exactly one drone.

    Parameters:
    N (int): Number of deliveries.
    m (int): Number of drones.
    sol (list): Solution list mapping deliveries to drones.

    Returns:
    int: 1 if constraint is met, 0 if a delivery is unassigned or assigned to multiple drones.
    """
    for j in range(N):
        assigned_drones = sum(sol[j + i * N] for i in range(m))
        if assigned_drones == 0:
            print(f"Delivery {j+1} is not completed.")
            return 0
        if assigned_drones > 1:
            print(f"Delivery {j+1} is assigned to multiple drones.")
            return 0
    return 1 


def check_time_conflicting(N, m, q, inters):
    """
    Verify that no time conflicts exist in the delivery schedule.

    Parameters:
    N (int): Number of deliveries.
    m (int): Number of drones.
    q (list): Schedule list mapping deliveries to drones.
    inters (list of lists): Conflict matrix indicating which deliveries overlap.

    Returns:
    int: 1 if no time conflicts exist, otherwise 0.
    """
    for i in range(m):
        for t in range(N):
            for k in range(t + 1, N):  # Avoid redundant checks
                if inters[t][k] != 0 and q[i * N + k] + q[i * N + t] > 1:
                    print(f"Deliveries {t+1} and {k+1} conflict.")
                    return 0
    return 1

def calculate_conflicts(N, I):
    """
    Calculate the number of conflicts in deliveries and determine if there's at least one conflict.

    Parameters:
    N (int): Total number of deliveries.
    I (list): Intersection matrix indicating conflicts between deliveries.

    Returns:
    tuple: A tuple containing:
        - num_conf (int): Total number of conflicts.
        - atleastoneconf (bool): True if there's at least one conflict, False otherwise.
        - inters (list): List of conflict vectors for each delivery.
    """
    num_conf = 0
    inters = []

    for t in range(N):
        N_t = find_intersection(N, I, t)  # N_t is the vector of conflicts for delivery t
        inters.append(N_t)
        num_conf += sum(1 for i in range(t + 1, N) if N_t[i] != 0)

    atleastoneconf = num_conf > 0
    return atleastoneconf, inters

def construct_hamiltonian(N, m, en_bud, c, inters, P, there_are_conflicts):
    """
    Constructs the Hamiltonian for the QUBO model considering delivery assignments, energy constraints, and time conflicts.

    Parameters:
    N (int): Number of deliveries.
    m (int): Number of drones.
    en_bud (float): Energy budget.
    c (list): Energy cost per delivery.
    inters (list of lists): Conflict matrix indicating time conflicts between deliveries.
    P (dict): Penalty weights for different constraints.
    there_are_conflicts (bool): Whether there are time conflicts among deliveries.

    Returns:
    BinaryQuadraticModel: The constructed Hamiltonian as a QUBO model.
    """
    # Calculate the number of logical qubits for both the improved and standard QUBO models
    num_qub = m * N + m * math.ceil(math.log2(en_bud)) + m * there_are_conflicts

    num_qub_standard = 2*m + num_qub + math.ceil(math.log2(N))*m

    # Initialize binary variables
    q = Array.create('q', shape=num_qub, vartype='BINARY')
    
    # Initialize Hamiltonian
    H = 0
    
    # Objective function: Assigning deliveries to drones efficiently
    for i in range(m):
        c_par = sum(q[j + i * N] for j in range(N))
        H += c_par * (N - c_par)
    
    # Constraint 1: Each delivery must be assigned to exactly one drone
    for j in range(N):
        D = sum(q[j + i * N] for i in range(m))
        H += P['all_deliveries'] * (D - 1) ** 2
    
    # Constraint 2: Energy budget must not be exceeded for each drone
    for i in range(m):
        A = sum(c[j] * q[j + i * N] for j in range(N))
        B = sum(q[N * m + k + i * math.ceil(math.log2(en_bud))] * 2**k for k in range(math.ceil(math.log2(en_bud))))
        H += P['energy_budget'] * (A + B - en_bud) ** 2
    
    # Constraint 3: Avoid time-conflicting deliveries
    if there_are_conflicts:
        for i in range(m):
            for t in range(N):
                for k in range(t + 1, N):  # Ensuring k > t
                    if inters[t][k] != 0:
                        H += P['time_consistency'] * (q[i * N + k] + q[i * N + t] + q[m * N + m * math.ceil(math.log2(en_bud)) + i] - 1) ** 2
    
    return H, num_qub, num_qub_standard

def count_active_drones(solution, num_drones, num_deliveries):
    """
    Count the number of drones that have been assigned at least one delivery.

    Parameters:
    - solution (list): A flattened list representing the assignment matrix where each element indicates
                       whether a drone is assigned to a delivery (1) or not (0).
    - num_drones (int): Total number of drones.
    - num_deliveries (int): Total number of deliveries.

    Returns:
    - int: The number of drones with at least one assigned delivery.
    """
    active_drones = 0

    for i in range(num_drones):
        # Extract the segment of the solution corresponding to the current drone
        drone_assignments = solution[i * num_deliveries : (i + 1) * num_deliveries]

        # Check if the drone has any assigned deliveries
        if any(drone_assignments):
            active_drones += 1

    return active_drones

def compute_objective_value(solution, num_drones, num_deliveries):
    """
    Compute the value of the improved objective function for the given solution.

    Parameters:
    - solution (list): A flattened list representing the assignment matrix where each element indicates
                       whether a drone is assigned to a delivery (1) or not (0).
    - num_drones (int): Total number of drones.
    - num_deliveries (int): Total number of deliveries.

    Returns:
    - int: The computed value of the objective function.
    """
    objective_value = 0

    for i in range(num_drones):
        # Extract the segment of the solution corresponding to the current drone
        drone_assignments = solution[i * num_deliveries : (i + 1) * num_deliveries]

        # Count the number of deliveries assigned to the current drone
        delivery_count = sum(drone_assignments)

        # Update the objective function value
        objective_value += delivery_count * (num_deliveries - delivery_count)

    return objective_value


num_iterations = 10 # Iterations for the statistics
m = 4 # Number of drones available in the depot 
loaded_data = np.load('benchmark_10_12.npy', allow_pickle=True) # Loaded Data of the Benchmark
benchmark_df = pd.DataFrame(loaded_data, columns=['Instance', '# deliveries', 'battery budget', 'distribution', 'costs', 'time intervals'])
num_instances = len(benchmark_df) # Number of Instances comprising the Benchmark
instances = list(range(num_instances))
k_par = 120 # Extra weight given to the most delicate constraint in the QUBO formulation 

checks_multiple = [0] * num_instances
feas_sol_with_less_drones = [0] * num_instances
individual_checks_of_optimal_sol = [0] * num_instances
avg_qpu_time = [0] * num_instances
H_x_QA_multiple = [0] * num_instances
avg_H_x_QA = [0] * num_instances 
avg_num_drones_involved = [0] * num_instances
best_sol = [0] * num_instances
best_drones = [0] * num_instances
avg_feasibility = [0] * num_instances
avg_times = [0] * num_instances 
avg_num_qubits = [0] * num_instances
num_qub = [0] * num_instances
num_qub_standard = [0] * num_instances

for n in range (num_instances): # Iterate over the instances of the benchmark
    
    qpu_time = [0] * num_iterations
    elapsed_times_QA = [0] * num_iterations
    phys_qub = [0] * num_iterations
    feasible = [0] * num_iterations
    check_b = [0] * num_iterations
    check_tc = [0] * num_iterations
    check_ad = [0] * num_iterations
    involved_drones = [0] * num_iterations
    H_x_QA = [0] * num_iterations
    
    N = int(loaded_data[n][1]) # Number of deliveries comprising instance n
    en_bud = int(loaded_data[n][2]) # Energy budget comprising instance n
    c = loaded_data[n][4] # list of costs associated with deliveries comprising instance n
    I = loaded_data[n][5] # Time intervals comprising instance n
    there_are_conflicts, inters = calculate_conflicts(N, I) 
    # there_are_conflicts = counter for having at least one conflict (used in the variables count)
    # inters = list of conflict vectors for each delivery
    
    P = {
    "all_deliveries": k_par*m*N**2, 
        
    "energy_budget": m*N**2, 
    
    "time_consistency": m*N**2,
    } # dictionary of weights to be assigned to the relative constraints in the QUBO form 

    H, num_qub[n], num_qub_standard[n] = construct_hamiltonian(N, m, en_bud, c, inters, P, there_are_conflicts)
    # QUBO Hamiltonian encoding the solution to the optimization problem, number of physical qubits
    # in the improved (proposed) QUBO model, and in the standard QUBO model, respectively
    
    for p in range(num_iterations):
    # Iterate to make statistics: find num_iterations times a solution with the data of instance n

        # Compile the problem Hamiltonian to get a model
        model = H.compile()
        # Transform the model in a binary quadratic model
        bqm = model.to_bqm() 
        # Use EmbeddingComposite to embed our bqm in the chosen qpu
        sampler = EmbeddingComposite(qpu) 
        
        # Record the start time to measure total execution duration
        start_time = time.time()
        
        # Sample from the Binary Quadratic Model using the quantum sampler
        sampleset = sampler.sample(bqm, return_embedding=True, num_reads=1000)
        
        # Extract the embedding information from the sample set: a dictionary where each key corresponds 
        # to a logical variable, and the associated value is a list of physical qubits (chains) on the QPU
        # representing that logical variable. 
        embedding = sampleset.info["embedding_context"]["embedding"]
        
        # Retrieve the QPU access time from the sample set's timing information
        qpu_access_time = sampleset.info["timing"]["qpu_access_time"]
        
        # Store the QPU access time for averaging over iterations
        qpu_time[p] = qpu_access_time
        
        # Print the detailed timing information provided by the sampler
        print(f"Timing information: {sampleset.info['timing']}")
        
        # Calculate the total number of physical qubits used in the embedding
        total_physical_qubits = sum(len(chain) for chain in embedding.values())
        print(f"Total number of physical qubits used: {total_physical_qubits}")
        
        # Calculate and print the total execution time
        total_execution_time = time.time() - start_time
        print(f"Total execution time: {total_execution_time:.2f} seconds")
        
        # Store the physical qubits involved for optimizing in iteration p
        phys_qub[p] = total_physical_qubits 
        
        # Decode samples and retrieve the one with the best energy value
        decoded_samples = model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        print(f"Best sample: {best_sample}")
        
        # Record the stop time, measure total execution duration for iteration p, and store for averaging
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times_QA[p] = elapsed_time # Time for QA imrpoved
        
        # Retrieve the solution matrix x_ij = 1 if drone i is assigned to delivery j
        sol=[]
        for i in range(N*m):
            key='q['+str(i)+']';
            sol.append(best_sample.sample[key])
        print("Optimal configuration as a matrix:")
        print(f"{np.array(sol).reshape(m, N)}")
  
        # Initialize a list to store the results of constraint checks
        # Each element corresponds to a specific constraint:
        # check_results[0] -> Battery constraint
        # check_results[1] -> Time-conflict constraint
        # check_results[2] -> Delivery assignment constraint
        check_results = [
            check_battery(en_bud, N, m, sol, c),
            check_time_conflicting(N, m, sol, inters),
            check_each_delivery(N, m, sol)
        ] 
        
        # Store individual constraint check trackers, to compute individual
        # percentages of constraints satisfaction
        check_b[p] = check_results[0]  # Battery constraint result
        check_tc[p] = check_results[1]  # Time-conflict constraint result
        check_ad[p] = check_results[2]  # Delivery assignment constraint result
        
        # Determine if all constraints are satisfied
        # A solution is considered FEASIBLE if all constraints are satisfied.
        feasible[p] = int(all(check_results))  # Convert boolean result to integer (1 for True, 0 for False)     
        
        # Store the number of drones involved according to schedule found in iteration p
        involved_drones[p] = count_active_drones(sol, m, N)
        
        # Evaluate the objective function on the solution found in attempt p, and store
        H_x_QA[p] = compute_objective_value(sol, m, N) 
    
    # Check if all elements in 'feasible' are 0
    if all(not f for f in feasible):
        
        print(f"No feasible solution for instance {n} was found in {num_iterations} attempts.")
        
    else:
        # Calculate average elapsed time, excluding infeasible attempts
        avg_times[n] = sum(elapsed_times_QA[p] for p in range(num_iterations) if feasible[p]) / sum(feasible)
        
        # Calculate average QPU access time, excluding infeasible attempts
        avg_qpu_time[n] = sum(qpu_time[p] for p in range(num_iterations) if feasible[p]) / sum(feasible)
        
        # Calculate average number of drones involved, excluding infeasible attempts
        avg_num_drones_involved[n] = sum(involved_drones[p] for p in range(num_iterations) if feasible[p]) / sum(feasible)
        
        # Calculate average objective function value, excluding infeasible attempts
        avg_H_x_QA[n] = sum(H_x_QA[p] for p in range(num_iterations) if feasible[p]) / sum(feasible)
        
        # Identify the best solution among feasible attempts
        best_sol[n] = min((H_x_QA[p] for p in range(num_iterations) if feasible[p]), default=None)
        
        # Identify the best number of drones among feasible attempts
        feas_sol_with_less_drones[n] = min((involved_drones[p] for p in range(num_iterations) if feasible[p]), default=None)
        
    # Identify the best number of drones among all attempts, to check individual constraints
    # and see which are satisfied and how much on average
    best_drones[n] = min((involved_drones[p] for p in range(num_iterations)), default=None)
    
    # Find the index of the solution involving the least number of drones among all the attempts 
    optimal_sol_among_all = next((p for p in range(num_iterations) if involved_drones[p] == best_drones[n]), None)
    
    # Store the feasibility individual checks for the best solution attempt
    # Note that it can happen that the optimal_sol_among_all is also feasible,
    # and corresponds to feas_sol_with_less_drones, then we will have 1,1,1.
    individual_checks_of_optimal_sol[n] = [
        check_b[optimal_sol_among_all],
        check_tc[optimal_sol_among_all],
        check_ad[optimal_sol_among_all]
        ]
    
    # Calculate the average feasibility across num_iterations attempts for instance n
    avg_feasibility[n] = sum(feasible) / num_iterations
    
    # Calculate the average satisfaction of each constraint across all attempts for instance n
    checks_multiple[n] = [
        sum(check_b) / num_iterations,
        sum(check_tc) / num_iterations,
        sum(check_ad) / num_iterations
    ]
    
    # Calculate the average number of physical qubits used across all num_iterations optimizations of instance n
    avg_num_qubits[n] = sum(phys_qub) / num_iterations


# Save data as a CSV file named DDPP_results
csv_file = 'results.csv'

# Define the header for the CSV
header = [
    'Instance', 'Log Qubits (improved)', 'Log Qubits (standard)', 'Avg_Time', 'Avg_QPU_Time', 'Avg_Num_Drones_Involved', 'Avg_H_x_QA',
    'Best_Sol', 'Feas_Sol_With_Less_Drones', 'Best_Drones',
    'Check_B', 'Check_TC', 'Check_AD',
    'Avg_Feasibility', 'Avg_Check_B', 'Avg_Check_TC', 'Avg_Check_AD',
    'Avg Physical Qubits'
]

# Write data to CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    for i in range(n):
        row = [
            instances[i], num_qub[i], num_qub_standard[i], avg_times[i], avg_qpu_time[i], avg_num_drones_involved[i], avg_H_x_QA[i],
            best_sol[i], feas_sol_with_less_drones[i], best_drones[i],
            *individual_checks_of_optimal_sol[i],
            avg_feasibility[i], *checks_multiple[i],
            avg_num_qubits[i]
        ]
        writer.writerow(row)

print(f"Data has been successfully saved to {csv_file}.")
        





