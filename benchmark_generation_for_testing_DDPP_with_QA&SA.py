import pandas as pd
import numpy as np
import random

# Set a seed for reproducibility
random.seed(123)
np.random.seed(123)

# Define the number of instances to generate
num_instances = 12

# Define options for the number of deliveries, battery budgets, and distributions
num_deliveries_options = [10, 12]
battery_budget_options = [50, 70, 100]
distribution_options = ['gaussian', 'uniform']

# Initialize a list to collect instance data
benchmark_data = []

# Generate random instances
for instance_id in range(1, num_instances + 1):
    # Randomly select parameters for the current instance
    num_deliveries = random.choice(num_deliveries_options)
    
    # alternatively, if we generate num_instances = len(num_deliveries_options) and we want to test
    # on an increasing and precise number of deliveries, we can pick up all num_deliveries_options one at a time:
    # num_deliveries = num_deliveries_options[instance_id - 1]
    
    battery_budget = random.choice(battery_budget_options)
    distribution = random.choice(distribution_options)
    
    # Generate costs based on the selected distribution
    if distribution == 'gaussian':
        # Generate costs from a Gaussian distribution and clip to [0, battery_budget]
        costs = np.random.normal(loc=battery_budget / 2, scale=10, size=num_deliveries)
        costs = np.clip(costs, 0, battery_budget)
    else:  # 'uniform' distribution
        # Generate costs from a uniform distribution between 0 and battery_budget
        costs = np.random.uniform(0, battery_budget, size=num_deliveries)
    
    # Round costs to one decimal place
    costs = np.round(costs, 1).tolist()
    
    # Generate random time intervals for each delivery
    time_intervals = []
    for _ in range(num_deliveries):
        start_time = random.randint(8, 19)  # Start time between 8 AM and 7 PM
        end_time = random.randint(start_time + 1, min(start_time + 3, 20))  # End time between start_time+1 and 8 PM
        time_intervals.append([start_time, end_time])
    
    # Append the generated data as a dictionary to the benchmark_data list
    benchmark_data.append({
        'Instance': instance_id,
        '# deliveries': num_deliveries,
        'battery budget': battery_budget,
        'distribution': distribution,
        'costs': costs,
        'time intervals': time_intervals
    })

# Create a DataFrame from the collected benchmark data
benchmark_df = pd.DataFrame(benchmark_data)

# Generate a filename based on the number of deliveries options
filename = f"benchmark_{'_'.join(map(str, num_deliveries_options))}.csv"

# Save the DataFrame to a CSV file without the index
benchmark_df.to_csv(filename, index=False)

print(f"Benchmark saved as {filename}")



