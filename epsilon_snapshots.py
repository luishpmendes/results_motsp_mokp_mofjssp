import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

solvers = ["NSGA-II", "NSPSO", "MOEA/D-DE", "MHACO", "IHS", "NS-BRKGA", "NS-BRKGA + IPR"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#8c7e6e", "#738191"]

# Load the data
metrics_snapshots_filename = 'metrics_snapshots.csv'
metrics_snapshots_df = pd.read_csv(metrics_snapshots_filename)

# Set target values for each metric
target_value = 0.50

# Filter data for the Hypervolume Ratio
epsilon_snapshots_df = metrics_snapshots_df[metrics_snapshots_df['metric name'] == 'Multiplicative Epsilon Indicator']

# Initialize a dictionary to hold the cumulative distribution for each solver
cumulative_distribution = {solver: [] for solver in solvers}

# Get all unique time values from epsilon_snapshots_df
time_values = epsilon_snapshots_df['snapshot time'].unique()
time_values.sort()

num_instances = len(epsilon_snapshots_df['instance'].unique())
num_seeds = len(epsilon_snapshots_df['seed'].unique())

# Calculate the cumulative distribution for each solver
for solver in solvers:
    print(solver)
    epsilon_snapshots_solver_df = epsilon_snapshots_df[epsilon_snapshots_df['solver'] == solver]
    for time in time_values:
        # For each time, calculate the proportion of executions where the solver's performance is at least the target value
        epsilon_snapshots_solver_time_df = epsilon_snapshots_solver_df[epsilon_snapshots_solver_df['snapshot time'] <= time]
        percentage_meeting_target = 100 * np.mean(epsilon_snapshots_solver_time_df['metric value'] >= target_value) if len(epsilon_snapshots_solver_time_df) > num_instances * num_seeds else 0
        cumulative_distribution[solver].append(percentage_meeting_target)

num_any_zeros = 0
while True:
    any_zeros = False
    for solver in solvers:
        if cumulative_distribution[solver][num_any_zeros] == 0:
            any_zeros = True
            break
    if any_zeros:
        num_any_zeros += 1
    else:
        break

# Remove the first num_any_zeros elements from the cumulative distribution
time_values = time_values[num_any_zeros:]
for solver in solvers:
    cumulative_distribution[solver] = cumulative_distribution[solver][num_any_zeros:]

# Convert the cumulative distribution to a DataFrame for easier plotting
cumulative_distribution_df = pd.DataFrame(cumulative_distribution, index=time_values)

cumulative_distribution_df.to_csv('epsilon_snapshots.csv')

# Plot the performance profile
plt.figure()
for i in range(len(solvers)):
    plt.plot(cumulative_distribution_df.index, cumulative_distribution_df[solvers[i]], label=solvers[i], marker = (i + 3, 2, 0), color = colors[i], alpha = 0.80, markevery = 0.02)
plt.title('Time-to-Target for Multiplicative Epsilon Indicator')
plt.xlabel('Time')
plt.ylabel('Percentage of Executions')
plt.xscale("log")
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Save the plot
plt.savefig('epsilon_snapshots.png')
