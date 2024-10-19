from functools import partial
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

solvers = ["NSGA-II", "NSPSO", "MOEA/D-DE", "MHACO", "IHS", "NS-BRKGA"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#8c7e6e", "#738191"]

# Load the data
metrics_snapshots_filename = 'metrics_snapshots.csv'
metrics_snapshots_df = pd.read_csv(metrics_snapshots_filename)

# Set target deviation for metric
target_deviation = 9.748801264859514

# Filter data for the Modified Inverted Generational Distance
igd_snapshots_df = metrics_snapshots_df[metrics_snapshots_df['metric name'] == 'Modified Inverted Generational Distance']

# Find the best (min) Modified Inverted Generational Distance 
best_igd_per_instance = igd_snapshots_df.groupby('instance')['metric value'].min()

# Calculate the ratio of each solver's Modified Inverted Generational Distance to the best Modified Inverted Generational Distance
igd_ratio_snapshots_df = igd_snapshots_df.copy()
igd_ratio_snapshots_df['metric value'] = igd_snapshots_df.apply(lambda row: row['metric value'] / best_igd_per_instance[row['instance']], axis=1)

# Initialize a dictionary to hold the cumulative distribution for each solver
cumulative_distribution = {solver: [] for solver in solvers}

instances = igd_snapshots_df['instance'].unique()
seeds = igd_snapshots_df['seed'].unique()

time_values = []
igd_snapshots_of_solver_instance_seed: dict[str, dict[str, dict[int, pd.DataFrame]]] = {}
for solver in solvers:
    igd_snapshots_of_solver_instance_seed[solver] = {}
    for instance in instances:
        igd_snapshots_of_solver_instance_seed[solver][instance] = {}
        for seed in seeds:
            igd_snapshots_of_solver_instance_seed[solver][instance][seed] = igd_ratio_snapshots_df[(igd_ratio_snapshots_df['solver'] == solver) & (igd_ratio_snapshots_df['instance'] == instance) & (igd_ratio_snapshots_df['seed'] == seed)]
            time_values_temp = igd_snapshots_of_solver_instance_seed[solver][instance][seed]['snapshot time'].unique()
            if len(time_values) == 0 or time_values[0] < time_values_temp[0]:
                time_values = time_values_temp

igd_snapshots_of_solver_instance_seed_time: dict[str, dict[str, dict[int, dict[float, pd.DataFrame]]]] = {}
for solver in solvers:
    igd_snapshots_of_solver_instance_seed_time[solver] = {}
    for instance in instances:
        igd_snapshots_of_solver_instance_seed_time[solver][instance] = {}
        for seed in seeds:
            igd_snapshots_of_solver_instance_seed_time[solver][instance][seed] = {}
            for time in time_values:
                igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time] = igd_snapshots_of_solver_instance_seed[solver][instance][seed][igd_snapshots_of_solver_instance_seed[solver][instance][seed]['snapshot time'] <= time]
                greatest_time = igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time]['snapshot time'].max()
                igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time] = igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time][igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time]['snapshot time'] == greatest_time]

# Calculate the cumulative distribution for each solver
for solver in solvers:
    print(solver)
    for time in time_values:
        num_meeting_target = 0.0
        num_not_meeting_target = 0.0
        for instance in instances:
            for seed in seeds:
                if not igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time].empty > 0:
                    if igd_snapshots_of_solver_instance_seed_time[solver][instance][seed][time]["metric value"].values[0] <= target_deviation:
                        num_meeting_target += 1
                    else:
                        num_not_meeting_target += 1
        if num_meeting_target + num_not_meeting_target == len(instances) * len(seeds):
            cumulative_distribution[solver].append(num_meeting_target / (num_meeting_target + num_not_meeting_target))
        else:
            cumulative_distribution[solver].append(-1.0)

num_any_negatives = 0
while True:
    any_negatives = False
    for solver in solvers:
        if cumulative_distribution[solver][num_any_negatives] < 0:
            any_negatives = True
            break
    if any_negatives:
        num_any_negatives += 1
    else:
        break

# Remove the first num_any_zeros elements from the cumulative distribution
time_values = time_values[num_any_negatives:]
for solver in solvers:
    cumulative_distribution[solver] = cumulative_distribution[solver][num_any_negatives:]

# Convert the cumulative distribution to a DataFrame for easier plotting
cumulative_distribution_df = pd.DataFrame(cumulative_distribution, index=time_values)

cumulative_distribution_df.to_csv('igd_snapshots.csv')

# Plot the performance profile
plt.figure()
plt.xlabel('Time')
plt.ylabel('Fraction of Executions')
plt.grid(alpha=0.5, color='gray', linestyle='dashed', linewidth=0.5, which='both')
for i in range(len(solvers)):
    plt.plot(cumulative_distribution_df.index, cumulative_distribution_df[solvers[i]], label=solvers[i], marker = (i + 3, 2, 0), color = colors[i], alpha = 0.80)
plt.xscale("log")
plt.yscale("function", functions=(partial(np.power, 10.0), np.log10))
plt.legend(loc='best')
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
# Save the plot
plt.savefig('igd_snapshots.png')
plt.close()