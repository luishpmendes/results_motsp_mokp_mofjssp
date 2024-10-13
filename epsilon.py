from functools import partial
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
metrics_filename = 'metrics.csv'
metrics_df = pd.read_csv(metrics_filename)

# Filter the data for Multiplicative Epsilon Indicator
epsilon_df = metrics_df[metrics_df['metric name'] == 'Multiplicative Epsilon Indicator']

# Find the best (max) Multiplicative Epsilon Indicator 
best_epsilon_per_instance = epsilon_df.groupby('instance')['metric value'].max()

# Calculate the ratio of each solver's Multiplicative Epsilon Indicator to the best Multiplicative Epsilon Indicator
epsilon_ratio = epsilon_df.copy()
epsilon_ratio['metric value'] = epsilon_df.apply(lambda row: best_epsilon_per_instance[row['instance']] / row['metric value'], axis=1)

# Initialize a dictionary to hold the cumulative distribution for each solver
cumulative_distribution = {solver: [] for solver in epsilon_ratio['solver'].unique()}

# Get all unique rho values from epsilon_data
rho_values = epsilon_ratio['metric value'].unique()
rho_values.sort()

# Calculate the cumulative distribution for each solver
for solver in cumulative_distribution.keys():
    for rho in rho_values:
        # For each rho, calculate the proportion of executions where the solver's performance is within rho times the best performance
        percentage_within_rho = np.mean(epsilon_ratio[epsilon_ratio['solver'] == solver]['metric value'] <= rho)
        cumulative_distribution[solver].append(percentage_within_rho)

# Convert the cumulative distribution to a DataFrame for easier plotting
cumulative_distribution_df = pd.DataFrame(cumulative_distribution, index=rho_values)

cumulative_distribution_df.to_csv('epsilon.csv')

solvers = ["NSGA-II", "NSPSO", "MOEA/D-DE", "MHACO", "IHS", "NS-BRKGA"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#8c7e6e", "#738191"]

# Plot the performance profile
plt.figure()
plt.xlabel('Deviation from best Multiplicative Epsilon Indicator')
plt.ylabel('Fraction of Executions')
plt.grid(alpha=0.5, color='gray', linestyle='dashed', linewidth=0.5, which='both')
for i in range(len(solvers)):
    plt.plot(cumulative_distribution_df.index, cumulative_distribution_df[solvers[i]], label=solvers[i], marker = (i + 3, 2, 0), color = colors[i], alpha = 0.80, markevery = 0.02)
plt.xscale("log")
plt.yscale("function", functions=(partial(np.power, 10.0), np.log10))
plt.legend(loc='best')
plt.gca().xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
# Save the plot
plt.savefig('epsilon.png')
plt.close()
