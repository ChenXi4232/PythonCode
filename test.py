import matplotlib.pyplot as plt

# Data for A* algorithm
expanded_nodes_astar = [
    [8, 6, 18, 21, 34, 28, 78, 58, 72, 486, 2083],
    [11, 6, 19, 31, 40, 65, 83, 65, 88, 487, 10001]
]

# Time taken for A* algorithm (milliseconds)
time_taken_astar = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 21],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 4, 68]
]

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
labels = ['Test {}'.format(i) for i in range(11)]

for i in range(2):
    axes[i].plot(labels, expanded_nodes_astar[i],
                 marker='o', label='Expanded Nodes')
    axes[i].set_ylabel('Expanded Nodes')
    axes[i].twinx().plot(labels, time_taken_astar[i], marker='s',
                         color='orange', label='Time Taken (ms)')
    axes[i].set_title('A* Algorithm (Test Set {})'.format(i))
    axes[i].set_xlabel('Test Cases')

fig.tight_layout()
plt.show()
