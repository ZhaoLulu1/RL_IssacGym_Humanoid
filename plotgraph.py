import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Define the file name and color
file = 'reward_log'

# Define a more vibrant Morandi color palette
vibrant_morandi_palette = ['#FF8374', '#FF5555', '#E83F6F', '#FF3A71', '#FFAAA5']

# Set the vibrant Morandi color palette
sns.set_palette(vibrant_morandi_palette)

# Set a more visually pleasing style
sns.set_style("whitegrid")
sns.set_context("notebook")

# Read the data from the Excel file and skip the first row
df = pd.read_csv(file + '.csv', usecols=[0], skiprows=1)
transformed_data = df  # Transformation equation can be applied here if needed

# Find the last index where a datapoint is below 0
last_negative_index = len(df)
for i in range(len(df) - 1, -1, -1):
    value = df.iloc[i, 0]
    if value < 0:
        last_negative_index = i
        break
last_negative_index = int(last_negative_index)

# Get the data points after the last negative point
data_after_last_negative = transformed_data.iloc[last_negative_index + 1:]

# Smooth the data using LOWESS
smooth_data = lowess(data_after_last_negative.values.flatten(), range(len(data_after_last_negative)),
                     frac=0.02)

# Create a new figure with better proportions
plt.figure(figsize=(10, 6))

# Plot the smoothed data with a solid line and add markers for data points
plt.plot(range(len(smooth_data[:, 1])), smooth_data[:, 1], label=file, linewidth=1.5,  # Reduce linewidth
         alpha=0.8, marker='o', markersize=4)

# Add labels and legend with improved font size and style
plt.xlabel('Iterations', fontsize=14, fontweight='bold')
plt.ylabel('Reward', fontsize=14, fontweight='bold')
plt.title('Mean Rewards', fontsize=16, fontweight='bold')
legend = plt.legend(fontsize=12, title=file, title_fontsize=14)  # Adjust legend font size and title font size
legend.get_title().set_fontweight('bold')  # Set legend title font weight

# Customize tick labels font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Remove the top and right spines
sns.despine(offset=10, trim=True)

# Add a grid with slightly reduced alpha for better readability
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
