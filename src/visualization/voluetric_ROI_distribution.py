import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the data from the txt file
data = np.loadtxt(r'D:\User_Behavior\volumetric_data\volumetric_ROI_power\News_interviewing_user5.txt')

# Set the style of the plot
plt.style.use('ggplot')

# Create a histogram of the data
n_bins = 50  # You can adjust this value to change the number of bins

# Create a color map with a gradient from red to green
color_map = cm.get_cmap('RdYlGn')
new_color_map = color_map(np.linspace(0.1, 1.0, 256))  # Adjust the color map to be more green
new_color_map[128:129, :] = [0.5, 0.5, 0.5, 1.0]  # Replace the middle color with dark grey

# Create the histogram and color the bars with the color map
counts, bins, patches = plt.hist(data, bins=n_bins, edgecolor='white')
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = (bin_centers - min(bin_centers)) / (max(bin_centers) - min(bin_centers))
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', new_color_map[int(c*255)])

# Add labels and titles
plt.xlabel('${F_a}$', fontsize=15,color='black',alpha=1)
plt.ylabel('#Cubes', fontsize=15,color='black',alpha=1)
#plt.title('Distribution of Volumetric Cube Importance', fontsize=18)
plt.xlim(0,100)
# Add a grid to the plot
plt.grid(axis='y', alpha=1)

# Add text showing the mean and standard deviation of the data
mean = np.mean(data)
std = np.std(data)
text = f"Mean: {mean:.2f}\nStd. dev.: {std:.2f}"
plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', alpha=0.5),fontsize=17)

# Adjust the spacing between subplots
plt.subplots_adjust(left=0.15, bottom=0.15)

plt.xlabel('ROI Level', fontsize=15, color='black', alpha=1)
plt.ylabel('#Cubes', fontsize=15, color='black', alpha=1)

# Show the plot
plt.show()