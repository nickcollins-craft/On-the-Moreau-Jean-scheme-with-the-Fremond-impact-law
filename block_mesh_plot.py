"""
This is a code that takes in two previously saved images of the block mesh and the deformed block mesh and plots them side by side, and adds a manually defined colour bar to the second. It then saves the plot as a pdf file.
"""

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from path_file import *


# Plot time!
width_in_inches = (8.27-2*1.5/2.54)
# In this case the width ratios are manually enforced to produce a plot where both meshes have approximately the same height
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width_in_inches, 0.5*0.65*width_in_inches), width_ratios=[1, 1.15])

# TeX the written elements so that it looks good
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the images
mesh_image = mpimg.imread(data_folder + 'sliding_block_mesh.png')
deformed_mesh_image = mpimg.imread(data_folder + 'deformed_sliding_block_mesh.png')

# Plot the raw mesh
ax0 = plt.subplot(121)
plt.imshow(mesh_image)
# Get rid of the axis
ax0.axis('off')
# Add the subplot label
text_x_pos = -0.05
text_y_pos = 0.99
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)

# Plot the deformed mesh
ax1 = plt.subplot(122)
plt.imshow(deformed_mesh_image)
# Do the colour bar, specifying colour scheme and range
cmap = matplotlib.cm.plasma
norm = matplotlib.colors.Normalize(vmin=0.69, vmax=0.72)
# Get rid of the axis
ax1.axis('off')
# Place the colour bar and set its width
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
# Add the colour bar and label it
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', label='Total displacement (mm)')
# Add the subplot label
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

# Make sure the plot is tight
plt.tight_layout()

# Save as a pdf file
save_file_name = figure_folder + 'block_mesh_version_2.pdf'
plt.savefig(save_file_name, dpi=320)

# Show the plot
plt.show()
