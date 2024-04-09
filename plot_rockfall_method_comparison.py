"""
This is a script to plot the results of the rockfall impact on a wall model. It requires the specification of the directory from which to load the results, as well as the method.
It then loads the results and plots the comparison of the dissipation using both the Moreau–Jean and Moreau–Jean–Frémond method. The results are saved as a PDF file in the specified directory.
"""

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import h5py
from path_file import *


# Get the load folder of the Moreau–Jean method
load_folder_MJ = data_folder + 'Wall/MJ/'
# Load the data
data_file_MJ = h5py.File(load_folder_MJ + 'rock_protection_wall_NewtonNSL.hdf5', 'r')
# The data is stored in the 'data' group, so we wil get the information from there
energy_work = data_file_MJ['data']['energy_work'][:, :]
# Close the data file
data_file_MJ.close()
# Get the time, which is stored in the first column of energy_work. It is strictly non-decreasing, but may not be strictly increasing
time_MJ = energy_work[:, 0]
# Get the unique time values
unique_time_MJ = np.unique(time_MJ)
# Get the dissipation of the MJ method
normal_contact_work_from_energy_MJ = np.cumsum(energy_work[:, 3])
tangent_contact_work_from_energy_MJ = np.cumsum(energy_work[:, 4])
dissipatated_contact_and_friction_energy_MJ = normal_contact_work_from_energy_MJ + tangent_contact_work_from_energy_MJ

# Get the load folder of the Moreau–Jean–Frémond method
load_folder_MJF = data_folder + 'Wall/MJF/'
# Load the data
data_file_MJF = h5py.File(load_folder_MJF + 'rock_protection_wall_FremondNSL.hdf5', 'r')
# The data is stored in the 'data' group, so we wil get the information from there
energy_work = data_file_MJF['data']['energy_work'][:, :]
# Close the data file
data_file_MJF.close()
# Get the time, which is stored in the first column of energy_work. It is strictly non-decreasing, but may not be strictly increasing
time_MJF = energy_work[:, 0]
# Get the unique time values
unique_time_MJF = np.unique(time_MJF)
# Get the dissipation of the MJF method
normal_contact_work_from_energy_MJF = np.cumsum(energy_work[:, 3])
tangent_contact_work_from_energy_MJF = np.cumsum(energy_work[:, 4])
dissipatated_contact_and_friction_energy_MJF = normal_contact_work_from_energy_MJF + tangent_contact_work_from_energy_MJF


# Plot the results
width_in_inches = (8.27 - 2*1.5/2.54)/1.5
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_in_inches, 0.625*width_in_inches))
colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
line_split = ['-', '--', '-.', ':', '-', '--']
# TeX the written elements so that it looks good
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot the individual energetic quantities
ax0 = plt.subplot(111)
ax0.set_prop_cycle('color', colour_split)
ax0.plot(time_MJ, dissipatated_contact_and_friction_energy_MJ, linestyle=line_split[0], label=r'Moreau--Jean')
ax0.plot(unique_time_MJF, dissipatated_contact_and_friction_energy_MJF, linestyle=line_split[1], label=r'Moreau--Jean--Frémond')
ax0.set_xlim([0, np.max([unique_time_MJ[-1], unique_time_MJF[-1]])])
ax0.set_xlabel(r'$t$ (s)')
ax0.set_ylim([-8.5E7, 0.5E7])
ax0.set_ylabel(r'Change in Energy (N$\cdot$m)')
ax0.minorticks_on()
ax0.legend()

# Tighten the layout
plt.tight_layout()

# Save as a pdf
save_file_name = figure_folder + '_'.join(['Wall', 'method', 'dissipation', 'comparison', 'version', '2']) + '.pdf'
plt.savefig(save_file_name)

# Show the figure
plt.show()
