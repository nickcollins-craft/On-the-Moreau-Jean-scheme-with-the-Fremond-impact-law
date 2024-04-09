"""
This is a script to plot the results of the rockfall impact on a wall model. It requires the specification of the directory from which to load the results, as well as the method. 
It then loads the results and plots the work and energies. The results are saved as a PDF file in the specified directory.
"""

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import h5py
from path_file import *


# Loop through the methods and process the data to produce the plots
for method in ['MJ', 'MJF']:
    # Get the load folder
    load_folder = data_folder + 'Wall/' + method + '/'
    # Load the data
    if method == 'MJ':
        file_name = load_folder + 'rock_protection_wall_NewtonNSL.hdf5'
    else:
        file_name = load_folder + 'rock_protection_wall_FremondNSL.hdf5'
    data_file = h5py.File(file_name, 'r')
    # The data is stored in the 'data' group, so we wil get the information from there
    cf_work = data_file['data']['cf_work'][:, :]
    energy_work = data_file['data']['energy_work'][:, :]
    # Close the data file
    data_file.close()
    # Get the time, which is stored in the first column of cf_work. It is strictly non-decreasing, but may not be strictly increasing
    time = cf_work[:, 0]
    # Get the unique time values
    unique_time = np.unique(time)
    # Get the normal and tangential contact works
    normal_contact_work = cf_work[:, 2]
    tangential_contact_work = cf_work[:, 3]
    # Get the kinetic energy
    kinetic_energy = energy_work[:, 1]
    external_work = energy_work[:, 2]
    # Now we will get the work done at each step, as well as the cumulative work done and the min and max values of the work
    normal_contact_work_at_each_time_step = np.zeros(len(unique_time))
    tangential_contact_work_at_each_time_step = np.zeros(len(unique_time))
    min_normal_work = np.zeros(len(unique_time))
    max_normal_work = np.zeros(len(unique_time))
    min_tangential_work = np.zeros(len(unique_time))
    max_tangential_work = np.zeros(len(unique_time))
    for i in range(len(unique_time)):
        # Get the indices of the time steps that are equal to the current time
        indices = np.where(time == unique_time[i])
        # Get the normal and tangential contact work at the current time step
        normal_contact_work_at_each_time_step[i] = np.sum(normal_contact_work[indices])
        tangential_contact_work_at_each_time_step[i] = np.sum(tangential_contact_work[indices])
        # Get the min and max values of the normal and tangential contact work.
        min_normal_work[i] = np.min([0, np.min(normal_contact_work[indices])])
        max_normal_work[i] = np.max([0, np.max(normal_contact_work[indices])])
        min_tangential_work[i] = np.min([0, np.min(tangential_contact_work[indices])])
        max_tangential_work[i] = np.max([0, np.max(tangential_contact_work[indices])])

    external_work_from_energy = np.cumsum(energy_work[:, 2])
    normal_contact_work_from_energy = np.cumsum(energy_work[:, 3])
    tangent_contact_work_from_energy = np.cumsum(energy_work[:, 4])
    dissipatated_contact_and_friction_energy = normal_contact_work_from_energy + tangent_contact_work_from_energy
    total_energy = kinetic_energy - external_work_from_energy
    # Find the index where the rock is released (choosing 1000 as an arbitrary threshold that is not too close to zero and captures the change from order 1 to order 10^7)
    release_index = np.argwhere(kinetic_energy > 1000)[0][0]
    # Get the change in the energy from the point where the rock is released onwards
    Delta_E = total_energy[release_index:] - total_energy[release_index]


    # Plot the results
    width_in_inches = (8.27 - 2*1.5/2.54)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(width_in_inches, 1.25*width_in_inches))
    colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
    line_split = ['-', '--', '-.', ':', '-', '--']
    # TeX the written elements so that it looks good
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot the individual energetic quantities
    ax0 = plt.subplot(411)
    ax0.set_prop_cycle('color', colour_split)
    ax0.plot(unique_time, external_work_from_energy, linestyle=line_split[0], label=r'Total work done by the forces')
    ax0.plot(unique_time, kinetic_energy, linestyle=line_split[1], label=r'Kinetic energy')
    ax0.plot(unique_time, dissipatated_contact_and_friction_energy, linestyle=line_split[3], label=r'Dissipation')
    ax0.set_xlim([0, unique_time[-1]])
    ax0.set_xlabel(r'$t$ (s)')
    ax0.set_ylim([-9E7, 1E8])
    ax0.set_ylabel(r'Energy (N$\cdot$m)')
    ax0.minorticks_on()
    text_x_pos = 0.0075
    text_y_pos = 0.91
    ax0.text(text_x_pos, text_y_pos, r'$(a)$', transform=ax0.transAxes, fontsize=10)
    ax0.legend(loc=1, ncol=2)

    # Plot the sum of the energies with the work less dissipation
    ax1 = plt.subplot(412)
    ax1.set_prop_cycle('color', colour_split)
    ax1.plot(unique_time[release_index:], dissipatated_contact_and_friction_energy[release_index:], linestyle=line_split[0], label=r'Cumulative dissipation')
    ax1.plot(unique_time[release_index:], Delta_E, linestyle=line_split[1], label=r'Change in total energy')
    ax1.set_xlim([0, unique_time[-1]])
    ax1.set_xlabel(r'$t$ (s)')
    ax1.set_ylim([-8.5E7, 1.5E7])
    ax1.set_ylabel(r'Change in Energy (N$\cdot$m)')
    ax1.minorticks_on()
    ax1.text(text_x_pos, text_y_pos, r'$(b)$', transform=ax1.transAxes, fontsize=10)
    ax1.legend(loc=1)

    # Plot the maximum values of w_t
    ax2 = plt.subplot(413)
    ax2.set_prop_cycle('color', colour_split)
    ax2.plot(unique_time, max_tangential_work, linestyle=line_split[0], label=r'$\max(0, w_{\hbox{\tiny{T}}})$')
    ax2.set_xlim([0, unique_time[-1]])
    ax2.set_xlabel(r'$t$ (s)')
    if method == 'MJF':
        ax2.set_ylim([-0.05, 90])
    else:
        ax2.set_ylim([-50, 9.5E5])
    ax2.set_ylabel(r'Tangential work (N$\cdot$m)')
    ax2.minorticks_on()
    ax2.text(text_x_pos, text_y_pos, r'$(c)$', transform=ax2.transAxes, fontsize=10)
    ax2.legend(loc=1)

    # Plot the minimum values of w_t
    ax3 = plt.subplot(414)
    ax3.set_prop_cycle('color', colour_split)
    ax3.plot(unique_time, -min_tangential_work, linestyle=line_split[0], label=r'$-\min(0, w_{\hbox{\tiny{T}}})$')
    ax3.set_xlim([0, unique_time[-1]])
    ax3.set_xlabel(r'$t$ (s)')
    ax3.set_ylim([-1E5, 2.6E7])
    ax3.set_ylabel(r'Tangential work (N$\cdot$m)')
    ax3.minorticks_on()
    ax3.text(text_x_pos, text_y_pos, r'$(d)$', transform=ax3.transAxes, fontsize=10)
    ax3.legend(loc=1)

    # Tighten the layout
    plt.tight_layout()

    # Save as a pdf
    save_file_name = figure_folder + '_'.join(['Wall', method, 'energy', 'version', '3']) + '.pdf'
    plt.savefig(save_file_name)

    # Show the figure
    plt.show()
