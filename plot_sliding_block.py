"""
This is a script to plot the results of the sliding block model. It requires the specification of the directory from which to load the results, as well as the method. It then loads the results and plots the velocities, 
positions, local velocities, contact impulses, work, and energies. The results are saved as a PDF file in the specified directory.
"""

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
from path_file import *


# Loop through the methods and process the data to produce the plots
for method in ['MJ', 'MJF']:
    # Get the load folder
    load_folder = data_folder + 'SlidingBlock/' + method + '/'
    # Load the data
    t = np.load(load_folder + 't' + '.npy')
    work_tangent = np.load(load_folder + 'w_t' + '.npy')
    kinetic_energy = np.load(load_folder + 'E_k' + '.npy')
    potential_energy = np.load(load_folder + 'E_p' + '.npy')
    strain_energy = np.load(load_folder + 'E_strain' + '.npy')
    dissipated_contact_and_friction_energy = np.load(load_folder + 'D_contact' + '.npy')
    total_work = np.load(load_folder + 'W' + '.npy')
    total_energy = np.load(load_folder + 'E' + '.npy')

    # Get the combination of the work and the dissipation
    work_with_dissipation = total_work + dissipated_contact_and_friction_energy

    # Get the maximum value of w_t at each time step
    max_w_t = np.maximum(np.zeros((len(work_tangent))), np.max(work_tangent, axis=1))
    # Get the minimum value of w_t at each time step
    min_w_t = np.minimum(np.zeros((len(work_tangent))), np.min(work_tangent, axis=1))

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
    ax0.plot(t, total_work, linestyle=line_split[0], label=r'Total work')
    ax0.plot(t, kinetic_energy, linestyle=line_split[1], label=r'Kinetic energy')
    ax0.plot(t, potential_energy, linestyle=line_split[2], label=r'Potential energy')
    ax0.plot(t, strain_energy, linestyle=line_split[3], label=r'Strain energy')
    ax0.plot(t, dissipated_contact_and_friction_energy, linestyle=line_split[4], label=r'Dissipation')
    ax0.set_xlim([0, t[-1]])
    ax0.set_xlabel(r'$t$ (ms)')
    ax0.set_ylim([-800, 1400])
    ax0.set_ylabel(r'Energy (N$\cdot$mm)')
    ax0.minorticks_on()
    text_x_pos = 0.0075
    text_y_pos = 0.92
    ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)
    ax0.legend(loc=9, ncol=2)

    # Plot the sum of the energies with the work less dissipation
    ax1 = plt.subplot(412)
    ax1.set_prop_cycle('color', colour_split)
    ax1.plot(t, work_with_dissipation, linestyle=line_split[0], label=r'Total work + dissipation')
    ax1.plot(t, total_energy, linestyle=line_split[1], label=r'Total energy')
    ax1.set_xlim([0, t[-1]])
    ax1.set_xlabel(r'$t$ (ms)')
    ax1.set_ylim([0, 350])
    ax1.set_ylabel(r'Energy (N$\cdot$mm)')
    ax1.minorticks_on()
    ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)
    ax1.legend(loc=1)

    # Plot the maximum values of w_t
    ax2 = plt.subplot(413)
    ax2.set_prop_cycle('color', colour_split)
    ax2.plot(t, max_w_t, linestyle=line_split[0], label=r'$\max(0, w_{\hbox{\tiny{T}}})$')
    ax2.set_xlim([0, t[-1]])
    ax2.set_xlabel(r'$t$ (ms)')
    ax2.set_ylim([-0.0005, 0.004])
    ax2.set_ylabel(r'Tangential work (N$\cdot$mm)')
    ax2.minorticks_on()
    ax2.text(text_x_pos, text_y_pos, '$(c)$', transform=ax2.transAxes, fontsize=10)
    ax2.legend(loc=1)

    # Plot the minimum values of w_t
    ax3 = plt.subplot(414)
    ax3.set_prop_cycle('color', colour_split)
    ax3.plot(t, -min_w_t, linestyle=line_split[0], label=r'$-\min(0, w_{\hbox{\tiny{T}}})$')
    ax3.set_xlim([0, t[-1]])
    ax3.set_xlabel(r'$t$ (ms)')
    ax3.set_ylim([-0.005, 0.09])
    ax3.set_ylabel(r'Tangential work (N$\cdot$mm)')
    ax3.minorticks_on()
    ax3.text(text_x_pos, text_y_pos, '$(d)$', transform=ax3.transAxes, fontsize=10)
    ax3.legend(loc=1)

    # Tighten the layout
    plt.tight_layout()

    # Save as a pdf
    save_file_name = figure_folder + '_'.join(['SlidingBlock', method, 'energy', 'version', '2']) + '.pdf'
    plt.savefig(save_file_name)

    # Show the figure
    plt.show()