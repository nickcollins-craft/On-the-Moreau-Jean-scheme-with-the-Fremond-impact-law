"""
This is a script to plot the results of the rocking block model. It requires the specification of the directory from which to load the results, as well as the method. It then loads the results and plots the velocities, 
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
    load_folder = data_folder + 'RockingBlock/' + method + '/'
    # Load the data
    t = np.load(load_folder + 't' + '.npy')
    v = np.load(load_folder + 'v' + '.npy')
    q = np.load(load_folder + 'q' + '.npy')
    u = np.load(load_folder + 'u' + '.npy')
    p = np.load(load_folder + 'p' + '.npy')
    work_normal = np.load(load_folder + 'w_n' + '.npy')
    work_tangent = np.load(load_folder + 'w_t' + '.npy')
    kinetic_energy = np.load(load_folder + 'E_k' + '.npy')
    potential_energy = np.load(load_folder + 'E_p' + '.npy')
    strain_energy = np.load(load_folder + 'E_strain' + '.npy')
    dissipated_contact_and_friction_energy = np.load(load_folder + 'D_contact' + '.npy')
    total_work = np.load(load_folder + 'W' + '.npy')
    total_energy = np.load(load_folder + 'E' + '.npy')

    # Plot the results
    width_in_inches = (8.27 - 2*1.5/2.54)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(width_in_inches, 1.3*width_in_inches))
    colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
    line_split = ['-', '--', '-.', ':', '-', '--']
    # TeX the written elements so that it looks good
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot the velocities at the end of each time step
    ax0 = plt.subplot(611)
    ax0.set_prop_cycle('color', colour_split)
    ax0.plot(t, v[:, 0], linestyle=line_split[0], label=r'$v_{x}$')
    ax0.plot(t, v[:, 1], linestyle=line_split[1], label=r'$v_{y}$')
    ax0.plot(t, v[:, 2], linestyle=line_split[2], label=r'$\dot{\gamma}$')
    ax0.set_xlim([0, t[-1]])
    ax0.set_xlabel(r'$t$')
    ax0.set_ylim([1.1*np.min(v), 1.1*np.max(v)])
    ax0.set_ylabel(r'Velocity')
    ax0.minorticks_on()
    text_x_pos = 0.0075
    text_y_pos = 0.87
    ax0.text(text_x_pos, text_y_pos, r'$(a)$', transform=ax0.transAxes, fontsize=10)
    plt.legend(loc=1)

    # Plot the positions at the end of each time step
    ax1 = plt.subplot(612)
    ax1.set_prop_cycle('color', colour_split)
    ax1.plot(t, q[:, 0], linestyle=line_split[0], label=r'$x$')
    ax1.plot(t, q[:, 1], linestyle=line_split[1], label=r'$y$')
    ax1.plot(t, q[:, 2], linestyle=line_split[2], label=r'$\gamma$')
    ax1.set_xlim([0, t[-1]])
    ax1.set_xlabel(r'$t$')
    ax1.set_ylim([1.1*np.min(q), 0.75])
    ax1.set_ylabel(r'Position')
    ax1.minorticks_on()
    ax1.text(text_x_pos, text_y_pos, r'$(b)$', transform=ax1.transAxes, fontsize=10)
    plt.legend(loc=1)

    # Plot the local velocities
    ax2 = plt.subplot(613)
    ax2.set_prop_cycle('color', colour_split)
    ax2.plot(t, u[:, 0], linestyle=line_split[0], label=r'$u_{\hbox{\tiny{N,1}}}$')
    ax2.plot(t, u[:, 1], linestyle=line_split[1], label=r'$u_{\hbox{\tiny{T,1}}}$')
    ax2.plot(t, u[:, 2], linestyle=line_split[2], label=r'$u_{\hbox{\tiny{N,2}}}$')
    ax2.plot(t, u[:, 3], linestyle=line_split[3], label=r'$u_{\hbox{\tiny{T,2}}}$')
    ax2.set_xlim([0, t[-1]])
    ax2.set_xlabel(r'$t$')
    ax2.set_ylim([1.1*np.min(u), 1.4*np.max(u)])
    ax2.set_ylabel(r'Relative velocity')
    ax2.minorticks_on()
    ax2.text(text_x_pos, text_y_pos, r'$(c)$', transform=ax2.transAxes, fontsize=10)
    plt.legend(loc=1, ncol=2)

    # Plot the contact impulses
    ax3 = plt.subplot(614)
    ax3.set_prop_cycle('color', colour_split)
    ax3.plot(t, p[:, 0], linestyle=line_split[0], label=r'$p_{\hbox{\tiny{N,1}}}$')
    ax3.plot(t, p[:, 1], linestyle=line_split[1], label=r'$p_{\hbox{\tiny{T,1}}}$')
    ax3.plot(t, p[:, 2], linestyle=line_split[2], label=r'$p_{\hbox{\tiny{N,2}}}$')
    ax3.plot(t, p[:, 3], linestyle=line_split[3], label=r'$p_{\hbox{\tiny{T,3}}}$')
    ax3.set_xlim([0, t[-1]])
    ax3.set_xlabel(r'$t$')
    ax3.set_ylim([1.1*np.min(p), 1.1*np.max(p)])
    ax3.set_ylabel(r'Impulse')
    ax3.minorticks_on()
    ax3.text(text_x_pos, text_y_pos, r'$(d)$', transform=ax3.transAxes, fontsize=10)
    plt.legend(loc=1, ncol=2)

    # Plot the increment of the work
    ax4 = plt.subplot(615)
    ax4.set_prop_cycle('color', colour_split)
    ax4.plot(t, work_normal[:, 0], linestyle=line_split[0], label=r'$w_{\hbox{\tiny{N,1}}}$')
    ax4.plot(t, work_tangent[:, 0], linestyle=line_split[1], label=r'$w_{\hbox{\tiny{T,1}}}$')
    ax4.plot(t, work_normal[:, 1], linestyle=line_split[2], label=r'$w_{\hbox{\tiny{N,2}}}$')
    ax4.plot(t, work_tangent[:, 1], linestyle=line_split[3], label=r'$w_{\hbox{\tiny{T,2}}}$')
    ax4.set_xlim([0, t[-1]])
    ax4.set_xlabel(r'$t$')
    if method == 'MJ':
        ax4.set_ylim([1.1*np.min(work_tangent), 1.1*np.max(work_tangent)])
    else:
        ax4.set_ylim([1.1*np.min(work_tangent), 0.015])
    ax4.set_ylabel(r'Work')
    ax4.minorticks_on()
    ax4.text(text_x_pos, text_y_pos, r'$(e)$', transform=ax4.transAxes, fontsize=10)
    plt.legend(loc=1, ncol=2)

    # Plot the total energies
    ax5 = plt.subplot(616)
    ax5.set_prop_cycle('color', colour_split)
    ax5.plot(t, kinetic_energy, linestyle=line_split[0], label=r'Kinetic energy')
    ax5.plot(t, potential_energy, linestyle=line_split[1], label=r'Potential energy')
    ax5.plot(t, dissipated_contact_and_friction_energy, linestyle=line_split[2], label=r'Dissipated energy')
    ax5.plot(t, total_energy, linestyle=line_split[3], label=r'Total energy')
    ax5.set_xlim([0, t[-1]])
    ax5.set_xlabel(r'$t$')
    ax5.set_ylim([-0.75, 1.25*np.max(total_energy)])
    ax5.set_ylabel(r'Energy')
    ax5.minorticks_on()
    ax5.text(text_x_pos, text_y_pos, r'$(f)$', transform=ax5.transAxes, fontsize=10)
    plt.legend(loc=5, ncol=2)

    # Tighten the layout
    plt.tight_layout()

    # Save as a pdf
    save_file_name = figure_folder + '_'.join(['RockingBlock', method, 'full_variables', 'version', '1']) + '.pdf'
    plt.savefig(save_file_name)

    # Show the figure
    plt.show()