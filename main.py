"""
This is a code to run the three models defined in these Python scripts, namely the impacting stick, the rocking block and the sliding block. It is a simple script that runs the Moreau-Jean and 
Moreau-Jean Frémond models for a given model and method, and saves the results to a directory. In the case of the sliding block model, the script also writes .vtk files for visualisation purposes.
To run the script, first set the global data_folder variable in path_file.py with your preferred save location, then navigate the terminal to the folder this code is located in. Build the sliding block
mesh by typeing the command "gmsh sliding_block_mesh.geo -1 -2" (without quotation marks), and then use the following command: "siconos main.py" (again, without quotation marks).
This assumes that you have gmsh, meshio and Siconos installed in the currently active Python environment. The script will then run the simulations and save the results to the specified directory.
"""

import numpy as np
import os
from MoreauJeanLinearOneStep import *
from MoreauJeanFrémondLinearOneStep import *
from models import *
from path_file import *


# Initialise the θ value and the tolerance
θ = 0.5
tol = 1e-10


# Now, cycle through the models and methods to run the simulations
for model in ['Rod', 'RockingBlock', 'SlidingBlock']:
    print('---- model ', model)
    for method in ['MJ', 'MJF']:
        print('---- method ', method)

        # Create a directory to store the results (if it doesn't already exist)
        model_directory = data_folder + model + '/'
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        directory = model_directory + method + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Initialise the model and the initial position and velocity, depending on the chosen model
        if model == 'Rod':
            # Initialise the rod model and the initial position and velocity via setting the initial angle
            ds = Rod_model()
            initial_angle = np.pi/4
            q_i = np.array([0.5*ds._l*np.sin(initial_angle),
                        0.5*ds._l*np.cos(initial_angle) + 0.01,
                        initial_angle])
            v_i = np.array([-0.5, 0.1, 0.1]) 
            # Set the total time and the time step
            T = 0.2
            h = 1e-04
        elif model == 'RockingBlock':
            # Initialise the rocking block model
            ds = RockingBlock_model()
            q_i = np.array([0., 0.6, 0.])
            v_i = np.array([0., -0.2, 1.])
            # Set the total time and the time step
            T = 1.
            h = 1e-04
        elif model == 'SlidingBlock':
            # Initialise the sldiing block model
            model = 'SlidingBlock'
            ds = SlidingBlock_model()
            # Create the initial displacement and velocity vectors
            q_i = np.zeros((ds._n_dof))
            v_i = np.zeros_like(q_i)
            # Initialise the displacement vector with an elastic deformation
            q_i = ds.initial_displacements(q_i)
            # Set the total time and the time step
            T = 1.0
            h = 1e-04
            # Set the .vtk writing folder
            vtk_write_folder = directory + '/' + 'vtk/'
            # Check that the write folder exists and make it if not
            if not os.path.isdir(vtk_write_folder):
                os.mkdir(vtk_write_folder)
            # Write the initial values to vtk
            ds._block_mesh.vtk_prepare_output_displacement(q_i)

            # Output strain for post-processing
            ε = ds._fem_model.compute_strain_at_gauss_points(q_i)
            ds._block_mesh.vtk_prepare_output_strain(ε)

            # Output stress for post-processing
            σ = ds._fem_model.compute_stress_at_gauss_points(np.array(ε), ds._material.D(2))
            ds._block_mesh.vtk_prepare_output_stress(σ)

            # File writing
            foutput = '{0}{1:03d}.vtk'.format('Sliding_Block_', 0)
            file_write = vtk_write_folder + foutput
            ds._block_mesh.vtk_finalize_output(file_write)
        else:
            raise ValueError('The model ' + model + ' is not defined')
        
        # Set the number of time steps
        N = int(T/h)

        # Initialise the lists of values
        t_i = 0.0
        v = np.copy(v_i)
        q = np.copy(q_i)
        t = np.array(t_i)
        # Intialise the H matrix and the local relative velocities
        H = ds.H(q_i)
        u = H @ v_i
        # Initialise the number of contact points
        n_contact = int(H.shape[0]/2)
        # Initialise the normal and tangential work values
        work_normal = np.zeros((1, n_contact))
        work_tangent = np.zeros((1, n_contact))
        # Initialise the contact impulses
        p = np.zeros(H.shape[0])
        # Initialise the gap function values
        g_i = ds.g(q_i)
        g = np.copy(g_i)
        # Set the initial mid-step velocities and times
        v_k_θ = np.empty((0, np.shape(v)[0]), dtype=float)
        t_k_θ = np.empty((0), dtype=float)

        # Run the simulation by looping over the time steps
        for i in range(N):
            print('---- step ', i)
            t_f = t_i + h
            if method == 'MJ':
                v_f, q_f, work_normal_f, work_tangent_f, u_f, g_f, p_f = MoreauJeanLinearOneStep(t_i, t_f, q_i, v_i, θ, ds, tol)
            elif method =='MJF':
                v_f, q_f, work_normal_f, work_tangent_f, u_f, g_f, p_f = MoreauJeanLinearVariantOneStep(t_i, t_f, q_i, v_i, θ, ds, tol)
            else:
                raise ValueError('The method ' + method + ' is not defined')
            
            # Stack the output arrays on the end of the current arrays
            v = np.vstack((v, v_f))
            q = np.vstack((q, q_f))
            t = np.vstack((t, t_f))
            work_normal = np.vstack((work_normal, work_normal_f))
            work_tangent = np.vstack((work_tangent, work_tangent_f))
            u = np.vstack((u, u_f))
            g = np.vstack((g, g_f))
            p = np.vstack((p, p_f))

            # If the model is the sliding block, we calculate the stresses and strains (at Gauss points) and output them to vtk, along with the displacements at the nodes
            if model == 'SlidingBlock':
                # Prepare to write to vtk
                ds._block_mesh.vtk_prepare_output_displacement(q[-1, :])

                # Output strain for post-processing
                ε = ds._fem_model.compute_strain_at_gauss_points(q[-1, :])
                ds._block_mesh.vtk_prepare_output_strain(ε)

                # Output stress for post-processing
                σ = ds._fem_model.compute_stress_at_gauss_points(np.array(ε), ds._material.D(2))
                ds._block_mesh.vtk_prepare_output_stress(σ)

                # File writing
                foutput = '{0}{1:03d}.vtk'.format('Sliding_Block_', i + 1)
                file_write = vtk_write_folder + foutput
                ds._block_mesh.vtk_finalize_output(file_write)

            # Calculate the mid-step velocities and times
            v_k_θ = np.vstack((v_k_θ, θ*v[-1, :] + (1 - θ)*v[-2, :]))
            t_k_θ = np.append(t_k_θ, θ*t_f + (1 - θ)*t_i)
            # Update the values for the next step
            v_i = v_f
            q_i = q_f
            t_i = t_f

        # Calculate the energetic properties using the simulation results
        kinetic_energy = np.empty((0, 1), dtype=float)
        potential_energy = np.empty((0, 1), dtype=float)
        dissipated_contact_and_friction_energy = np.empty((0, 1), dtype=float)
        strain_energy = np.empty((0, 1), dtype=float)
        total_work = np.empty((0, 1), dtype=float)
        total_energy = np.empty((0, 1), dtype=float)
        for i in range(v.shape[0]):
            # Get the kinetic energy
            kinetic_energy = np.vstack((kinetic_energy, ds.kinetic_energy(q[i, :], v[i, :])))
            # Get the gravitational potential energy
            potential_energy = np.vstack((potential_energy, ds.potential_energy(q[i, :])))
            # Get the strain energy
            strain_energy = np.vstack((strain_energy, ds.strain_energy(q[i, :])))
            # Get the dissipated energy (which is the sum of the normal impact and tangential frictional work), and the work input to the system
            if i == 0:
                dissipated_contact_and_friction_energy = np.vstack((dissipated_contact_and_friction_energy, np.sum(work_normal[i, :]) + np.sum(work_tangent[i, :])))
                # Take account of any initial energy in the system (e.g. pre-existing strain energy, potential energy, etc.)
                initial_energy = kinetic_energy[i] + potential_energy[i] + strain_energy[i]
                total_work = total_work = np.vstack((total_work, initial_energy + h*θ*ds.power_input(t[i], v[i])))
            else:
                dissipated_contact_and_friction_energy = np.vstack((dissipated_contact_and_friction_energy, dissipated_contact_and_friction_energy[-1] + np.sum(work_normal[i, :]) + np.sum(work_tangent[i, :])))
                total_work = np.vstack((total_work, total_work[-1] + h*(θ*ds.power_input(t[i], v[i]) + (1 - θ)*ds.power_input(t[i - 1], v[i - 1]))))

            # Get the total energy (which is the kinetic plus the potential plus the strain minus the dissipation)
            total_energy = np.vstack((total_energy, kinetic_energy[-1] + potential_energy[-1] + strain_energy[-1]))

        # Save the results to the directory
        np.save(os.path.join(directory, 't' + '.npy'), t)
        np.save(os.path.join(directory, 't_theta' + '.npy'), t_k_θ)
        np.save(os.path.join(directory, 'v' + '.npy'), v)
        np.save(os.path.join(directory, 'v_theta' + '.npy'), v_k_θ)
        np.save(os.path.join(directory, 'q' + '.npy'), q)
        np.save(os.path.join(directory, 'u' + '.npy'), u)
        np.save(os.path.join(directory, 'g' + '.npy'), g)
        np.save(os.path.join(directory, 'p' + '.npy'), p)
        np.save(os.path.join(directory, 'w_n' + '.npy'), work_normal)
        np.save(os.path.join(directory, 'w_t' + '.npy'), work_tangent)
        np.save(os.path.join(directory, 'E_k' + '.npy'), kinetic_energy)
        np.save(os.path.join(directory, 'E_p' + '.npy'), potential_energy)
        np.save(os.path.join(directory, 'E_strain' + '.npy'), strain_energy)
        np.save(os.path.join(directory, 'D_contact' + '.npy'), dissipated_contact_and_friction_energy)
        np.save(os.path.join(directory, 'W' + '.npy'), total_work)
        np.save(os.path.join(directory, 'E' + '.npy'), total_energy)

        # Save files in a format favourable for tikz/gnuplot plotting
        file_kinematic_and_kinetic_txt = os.path.join(directory, model + '_' + method + '_' + 'kinematic_and_kinetic' + '.dat')
        output = np.concatenate((t, q, v, u, g, p, work_normal, work_tangent), axis=1)
        np.savetxt(file_kinematic_and_kinetic_txt, output)
        file_energetic_txt = os.path.join(directory, model + '_' + method + '_' + 'energetic' + '.dat')
        output = np.concatenate((t, p, work_normal, work_tangent, kinetic_energy, potential_energy, strain_energy, dissipated_contact_and_friction_energy, total_work, total_energy), axis=1)
        np.savetxt(file_energetic_txt, output)
