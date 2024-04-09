"""
This is a file that defines the individual models that are used in the main.py script. The models are defined as classes, with a variety of methods that return 
quantities useful in the Moreau–Jean and Moreau–Jean–Frémond models.
"""

import numpy as np
from fem import *
from mesh import *


class Rod_model:
    """
    Define a model that creates a rigid rod with Newton contact law and Coulomb friction
    """
    # Initialise the model
    def __init__(self):
        # Define the length of the rod
        self._l = 1.
        # Define the mass of the rod
        self._m = 1.
        # Obtain the rotational inertia of the rod rotating about its own central axis
        self._inertia = (self._m*self._l**2)/12.
        # Define the restitution coefficient
        self._e = 1.0
        # Define the friction coefficient
        self._μ = 0.01
        # Define the value of the gravitational acceleration
        self._gravity = 10.
        # Define that the model is not linear
        self._isLinear = False
        # Define that the model does not have displacement control
        self._isDisplacementControl = False

    # Define the mass matrix
    def mass(self, q):
        M = np.eye(3)*self._m
        # The third entry is the rotational inertia
        M[2, 2] = self._inertia
        return M

    # Define the gradient of the internal forces (which gives the stiffness and viscous damping matrices)
    def nabla_f_int(self, t_i, q_i, v_i):
        K = np.zeros((3, 3))
        C = np.zeros((3, 3))
        return K, C

    # Define the external forces applied to the rod
    def f_ext(self, t):
        # The rod is subject to gravity
        f = [0., -self._m*self._gravity, 0.]
        return np.array(f)

    # Define the gap function g(q) that gives the distance between the rod and the ground
    def g(self, q):
        y = q[1] - 0.5*self._l*np.cos(q[2])
        return np.array([y])

    # Define a function to get the H matrix (which depends on the current position of the rod)
    def H(self, q):
        H = np.array([[0., 1., 0.5*self._l*np.sin(q[2])],
                      [-1., 0., -0.5*self._l*np.cos(q[2])]])
        return H
    
    # Define a function to get the kinetic energy
    def kinetic_energy(self, q, v):
        E_K = 0.5*np.dot(v, self.mass(q) @ v)
        return E_K
    
    # Define a function to get the gravitational potential energy
    def potential_energy(self, q):
        E_P = self._m*self._gravity*q[1]
        return E_P
    
    # Define a function to get the elastic strain energy
    def strain_energy(self, q):
        E_Ψ = 0.0
        return E_Ψ
    
    # Define a function to get the power input to the system
    def power_input(self, t, v):
        P_in = self.f_ext(t) @ v
        return P_in


class RockingBlock_model:
    """
    Define a model that creates a rigid block with Newton contact law and Coulomb friction
    """
    # Initialise the model
    def __init__(self):
        # Define the width of the block
        self._b = 1.
        # Define the height of the block
        self._l = 1.
        # Define the mass of the block
        self._m = 1.
        # Define the density of the block
        self._ρ = self._b*self._l/self._m
        # Obtain the rotational inertia of the block rotating about its own central axis
        self._inertia = (self._b*self._l/12.)*self._ρ
        # Define the restitution coefficient
        self._e = 1.0
        # Define the friction coefficient
        self._μ = 0.1
        # Define the value of the gravitational acceleration
        self._gravity = 10.
        # Define that the model is not linear
        self._isLinear = False
        # Define that the model does not have displacement control
        self._isDisplacementControl = False

    # Define the mass matrix
    def mass(self, q):
        M = np.eye(3)*self._m
        # The third entry is the rotational inertia
        M[2, 2] = self._inertia
        return M

    # Define the gradient of the internal forces (which gives the stiffness and viscous damping matrices)
    def nabla_f_int(self, t_i, q_i, v_i):
        K = np.zeros((3, 3))
        C = np.zeros((3, 3))
        return K, C

    # Define the external forces applied to the block
    def f_ext(self, t):
        # The block is subject to gravity
        f = [0., -self._m*self._gravity, 0.]
        return np.array(f)

    # Define the gap function g(q) that gives the distance between the block and the ground
    def g(self, q):
        y = [q[1] - 0.5*self._b*np.sin(q[2]) - 0.5*self._l*np.cos(q[2]),
             q[1] + 0.5*self._b*np.sin(q[2]) - 0.5*self._l*np.cos(q[2])]
        return np.array(y)

    # Define a function to get the H matrix (which depends on the current position of the block)
    def H(self, q):
        H = np.array([[0., 1., -0.5*self._b*np.cos(q[2]) + 0.5*self._l*np.sin(q[2])],
                       [-1., 0., -0.5*self._b*np.sin(q[2]) - 0.5*self._l*np.cos(q[2])],
                       [0., 1., 0.5*self._b*np.cos(q[2]) + 0.5*self._l*np.sin(q[2])],
                       [-1., 0., 0.5*self._b*np.sin(q[2]) - 0.5*self._l*np.cos(q[2])]])
        return H
    
    # Define a function to get the kinetic energy
    def kinetic_energy(self, q, v):
        E_K = 0.5*np.dot(v, self.mass(q) @ v)
        return E_K
    
    # Define a function to get the gravitational potential energy
    def potential_energy(self, q):
        E_P = self._m*self._gravity*q[1]
        return E_P
    
    # Define a function to get the elastic strain energy
    def strain_energy(self, q):
        E_Ψ = 0.0
        return E_Ψ
    
    # Define a function to get the power input to the system
    def power_input(self, t, v):
        P_in = self.f_ext(t) @ v
        return P_in


class SlidingBlock_model:
    """
    Define a model that creates a two-dimensional damped elastic block with Newton contact law and Coulomb friction
    """
    # Initialise the model
    def __init__(self):
        # Set the model parameters
        # Create a material, using the parameters from Berman et al. (2020), for PMMA. We will use units in N, mm, ms, g, MPa etc
        self._E = 5.75E3
        self._ρ = 1.17E-3
        self._ν = 0.358
        # Set the block thickness
        self._block_thickness = 15.0
        # Define the restitution coefficient
        self._e = 0.
        # Define the friction coefficient
        self._μ = 0.5
        # Define the value of the gravitational acceleration (in mm/ms^2)
        self._gravity = 1E-3
        # Define the model as linear
        self._isLinear = True
        # Define the model as having displacement control
        self._isDisplacementControl = True
        
        # Define the magnitude of the applied traction
        self._applied_force_magnitude = 2.
        # Define the frequency of the applied traction
        self._applied_force_frequency = 2.
        # Define the magnitude of the confining displacements
        self._confining_displacement_magnitude = -0.005
        # Define the magnitude of the confining velocities
        self._confining_velocity_magnitude = 0.0

        # Create the bulk material using the parameters
        self._material = material(self._E, self._ν, self._ρ, load_type='plane_stress')

        # Obtain the mesh
        self._mesh_filename = 'sliding_block_mesh'
        self._block_mesh = gmsh_mesh(self._mesh_filename)
        self._mesh = self._block_mesh._mesh
        self._physical_name = self._block_mesh._physical_name
        self._n_nodes = self._mesh.points.shape[0]
        # Create lists to store the nodes and dofs that have boundary conditions applied
        self._applied_force_nodes = []
        self._applied_force_dofs = []
        self._contact_nodes = []
        self._contact_dofs = []
        self._contact_dofs_normal = []
        self._applied_force_y_position = []
        self._confining_displacement_nodes = []
        self._confining_displacement_dofs = []
        # Now loop through the physical names to find the nodes that have boundary conditions applied
        for fd in self._physical_name:
            if 'Contact BC' in fd:
                for n in self._physical_name[fd]['nodes']:
                    self._contact_nodes.extend([n])
                    # Contact phenomena are both normal and tangential, so we take both dofs
                    self._contact_dofs.extend([2*n, 2*n + 1])
                    # However, contact is detected only in the normal direction, so we only take the normal dof (which is y in this case)
                    self._contact_dofs_normal.extend([2*n + 1])
            if 'Applied force' in fd:
                for n in self._physical_name[fd]['nodes']:
                    self._applied_force_nodes.extend([n])
                    # The force is only applied horizontally, so we only take the x dof
                    self._applied_force_dofs.extend([2*n])
                    # It is also convenient to track the y position of the applied force
                    self._applied_force_y_position.extend([self._mesh.points[n][1]])
            if 'Confining displacement' in fd:
                for n in self._physical_name[fd]['nodes']:
                    self._confining_displacement_nodes.extend([n])
                    # The displacement is only applied vertically, so we only take the y dof
                    self._confining_displacement_dofs.extend([2*n + 1])
        # Calculate a matrix giving the distances of the applied force points from each other
        distance_matrix_force = np.zeros((len(self._applied_force_nodes), len(self._applied_force_nodes)))
        for node_index_1 in range(np.shape(distance_matrix_force)[0]):
            for node_index_2 in range(np.shape(distance_matrix_force)[1]):
                distance_matrix_force[node_index_1, node_index_2] = np.abs(self._applied_force_y_position[node_index_1] - self._applied_force_y_position[node_index_2])

        # Declare the fixed dofs
        self._fixed_dofs = self._confining_displacement_dofs
        # Create the control boolean vector
        self._control = np.zeros((2*self._n_nodes), dtype=bool)
        self._control[self._fixed_dofs] = True

        # Now create the finite element model
        self._fem_model = fem_model(self._mesh, thickness=self._block_thickness, mass_type='consistent')
        # Compute the stiffness and mass matrices
        self._fem_model.compute_stiffness_matrix(self._material)
        self._fem_model.compute_mass_matrix(self._material)
        self._n_dof = np.shape(self._fem_model.K)[0]
        # Create the viscous damping matrix
        self._C = np.zeros_like(self._fem_model.K)

        # Now, we find the tributary areas of the applied force nodes
        self._applied_force_area = np.zeros((len(self._applied_force_nodes)))
        # Iterate through the nodes and calculate the distances of the nodes above and below
        for node_index in range(len(self._applied_force_nodes)):
            # Get the case for the bottom boundary, with no node below
            if self._applied_force_y_position[node_index] == np.min(self._applied_force_y_position):
                # Add half the area above, from the distance to the next node above, taking the second entry to ignore the self-distance
                self._applied_force_area[node_index] = self._applied_force_area[node_index] + 0.5*self._fem_model._thickness*np.sort(distance_matrix_force[node_index, :])[1]
            # Now get the case for the upper boundary, with no node above
            elif self._applied_force_y_position[node_index] == np.max(self._applied_force_y_position):
                # Add half the area to the node below, from the distance to the next node below, taking the second entry to ignore the self-distance
                self._applied_force_area[node_index] = self._applied_force_area[node_index] + 0.5*self._fem_model._thickness*np.sort(distance_matrix_force[node_index, :])[1]
            # Now solve all the other nodes
            else:
                # Add half the area above and below the nodes, taking the second and third entries to ignore the self-distance
                self._applied_force_area[node_index] = self._applied_force_area[node_index] + 0.5*self._fem_model._thickness*np.sort(distance_matrix_force[node_index, :])[1]
                self._applied_force_area[node_index] = self._applied_force_area[node_index] + 0.5*self._fem_model._thickness*np.sort(distance_matrix_force[node_index, :])[2]

        # Create the constant gravity vector
        gravity = np.zeros((2*self._n_nodes))
        gravity[1::2] = -self._gravity
        # The block is subject to gravity
        self._f_gravity = self._fem_model.M @ gravity

        # Create the contact matrix (which is the same for all time steps)
        self._H = np.zeros((2*len(self._contact_nodes), 2*self._n_nodes))
        for i in range(len(self._contact_nodes)):
            self._H[2*i, 2*self._contact_nodes[i] + 1] = 1.
            self._H[2*i + 1, 2*self._contact_nodes[i]] = 1.

    # Define the mass matrix
    def mass(self, q):
        return self._fem_model.M

    # Define the gradient of the internal forces (which gives the stiffness and viscous damping matrices)
    def nabla_f_int(self, t_i, q_i, v_i):
        return self._fem_model.K, self._C
    
    # Define the external forces applied to the block
    def f_ext(self, t):
        # Set to zero initially
        f = np.zeros((self._n_dof))
        # The block is subject to gravity
        f[:] = self._f_gravity
        # The applied force is a square wave applied to the left-hand side of the block
        f[self._applied_force_dofs] = f[self._applied_force_dofs] + self._applied_force_magnitude*self._applied_force_area*np.sign(np.sin(2*np.pi*self._applied_force_frequency*t))
        return np.array(f)
    
    # Define the gap function g(q) that gives the distance between the block and the ground
    def g(self, q):
        # In this case, we just take the normal distance, because our surface is flat (and presumptively always in contact)
        y = q[self._contact_dofs_normal]
        return np.array(y)
    
    # Define a function to get the H matrix
    def H(self, q):
        return self._H
    
    # Define a function that computes the augmented mass matrix
    def compute_M_hat(self, h, θ):
        # Check if we have already computed the augmented mass matrix, and return it if so
        if hasattr(self, '_M_hat'):
            return self._M_hat
        # If not, compute it and store it
        self._M_hat = self._fem_model.M + ((h*θ)**2)*self._fem_model.K
        return self._M_hat
    
    # Define a function to get the inverse of the modified augmented mass matrix or stiffness matrix, depending on what we want to do
    def compute_modified_matrix_for_boundary_conditions(self, matrix, control):
        # Take a copy of the matrix. This works equally well for the stiffness matrix if we want to do elastic quasi-static behaviour using K, or for dynamics by using M_hat
        matrix_bar = matrix.copy()
        # Now loop through the controlled velocities (or displacements) and make the modifications that will help to enforce the boundary conditions
        for i in range(np.shape(control)[0]):
            if control[i]:
                # Set the row and column of the controlled velocity (or displacement) to zero, except for the diagonal entry
                matrix_bar[i, :] = 0.0
                matrix_bar[:, i] = 0.0
                matrix_bar[i, i] = matrix[i, i]
        # Take the inverse of the modified matrix
        matrix_inv = np.linalg.inv(matrix_bar)
        return matrix_inv
    
    # Define a function to get the inverse of the augmented mass matrix
    def compute_M_hat_inv(self, h, θ):
        # Check if we have already computed the inverse of the augmented mass matrix, and return it if so
        if hasattr(self, '_M_hat_inv'):
            return self._M_hat_inv
        # If not, compute it and store it
        self._M_hat_inv = self.compute_modified_matrix_for_boundary_conditions(self.compute_M_hat(h, θ), self._control)
        return self._M_hat_inv
    
    # Define a function which inputs the augmented mass matrix, the free-flight impulse, the velocity and the control vector, to enforce the boundary conditions.
    # It works equally well with force and displacment for R and v, substituting the stiffness matrix for M_hat to get elastic quasi-static behaviour.
    def boundary_condition_enforcement(self, M_hat, R, v, control):
        R_bar = R.copy()
        # Loop through the controlled velocities
        for i in range(np.shape(v)[0]):
            # If it's a controlled velocity, calculate the impulse from the augmented mass matrix.
            if control[i]:
                R_bar[i] = M_hat[i, i]*v[i]
                # Then subtract its effect from all the other entries
                for j in range(np.shape(M_hat)[0]):
                    if j != i:
                        # Make sure we don't interfere with any of the other controlled velocities
                        if not control[j]:
                            R_bar[j] = R_bar[j] - M_hat[j, i]*v[i]
        return R_bar
    
    # Define the function which determines the initial displacements after being subjected to the confining displacement
    def initial_displacements(self, q):
        # Set the initial displacements to be the confining displacement at the top boundary
        q[self._confining_displacement_dofs] = self._confining_displacement_magnitude
        # Set the initial displacemnts to be zero at the contact nodes (both vertically and horizontally)
        q[self._contact_dofs] = 0.0
        # Create the list of controlled dofs (with known displacements)
        controlled_dofs = np.concatenate((self._confining_displacement_dofs, self._contact_dofs))
        # Create the boolean vector of controlled dofs
        controlled_dofs_bool = np.zeros((2*self._n_nodes), dtype=bool)
        controlled_dofs_bool[controlled_dofs] = True
        # Compute the inverse of the modified K_bar matrix
        K_bar_inv = self.compute_modified_matrix_for_boundary_conditions(self._fem_model.K, controlled_dofs_bool)
        # Get the modified forces which would impose these displacements at the controlled dofs (modified from a vector of zero forces)
        F_bar = self.boundary_condition_enforcement(self._fem_model.K, np.zeros_like(q), q, controlled_dofs_bool)
        # Solve the system to get the initial displacements
        q = K_bar_inv @ F_bar
        return q
    
    # Define a function that returns the velocities on the fixed dofs
    def fixed_dof_values(self, t):
        # Set the fixed dofs to be the confining velocity
        v = np.zeros_like(self._fixed_dofs)
        v[:] = self._confining_velocity_magnitude
        return v
    
    # Define a function to get the kinetic energy
    def kinetic_energy(self, q, v):
        E_K = 0.5*v @ self.mass(q) @ v
        return E_K
    
    # Define a function to get the gravitational potential energy
    def potential_energy(self, q):
        E_P = np.dot(self._f_gravity[1::2], q[1::2])
        return E_P
    
    # Define a function to get the elastic strain energy
    def strain_energy(self, q):
        E_Ψ = 0.5*q @ self._fem_model.K @ q
        return E_Ψ
    
    # Define a function to get the power input to the system
    def power_input(self, t, v):
        P_in = self.f_ext(t)[self._applied_force_dofs] @ v[self._applied_force_dofs]
        return P_in