"""
This is a code to define a minimal finite element model using linear triangular elements, and to compute the stiffness and mass matrices of the model. The model is defined in 2D, and the material is assumed to be linear elastic 
and isotropic. The small deformation and linear finite element framework is used, and each element has a single Gauss point. The model is defined in a way that allows for the use of plane stress or plane strain assumptions.
"""

import numpy as np


# Define the bulk material
class material:
    # Initialise its value with the Young's modulus, Poisson's ratio, a density, and whether we make a plane stress or plane strain assumption
    def __init__(self, E, ν, ρ, load_type='plane_stress'):
        # Young's modulus
        self._E = E
        # Poisson's ratio
        self._ν = ν
        # Density
        self._ρ = ρ
        # Load type
        self._load_type = load_type

    # Define the element stiffness tensor for 2D, depending on whether it's plane strain or plane stress. This matrix acts on a vector [ε_xx, ε_yy, γ_xy] to return a vector [σ_xx, σ_yy, σ_xy]
    def D(self, dim):
        if (dim == 2):
            D = np.zeros((3, 3))
            if self._load_type == 'plane_strain':
                coef = self._E/((1 + self._ν)*(1 - 2*self._ν))
                # Plain strain matrix
                D[0, 0] = coef*(1.0 - self._ν)
                D[0, 1] = coef*self._ν
                D[0, 2] = 0.0

                D[1, 0] = D[0, 1]
                D[1, 1] = D[0, 0]
                D[1, 2] = 0.0

                D[2, 0] = 0.0
                D[2, 1] = 0.0
                D[2, 2] = coef*(1.0 - 2*self._ν)/2
            elif self._load_type == 'plane_stress':
                coef = self._E/(1 - self._ν**2)
                # Plain stress matrix
                D[0, 0] = coef
                D[0, 1] = coef*self._ν
                D[0, 2] = 0.0

                D[1, 0] = D[0, 1]
                D[1, 1] = D[0, 0]
                D[1, 2] = 0.0

                D[2, 0] = 0.0
                D[2, 1] = 0.0
                D[2, 2] = coef*(1.0 - self._ν)/2
            else:
                # If it's not one of the two allowable load types, raise an error
                raise ValueError('self._load_type', 'load type not implemented, please enter plane_strain or plane_stress')
        else:
            # If in 3D, raise an error
            raise ValueError('dim', 'dimension not implemented')
        return D


# Define the finite element
class fem_element:
    # Allocate the type of finite element, and the number of nodes and dimensions
    def __init__(self):
        self._fem_type = 'T3'
        self._n_nodes = 3
        self._dim = 2

    # Define the appropriate shape functions  in the internal triangular coordinates ξ and η, given those internal coordinates
    def shape_functions(self, ξ, η):
        N = np.zeros(3)
        N[0] = 1.0 - ξ - η
        N[1] = ξ
        N[2] = η
        return N

    # Define the derivatives of the shape functions w.r.t. the internal coordinates
    def shape_functions_derivatives(self, ξ, η):
        Nξ = np.zeros(3)
        Nξ[0] = -1.0
        Nξ[1] = 1.0
        Nξ[2] = 0.0
        Nη = np.zeros(3)
        Nη[0] = -1.0
        Nη[1] = 0.0
        Nη[2] = 1.0
        return Nξ, Nη


# Define the finite element model as a whole
class fem_model:
    # Load in the mesh and allocate a thickness and mass matrix type
    def __init__(self, mesh, thickness=None, mass_type='consistent'):
        self._mesh = mesh
        self._thickness = thickness
        self._mass_type = mass_type
        # Compute the number of degrees of freedom (dof)
        self._n_dof = 2*np.shape(mesh.points)[0]
        # Initialise the stiffness matrix
        self.K = np.zeros((self._n_dof, self._n_dof))
        # Initialise the mass matrix
        self.M = np.zeros((self._n_dof, self._n_dof))
        # Initialise the damping matrix
        self.C = np.zeros((self._n_dof, self._n_dof))

    # Report the number of dof and the stiffness and mass matrices
    def __str__(self):
        message = """
        number of dof = {0}
        stiffness matrix = {1}
        mass matrix = {2}
        """.format(self._n_dof, self.K, self.M)
        return message

    # Return the location of the Gauss points in internal coordinates, plus weighting factors
    def gauss_points(self):
        # First, designate the Gauss points and weighting factors for the element described in triangular coordinates
        ξ_coordinate = np.array([1./3.])
        η_coordinate = np.array([1./3.])
        weighting_factor = 0.5*np.array([1.0])
        return ξ_coordinate, η_coordinate, weighting_factor

    # Define the formula to calculate the area of an element, given the coordinates of its nodes
    def element_area(self, e):
        fem_e = fem_element()
        x = []
        y = []
        # Loop through each node in the element
        for i in range(fem_e._n_nodes):
            # Get the global node number
            n = e[i]
            # Get the x and y coordinates of that node
            x = np.append(x, self._mesh.points[n][0])
            y = np.append(y, self._mesh.points[n][1])
        # Once all coordinates are obtained, calculate the area of the element
        A = 0.5*abs(x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0]) + x[2]*(y[0] - y[1]))
        return A

    # Compute the Jacobian "J" matrix of an individual finite element e at a given Gauss point p
    def compute_elementary_Jacobian_matrix(self, e, p):
        fem_e = fem_element()
        p_ξ = p[0]
        p_η = p[1]
        Nξ, Nη = fem_e.shape_functions_derivatives(p_ξ, p_η)

        # Compute Jacobian matrix
        J = np.zeros((fem_e._dim, fem_e._dim))
        # Loop through each node and calculate its contribution to the Jacobian
        for i in range(fem_e._n_nodes):
            n = e[i]
            x = self._mesh.points[n][0]
            y = self._mesh.points[n][1]
            J[0, 0] = J[0, 0] + Nξ[i]*x
            J[0, 1] = J[0, 1] + Nξ[i]*y
            J[1, 0] = J[1, 0] + Nη[i]*x
            J[1, 1] = J[1, 1] + Nη[i]*y
        return J

    # Compute the "B" matrix of an individual finite element (which will give us the finite element estimate of the strain), where e is the element number, and p the Gauss point
    def compute_elementary_B_matrix(self, e, p):
        fem_e = fem_element()
        n_dof_e = fem_e._n_nodes*fem_e._dim
        p_ξ = p[0]
        p_η = p[1]
        Nξ, Nη = fem_e.shape_functions_derivatives(p_ξ, p_η)

        # Compute inverse of Jacobian matrix
        J_inv = np.linalg.inv(self.compute_elementary_Jacobian_matrix(e, p))

        # Compute the derivative w.r.t x and y of the shape function at each node
        Nx = np.zeros(fem_e._n_nodes)
        Ny = np.zeros(fem_e._n_nodes)
        # For each node in the element, calculate the contribution to the derivative
        for i in range(fem_e._n_nodes):
             Nx[i] = J_inv[0, 0]*Nξ[i] + J_inv[0, 1]*Nη[i]
             Ny[i] = J_inv[1, 0]*Nξ[i] + J_inv[1, 1]*Nη[i]

        # Construct the B matrix (its form is consistent with the choice of the representation of strain)
        B = np.zeros((3, n_dof_e))
        # Loop through each node in the element and use the previously calculated derivatives to construct "B"
        for i in range(fem_e._n_nodes):
            B[0, 2*i] = Nx[i]
            B[1, 2*i] = 0.0
            B[2, 2*i] = Ny[i]
            B[0, 2*i + 1] = 0.0
            B[1, 2*i + 1] = Ny[i]
            B[2, 2*i + 1] = Nx[i]
        return B

    # Define the stiffness matrix for an element, given the element and the material stiffness tensor
    def compute_elementary_stiffness_matrix(self, e, D):
        # Get the element information and proceed accordingly
        fem_e = fem_element()
        n_dof_e = fem_e._n_nodes*fem_e._dim
        # Loop over the Gauss points and fill in the stiffness matrix
        K_e = np.zeros((n_dof_e, n_dof_e))
        ξ_coordinate, η_coordinate, weighting_factor = self.gauss_points()
        for gauss_point_index in range(len(ξ_coordinate)):
            p = [ξ_coordinate[gauss_point_index], η_coordinate[gauss_point_index]]
            # Get the elementary B matrix and Jacobian determinant from the previous procedures
            det_J = np.linalg.det(self.compute_elementary_Jacobian_matrix(e, p))
            B = self.compute_elementary_B_matrix(e, p)
            # Fill in the stiffness matrix via multiplication of the appropriate terms
            K_e = K_e + weighting_factor[gauss_point_index]*det_J*self._thickness*(B.T @ D @ B)
        return K_e

    # Define the mass matrix for an element, given the element and the density
    def compute_elementary_mass_matrix(self, e, ρ):
        fem_e = fem_element()
        n_dof_e = fem_e._n_nodes*fem_e._dim
        A = self.element_area(e)
        if self._mass_type == 'lumped':
            M_e = (ρ*A*self._thickness/fem_e._n_nodes)*np.eye(n_dof_e)
        elif self._mass_type == 'consistent':
            M_e = 2.0*np.eye(n_dof_e)
            M_e[0, 2] = 1.0
            M_e[0, 4] = 1.0
            M_e[1, 3] = 1.0
            M_e[1, 5] = 1.0
            M_e[2, 0] = 1.0
            M_e[2, 4] = 1.0
            M_e[3, 1] = 1.0
            M_e[3, 5] = 1.0
            M_e[4, 0] = 1.0
            M_e[4, 2] = 1.0
            M_e[5, 1] = 1.0
            M_e[5, 3] = 1.0
            M_e = (ρ*A*self._thickness/12.)*M_e
        return M_e
        
    # Define the stiffness matrix at a structural level, given an element and its stiffness matrix
    def assemble_stiffness_matrix(self, e, K_e):
        # Initialse the dof list and the node counter
        dof_index_n = []
        n_cnt = 0
        # Loop through the nodes in the element
        for n in e:
            # Specify the global dofs of the node we consider
            dof_index_n = [2*n, 2*n + 1]
            # Initialise the node counter for the other nodes that are going to contribute stiffness to that dof
            m_cnt = 0
            # Loop through the nodes (again)
            for m in e:
                # Specify the global dofs of the node we consider
                dof_index_m = [2*m, 2*m + 1]
                # Loop through the local dofs
                for i in range(2):
                    for j in range(2):
                        # The stiffness terms contributed to the dofs at node n by the dofs at node m. Each pair gives a 2x2 contribution
                        # that is mapped into the global coordinates, and added to whatever is already contained in the global stiffness matrix.
                        self.K[dof_index_n[i], dof_index_m[j]] = self.K[dof_index_n[i], dof_index_m[j]] + K_e[i + n_cnt*2, j + m_cnt*2]
                m_cnt = m_cnt + 1
            n_cnt = n_cnt + 1
        return

    # Define the mass matrix at a structural level, given an element and its mass matrix
    def assemble_mass_matrix(self, e, M_e):
        # Initialse the dof list and the node counter
        dof_index_n = []
        n_cnt = 0
        # Loop through the nodes in the element
        for n in e:
            # Specify the global dofs of the node we consider
            dof_index_n = [2*n, 2*n + 1]
            # Initialise the node counter for the other nodes that are going to contribute mass to that dof
            m_cnt = 0
            # Loop through the nodes (again)
            for m in e:
                # Specify the global dofs of the node we consider
                dof_index_m = [2*m, 2*m + 1]
                # Loop through the local dofs
                for i in range(2):
                    for j in range(2):
                        # The mass terms contributed to the dofs at node n by the dofs at node m. Each pair gives a 2x2 contribution
                        # that is mapped into the global coordinates, and added to whatever is already contained in the global mass matrix.
                        self.M[dof_index_n[i], dof_index_m[j]] = self.M[dof_index_n[i], dof_index_m[j]] + M_e[i + n_cnt*2, j + m_cnt*2]
                m_cnt = m_cnt + 1
            n_cnt = n_cnt + 1
        return

    # Get the full global stiffness matrix by looping over all elements
    def compute_stiffness_matrix(self, material):
        # Loop over the triangle elements
        for e in self._mesh.cells_dict['triangle']:
            D = material.D(2)
            K_e = self.compute_elementary_stiffness_matrix(e, D)
            self.assemble_stiffness_matrix(e, K_e)
        return

    # Get the full global mass matrix by looping over all elements
    def compute_mass_matrix(self, material):
        # Loop over the triangle elements
        for e in self._mesh.cells_dict['triangle']:
            M_e = self.compute_elementary_mass_matrix(e, material._ρ)
            self.assemble_mass_matrix(e, M_e)
        return

    # Give the method for computing the strain at the Gauss point
    def compute_strain_at_gauss_points(self, u):
        # Loop over the triangle elements
        ε = []
        for e in self._mesh.cells_dict['triangle']:
            # Get the displacement of the element
            u_e = []
            for n in e:
                u_e.extend([u[2*n], u[2*n + 1]])
            # Get the Gauss point information and calculate the strain
            ξ_coordinate, η_coordinate, weighting_factor = self.gauss_points()
            p = [ξ_coordinate[0], η_coordinate[0]]
            B = self.compute_elementary_B_matrix(e, p)
            ε.extend(B @ u_e)
        return ε
    
    # Give the method for computing the stress at the Gauss point, given the strain and the material
    def compute_stress_at_gauss_points(self, ε, D):
        # Loop over the triangle elements
        σ = []
        for i in range(np.shape(self._mesh.cells_dict['triangle'])[0]):
            σ.extend(D @ ε[3*i:3*(i + 1)])
        return σ
