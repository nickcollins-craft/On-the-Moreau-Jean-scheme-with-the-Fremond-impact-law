"""
This is a code that performs the Moreau–Jean–Frémond linear one-step integration scheme for a nonsmooth contact dynamics problem. The function is called from another code (main.py) which maintains the global history.
The function takes the initial and final times, initial positions and velocities, the θ parameter, the model, and the tolerance for the solver as inputs, and returns the final velocity and position, the work done by the 
normal and tangential impulses, the local normal and tangential relative velocities, the gap vector, and the final contact impulse vector. The Model argument must be able to return the mass matrix, the external forces, 
the stiffness and damping matrices, the contact selection matrix, the gap vector, and the coefficients of restitution and friction. Where appropriate, the model must also be able to return the augmented mass matrix, the 
inverse of the augmented mass matrix, the nature of the boundary conditions, and a method of boundary condition enforcement. This structure allows exploitation of model linearity when it is present, and the ability to 
handle displacement-controlled degrees of freedom. The function uses the Siconos library to solve the frictional contact problem.
"""

import numpy as np
import siconos.numerics as sn


def MoreauJeanLinearVariantOneStep(t_i, t_f, q_i, v_i, θ, Model, tol):
    """
    This function implements the Moreau-Jean-Frémond time-stepping scheme for a nonsmooth contact dynamics problem.

    :param t_i: Initial time
    :param t_f: Final time
    :param q_i: Initial position
    :param v_i: Initial velocity
    :param θ: Parameter for the time-stepping scheme
    :param Model: The model of the system
    :param tol: Tolerance for the solver
    :return: The final velocity and position, the work done by the normal and tangential impulses, the local normal and tangential relative velocities, the gap vector, and the final contact impulse vector
    """
    # Get the time step size between the initial and final time
    h = t_f - t_i
    # Get the mass matrix from the model
    M = Model.mass(q_i)
    # Get the external forces from the model
    f_ext = Model.f_ext
    # Get the external forces at the mid-step
    f_θ = (θ*f_ext(t_f) + (1 - θ)*f_ext(t_i))
    # Get the gradient "internal forces" (the stiffness matrix and the damping matrix) from the model
    K, C = Model.nabla_f_int(t_i, q_i, v_i)
    # Calculate the inverse of the (possibly modified) augmented mass matrix, depending on the linearity of the model
    if Model._isLinear:
        # Calculate the augmented mass matrix
        M_hat = Model.compute_M_hat(h, θ)
        # Then calculate the inverse of the (modified) augmented mass matrix
        M_hat_inv = Model.compute_M_hat_inv(h, θ)
    else:
        # Then calculate the inverse of the (modified) augmented mass matrix, depending on whether the matrix needs to be modified for boundary conditions or not
        if Model._isDisplacementControl:
            M_hat_inv = Model.compute_M_hat_inv(h, θ)
        else:
            M_hat_inv = np.linalg.inv(M + ((h*θ)**2)*K + h*θ*C)
    # Calculate the free impulse (in the absence of contact) at the mid-step
    R_free_unmodified = M @ v_i - h*θ*K @ q_i + h*θ*f_θ
    # If the model has displacement controlled degrees of freedom, the boundary conditions need to be enforced. v_f here is actually v_k+θ, but this saves us from allocating a new variable
    v_f = np.zeros_like(v_i)
    if Model._isDisplacementControl:
        # Impose the velocity boundary conditions at the θ-point of the time step
        v_f[Model._fixed_dofs] = Model.fixed_dof_values((1 - θ)*t_i + θ*t_f)
        R_free = Model.boundary_condition_enforcement(M_hat, R_free_unmodified, v_f, Model._control)
    else:
        # Otherwise, the free impulse is the same as the unmodified free impulse
        R_free = R_free_unmodified
    # Calculate the free velocity (in the absence of contact) at the mid-step
    v_free = M_hat_inv @ R_free

    # Mid-step evaluation (which we set to be the initial displacements, but which could include a velocity projection if desired)
    q_m = q_i

    # Calculate the H matrix and the gap vector based on the mid-step evaluation
    H_m = Model.H(q_m)
    g_m = Model.g(q_m)

    # Initialise the contact forecast
    n_active = 0
    n_contact = int(H_m.shape[0]/2)
    b_i = (θ*(1. + Model._e) - 1.)*H_m @ v_i
    α_active = []
    # Loop through the contacts to find the active ones
    for α in range(n_contact):
        # If the gap is non-positive, then the contact is active
        if g_m[α] <= 0:
            # Increment the number of active contacts
            n_active = n_active + 1
            # Add the contact to the list of active contacts
            α_active.append(α)
            # Get the contact matrix for the set of active contacts
            H_contact = H_m[2*α:2*(α + 1), :]
            # If there is only one active contact, transform the matrix into a numpy array
            if n_active == 1:
                H_active = np.array(H_contact)
                b_active = np.array([b_i[2*α], 0])
            # Otherwise, concatenate the matrix and vector to the existing ones
            else:
                H_active = np.concatenate((H_active, H_contact), axis=0)
                b_active = np.concatenate((b_active, [b_i[2*α], 0]))
    
    # If there are active contacts, then a contact problem needs to be solved
    if n_active > 0:
        # Build the Frictional Contact Problem matrix and vector
        W = θ*H_active @ M_hat_inv @ H_active.T
        y = H_active @ v_free + b_active
        # Initialise the friction coefficient vector
        μ = np.ones(n_active)*Model._μ
        # Initialise the LCP problem using the Siconos library (which has a dedicated solver for frictional contact problems)
        fc2d = sn.FrictionContactProblem(2, W, y, μ)
        # Choose an option for the solver. We use the Non-Smooth Gauss-Seidel (NSGS) solver
        options = sn.SolverOptions(sn.SICONOS_FRICTION_2D_NSGS)
        # Give the Siconos solver a tolerance
        options.dparam[sn.SICONOS_DPARAM_TOL] = tol
        r = np.zeros_like(y)
        w = np.zeros_like(y)

        # This calls the Siconos solver and solves the LCP
        info = sn.fc2d_nsgs(fc2d, r, w, options)

    # Initialise the final contact impulse vector
    p_f = np.zeros(H_m.shape[0])
    # Loop through the active contacts and fill in the final contact impulse vector with the solution from the LCP
    i = 0
    for n_active in α_active:
        p_f[2*n_active:2*(n_active + 1)] = r[2*i:2*(i + 1)]
        i = i + 1
    # Now build the final contact impulse vector for all contacts (which will be zero if there are no active contacts)
    P_f = H_m.T @ p_f

    # Calculate the global final velocity and position. Note that the first v_f is actually the mid-step velocity.
    v_f = v_free + θ*M_hat_inv @ P_f
    v_f = (1./θ)*(v_f - (1 - θ)*v_i)
    q_f = q_i + h*(θ*v_f + (1 - θ)*v_i)

    # Calculate the local normal and tangential relative velocities
    u_f = H_m @ v_f
    u_i = H_m @ v_i

    # Calculate the gap vector at the final time
    g_f = Model.g(q_f)

    # Calculate the work done by the normal and tangential impulses, element wise (so this will return a vector of length n_contact)
    work_normal = np.multiply(θ*u_f[0::2] + (1 - θ)*u_i[0::2], p_f[0::2])
    work_tangent = np.multiply(θ*u_f[1::2] + (1 - θ)*u_i[1::2], p_f[1::2])

    # Return the final velocity and position, the work done by the normal and tangential impulses, the local normal and tangential relative velocities, the gap vector, and the final contact impulse vector
    return v_f, q_f, work_normal, work_tangent, u_f, g_f, p_f
