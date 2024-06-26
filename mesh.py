"""
This is a code to read in a mesh file in the gmsh format and make sense of it. It assumes the mesh has already been created using gmsh, and relies on the meshio package to sensibly interpret the mesh information.
Functions are also provided to output the results of a simulation to a .vtk file, which can be read in by Paraview for post-processing.
"""

import meshio
import numpy as np


# Define a class to read in and make sense of gmsh meshes
class gmsh_mesh():
    def __init__(self, filename):
        self._filename = filename
        self._output_point_data = {}
        self._output_cell_data = {}
        # Reading a mesh file in the gmsh format.
        mesh = meshio.read(
            filename + '.msh',
            file_format="gmsh",
        )
        self._mesh = mesh
        # .vtk output for writing
        meshio.write(
            filename + ".vtk",
            mesh,
            file_format="vtk42"  # optional if first argument is a path; inferred from extension
        )

        # Get cell block information and physical entity information
        # Create an empty list of "physical entity"
        physical_entity = {}
        cell_block_count = 0
        # Add some "existence checks", because each of the boundaries are represented as a separate cell block (in gmsh 4.1 0 8 mesh format), so the line cell
        # block will be overwritten each time it encounters a new one. Hence, add a different behaviour for when such a cell block type already exists
        self._vertex_exists = False
        self._line_exists = False
        self._triangle_exists = False
        # Also initialise some necessary values
        n_cell_vertex = 0
        n_cell_line = 0
        n_cell_triangle = 0
        vertex_idx = 0
        line_idx = 0
        triangle_idx = 0
        for cell_block in mesh.cells:
            if cell_block.type == 'vertex':
                if self._vertex_exists:
                    vertex_idx = cell_block_count
                    n_cell_vertex = n_cell_vertex + len(cell_block.data)
                    physical_entity['vertex'] = np.concatenate((physical_entity['vertex'], mesh.cell_data['gmsh:physical'][vertex_idx]))
                    # Add a count of the different number of vertices
                    self.vertex_counter = np.concatenate((self.vertex_counter, [len(cell_block.data)]))
                else:
                    vertex_idx = cell_block_count
                    n_cell_vertex = len(cell_block.data)
                    physical_entity['vertex'] = mesh.cell_data['gmsh:physical'][vertex_idx]
                    self._vertex_exists = True
                    # Add a count of the different number of vertices
                    self.vertex_counter = [len(cell_block.data)]
            elif cell_block.type == 'line':
                if self._line_exists:
                    line_idx = cell_block_count
                    n_cell_line = n_cell_line + len(cell_block.data)
                    physical_entity['line'] = np.concatenate((physical_entity['line'], mesh.cell_data['gmsh:physical'][line_idx]))
                    # Add a count of the different number of boundaries
                    self.boundary_counter = np.concatenate((self.boundary_counter, [len(cell_block.data)]))
                else:
                    line_idx = cell_block_count
                    n_cell_line = len(cell_block.data)
                    physical_entity['line'] = mesh.cell_data['gmsh:physical'][line_idx]
                    self._line_exists = True
                    # Add a count of the different number of boundaries
                    self.boundary_counter = [len(cell_block.data)]
            elif cell_block.type == 'triangle':
                if self._triangle_exists:
                    triangle_idx = cell_block_count
                    n_cell_triangle = n_cell_triangle + len(cell_block.data)
                    physical_entity['triangle'] = np.concatenate((physical_entity['triangle'], mesh.cell_data['gmsh:physical'][triangle_idx]))
                else:
                    triangle_idx = cell_block_count
                    n_cell_triangle = len(cell_block.data)
                    physical_entity['triangle'] = mesh.cell_data['gmsh:physical'][triangle_idx]
                    self._triangle_exists = True
            else:
                print("cell_block.type = ", cell_block.type)
                raise ValueError("cell block type not recognized")
            cell_block_count = cell_block_count + 1
            # Print out the information about the mesh
            print("cell_block.type = ", cell_block.type)
        summary = \
            """The mesh has {0} cells block:
            - cell block number {1} of type vertex composed of {2} cells
            - cell block number {3} of type line composed of {4} cells
            - cell block number {5} of type triangle composed of {6} cells
            """.format(cell_block_count, vertex_idx, n_cell_vertex, line_idx, n_cell_line, triangle_idx, n_cell_triangle)
        print(summary)

        # Attach the various quantities to the mesh entity
        self._physical_entity = physical_entity
        self._n_cell_vertex = n_cell_vertex
        self._n_cell_line = n_cell_line
        self._n_cell_triangle = n_cell_triangle
        self._vertex_idx = vertex_idx
        self._line_idx = line_idx
        self._triangle_idx = triangle_idx

        physical_name = {}
        for field in mesh.field_data:
            physical_name[field] = {}
            physical_name[field]['index'] = mesh.field_data[field][0]
            physical_name[field]['dimension'] = mesh.field_data[field][1]

        for field in physical_name:
            physical_name[field]['nodes'] = set()
            physical_name[field]['cells'] = []
            # Physical name defined on vertex
            if physical_name[field]['dimension'] == 0:
                if 'Applied force' in field:
                    element_count = 0
                    for e in mesh.cells_dict['vertex']:
                        if physical_entity['vertex'][element_count] == physical_name[field]['index']:
                            physical_name[field]['cells'].append(e)
                            physical_name[field]['nodes'].add(e[0])
                        element_count = element_count + 1
                if 'Contact BC' in field:
                    element_count = 0
                    for e in mesh.cells_dict['vertex']:
                        if physical_entity['vertex'][element_count] == physical_name[field]['index']:
                            physical_name[field]['cells'].append(e)
                            physical_name[field]['nodes'].add(e[0])
                        element_count = element_count + 1
                if 'Applied displacement' in field:
                    element_count = 0
                    for e in mesh.cells_dict['vertex']:
                        if physical_entity['vertex'][element_count] == physical_name[field]['index']:
                            physical_name[field]['cells'].append(e)
                            physical_name[field]['nodes'].add(e[0])
                        element_count = element_count + 1
            # Physical name defined on line
            elif physical_name[field]['dimension'] == 1:
                if 'Contact BC' in field:
                    element_count = 0
                    if self._line_exists:
                        for e in mesh.cells_dict['line']:
                            if physical_entity['line'][element_count] == physical_name[field]['index']:
                                physical_name[field]['cells'].append(e)
                                physical_name[field]['nodes'].add(e[0])
                                physical_name[field]['nodes'].add(e[1])
                            element_count = element_count + 1
                if 'Applied force' in field:
                    element_count = 0
                    if self._line_exists:
                        for e in mesh.cells_dict['line']:
                            if physical_entity['line'][element_count] == physical_name[field]['index']:
                                physical_name[field]['cells'].append(e)
                                physical_name[field]['nodes'].add(e[0])
                                physical_name[field]['nodes'].add(e[1])
                            element_count = element_count + 1
                if 'Confining displacement' in field:
                    element_count = 0
                    if self._line_exists:
                        for e in mesh.cells_dict['line']:
                            if physical_entity['line'][element_count] == physical_name[field]['index']:
                                physical_name[field]['cells'].append(e)
                                physical_name[field]['nodes'].add(e[0])
                                physical_name[field]['nodes'].add(e[1])
                            element_count = element_count + 1
            # Physical name defined on surfaces
            elif physical_name[field]['dimension'] == 2:
                if 'Bulk material' in field:
                    element_count = 0
                    if self._triangle_exists:
                        for e in mesh.cells_dict['triangle']:
                            if physical_entity['triangle'][element_count] == physical_name[field]['index']:
                                physical_name[field]['cells'].append(e)
                                physical_name[field]['nodes'].add(e[0])
                                physical_name[field]['nodes'].add(e[1])
                                physical_name[field]['nodes'].add(e[2])
                            element_count = element_count + 1
        self._physical_name = physical_name
        # Print out the information about the physical names associated with the mesh
        summary = \
            """The mesh has {0} physical names:
            """.format(len(physical_name))
        for field in mesh.field_data:
            summary = summary + '-name : {0}\n            -index {1}, dimension {2}\n            -nodes {3}\n            -cells: {4}\n'.format(
                field, physical_name[field]['index'], physical_name[field]['dimension'], physical_name[field]['nodes'], physical_name[field]['cells'])
        print(summary)

    # Arrange the displacements in a way that is suitable for output to a .vtk file
    def vtk_prepare_output_displacement(self, u):
        # Output u for post-processing
        self._mesh.point_data['u'] = np.column_stack((u[::2], u[1::2], 0.0*u[::2]))
        self._mesh.point_data['u_x'] = u[::2]
        self._mesh.point_data['u_y'] = u[1::2]

    # Arrange the stresses in a way that is suitable for output to a .vtk file
    def vtk_prepare_output_stress(self, σ):
        # Output total stress for post-processing
        self._mesh.cell_data['sigma(xx,yy,xy)'] = []
        if self._vertex_exists:
            # Loop through all the vertices and add in the zeros (for arbitrary sizes)
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['sigma(xx,yy,xy)'].append(np.array([[0, 0, 0] for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            # Loop through all the boundaries and add in the zeros (for arbitrary sizes)
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['sigma(xx,yy,xy)'].append(np.array([[0, 0, 0] for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['sigma(xx,yy,xy)'].append(np.column_stack((σ[::3], σ[1::3], σ[2::3])))

        # Output the individual stress components for post-processing as for the total stress
        self._mesh.cell_data['sigma_xx'] = []
        if self._vertex_exists:
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['sigma_xx'].append(np.array([0 for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['sigma_xx'].append(np.array([0 for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['sigma_xx'].append(np.array(σ[::3]))

        self._mesh.cell_data['sigma_yy'] = []
        if self._vertex_exists:
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['sigma_yy'].append(np.array([0 for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['sigma_yy'].append(np.array([0 for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['sigma_yy'].append(np.array(σ[1::3]))

        self._mesh.cell_data['sigma_xy'] = []
        if self._vertex_exists:
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['sigma_xy'].append(np.array([0 for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['sigma_xy'].append(np.array([0 for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['sigma_xy'].append(np.array(σ[2::3]))

    # Arrange the strains in a way that is suitable for output to a .vtk file
    def vtk_prepare_output_strain(self, ε):
        # Output the total strain for post-processing
        self._mesh.cell_data['epsilon(xx,yy,xy)'] = []
        if self._vertex_exists:
            # Loop through all the vertices and add in the zeros (for arbitrary sizes)
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['epsilon(xx,yy,xy)'].append(np.array([[0, 0, 0] for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            # Loop through all the boundaries and add in the zeros (for arbitrary sizes)
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['epsilon(xx,yy,xy)'].append(np.array([[0, 0, 0] for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['epsilon(xx,yy,xy)'].append(np.column_stack((ε[::3], ε[1::3], ε[2::3])))

        # Output the individual strain components for post-processing as for the total strain
        self._mesh.cell_data['epsilon_xx'] = []
        if self._vertex_exists:
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['epsilon_xx'].append(np.array([0 for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['epsilon_xx'].append(np.array([0 for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['epsilon_xx'].append(np.array(ε[::3]))

        self._mesh.cell_data['epsilon_yy'] = []
        if self._vertex_exists:
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['epsilon_yy'].append(np.array([0 for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['epsilon_yy'].append(np.array([0 for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['epsilon_yy'].append(np.array(ε[1::3]))

        self._mesh.cell_data['epsilon_xy'] = []
        if self._vertex_exists:
            for vertex_index in range(len(self.vertex_counter)):
                self._mesh.cell_data['epsilon_xy'].append(np.array([0 for k in range(self.vertex_counter[vertex_index])]))
        if hasattr(self, 'boundary_counter'):
            for boundary_index in range(len(self.boundary_counter)):
                self._mesh.cell_data['epsilon_xy'].append(np.array([0 for k in range(self.boundary_counter[boundary_index])]))
        self._mesh.cell_data['epsilon_xy'].append(np.array(ε[2::3]))

    # Write the output to a .vtk file
    def vtk_finalize_output(self, foutput, mode='write'):
        meshio.write_points_cells(# './vtk/' + foutput,
            foutput,
            points=self._mesh.points,
            cells=self._mesh.cells,
            # Optionally provide extra data on points, cells, etc.
            point_data=self._mesh.point_data,
            cell_data=self._mesh.cell_data,
            file_format="vtk"  # optional if first argument is a path; inferred from extension # field_data=field_data,
        )
