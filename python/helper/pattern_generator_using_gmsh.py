import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..')); sys.path.append(os.path.join(os.path.dirname(__file__), '../gmsh')); 
import numpy as np
import numpy.linalg as la
import mesher_helper
from mesher_helper import generate_mesh_from_embeddings_array_input, generate_mesh_from_embeddings_array_input_allow_boundary
import MeshFEM, homogenized_inflation
import igl


import parametric_pillows
def periodic_box_transform(box, point):
    ''' transform point outside the box to inside the box by applying shifts of integer multiples of the box size '''
    new_point = np.array(point)
    box_size_x = la.norm(box[1] - box[0])
    box_size_y = la.norm(box[1] - box[2])
    point_out = False
    if new_point[0] < box[0][0]:
        new_point[0] += box_size_x
        point_out = True
    elif new_point[0] > box[2][0]:
        new_point[0] -= box_size_x
        point_out = True
    if new_point[1] < box[0][1]:
        new_point[1] += box_size_y
        point_out = True
    elif new_point[1] > box[2][1]:
        new_point[1] -= box_size_y
        point_out = True
    return new_point, point_out

def get_edge_edge_intersection(edge1, edge2):
    # Assume the two edges are not colinear
    # Compute intersection of two line segments:
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    p = np.array(edge1[0])
    r = np.array(edge1[1] - edge1[0])
    q = np.array(edge2[0])
    s = np.array(edge2[1] - edge2[0])
    # Determine whether the line segments intersect
    r_cross_s = np.cross(r, s)[2]
    if (r_cross_s == 0):
        # The lines are parallel
        return None
    t = np.cross(q - p, s)[2] / r_cross_s
    u = np.cross(q - p, r)[2] / r_cross_s
    if (t >= 0 and t <= 1 and u >= 0 and u <= 1):
        return p + t * r
    return None

def get_intersection_point(box, edge):
    # Compute the intersection point between the edge and the box boundary:
    # box: 4 * 2 array
    # edge: 2 * 2 array
    # return: 2 * 1 array

    for i in range(4):
        boundary_edge = np.array([box[i], box[(i + 1) % 4]])
        intersection_point = get_edge_edge_intersection(boundary_edge, edge)
        if (intersection_point is not None):
            return intersection_point, i
    return None, -1

def get_cosine_dash_curve(h, amplitude, dash_point, avg_len_embeddings):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    dash_line = np.array([dash_point, mid_point * 2 - dash_point])

    dash_vector = dash_point - mid_point

    transformation = np.array([[dash_vector[0], -dash_vector[1]], [dash_vector[1], dash_vector[0]]])

    def base_cosine_function(x):
        # Cosine function defined between -1 and 1
        return [x, amplitude * np.cos(x * np.pi)]

    dash_len = la.norm(dash_line[1] - dash_line[0])
    num_seg = int(np.round(dash_len / avg_len_embeddings))
    base_cosine_curve = np.linspace(-1, 1, num_seg + 1)
    base_cosine_curve = np.array([base_cosine_function(x) for x in base_cosine_curve])

    transformed_cosine_curve = []
    new_pt_out = False
    edges = []

    transformed_point, prev_pt_out = periodic_box_transform(boundary_vxs, transformation @ base_cosine_curve[0])
    transformed_cosine_curve.append(transformed_point)
    for i in range(len(base_cosine_curve))[1:]:
        transformed_point, new_pt_out = periodic_box_transform(boundary_vxs, transformation @ base_cosine_curve[i])
        transformed_cosine_curve.append(transformed_point)
        if new_pt_out:
            return None, None, None, None
        if (prev_pt_out == new_pt_out):
            edges.append([i-1, i])
        prev_pt_out = new_pt_out
    transformed_cosine_curve = np.array(transformed_cosine_curve)

    # Pad a column of zeros to the transformed_cosine_curve
    transformed_cosine_curve = np.concatenate((transformed_cosine_curve, np.zeros((len(transformed_cosine_curve), 1))), axis=1)

    curve_edges = np.array(edges) + 1
    return boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges

def get_cosine_dash(h, avg_len_boundary, avg_len_embeddings, amplitude, dash_point, return_line_segments = False):
    boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges = get_cosine_dash_curve(h, amplitude, dash_point, avg_len_embeddings)
    if (boundary_vxs is None):
        return None, None, None
    if return_line_segments:
        return transformed_cosine_curve, curve_edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-9)

    return ipu, m, fusing_data

def get_cosine_curve(h, avg_len_boundary, avg_len_embeddings, amplitude, return_line_segments = False, end_threshold = 0.0, use_half_period = False):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    def base_cosine_function(x):
        # Cosine function defined between -1 and 1
        return [amplitude * (np.cos(x * np.pi)), x, 0.0]
    
    num_seg = int(np.round(w / avg_len_embeddings))
    print("num_seg: {}".format(num_seg))
    base_cosine_curve = np.linspace(-1 + end_threshold, (0 if use_half_period else 1) - end_threshold, num_seg + 1)
    base_cosine_curve = np.array([base_cosine_function(x) for x in base_cosine_curve])
    base_cosine_curve *= w * (1 if use_half_period else 0.5)
    base_cosine_curve[:, 1] += (w / 2) if use_half_period else 0
    # base_cosine_curve[:, 1] += amplitude * w / 2

    curve_edges = np.array([[i + 1, i + 2] for i in range(len(base_cosine_curve) - 1)])

    if return_line_segments:
        return base_cosine_curve, curve_edges, boundary_vxs, boundary_lines

    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, base_cosine_curve, curve_edges)
    print(sum(fusing_data))

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data


def get_dash_line(h, dash_point):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    dash_line = np.array([dash_point, mid_point * 2 - dash_point])
    dash_line = np.concatenate((dash_line, np.zeros((len(dash_line), 1))), axis=1)
    print(dash_line)
    edges = [[0, 1]]
    curve_edges = np.array(edges) + 1
    return boundary_vxs, boundary_lines, dash_line, curve_edges

def get_dash_region(h, dash_point):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    dash_line = np.array([dash_point, mid_point * 2 - dash_point])

    offset_normal = np.array([dash_point[1] - mid_point[1], mid_point[0] - dash_point[0]])
    offset_normal = offset_normal / la.norm(offset_normal)
    offset = 0.2
    offset_vector = offset_normal * offset
    # Write the rectangle formed by the dash line and the offset line
    dash_region = np.array([dash_line[0] + offset_vector, dash_line[0] - offset_vector, dash_line[1] - offset_vector, dash_line[1] + offset_vector])
    dash_region = np.concatenate((dash_region, np.zeros((len(dash_region), 1))), axis=1)

    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    curve_edges = np.array(edges) + 1
    return boundary_vxs, boundary_lines, dash_region, curve_edges

def get_dash(h, avg_len_boundary, avg_len_embeddings, dash_point, use_region = False):
    if (use_region):
        boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges = get_dash_region(h, dash_point)
    else:
        boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges = get_dash_line(h, dash_point)
    if (boundary_vxs is None):
        return None, None, None

    # return boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges
    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, transformed_cosine_curve, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data

def get_single_dash(h, avg_len_boundary, avg_len_embeddings, dash_point, use_region = False, return_line_segments = False):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    dash_line = np.array([dash_point, mid_point * 2 - dash_point])

    dash_line = np.concatenate((dash_line, np.zeros((len(dash_line), 1))), axis=1)

    edges = [[0, 1]]
    curve_edges = np.array(edges) + 1

    if return_line_segments:
        return boundary_vxs, boundary_lines, dash_line, curve_edges
    
    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, dash_line, curve_edges)


    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data


def get_double_zigzag_dash(h, avg_len_boundary, avg_len_embeddings, dash_point, use_region = False):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, h / 4])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    dash_line = np.array([dash_point, mid_point * 2 - dash_point])
    
    second_dash = np.array(dash_line)
    second_dash[:, 1] = - second_dash[:, 1]

    dash_line = np.concatenate((dash_line, second_dash), axis=0)


    from shapely.geometry import LineString

    def line_segment_intersection(segment1, segment2):
        line1 = LineString(segment1)
        line2 = LineString(segment2)
        intersection = line1.intersection(line2)

        if intersection.is_empty:
            return None
        else:
            return intersection.x, intersection.y


    # Define the two dashes
    dash1 = dash_line[:2]
    dash2 = dash_line[2:4]

    # Check if the dashes intersect
    intersection = line_segment_intersection(dash1, dash2)
    if intersection is not None:
        # If they intersect, clip them at the intersection point
        dash1[1] = intersection
        dash2[1] = intersection
        
    dash_line = np.concatenate((dash_line, np.zeros((len(dash_line), 1))), axis=1)

    edges = [[0, 1], [2, 3]]
    curve_edges = np.array(edges) + 1

    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, dash_line, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data

def get_connected_double_zigzag_dash(h, avg_len_boundary, avg_len_embeddings, dash_point, use_region = False):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, h / 4])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    dash_line = np.array([dash_point, mid_point * 2 - dash_point])
    
    second_dash = np.array(dash_line)
    second_dash[:, 1] = - second_dash[:, 1]

    dash_line = np.concatenate((dash_line, second_dash), axis=0)


    from shapely.geometry import LineString

    def line_segment_intersection(segment1, segment2):
        line1 = LineString(segment1)
        line2 = LineString(segment2)
        intersection = line1.intersection(line2)

        if intersection.is_empty:
            return None
        else:
            return intersection.x, intersection.y


    # Define the two dashes
    dash1 = dash_line[:2]
    dash2 = dash_line[2:4]

    # Check if the dashes intersect
    intersection = line_segment_intersection(dash1, dash2)
    if intersection is not None:
        # If they intersect, clip them at the intersection point
        dash1[1] = intersection
        dash2[1] = intersection
        dash_line = np.array([dash1[0], dash1[1], dash2[0]])
        edges = [[0, 1], [1, 2]]
    
    else:
        edges = [[0, 1],[1, 3], [3, 2]]    

    dash_line = np.concatenate((dash_line, np.zeros((len(dash_line), 1))), axis=1)

    curve_edges = np.array(edges) + 1

    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, dash_line, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data


import gmsh
import numpy as np

def get_pattern_with_rectangle_holes(h, ih, iw, avg_len):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add('square_with_rectangle_hole')

    # Create the main square
    main_square = gmsh.model.occ.addRectangle(0, 0, 0, h, h)

    # Calculate the center of the main square
    center_x = h / 2
    center_y = h / 2

    # Calculate the top left corner of the hole
    hole_x = center_x - iw / 2
    hole_y = center_y - ih / 2

    # Create the hole
    hole = gmsh.model.occ.addRectangle(hole_x, hole_y, 0, iw, ih)

    # Cut the hole from the main square
    _, removed = gmsh.model.occ.cut([(2, main_square)], [(2, hole)])

    # Synchronize the model (required before meshing)
    gmsh.model.occ.synchronize()

    # Generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeMin", avg_len)
    gmsh.option.setNumber("Mesh.MeshSizeMax", avg_len)
    gmsh.model.mesh.generate(2)

    
    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(2)

    v = node_coords.reshape((len(node_tags),3))
    f = mesher_helper.order_faces(v, elements_node_tags.reshape((len(element_tags),3)))
    m = MeshFEM.Mesh(v, np.array(f) - 1)

    # Finalize Gmsh
    gmsh.finalize()

    vertices = m.vertices()
    boundary_index = m.boundaryVertices()
    fusing_data = np.full(len(vertices), False)
    for vx in boundary_index:
        vx_coords = vertices[vx]
        if np.abs(vx_coords[0]) < 1e-2 or np.abs(vx_coords[0] - 5) < 1e-2:
            continue
        if np.abs(vx_coords[1]) < 1e-2 or np.abs(vx_coords[1] - 5) < 1e-2:
            continue
        fusing_data[vx] = True
    
    return m, fusing_data

def get_pattern_with_circular_holes(h, r, avg_len):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add('square_with_circular_hole')

    # Set the maximum length of the elements
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", avg_len)

    # Create the main square
    main_square = gmsh.model.occ.addRectangle(0, 0, 0, h, h)

    # Calculate the center of the main square
    center_x = h / 2
    center_y = h / 2

    # Create the hole
    hole = gmsh.model.occ.addDisk(center_x, center_y, 0, r, r)

    # Cut the hole from the main square
    _, removed = gmsh.model.occ.cut([(2, main_square)], [(2, hole)])

    # Synchronize the model (required before meshing)
    gmsh.model.occ.synchronize()

    # Generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeMin", avg_len)
    gmsh.option.setNumber("Mesh.MeshSizeMax", avg_len)
    gmsh.model.mesh.generate(2)
    
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(2)

    v = node_coords.reshape((len(node_tags),3))
    f = mesher_helper.order_faces(v, elements_node_tags.reshape((len(element_tags),3)))
    m = MeshFEM.Mesh(v, np.array(f) - 1)

    # Finalize Gmsh
    gmsh.finalize()

    vertices = m.vertices()
    boundary_index = m.boundaryVertices()
    fusing_data = np.full(len(vertices), False)
    for vx in boundary_index:
        vx_coords = vertices[vx]
        distance_to_center = np.sqrt((vx_coords[0] - center_x) ** 2 + (vx_coords[1] - center_y) ** 2)
        if np.abs(distance_to_center - r) < 1e-2:
            fusing_data[vx] = True
    
    return m, fusing_data


def get_pattern_with_elliptic_holes(h, rw, rh, avg_len, angle = 0):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add('square_with_circular_hole')

    # Set the maximum length of the elements
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", avg_len)

    # Create the main square
    main_square = gmsh.model.occ.addRectangle(0, 0, 0, h, h)

    # Calculate the center of the main square
    center_x = h / 2
    center_y = h / 2

    # Create the hole
    if (rw < rh):
        hole = gmsh.model.occ.addDisk(center_x, center_y, 0, rh, rw)
        gmsh.model.occ.rotate([(2, hole)], center_x, center_y, 0, 0, 0, 1, np.pi/2)
    else:
        hole = gmsh.model.occ.addDisk(center_x, center_y, 0, rw, rh)
    
    gmsh.model.occ.rotate([(2, hole)], center_x, center_y, 0, 0, 0, 1, np.deg2rad(angle))

    # Cut the hole from the main square
    _, removed = gmsh.model.occ.cut([(2, main_square)], [(2, hole)])

    # Synchronize the model (required before meshing)
    gmsh.model.occ.synchronize()

    # Generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeMin", avg_len)
    gmsh.option.setNumber("Mesh.MeshSizeMax", avg_len)
    gmsh.model.mesh.generate(2)
    
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(2)

    v = node_coords.reshape((len(node_tags),3))
    f = mesher_helper.order_faces(v, elements_node_tags.reshape((len(element_tags),3)))
    m = MeshFEM.Mesh(v, np.array(f) - 1)

    # Finalize Gmsh
    gmsh.finalize()

    vertices = m.vertices()
    boundary_index = m.boundaryVertices()
    fusing_data = np.full(len(vertices), False)
    for vx in boundary_index:
        vx_coords = vertices[vx]
        # rotate point back first. 
        vx_coords = np.array([vx_coords[0] - center_x, vx_coords[1] - center_y])
        vx_coords = np.array([vx_coords[0] * np.cos(np.deg2rad(-angle)) - vx_coords[1] * np.sin(np.deg2rad(-angle)), vx_coords[0] * np.sin(np.deg2rad(-angle)) + vx_coords[1] * np.cos(np.deg2rad(-angle))])

        ellipse_value = ((vx_coords[0]) ** 2 / rw ** 2) + ((vx_coords[1]) ** 2 / rh ** 2)
        if np.abs(ellipse_value - 1) < 1e-2:
            fusing_data[vx] = True
    
    return m, fusing_data


def get_pattern_with_elliptic_fused(h, rw, rh, avg_len, angle = 0):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add('square_with_circular_hole')

    # Set the maximum length of the elements
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", avg_len)

    # Create the main square
    main_square = gmsh.model.occ.addRectangle(0, 0, 0, h, h)

    # Calculate the center of the main square
    center_x = h / 2
    center_y = h / 2

    # Create the elliptic curve
    if (rw < rh):
        curve = gmsh.model.occ.addEllipse(center_x, center_y, 0, rh, rw)
        gmsh.model.occ.rotate([(1, curve)], center_x, center_y, 0, 0, 0, 1, np.pi/2)
    else:
        curve = gmsh.model.occ.addDisk(center_x, center_y, 0, rw, rh)
    
    gmsh.model.occ.rotate([(1, curve)], center_x, center_y, 0, 0, 0, 1, np.deg2rad(angle))

    out, _ = gmsh.model.occ.fragment([(2, main_square)], [(1, curve)])
    gmsh.model.occ.synchronize()

    # Generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeMin", avg_len)
    gmsh.option.setNumber("Mesh.MeshSizeMax", avg_len)
    gmsh.model.mesh.generate(2)
    
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(2)

    v = node_coords.reshape((len(node_tags),3))
    f = mesher_helper.order_faces(v, elements_node_tags.reshape((len(element_tags),3)))
    m = MeshFEM.Mesh(v, np.array(f) - 1)

    # Finalize Gmsh
    gmsh.finalize()

    vertices = m.vertices()
    fusing_data = np.full(len(vertices), False)
    for vx in range(len(vertices)):
        vx_coords = vertices[vx]
        # rotate point back first. 
        vx_coords = np.array([vx_coords[0] - center_x, vx_coords[1] - center_y])
        vx_coords = np.array([vx_coords[0] * np.cos(np.deg2rad(-angle)) - vx_coords[1] * np.sin(np.deg2rad(-angle)), vx_coords[0] * np.sin(np.deg2rad(-angle)) + vx_coords[1] * np.cos(np.deg2rad(-angle))])

        ellipse_value = ((vx_coords[0]) ** 2 / rw ** 2) + ((vx_coords[1]) ** 2 / rh ** 2)
        if ellipse_value < 1 + 1e-2:
            fusing_data[vx] = True
    
    return m, fusing_data


def get_pattern_with_hexagonal_holes(a, d, avg_len):
    '''
        Args:
            a: edge length of the tessellation hexagon
            d: offset for the tubes
    '''
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add('elliptic hole tessellation')

    # Set the maximum length of the elements
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", avg_len)

    dim = 2
    mid_height = np.sqrt(3) / 2 * a
    offset_height = d / np.sqrt(3)
    boundary_vxs = np.array([[0, mid_height - d, 0.0], [a / 2 - offset_height, mid_height - d, 0.0], [a - 2 * offset_height, 0, 0], [1.5 * a, 0, 0], [1.5 * a, d, 0], [a + offset_height, d, 0], [a / 2 + 2 * offset_height, mid_height, 0], [0, mid_height, 0]])
    
    boundary_point_tag = []
    for vx in boundary_vxs:
        boundary_point_tag.append(gmsh.model.occ.addPoint(*vx, meshSize=avg_len))

    line_tags = []
    for i in range(len(boundary_point_tag)):
        line_tag = gmsh.model.occ.addLine(boundary_point_tag[i], boundary_point_tag[(i+1)%len(boundary_point_tag)])
        line_tags.append(line_tag)

    wire_tag = gmsh.model.occ.addWire(line_tags)

    # All these curves should have the same orientation.
    srf_tag = gmsh.model.occ.addPlaneSurface([wire_tag])

    gmsh.model.occ.synchronize()

    mesh_size = avg_len
    # TODO: these functions might need to be updated to produce uniform mesh.
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.model.mesh.generate(dim)

    # Add these lines to fix the mesh
    gmsh.model.mesh.removeDuplicateNodes()

    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(dim)

    v = node_coords.reshape((len(node_tags),3))
    f = mesher_helper.order_faces(v, elements_node_tags.reshape((len(element_tags),3)))
    gmsh.finalize()

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return v, f

def get_bistable_pattern(h, dh, dw, frequency, avg_len):
    # Write two dash lines: the first one is vertical at the center of the unit cell with length h. The second one is split in half, attached horizontally to the boundary with length w. Check that w is smaller than h.
    
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    # First dash
    dash_point = np.array([dh * 0.75, dh * 0.75])
    dash_line = []
    first_dash_line = np.array([dash_point, mid_point * 2 - dash_point])
    dash_line.extend(first_dash_line)
    # Second dash
    height =  h / 2 - h / 10
    second_dash_line = np.array([[-w/2,height], [-w/2 + dw,height]])
    third_dash_line = np.array([[height, -h/ 2], [height,-h / 2 + dw]])
    dash_line.extend(second_dash_line)
    dash_line.extend(third_dash_line)

    dash_line = np.concatenate((dash_line, np.zeros((len(dash_line), 1))), axis=1)

    edges = [[2 * i, 2 * i + 1] for i in range(len(dash_line) // 2)]
    curve_edges = np.array(edges) + 1

    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len, avg_len, None, boundary_vxs, boundary_lines, dash_line, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data

def get_biaxial_pattern(h, dh, dw, frequency, avg_len):
    
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    # Vertical dash
    dash_point = np.array([0, dh * 0.5])
    dash_line = []
    first_dash_line = np.array([dash_point, mid_point * 2 - dash_point])
    dash_line.extend(first_dash_line)
    # Horizontal dash
    height = h / 4
    second_dash_line = np.array([[-w/2,height], [-w/2 + dw / 2,height]])
    third_dash_line = np.array([[w/2,height], [w/2 - dw / 2,height]])
    dash_line.extend(second_dash_line)
    dash_line.extend(third_dash_line)

    height = - h / 4
    second_dash_line = np.array([[-w/2,height], [-w/2 + dw / 2,height]])
    third_dash_line = np.array([[w/2,height], [w/2 - dw / 2,height]])
    dash_line.extend(second_dash_line)
    dash_line.extend(third_dash_line)

    dash_line = np.concatenate((dash_line, np.zeros((len(dash_line), 1))), axis=1)

    edges = [[2 * i, 2 * i + 1] for i in range(len(dash_line) // 2)]
    curve_edges = np.array(edges) + 1

    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len, avg_len, None, boundary_vxs, boundary_lines, dash_line, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data
        
def get_spiral(h, start_angle, alpha, target_radius, avg_len_boundary, avg_len_embeddings):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    mid_point = np.array([0, 0])
    alpha_rad = np.deg2rad(alpha)
    b = np.tan(alpha_rad)

    start_angle = np.deg2rad(start_angle)

    sqrtTerm = np.sqrt(1 + 1 / (b * b))
    rForTheta      = lambda th: np.exp(b * th)
    thetaForR      = lambda r: np.log(r) / b
    thetaForArclen = lambda s: (1.0 / b) * (np.log(s + sqrtTerm) - np.log((sqrtTerm)))
    arclenForTheta = lambda th: sqrtTerm * (np.exp(b * th) - 1.0)

    def thetasForRadiusInterval(rmin, rmax, scale):
        smin, smax = map(lambda r: arclenForTheta(thetaForR(r)), [rmin, rmax])
        nsubdiv = int(np.round((smax - smin) / avg_len_embeddings * scale))
        return thetaForArclen(np.linspace(smin, smax, nsubdiv))

    rmin, rmax = np.exp(b * start_angle), np.exp(b * (start_angle + 2.5 * np.pi))
    roppo = np.exp(b * (start_angle + 1.5 * np.pi))
    scale = target_radius / (rmax + roppo)

    thetas = thetasForRadiusInterval(rmin, rmax, scale)

    r = np.exp(np.tan(alpha_rad) * thetas)
    pts = (np.array([np.cos(thetas), np.sin(thetas), np.zeros(len(r))]) * r).transpose()
    edges = np.array([[i, i + 1] for i in range(len(pts) - 1)])

    shift = np.array([((rmax - roppo) / 2 * np.cos(start_angle + 2.5 * np.pi)), ((rmax - roppo) / 2 * np.sin(start_angle + 2.5 * np.pi)), 0])
    pts = pts - shift
    pts = pts * scale

    for pt in pts:
        _, new_pt_out = periodic_box_transform(boundary_vxs, pt)
        if new_pt_out:
            return None, None, None
        
    curve_edges = np.array(edges) + 1
    # return boundary_vxs, boundary_lines, pts, curve_edges
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, pts, curve_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_circular_arc(h, radius, start, end, avg_len_boundary, avg_len_embeddings):
    w = h
    boundary_vxs = np.array([[0, 0., 0.], [w, 0., 0.], [w, h, 0.], [0, h, 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    length = 2 * np.pi * radius / 4
    num_seg = int(np.round(length / avg_len_embeddings))
    base_arc_points = np.array([[radius * np.cos(x * np.pi / 2 + np.pi), radius * np.sin(x * np.pi / 2 + np.pi)] for x in np.linspace(start, end, num_seg + 1)]) + np.array([w, h])
    base_arc_points = np.concatenate((base_arc_points, np.zeros((len(base_arc_points), 1))), axis=1)

    arc_edges = np.array([[i, i + 1] for i in range(len(base_arc_points) - 1)]) + 1
    # return boundary_vxs, boundary_lines, base_arc_points, arc_edges
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, base_arc_points, arc_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_aeromorph(w, h, x, y, avg_len_boundary, avg_len_embeddings):
    boundary_vxs = np.array([[0, 0., 0.], [w, 0., 0.], [w, h, 0.], [0, h, 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    center = np.array([w/ 2, h / 2])
    points = np.array([[0, y], [x, 0], [0, -y], [-x, 0]]) + center
    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)

    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    # return boundary_vxs, boundary_lines, base_arc_points, arc_edges
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    for i in range(len(v)):
        abs_x = np.abs(v[i][0] - w / 2)
        abs_y = np.abs(v[i][1] - h / 2)
        # Check whether the point is below the line passing through (x, 0) and (0, y)
        if (abs_x * y + abs_y * x <= x * y):
            fusing_data[i] = True
    
    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data


def get_heart(h, x_len, y_len, avg_len_boundary, avg_len_embeddings):
    w = h
    w = h
    boundary_vxs = np.array([[0, 0., 0.], [w, 0., 0.], [w, h, 0.], [0, h, 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    # Creating equally spaced 100 data in range 0 to 2*pi
    theta = np.linspace(0.1, 2 * np.pi * 0.9, 50)

    # Generating x and y data
    x = x_len * ( np.sin(theta) ** 3 )
    y = y_len * np.cos(theta) - y_len / 16 * 10* np.cos(2*theta) - y_len / 16 *2 * np.cos(3*theta) - y_len / 16 * 1 *np.cos(4*theta)
    base_arc_points =  np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    transformation = np.array([[np.cos(np.pi / 4 * 3), - np.sin(np.pi / 4 * 3)], [np.sin(np.pi / 4 * 3), np.cos(np.pi / 4 * 3)]])
    base_arc_points = (transformation @ base_arc_points.T).T

    base_arc_points = np.concatenate((base_arc_points, np.zeros((len(base_arc_points), 1))), axis=1) + np.array([h / 2., h / 2., 0])
    arc_edges = np.array([[i, i + 1] for i in range(len(base_arc_points) - 1)]) + 1

    # return boundary_vxs, boundary_lines, base_arc_points, arc_edges
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, base_arc_points, arc_edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_three_star(h, avg_len_boundary, avg_len_embeddings, dash_point):
    w = h
    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    rotation = np.array([[np.cos(np.pi / 3 * 2), -np.sin(np.pi / 3 * 2)], [np.sin(np.pi / 3 * 2), np.cos(np.pi / 3 * 2)]])
    points = np.array([mid_point,dash_point,rotation @ dash_point, rotation @(rotation @ dash_point)])
    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)

    edges = np.array([[0, 1], [0, 2], [0, 3]]) + 1

    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_star(a, radius, angle, num_spikes = 3, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
    # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    center_points = np.array([[a / 2, a / 2]])

    rotation = np.array([[np.cos(np.pi / num_spikes * 2), -np.sin(np.pi / num_spikes * 2)], [np.sin(np.pi / num_spikes * 2), np.cos(np.pi / num_spikes * 2)]])

    dash_point = np.array([0, radius])
    default_rotation = np.array([[np.cos(angle / 180 * np.pi), -np.sin(angle / 180 * np.pi)], [np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]])
    dash_point = default_rotation @ dash_point
    star_points = [[0, 0]]
    for i in range(num_spikes):
        point = np.array(dash_point)
        for j in range(i):
            point = rotation @ point
        star_points.append(point)
    star_points = np.array(star_points)
    points = []
    for i in range(len(center_points)):
        points.extend(center_points[i] + star_points)

    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)
    points = points.tolist()

    edges = []
    for i in range(len(center_points)):
        edges.extend(np.array([[0, i + 1] for i in range(num_spikes)]) + 1 + (num_spikes + 1) * i)

    # for i in range(len(center_points)):
    #     # Assume that the star point is not out.
    #     for j in range(num_spikes):
    #         transformed_point, new_pt_out = periodic_box_transform(boundary_vxs, points[i * (num_spikes + 1) + j + 1])
    #         if (new_pt_out):
    #             # Replace i * 4 + j + 1 th point with intersection with between the edge with the boundary.
    #             # The edge is between i * (num_spikes + 1) and i * (num_spikes + 1) + j + 1
    #             intersection_point, box_edge = get_intersection_point(boundary_vxs, np.array([points[i * (num_spikes + 1)], points[i * (num_spikes + 1) + j + 1]]))
    #             if (intersection_point is None):
    #                 print("Error: intersection point is None")
    #                 return points, edges
    #             points[i * (num_spikes + 1) + j + 1] = (intersection_point - points[i * (num_spikes + 1)]) * 0.95 + points[i * (num_spikes + 1)]
    #             points.append(transformed_point)
    #             points.append(periodic_box_transform(boundary_vxs,  (intersection_point - points[i * (num_spikes + 1)]) * 1.05 + points[i * (num_spikes + 1)])[0])
    #             edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

    points = np.array(points)
    edges = np.array(edges)
    # return points, edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data
def get_three_star_hex(a, radius, num_spikes = 3, avg_len_boundary = 0.1, avg_len_embeddings = 0.1, return_line_segments = False):
    # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[0, 0, 0], [4 * a, 0, 0], [4 * a, 6 / np.sqrt(3) * a, 0], [0, 6 / np.sqrt(3) * a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    center_points = np.array([[a / 2., a / np.sqrt(3)], [a / 2 + 2 * a, a / np.sqrt(3)], [a / 2. + a, 4 * a / np.sqrt(3)], [a / 2 + 3 * a, 4 * a / np.sqrt(3)]])

    rotation = np.array([[np.cos(np.pi / num_spikes * 2), -np.sin(np.pi / num_spikes * 2)], [np.sin(np.pi / num_spikes * 2), np.cos(np.pi / num_spikes * 2)]])

    dash_point = [0, a / np.sqrt(3) * radius]
    star_points = [[0, 0]]
    for i in range(num_spikes):
        point = np.array(dash_point)
        for j in range(i):
            point = rotation @ point
        star_points.append(point)
    star_points = np.array(star_points)
    points = []
    for i in range(len(center_points)):
        points.extend(center_points[i] + star_points)

    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)
    points = points.tolist()

    edges = []
    for i in range(len(center_points)):
        edges.extend(np.array([[0, i + 1] for i in range(num_spikes)]) + 1 + (num_spikes + 1) * i)

    for i in range(len(center_points)):
        # Assume that the star point is not out.
        for j in range(num_spikes):
            transformed_point, new_pt_out = periodic_box_transform(boundary_vxs, points[i * (num_spikes + 1) + j + 1])
            if (new_pt_out):
                # Replace i * 4 + j + 1 th point with intersection with between the edge with the boundary.
                # The edge is between i * (num_spikes + 1) and i * (num_spikes + 1) + j + 1
                intersection_point, box_edge = get_intersection_point(boundary_vxs, np.array([points[i * (num_spikes + 1)], points[i * (num_spikes + 1) + j + 1]]))
                if (intersection_point is None):
                    print("Error: intersection point is None")
                    return points, edges
                points[i * (num_spikes + 1) + j + 1] = (intersection_point - points[i * (num_spikes + 1)]) * 0.95 + points[i * (num_spikes + 1)]
                points.append(transformed_point)
                points.append(periodic_box_transform(boundary_vxs,  (intersection_point - points[i * (num_spikes + 1)]) * 1.05 + points[i * (num_spikes + 1)])[0])
                edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

    points = np.array(points)
    edges = np.array(edges)
    if return_line_segments:
        return points, edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_three_star_hex_dash(a, radius, num_spikes = 3, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
    # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[0, 0, 0], [4 * a, 0, 0], [4 * a, 6 / np.sqrt(3) * a, 0], [0, 6 / np.sqrt(3) * a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    center_points = np.array([[a / 2., a / np.sqrt(3)], [a / 2 + 2 * a, a / np.sqrt(3)], [a / 2. + a, 4 * a / np.sqrt(3)], [a / 2 + 3 * a, 4 * a / np.sqrt(3)]])

    rotation = np.array([[np.cos(np.pi / num_spikes * 2), -np.sin(np.pi / num_spikes * 2)], [np.sin(np.pi / num_spikes * 2), np.cos(np.pi / num_spikes * 2)]])

    dash_point = [0, a / np.sqrt(3) * radius]
    # star_points = [[0, 0]]
    star_points = []
    for i in range(num_spikes):
        point = np.array(dash_point)
        for j in range(i):
            point = rotation @ point
        star_points.append(point * 0.5)
        star_points.append(point)
    star_points = np.array(star_points)
    points = []
    for i in range(len(center_points)):
        points.extend(center_points[i] + star_points)

    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)
    points = points.tolist()

    edges = []
    for i in range(len(center_points)):
        # Each star will have num_spikes * 2 points and num_spikes edges
        star_edges = np.array([[i * num_spikes * 2 + j * 2, i * num_spikes * 2 + j * 2 + 1] for j in range(num_spikes)])
        edges.extend(star_edges)

    points = np.array(points)
    edges = np.array(edges) + 1
    # return points, edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data


def get_three_star_tri(a, radius, avg_len_boundary, avg_len_embeddings):
    # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[0, 0, 0], [4 * a, 0, 0], [4 * a, 6 / np.sqrt(3) * a, 0], [0, 6 / np.sqrt(3) * a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    center_points = np.array([[a / 2., a / np.sqrt(3)], [a / 2 + 2 * a, a / np.sqrt(3)], [a / 2. + a, 4 * a / np.sqrt(3)], [a / 2 + 3 * a, 4 * a / np.sqrt(3)]])


    center_points_2 = np.array([4 * a, 0]) - center_points
    center_points_2[:, 1] = center_points[:, 1] - a / np.sqrt(3)
    center_points = np.concatenate((center_points, center_points_2))

    center_points_3 = np.array(center_points_2)
    center_points_3[:, 1] = center_points_2[:, 1] + 2 * a / np.sqrt(3)

    center_points = np.concatenate((center_points, center_points_3))

    center_points += np.array([0, a / np.sqrt(3) / 4])

    rotation = np.array([[np.cos(np.pi / 3 * 2), -np.sin(np.pi / 3 * 2)], [np.sin(np.pi / 3 * 2), np.cos(np.pi / 3 * 2)]])

    dash_point = [0, a / np.sqrt(3) * radius]
    star_points = np.array([[0, 0],dash_point,rotation @ dash_point, rotation @(rotation @ dash_point)])
    points = []
    for i in range(len(center_points)):
        points.extend(center_points[i] + star_points)

    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)
    points = points.tolist()

    edges = []
    for i in range(len(center_points)):
        edges.extend(np.array([[0, 1], [0, 2], [0, 3]]) + 1 + 4 * i)

    for i in range(len(center_points)):
        # Assume that the star point is not out.
        for j in range(3):
            transformed_point, new_pt_out = periodic_box_transform(boundary_vxs, points[i * 4 + j + 1])
            if (new_pt_out):
                # Replace i * 4 + j + 1 th point with intersection with between the edge with the boundary.
                # The edge is between i * 4 and i * 4 + j + 1
                intersection_point, box_edge = get_intersection_point(boundary_vxs, np.array([points[i * 4], points[i * 4 + j + 1]]))
                if (intersection_point is None):
                    print("Error: intersection point is None")
                    return points, edges
                points[i * 4 + j + 1] = (intersection_point - points[i * 4]) * 0.9 + points[i * 4]
                points.append(transformed_point)
                points.append(periodic_box_transform(boundary_vxs,  (intersection_point - points[i * 4]) * 1.1 + points[i * 4])[0])
                edges.append([len(points) - 2 + 1, len(points) - 1 + 1])



    points = np.array(points)
    edges = np.array(edges)

    # return points, edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

import random
def get_random_grid_dots(avg_len_boundary, avg_len_embeddings, num_grid, num_dots):
    dots = []
    h = 5.
    w = h
    margin = h / num_grid / 2

    xs = np.linspace(-h / 2. + margin, h / 2. - margin, num_grid)
    ys = np.linspace(-h / 2. + margin, h / 2. - margin, num_grid)
    dots = np.array([[x, y, 0.] for x in xs for y in ys])
    dot_selector = random.sample(range(len(dots)), num_dots)
    dots = dots[dot_selector]

    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    mid_point = np.array([0, 0])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary,avg_len_embeddings, dots, boundary_vxs, boundary_lines, [], [])
    m = MeshFEM.Mesh(v, np.array(f) - 1)
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)
    return ipu, m, fusing_data, dots

from shapely.geometry import LineString
from shapely.ops import unary_union

def pipes_intersect(pipe1, pipe2):
    # Convert the pipes to LineString objects
    line1 = LineString(pipe1)
    line2 = LineString(pipe2)

    # Check if the lines intersect
    return line1.intersects(line2)

def clip_pipe(pipe, margin):
    # Convert the pipe to a LineString object
    line = LineString(pipe)

    # Calculate the length to clip off each end of the line
    clip_length = margin / 2

    # Clip the line
    clipped_line = line.parallel_offset(clip_length, 'left').parallel_offset(clip_length, 'right')

    # Convert the clipped line back to a pipe
    clipped_pipe = [list(clipped_line.coords[0]), list(clipped_line.coords[-1])]

    return clipped_pipe

def remove_intersecting_pipes(pipes, margin):
    # Convert the pipes to LineString objects
    lines = [LineString(pipe) for pipe in pipes]

    # Combine the lines into a single MultiLineString object
    multi_line = unary_union(lines)

    # Clip the lines
    clipped_lines = [clip_pipe(list(line.coords), margin) for line in multi_line]

    return clipped_lines

def get_random_grid_pipes(avg_len_boundary, avg_len_embeddings, num_grid, num_pipes, return_line_segments = False):
    pipes = []
    h = 5.
    w = h
    margin = h / num_grid / 2

    pipe_length = h / num_grid

    xs = np.linspace(-h / 2. + margin, h / 2. - margin, num_grid)
    ys = np.linspace(-h / 2. + margin, h / 2. - margin, num_grid)
    grid_points = np.array([[x, y, 0.] for x in xs for y in ys])

    # Randomly select grid points to be the start of a pipe
    pipe_starts = random.sample(list(grid_points), num_pipes)

    vertices = []
    edges = []

    for i, start in enumerate(pipe_starts):
        # Randomly choose the orientation of the pipe
        orientation = random.choice(['vertical', 'horizontal'])

        if orientation == 'vertical':
            end = [start[0], start[1] + pipe_length, 0.]
        else:  # orientation == 'horizontal'
            end = [start[0] + pipe_length, start[1], 0.]

        # Add the vertices of the pipe to the vertices list
        vertices.extend([start, end])

        # Add the edge of the pipe to the edges list
        edges.append([2*i, 2*i+1])

    # Convert the vertices and edges lists to numpy arrays
    vertices = np.array(vertices)
    edges = np.array(edges) + 1  # The +1 is to match the 1-based indexing used in the meshing function

    boundary_vxs = np.array([[-w/2., -h/2., 0.], [w/2., -h/2., 0.], [w/2., h/2., 0.], [-w/2., h/2., 0.]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    if return_line_segments:
        return vertices, edges, boundary_vxs, boundary_lines
    
    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary,avg_len_embeddings, None, boundary_vxs, boundary_lines, vertices, edges)
    m = MeshFEM.Mesh(v, np.array(f) - 1)
    return m, fusing_data, pipes

from scipy.spatial import Voronoi

def get_voronoi_pipes(points, boundary, shrink_percentage):

    # Calculate the width and height of the boundary
    width = boundary[1][0] - boundary[0][0]
    height = boundary[1][1] - boundary[0][1]

    # Duplicate the points in the neighboring regions of the boundary
    points_above = points + [0, height]
    points_below = points - [0, height]
    points_right = points + [width, 0]
    points_left = points - [width, 0]
    points = np.concatenate((points, points_above, points_below, points_right, points_left))

    # Generate Voronoi diagram
    vor = Voronoi(points)

    vertices = []
    edges = []

    # Calculate if a point is outside the boundary
    def is_outside(point, boundary):
        return (point < boundary[0]).any() or (point > boundary[1]).any()

    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # Ignore ridges that go to infinity
            start, end = vor.vertices[ridge]
            start = np.append(start, 0.)  # Add z-coordinate
            end = np.append(end, 0.)  # Add z-coordinate

            # Calculate the midpoint of the edge
            midpoint = (start + end) / 2

            # Move the start and end vertices towards the midpoint by the shrink percentage
            start = start + shrink_percentage * (midpoint - start)
            end = end + shrink_percentage * (midpoint - end)

            # Only remove the edge if both vertices are outside the boundary
            if is_outside(start[:2], boundary) and is_outside(end[:2], boundary):
                continue

            vertices.extend([start, end])

            # Add the edge of the ridge to the edges list
            i = len(vertices) // 2 - 1
            edges.append([2*i, 2*i+1])

    # Convert the vertices and edges lists to numpy arrays
    vertices = np.array(vertices)
    edges = np.array(edges) + 1  # The +1 is to match the 1-based indexing used in the meshing function

    return vertices, edges

def generate_voronoi_mesh(points, avg_len_boundary=0.1, avg_len_embeddings=0.1, shrink_percentage = 0.2, return_line_segments=False):
    # Define the boundary vertices and lines
    boundary_vxs = np.array([[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1
    boundary = np.array([[-2.5, -2.5], [2.5, 2.5]])  # Define the boundary for the Voronoi diagram

    # Call the get_voronoi_pipes function with the boundary
    vertices, edges = get_voronoi_pipes(points, boundary, shrink_percentage)
    if return_line_segments:
        return vertices, edges, boundary_vxs, boundary_lines
    # Generate the mesh from the vertices and edges
    v, f, fusing_data = generate_mesh_from_embeddings_array_input_allow_boundary(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, vertices, edges)
    m = MeshFEM.Mesh(v, np.array(f) - 1)

    return m, fusing_data, vertices, edges

def get_log_star(a, radius, alpha=70, edgeLength=0.02, minDist=0.1, margin=.0, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
     # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1
    points, edges = parametric_pillows.logSpiralPlot(alpha = alpha, edgeLength = edgeLength, minDist = minDist, margin = margin, use_boundary = False)    
    points = np.concatenate((points, np.zeros((len(points), 1))), axis=1)
    points *= radius
    edges = np.array(edges) + 1
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_bidirectional_pipe(a, num_pipes, margin, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
     # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    points = []
    edges = []
    for pos in np.linspace(margin * 3, a - margin * 3, num_pipes):
        points.append([pos, margin, 0.])
        points.append([pos, np.min([pos - margin, a - pos - margin]), 0.])
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

        points.append([pos, a - margin, 0])
        points.append([pos, a -  np.min([pos - margin, a - pos - margin]), 0])
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

        points.append([ margin, pos, 0])
        points.append([  np.min([pos - margin, a - pos - margin]), pos, 0])
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

        points.append([ a - margin, pos, 0])
        points.append([ a -  np.min([pos - margin, a - pos - margin]), pos, 0])
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

    edges = np.array(edges) 
    points = np.array(points)
    # return points, edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data



def get_center_pipe(a, num_pipes, margin, radius, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
     # write boundary vxs covering the box with width 4 * a and height 6 / sqrt(3) * a, starting from [0, 0]
    boundary_vxs = np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1
    center = np.array([a / 2, a / 2, 0])
    points = []
    edges = []
    for pos in np.linspace(margin * 3, a - margin * 3, num_pipes):
        points.append([pos, margin, 0.])
        ray = np.array(points[-1]) - center
        ray /= la.norm(ray)
        points.append(ray * radius + center)
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

        points.append([pos, a - margin, 0])
        ray = np.array(points[-1]) - center
        ray /= la.norm(ray)
        points.append(ray * radius + center)
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

        points.append([ margin, pos, 0])
        ray = np.array(points[-1]) - center
        ray /= la.norm(ray)
        points.append(ray * radius + center)
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

        points.append([ a - margin, pos, 0])    
        ray = np.array(points[-1]) - center
        ray /= la.norm(ray)
        points.append(ray * radius + center)
        edges.append([len(points) - 2 + 1, len(points) - 1 + 1])

    edges = np.array(edges) 
    points = np.array(points)
    # return points, edges, boundary_vxs, boundary_lines
    v, f, fusing_data = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, points, edges)

    m = MeshFEM.Mesh(v, np.array(f) - 1)

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusing_data, epsilon = 1e-5)

    return ipu, m, fusing_data

def get_diamond_cuts(h, a, b, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
    h = 5
    a = 1
    b = 2
    # Write the list of points describing boundary of diamond cuts.
    points = np.array([[0., 0., 0.], [h / 2 - a, 0., 0.], [h / 2, b, 0.], [h / 2 + a, 0., 0.], [h, 0., 0.], [0, h, 0], [h / 2 - a, h, 0], [h / 2, h - b, 0], [h / 2 + a, h, 0], [h, h, 0], [h/2 - b, h / 2, 0], [h / 2, h / 2 - a, 0], [h / 2 + b, h / 2, 0], [h / 2, h / 2 + a, 0]])
    edges = np.array([[4, 3], [3, 2], [2, 1], [1, 0], [0, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 4], [10, 11], [11, 12], [12, 13], [13, 10]])

    upsample_edges = []
    upsample_points = []
    resolution = 10
    for i in range(len(edges)):
        upsample_points.extend(np.linspace(points[edges[i][0]], points[edges[i][1]], resolution + 1))
        upsample_edges.extend([[i * (resolution + 1) + j, i * (resolution + 1) + j + 1] for j in range(resolution)])

    upsample_points = np.array(upsample_points)
    upsample_edges = np.array(upsample_edges)

def get_square_pillow_helper(a, b, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
    points = np.array([[0., 0., 0.], [a, 0., 0.], [a, b, 0.], [0, b, 0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    boundary_vxs = np.array([[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]])
    boundary_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + 1

    v, f, marker = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, boundary_vxs, boundary_lines, [], [])

    m = MeshFEM.Mesh(v, np.array(f) - 1)
    # Fuse boundary
    marker[m.boundaryVertices()] = True
    return m, marker

def get_square_pillow(a, b, avg_len_boundary = 0.1, avg_len_embeddings = 0.1, scale = 1):
    if a != b:                
        ipu = homogenized_inflation.InflatablePeriodicUnit(*get_square_pillow_helper(a, b, avg_len_boundary, avg_len_embeddings), epsilon = 1e-5)
    else:
        scale = scale / 5
        m, marker = get_square_pillow_helper(5, 5, avg_len_boundary, avg_len_embeddings)
        m = MeshFEM.Mesh(m.vertices() * scale, m.elements())
        ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    return ipu, m, marker


def get_four_square(h, type_index, avg_len_boundary = 0.1, avg_len_embeddings = 0.1):
    points = []
    # List the boundary points of five arrangements of four squares. 
    if type_index == 0:
        points = np.array([[0., 0, 0], [4 * h, 0, 0], [4 * h, h, 0], [0, h, 0]])
    elif type_index == 1:
        points = np.array([[0., 0, 0], [3 * h, 0, 0], [3 * h, 2 * h, 0], [2 * h, 2 * h, 0], [2 * h, h, 0], [0, h, 0]])
    elif type_index == 2:
        points = np.array([[0., 0, 0], [3 * h, 0, 0], [3 * h, h, 0], [2 * h, h, 0], [2 * h, 2 * h, 0], [h, 2 * h, 0], [h, h, 0], [0, h, 0]])
    elif type_index == 3:
        points = np.array([[0., 0, 0], [2 * h, 0, 0], [2 * h, h, 0], [3 * h, h, 0], [3 * h, 2 * h, 0], [h, 2 * h, 0], [h, h, 0], [0, h, 0]])
    elif type_index == 4:
        points = np.array([[0., 0, 0], [2 * h, 0, 0], [2 * h, 2 * h, 0], [0, 2 * h, 0]])

    edges = np.array([[i, (i + 1) % len(points)] for i in range(len(points))]) + 1

    v, f, marker = generate_mesh_from_embeddings_array_input(avg_len_boundary, avg_len_embeddings, None, points, edges, [], [])

    m = MeshFEM.Mesh(v, np.array(f) - 1)
    # Fuse boundary
    marker[m.boundaryVertices()] = True
    isheet = homogenized_inflation.InflatableSheet(m, marker)
    return isheet, m, marker