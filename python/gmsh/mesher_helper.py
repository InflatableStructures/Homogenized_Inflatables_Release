import gmsh
import sys; sys.path.append('../')
import numpy as np
import time
import igl
def generate_mesh_from_embeddings_array_input(mesh_size_at_boundary, mesh_size_at_embeddings, embed_pts, boundary_vxs, boundary_lines, embedding_vxs, embeddings_lines, gui = False, tolerance = 1e-6):
    # print("inputs ", mesh_size_at_boundary, mesh_size_at_embeddings, embed_pts, boundary_vxs, boundary_lines, embedding_vxs, embeddings_lines)
    start_time = time.time()
    # embed_pts=None
    # file_boundary = open(filename_boundary)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("mesher_from_embeddings")

    # Read file
    dim = 2
    for vx in boundary_vxs:
        last_point_tag = gmsh.model.geo.addPoint(*vx, mesh_size_at_boundary)
    for line in boundary_lines:
        gmsh.model.geo.addLine(*line)


    # Generate planar surface
    loop_tag = gmsh.model.geo.addCurveLoop(np.array(range(1,len(boundary_lines)+1)))
    srf_tag = gmsh.model.geo.addPlaneSurface([loop_tag])

    # Synchronize
    gmsh.model.geo.synchronize()

    # Build embeddings

    embed_tags = []
    temp_pts = []
    embed_edges = []
    for vx in embedding_vxs:
        gmsh.model.geo.addPoint(*vx, mesh_size_at_embeddings)
        temp_pts.append(vx)

    for line in embeddings_lines:
        embed_tags.append(gmsh.model.geo.addLine(*(last_point_tag + line)))
        embed_edges.append([temp_pts[line[0]-1], temp_pts[line[1]-1]])


    # Synchronize
    gmsh.model.geo.synchronize()

    # Add embeddings to surface
    gmsh.model.mesh.embed(dim-1, embed_tags, dim, srf_tag)

    if embed_pts is not None:
        embed_tags = []
        for vx in embed_pts:
            tag = gmsh.model.geo.addPoint(*vx, mesh_size_at_embeddings)
            embed_tags.append(tag)

        # Synchronize
        gmsh.model.geo.synchronize()

        # Add embeddings to surface
        gmsh.model.mesh.embed(dim-2, embed_tags, dim, srf_tag)

    # Generate mesh
    if mesh_size_at_boundary<mesh_size_at_embeddings: mesh_size = mesh_size_at_boundary
    else : mesh_size = mesh_size_at_embeddings
    # TODO: these functions might need to be updated to produce uniform mesh.
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.model.mesh.generate(dim)

    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(dim)

    # Get new nodes and faces
    v = node_coords.reshape((len(node_tags),3))
    f = order_faces( v, elements_node_tags.reshape((len(element_tags),3)) )

    if embed_edges != []:
        fusing_data = generate_fusing_data_from_lines(v, embed_edges, tolerance)
    else:
        fusing_data = np.array([False] * len(v))
    if embed_pts is not None:
        for p in embed_pts:
            idx = find_point_closest_point(v, p)
            fusing_data[idx] = True
    if(gui): gmsh.fltk.run()
    # End gmsh
    gmsh.finalize()
    return v, f, fusing_data

def generate_mesh_from_embeddings_array_input_allow_boundary(mesh_size_at_boundary, mesh_size_at_embeddings, embed_pts, boundary_vxs, boundary_lines, embedding_vxs, embeddings_lines, gui = False, tolerance = 1e-6):
    # print("inputs ", mesh_size_at_boundary, mesh_size_at_embeddings, embed_pts, boundary_vxs, boundary_lines, embedding_vxs, embeddings_lines)
    start_time = time.time()
    # embed_pts=None
    """Generate mesh from embeddings"""

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("mesher_from_embeddings")

    # Read file
    dim = 2

    x_shift = np.max(boundary_vxs[:, 0]) - np.min(boundary_vxs[:, 0])
    y_shift = np.max(boundary_vxs[:, 1]) - np.min(boundary_vxs[:, 1])

    dim_shift = [x_shift, y_shift, 0]
    lower_x = np.min(boundary_vxs[:, 0])
    lower_y = np.min(boundary_vxs[:, 1])
    upper_x = np.max(boundary_vxs[:, 0])
    upper_y = np.max(boundary_vxs[:, 1])
    z = 0
    boundary_tag = 1
    gmsh.model.occ.addRectangle(lower_x, lower_y, z, x_shift, y_shift, boundary_tag)

    embed_tags = []
    temp_pts = []
    embed_edges = []
    point_index_to_tag = {}
    for idx,vx in enumerate(embedding_vxs):
        point_index_to_tag[idx + 1] = gmsh.model.occ.addPoint(*vx)
        temp_pts.append(vx)

    for line in embeddings_lines:
        embed_tags.append(gmsh.model.occ.addLine(point_index_to_tag[line[0]], point_index_to_tag[line[1]]))
        embed_edges.append([temp_pts[line[0]-1], temp_pts[line[1]-1]])

    # Remove elements that are outside of the bounding box
    out, _ = gmsh.model.occ.fragment([(2, boundary_tag)], [(1, i) for i in embed_tags])
    gmsh.model.occ.synchronize()

    # Ask OpenCASCADE to compute more accurate bounding boxes of entities using
    # the STL mesh:
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

    # We then retrieve all the volumes in the bounding box of the original cube,
    # and delete all the parts outside it:
    eps = 1e-3
    vin = gmsh.model.getEntitiesInBoundingBox(lower_x - eps, lower_y-eps, z, upper_x + eps, upper_y + eps, z, 2)
    for v in vin:
        out.remove(v)
    gmsh.model.removeEntities(out, True)  # Delete outside parts recursively

    for curr_dim in range(2):
        bbox = [lower_x - eps,lower_y - eps, z, lower_x + eps, lower_y + eps, z]
        bbox[3 + curr_dim] += dim_shift[curr_dim]
        # First we get all surfaces on the left:
        sxmin = gmsh.model.getEntitiesInBoundingBox(*bbox, 1)
        translation = [1, 0, 0, 0, 
                       0, 1, 0, 0, 
                       0, 0, 1, 0, 
                       0, 0, 0, 1]
        translation[3 + 4 * curr_dim] += dim_shift[curr_dim]

        for i in sxmin:
            # Then we get the bounding box of each left surface
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
            # We translate the bounding box to the right and look for surfaces inside
            # it:
            new_bbox = [xmin - eps, ymin - eps, zmin - eps, xmax + eps, ymax + eps, zmax + eps]
            new_bbox[curr_dim] += dim_shift[curr_dim]
            new_bbox[curr_dim+ 3] += dim_shift[curr_dim]

            sxmax = gmsh.model.getEntitiesInBoundingBox(*new_bbox, 1)
            # For all the matches, we compare the corresponding bounding boxes...
            for j in sxmax:
                xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j[0], j[1])
                if (curr_dim == 0):
                    xmin2 -= dim_shift[curr_dim]
                    xmax2 -= dim_shift[curr_dim]
                elif (curr_dim == 1):
                    ymin2 -= dim_shift[curr_dim]
                    ymax2 -= dim_shift[curr_dim]

                # ...and if they match, we apply the periodicity constraint
                if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps and
                    abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps and
                    abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                    gmsh.model.mesh.setPeriodic(j[0], [j[1]], [i[1]], translation)

    # Generate mesh
    if mesh_size_at_boundary<mesh_size_at_embeddings: mesh_size = mesh_size_at_boundary
    else : mesh_size = mesh_size_at_embeddings
    # TODO: these functions might need to be updated to produce uniform mesh.
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.model.mesh.generate(dim)

    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(dim)

    # Get new nodes and faces
    v = node_coords.reshape((len(node_tags),3))
    f = order_faces( v, elements_node_tags.reshape((len(element_tags),3)) )

    if embed_edges != []:
        fusing_data = generate_fusing_data_from_lines(v, embed_edges, tolerance)
    else:
        fusing_data = np.array([False] * len(v))
    if embed_pts is not None:
        for p in embed_pts:
            idx = find_point_closest_point(v, p)
            fusing_data[idx] = True
    if(gui): gmsh.fltk.run()
    # End gmsh
    gmsh.finalize()
    return v, f, fusing_data

def get_hole_wire_tags(hole_vxs, mesh_size_at_embeddings):
    hole_point_tags = []
    for hole_vx in hole_vxs:
        hole_point_tags.append([gmsh.model.occ.addPoint(*vx, meshSize=mesh_size_at_embeddings) for vx in hole_vx])

    hole_line_tags = []
    for hole_point_tag in hole_point_tags:
        hole_line_tag = []
        for i in range(len(hole_point_tag)):
            hole_line_tag.append(gmsh.model.occ.addLine(hole_point_tag[i], hole_point_tag[(i+1)%len(hole_point_tag)]))
        hole_line_tags.append(hole_line_tag)

    hole_wire_tags = [gmsh.model.occ.addWire(hole_line_tag) for hole_line_tag in hole_line_tags]
    return hole_wire_tags

def generate_mesh_non_periodic(mesh_size_at_embeddings, boundary_vxs, boundary_hole_vxs, non_boundary_holes_vxs, embedding_vxs, embeddings_lines, gui = False, tolerance = 1e-6):
    import gmsh
    import time
    import numpy as np
    start_time = time.time()

    gmsh.initialize()
    gmsh.model.add("mesher_from_embeddings")
    gmsh.option.setNumber("General.Verbosity", 0)

    dim = 2

    boundary_point_tag = []
    for vx in boundary_vxs:
        boundary_point_tag.append(gmsh.model.occ.addPoint(*vx, meshSize=mesh_size_at_embeddings))

    line_tags = []
    for i in range(len(boundary_point_tag)):
        line_tag = gmsh.model.occ.addLine(boundary_point_tag[i], boundary_point_tag[(i+1)%len(boundary_point_tag)])
        line_tags.append(line_tag)

    wire_tag = gmsh.model.occ.addWire(line_tags)
    non_boundary_hole_wire_tags = get_hole_wire_tags(non_boundary_holes_vxs, mesh_size_at_embeddings)
    # All these curves should have the same orientation.
    srf_tag = gmsh.model.occ.addPlaneSurface([wire_tag] + non_boundary_hole_wire_tags)


    boundary_time = time.time()
    print("Boundary processing took ", boundary_time - start_time, " seconds")

    if boundary_hole_vxs is not None:
        print("Processing {} holes".format(len(boundary_hole_vxs)))
        # Assume hole_vxs is a list of vertices for the holes
        hole_wire_tags = get_hole_wire_tags(boundary_hole_vxs, mesh_size_at_embeddings)
        hole_srf_tags = [gmsh.model.occ.addPlaneSurface([hole_wire_tag]) for hole_wire_tag in hole_wire_tags]
        # Subtract the holes from the main surface
        for hole_srf_tag in hole_srf_tags:
            cut_result = gmsh.model.occ.cut([(2, srf_tag)], [(2, hole_srf_tag)])
            if cut_result[0]:  # Check if the cut operation produced a new surface
                srf_tag = cut_result[0][0][1]  # Update the surface tag with the result of the cut
            else:
                print(f"Warning: Cut operation with hole_wire_tag {hole_srf_tag} did not produce a new surface")

    hole_time = time.time()
    print("Hole processing took ", hole_time - boundary_time, " seconds")

    temp_pts = []
    point_index_to_tag = {}
    for idx,vx in enumerate(embedding_vxs):
        point_index_to_tag[idx] = gmsh.model.occ.addPoint(*vx)
        temp_pts.append(vx)

    embed_tags = []
    embed_edges = []
    for line in embeddings_lines:
        embed_tags.append(gmsh.model.occ.addLine(point_index_to_tag[line[0]], point_index_to_tag[line[1]]))
        embed_edges.append([temp_pts[line[0]], temp_pts[line[1]]])    
    gmsh.model.occ.synchronize()

    print(len(point_index_to_tag), len(embed_tags))
    embedding_time = time.time()
    print("Embedding processing took ", embedding_time - hole_time, " seconds")

    # gmsh.model.mesh.embed(dim-1, embed_tags, dim, srf_tag)

    out, _ = gmsh.model.occ.fragment([(2, srf_tag)], [(1, i) for i in embed_tags])

    gmsh.model.occ.synchronize()

    sync_time = time.time()
    print("Sync processing took ", sync_time - embedding_time, " seconds")

    mesh_size = mesh_size_at_embeddings
    # TODO: these functions might need to be updated to produce uniform mesh.
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.model.mesh.generate(dim)

    # Add these lines to fix the mesh
    gmsh.model.mesh.removeDuplicateNodes()

    meshing_time = time.time()
    print("Meshing took ", meshing_time - sync_time, " seconds")

    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(dim)

    v = node_coords.reshape((len(node_tags),3))
    f = order_faces(v, elements_node_tags.reshape((len(element_tags),3)))

    get_entity_time = time.time()
    print("Getting entities took ", get_entity_time - meshing_time, " seconds")

    print("beginning to generate fusing data")

    if len(embed_edges) != 0:
        fusing_data = generate_fusing_data_from_lines(v, embed_edges, tolerance)
    else:
        fusing_data = np.array([False] * len(v))

    print("Compupting fusing data took ", time.time() - get_entity_time, " seconds")

    if(gui): gmsh.fltk.run()

    gmsh.finalize()

    return v, f, fusing_data

def generate_mesh_from_regions(filename_regions, filename_edges, mesh_size=1.0, gui=False):
    ''''Generate mesh from regions'''
    # Import mesh
    file_regions = open(filename_regions)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("mesher_from_regions")

    # Tags
    region_tags = []
    dim = 2
    # Read file
    for line in file_regions.readlines():
        data = [d.strip() for d in line.split(' ')]
        
        # Generate points
        if(data[0]=="v"):
            gmsh.model.geo.addPoint(float(data[1]),float(data[2]),float(data[3]), mesh_size)

        # Generate lines
        if(data[0]=="l"):
            gmsh.model.geo.addLine(int(data[1]),int(data[2]))

        # Generate planar surfaces
        if(data[0]=="pl"):
            loop_tag = gmsh.model.geo.addCurveLoop([int(d) for d in data[1:]])
            srf_tag = gmsh.model.geo.addPlaneSurface([loop_tag])
            region_tags.append(srf_tag)
    
    # Build fusing lines
    file_edges= open(filename_edges)
    temp_pts = []
    edges = []
    for line in file_edges.readlines():
        data = [d.strip() for d in line.split(' ')]
        if(data[0]=="v"):
            temp_pts.append( [float(data[1]),float(data[2]),float(data[3])] )
            
        if(data[0]=="l"):
            edges.append([temp_pts[int(data[1])-1], temp_pts[int(data[2])-1]])

    # Generate surface loop
    gmsh.model.geo.addSurfaceLoop(region_tags)
    gmsh.model.geo.synchronize()

    # Generate mesh
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(dim)

    node_tags, node_coords, node_param = gmsh.model.mesh.getNodes()
    element_tags, elements_node_tags = gmsh.model.mesh.getElementsByType(dim)

    # Get nodes and faces
    v = node_coords.reshape((len(node_tags),3))
    f = order_faces( v, elements_node_tags.reshape((len(element_tags),3)) )

    fusing_data = fusing_data = generate_fusing_data_from_lines(v, edges, 1e-3)

    # Open gmsh GUI
    if(gui): gmsh.fltk.run()

    # End gmsh
    gmsh.finalize()
    
    return v, f, fusing_data

def write_obj(vertices, faces, filename):
    """Write obj file"""
    file = open(filename, 'w')
    for v in vertices:
        file.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
    for f in faces:
        file.write('f ' + str(f[0]) + ' ' + str(f[1]) + ' ' + str(f[2]) + '\n')
    file.close()



def order_faces(vertices, faces):
    """Order faces in a counter-clockwise fashion"""
    v_coords = vertices[faces - 1]
    n = np.cross(v_coords[:, 1] - v_coords[:, 0], v_coords[:, 2] - v_coords[:, 0])
    return np.array([faces[i][[0, 1, 2]] if n[i][2] > 0 else faces[i][[0, 2, 1]] for i in range(len(n))])

def generate_fusing_data_from_lines(vertices, edges, min_distance = float("inf")):
    """
    Generate fusing data (Vectorized)
    """
    edges = np.array(edges)[:, :, :2]
    query_vertices = np.array(vertices)[:, :2]
    mesh_vertices = edges.reshape((-1, 2))
    mesh_edges = [[2 * i, 2 * i + 1] for i in range(len(edges))]

    # Construct an AABB tree for the edges
    tree = igl.AABB_f64_2()
    tree.init(mesh_vertices, mesh_edges)
    # Find the closest point on the edges for each vertex
    indicator = tree.squared_distance(mesh_vertices, mesh_edges, query_vertices)
    indicator = indicator < min_distance
    return indicator

def generate_fusing_data_from_lines_numpy(vertices, edges, min_distance = float("inf")):
    """
    Generate fusing data (Vectorized)
    """
    edges = np.array(edges)
    vertices = np.array(vertices)

    # Compute norms using broadcasting instead of tiling
    first_point_norm = np.linalg.norm(vertices[:, None] - edges[:, 0], axis=2)

    second_point_norm = np.linalg.norm(vertices[:, None] - edges[:, 1], axis=2)


    # Compute edge lengths and reshape for broadcasting
    edge_len = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)[None, :]

    # Use in-place operation to compute indicator
    indicator = first_point_norm + second_point_norm
    indicator -= edge_len
    indicator = indicator < min_distance

    # Compute fusing data
    fusing_data = np.any(indicator, axis=1)

    return fusing_data        

def find_line_closest_points(point_set, line, min_distance):
    """
    Finds the closest points in a given point set to a given line segment, and returns their indexes.
    """
    def point_in_line_segment(pt, line, tolerance = 1e-6):
        pt = np.array(pt)
        line = np.array(line)
        return (np.linalg.norm(pt - line[0]) + np.linalg.norm(pt - line[1]) - np.linalg.norm(line[0] - line[1]) < tolerance)
    closest_points = []
    for i, point in enumerate(point_set):
        if point_in_line_segment(point, line):
            closest_points.append(i)
    
    return closest_points

def find_point_closest_point(point_set, point):
    """
    Finds the closest point in a given point set to a given point, and returns its index.
    """
    distances = np.sqrt(np.sum((point_set - point)**2, axis=1)) 
    return np.argmin(distances)
