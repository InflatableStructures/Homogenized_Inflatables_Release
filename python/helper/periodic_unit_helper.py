import homogenized_inflation, numpy as np, importlib, fd_validation, visualization, parametric_pillows, wall_generation
from numpy.linalg import norm
import numpy.linalg as la
import MeshFEM, parallelism, benchmark, utils
import igl

def get_mesh_input():
    N = 100
    h = 1
    base_d = 0.05
    base_w = base_d * 5
    base_triArea = 0.001
    numSegments = 100
    tilt_n = 20
    total_width = base_d + base_w
    clipping_lN = 50
    clipping_uN = 51

    use_non_empty_wall = True

    freq = 1
    amplitude = 0.3
    tilt_n = 1

    d = base_d / freq if use_non_empty_wall else 0


    m, fuseMarkers, brdyWallMarkers = parametric_pillows.sinusoid(N = N, h = h, d = 2.5 * d, w = base_w / freq, triArea = 40 * base_triArea / freq**2 * base_w, numSegments = int(4 * freq), freq = np.pi * freq, amplitude = amplitude, tilt_n = tilt_n, clipping_lN = clipping_lN, clipping_uN = clipping_uN, target_length = None, use_periodic = True, epsilon = 1e-3 / freq)

    return m, fuseMarkers, brdyWallMarkers

def get_center_fixedVars(ipu):
    fixedVxIdx = ipu.get_IPU_vidx_for_inflatable_vidx(ipu.sheet.center_non_fused_vx_idx())
    fixedVars = np.arange(3 + np.array(fixedVxIdx) * 3, 3 + 3 + np.array(fixedVxIdx) * 3)
    return fixedVars

import sparse_matrices
import scipy
def getScipySparseMatrixFromCSC(csc, reflect = False):
    if reflect:
        csc = csc.toSymmetryMode(sparse_matrices.SymmetryMode.NONE)
    data = csc.Ax
    row = csc.Ai
    col_ends = csc.Ap
    col = []
    col_start = 0
    for col_idx, j in enumerate(col_ends[1:]):
        col.extend([col_idx] * (j - col_start))
        col_start = j
    H_csc = scipy.sparse.csc_matrix((data, (row, col)), shape = (len(col_ends) - 1, len(col_ends) - 1))
    return H_csc

def getNumpyArrayFromCSC(csc, reflect = False):
    H_csc = getScipySparseMatrixFromCSC(csc, reflect)
    H = H_csc.toarray()
    return H

def rotate_2D_around_center(pt, center, angle, scale = 1):
    pt = np.array(pt)
    center = np.array(center)
    return np.dot(pt - center, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])) * scale + center


def getBox(triArea):
    avg_len = triArea ** 0.5

    def get_scaled_box(pts, scale = 1.1):
        cm = np.mean(np.array(pts), axis = 0)
        scaled_pts = (np.array(pts) - cm) * scale + cm
        return igl.bounding_box(scaled_pts)

    def get_scaled_square_box(pts, scale = 1.1):
        bbox_vx, bbox_edges = get_scaled_box(pts, scale)
        box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
        box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])
        if box_width > box_height:
            height_scale = box_width / box_height
            height_mean = np.mean(bbox_vx, axis = 0)[1]
            bbox_vx[:, 1] = (bbox_vx[:, 1] - height_mean) * height_scale + height_mean
        else:
            width_scale = box_height / box_width
            width_mean = np.mean(bbox_vx, axis = 0)[0]
            bbox_vx[:, 0] = (bbox_vx[:, 0] - width_mean) * width_scale + width_mean
        return bbox_vx, bbox_edges

    bbox_vx, bbox_edges = get_scaled_square_box([[0, 0], [1, 1]], scale = 1.5)

    box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
    box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])

    num_width_seg = int(np.round(box_width / avg_len))
    num_height_seg = int(np.round(box_height / avg_len))

    top_y = bbox_vx[0][1]
    right_x = bbox_vx[0][0]
    bot_y = bbox_vx[3][1]
    left_x = bbox_vx[3][0]

    top_y,right_x, bot_y, left_x

    num_width_seg, num_height_seg

    bbox_vx = np.concatenate((np.linspace(bbox_vx[0], bbox_vx[1], num_height_seg),  np.linspace(bbox_vx[1], bbox_vx[3], num_width_seg)[1:], np.linspace(bbox_vx[3], bbox_vx[2], num_height_seg)[1:], np.linspace(bbox_vx[2], bbox_vx[0], num_width_seg)[1:-1]))

    bbox_edges = [[i, i + 1] for i in np.arange(len(bbox_vx) - 1)] + [[len(bbox_vx) - 1, 0]]

    n_vx = [] + list(bbox_vx)
    n_edge = [] + list(np.array(bbox_edges))
    return n_vx, n_edge

def get_scaled_box(pts, scale = 1.1):
    cm = np.mean(np.array(pts), axis = 0)
    scaled_pts = (np.array(pts) - cm) * scale + cm
    return igl.bounding_box(scaled_pts)

def get_scaled_square_box(pts, scale = 1.1):
    bbox_vx, bbox_edges = get_scaled_box(pts, scale)
    box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
    box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])
    if box_width > box_height:
        height_scale = box_width / box_height
        height_mean = np.mean(bbox_vx, axis = 0)[1]
        bbox_vx[:, 1] = (bbox_vx[:, 1] - height_mean) * height_scale + height_mean
    else:
        width_scale = box_height / box_width
        width_mean = np.mean(bbox_vx, axis = 0)[0]
        bbox_vx[:, 0] = (bbox_vx[:, 0] - width_mean) * width_scale + width_mean
    return bbox_vx, bbox_edges

def get_target_box(pts, target_w, target_h):
    bbox_vx, bbox_edges = igl.bounding_box(np.array(pts))
    box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
    box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])
    if target_h > box_height:
        height_scale = target_h / box_height
        height_mean = np.mean(bbox_vx, axis = 0)[1]
        bbox_vx[:, 1] = (bbox_vx[:, 1] - height_mean) * height_scale + height_mean
    if target_w > box_width:
        width_scale = target_w / box_width
        width_mean = np.mean(bbox_vx, axis = 0)[0]
        bbox_vx[:, 0] = (bbox_vx[:, 0] - width_mean) * width_scale + width_mean
    return bbox_vx, bbox_edges

def get_rectangle_wall(h, w, avg_len, angle = 0.0):
    triArea = avg_len ** 2
    bbox_vx, bbox_edges = igl.bounding_box(np.array([[0., 0], [w, 0], [0, h], [w, h]]))
    center = np.array([w / 2, h / 2])
    # print(center)

    for i in range(len(bbox_vx)):
        bbox_vx[i] = rotate_2D_around_center(bbox_vx[i], center, angle)

    box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
    box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])

    num_width_seg = int(np.round(box_width / avg_len))
    num_height_seg = int(np.round(box_height / avg_len))

    top_y = bbox_vx[0][1]
    right_x = bbox_vx[0][0]
    bot_y = bbox_vx[3][1]
    left_x = bbox_vx[3][0]

    top_y,right_x, bot_y, left_x

    bbox_vx = np.concatenate((np.linspace(bbox_vx[0], bbox_vx[1], num_height_seg),  np.linspace(bbox_vx[1], bbox_vx[3], num_width_seg)[1:], np.linspace(bbox_vx[3], bbox_vx[2], num_height_seg)[1:], np.linspace(bbox_vx[2], bbox_vx[0], num_width_seg)[1:-1]))

    bbox_edges = [[i, i + 1] for i in np.arange(len(bbox_vx) - 1)] + [[len(bbox_vx) - 1, 0]]

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(bbox_vx, bbox_edges, triArea)


    edges = []

    def append_edges(a, b):
        edges.append([*a])

    m.visitEdges(append_edges)
    return list(m.vertices()[:, :2]), edges

def point_in_box(pt, box, inclusive = True, tolerance = 1e-6, angle = 0, scale = 1):
    center = np.mean(box[0], axis = 0)[:2]
    # print(box[0])
    # print("point in box ", center)
    pt = rotate_2D_around_center(pt[:2], center, -angle, scale)

    box_vx = box[0]
    min_x = np.min(box_vx[:, 0])
    max_x = np.max(box_vx[:, 0])
    min_y = np.min(box_vx[:, 1])
    max_y = np.max(box_vx[:, 1])
    if inclusive:
        if pt[0] - min_x >= -tolerance and pt[0] - max_x <= tolerance and pt[1] - min_y >= -tolerance and pt[1] - max_y <= tolerance:
            return True
    else:
        if pt[0] > min_x and pt[0] < max_x and pt[1] > min_y and pt[1] < max_y:
            return True
    return False

def get_discretized_box(pts, scale, avg_len, shift = np.array([0.0, 0.0]), use_square_boundary = False, use_even = False):
    if use_square_boundary:
        bbox_vx, bbox_edges = get_scaled_square_box(pts, scale = scale)
    else:
        bbox_vx, bbox_edges = get_scaled_box(pts, scale = scale)
    bbox_vx += shift

    box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
    box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])

    num_width_seg = int(np.round(box_width / avg_len))
    if use_even and num_width_seg % 2 == 0:
        num_width_seg += 1
    num_height_seg = int(np.round(box_height / avg_len))
    if use_even and num_height_seg % 2 == 0:
        num_height_seg += 1

    top_y = bbox_vx[0][1]
    right_x = bbox_vx[0][0]
    bot_y = bbox_vx[3][1]
    left_x = bbox_vx[3][0]

    bbox_vx = np.concatenate((np.linspace(bbox_vx[0], bbox_vx[1], num_height_seg),  np.linspace(bbox_vx[1], bbox_vx[3], num_width_seg)[1:], np.linspace(bbox_vx[3], bbox_vx[2], num_height_seg)[1:], np.linspace(bbox_vx[2], bbox_vx[0], num_width_seg)[1:-1]))

    bbox_edges = [[i, i + 1] for i in np.arange(len(bbox_vx) - 1)] + [[len(bbox_vx) - 1, 0]]
    return bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x

def get_discretized_box_from_w_h(w, h, scale, avg_len, shift = np.array([0.0, 0.0]), use_even = False):
    pts = [[0.0, 0.0], [0.0, w], [h, 0.0], [h, w]]
    return get_discretized_box(pts, scale, avg_len, shift, use_even = use_even)

def embed_rectangle_wall_in_box(pts, edges, avg_len, scale = 1):
    bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x = get_discretized_box(pts, scale, avg_len)

    n_vx = pts + list(bbox_vx)
    n_edge = edges + list(np.array(bbox_edges) + len(pts))
    return n_vx, n_edge, top_y, right_x, bot_y, left_x

def get_mesh_from_embedded_wall(pts, edges, avg_len, scale, triArea):
    n_vx, n_edge, top_y, right_x, bot_y, left_x = embed_rectangle_wall_in_box(pts, edges, avg_len, scale)
    wall_box = get_scaled_box(pts, 1)

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(n_vx, n_edge, triArea, flags="Y")
    marker = [(point_in_box(pt, wall_box)) for pt in m.vertices()]
    return m, marker, n_vx, n_edge

def get_dashline(h, w, avg_len):    
    triArea = avg_len ** 2
    pts, edges = get_rectangle_wall(h, w, avg_len)
    m, marker, n_vx, n_edge = get_mesh_from_embedded_wall(pts, edges, avg_len, 5 / w, triArea)
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    return ipu, n_vx, n_edge, m, marker

def shift_points(points, shift):
    points = list(np.array(points) + shift)

def embed_shifted_rectangle_walls_in_box(h, w, pts, edges, avg_len, shift, scale = 1, two_dash = True, angle = 0.0, opposite_angle = False):
    bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x = get_discretized_box(pts, scale, avg_len, use_square_boundary = True)
    if two_dash:
        if (opposite_angle):
            oppo_pts, oppo_edges = get_rectangle_wall(h, w, avg_len, -1 * angle)
            n_vx =  list(bbox_vx) + list(np.array(oppo_pts) + shift)
            n_edge = bbox_edges + list(np.array(oppo_edges) + len(bbox_vx))
        else:
            n_vx =  list(bbox_vx) + list(np.array(pts) + shift)
            n_edge = bbox_edges + list(np.array(edges) + len(bbox_vx))
    else:
        n_vx = list(bbox_vx)
        n_edge = bbox_edges

    nn_vx =  list(n_vx) + list(np.array(pts) - shift)
    nn_edge = n_edge + list(np.array(edges) + len(n_vx))

    return nn_vx, nn_edge, top_y, right_x, bot_y, left_x

def get_mesh_from_embedded_wall_shifted(h, w, pts, edges, avg_len, shift, scale, triArea, two_dash = True, angle = 0.0, opposite_angle = False):
    n_vx, n_edge, top_y, right_x, bot_y, left_x = embed_shifted_rectangle_walls_in_box(h, w, pts, edges, avg_len, shift, scale, two_dash, angle, opposite_angle)
    wall_box_1 = get_scaled_box(np.array([[0., 0], [w, 0], [0, h], [w, h]]) - shift, 1)
    wall_box_2 = get_scaled_box(np.array([[0., 0], [w, 0], [0, h], [w, h]]) + shift, 1)

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(n_vx, n_edge, triArea, flags="Y")
    marker = [(point_in_box(pt, wall_box_1, angle = angle, scale = 1, inclusive = True,tolerance = 1e-3) or (two_dash and point_in_box(pt, wall_box_2, angle = -angle if opposite_angle else angle, scale = 1, inclusive = True,tolerance = 1e-3))) for pt in m.vertices()]
    return m, marker, n_vx, n_edge

def get_shifted_dashline(h, w, avg_len, shift, two_dash = True, angle = 0.0, opposite_angle = False):    
    triArea = avg_len ** 2
    pts, edges = get_rectangle_wall(h, w, avg_len, angle)
    scale = 5 / (max(max(igl.bounding_box(np.array(pts))[0][:, 1]) - min(igl.bounding_box(np.array(pts))[0][:, 1]), max(igl.bounding_box(np.array(pts))[0][:, 0]) - min(igl.bounding_box(np.array(pts))[0][:, 0])))

    m, marker, n_vx, n_edge = get_mesh_from_embedded_wall_shifted(h, w, pts, edges, avg_len, shift, scale, triArea, two_dash, angle, opposite_angle)
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    return ipu, n_vx, n_edge, m, marker

def point_in_line_segment(pt, line, tolerance = 1e-6):
    pt = np.array(pt)
    line = np.array(line)
    return (np.linalg.norm(pt - line[0]) + np.linalg.norm(pt - line[1]) - np.linalg.norm(line[0] - line[1]) < tolerance)

def get_zero_area_dashline(h,w, avg_len, dash_point):
    triArea = avg_len ** 2
    box_points = np.array([[-w/2., -h/2.], [w/2., -h/2.], [w/2., h/2.], [-w/2., h/2.]])
    mid_point = np.array([0, 0])
    dash_line_1 = np.array([dash_point, mid_point * 2 - dash_point])
    
    dash_line_len = np.linalg.norm(dash_line_1[0] - dash_line_1[1])
    num_seg = int(np.round(dash_line_len / avg_len))
    dash_line_1_discrete = np.linspace(dash_line_1[0], dash_line_1[1], num_seg)

    bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x = get_discretized_box(box_points, 1, avg_len)
    n_vx = list(bbox_vx) + list(dash_line_1_discrete)
    dash_line_1_edges = np.concatenate((np.array(range(len(bbox_vx), len(bbox_vx) + num_seg - 1)).reshape(num_seg - 1, 1), np.array(range(len(bbox_vx), len(bbox_vx) + num_seg - 1)).reshape(num_seg - 1, 1) + 1), axis = 1)
    n_edge = bbox_edges + dash_line_1_edges.tolist()
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(n_vx, n_edge, triArea, flags="Y")

    marker = [point_in_line_segment(pt[:2], dash_line_1, tolerance = 1e-6) for pt in m.vertices()]
    return m, marker, n_vx, n_edge

def get_cross(h,w, avg_len, orientation = 0, center = True):
    triArea = avg_len ** 2
    box_points = np.array([[-w/2., -h/2.], [w/2., -h/2.], [w/2., h/2.], [-w/2., h/2.]])
    shift = 0 if center else h / 4
    mid_point = np.array([0, shift])

    # vertical dashlines
    dash_point = np.array([0, 2 + shift])
    dash_line_1 = np.array([dash_point, mid_point * 2 - dash_point])
    # horizontal dashlines
    dash_point = np.array([2, shift])
    dash_line_2 = np.array([dash_point, mid_point * 2 - dash_point])

    draw_dash_1 = np.array([dash_line_1[0], mid_point])
    draw_dash_2 = np.array([dash_line_1[1], mid_point])
    draw_dash_3 = np.array([dash_line_2[0], mid_point])
    draw_dash_4 = np.array([dash_line_2[1], mid_point])


    dash_line_len = np.linalg.norm(draw_dash_1[0] - draw_dash_1[1])
    num_seg = int(np.round(dash_line_len / avg_len))
    dash_line_1_discrete = np.linspace(draw_dash_1[0], draw_dash_1[1], num_seg)
    dash_line_2_discrete = np.linspace(draw_dash_2[0], draw_dash_2[1], num_seg)
    dash_line_3_discrete = np.linspace(draw_dash_3[0], draw_dash_3[1], num_seg)
    dash_line_4_discrete = np.linspace(draw_dash_4[0], draw_dash_4[1], num_seg)

    bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x = get_discretized_box(box_points, 1, avg_len)

    # if (center):
    #     box = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    #     n_bbox_vx, n_bbox_edges, _, _, _, _ = get_discretized_box(box * (2.1), 1, avg_len)
    #     bbox_edges = bbox_edges + (np.array(n_bbox_edges) + len(bbox_vx)).tolist()
    #     bbox_vx = bbox_vx.tolist() + n_bbox_vx.tolist()

    extra_points = list(dash_line_1_discrete) + list(dash_line_2_discrete[:-1]) + list(dash_line_3_discrete[:-1]) + list(dash_line_4_discrete[:-1])

    n_vx = list(bbox_vx) + extra_points
    if (not center):
        n_vx += (np.array(extra_points) + np.array([0, - h / 2])).tolist()
    dash_line_1_edges = np.concatenate((np.array(range(len(bbox_vx), len(bbox_vx) + num_seg - 1)).reshape(num_seg - 1, 1), np.array(range(len(bbox_vx), len(bbox_vx) + num_seg - 1)).reshape(num_seg - 1, 1) + 1), axis = 1)

    dash_line_2_edges = np.concatenate((
                np.array(range(len(bbox_vx) + num_seg, len(bbox_vx) + 2 * num_seg - 1 - 1)).reshape(num_seg - 1 - 1, 1), 
                np.array(range(len(bbox_vx) + num_seg, len(bbox_vx) + 2 * num_seg - 1 - 1)).reshape(num_seg - 1 - 1, 1) + 1), axis = 1)
    dash_line_3_edges = np.concatenate((
                np.array(range(len(bbox_vx) + 2 * num_seg - 1, len(bbox_vx) + 3 * num_seg - 1 - 1 - 1)).reshape(num_seg - 1 - 1, 1), 
                np.array(range(len(bbox_vx) + 2 * num_seg - 1, len(bbox_vx) + 3 * num_seg - 1 - 1 - 1)).reshape(num_seg - 1 - 1, 1) + 1), axis = 1)
    dash_line_4_edges = np.concatenate((
                np.array(range(len(bbox_vx) + 3 * num_seg - 2, len(bbox_vx) + 4 * num_seg - 4)).reshape(num_seg - 1 - 1, 1), 
                np.array(range(len(bbox_vx) + 3 * num_seg - 2, len(bbox_vx) + 4 * num_seg - 4)).reshape(num_seg - 1 - 1, 1) + 1), axis = 1)

    extra_edges = dash_line_1_edges.tolist() + dash_line_2_edges.tolist() + dash_line_3_edges.tolist() + dash_line_4_edges.tolist() + [[len(bbox_vx) + num_seg - 1, len(bbox_vx) + num_seg + num_seg - 1 - 1], [len(bbox_vx) + num_seg - 1, len(bbox_vx) + 2 * num_seg + num_seg - 3], [len(bbox_vx) + num_seg - 1, len(bbox_vx) + 3 * num_seg + num_seg - 4]]

    n_edge = bbox_edges + extra_edges
    if (not center):
        n_edge += (np.array(extra_edges) + len(extra_points)).tolist()
    # return n_vx, n_edge
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(n_vx, n_edge, triArea, flags="Y")


    if orientation == 0:
        marker = [(point_in_line_segment(pt[:2], draw_dash_1, tolerance = 1e-6) or point_in_line_segment(pt[:2], draw_dash_2, tolerance = 1e-6)) or ((not center) and ((point_in_line_segment(pt[:2], draw_dash_1 + np.array([0, - h / 2]), tolerance = 1e-6) or point_in_line_segment(pt[:2], draw_dash_2 + np.array([0, - h / 2]), tolerance = 1e-6)))) for pt in m.vertices()]
    else:
        marker = [(point_in_line_segment(pt[:2], draw_dash_3, tolerance = 1e-6) or point_in_line_segment(pt[:2], draw_dash_4, tolerance = 1e-6)) or ((not center) and ((point_in_line_segment(pt[:2], draw_dash_3 + np.array([0, - h / 2]), tolerance = 1e-6) or point_in_line_segment(pt[:2], draw_dash_4 + np.array([0, - h / 2]), tolerance = 1e-6)))) for pt in m.vertices()]

    return m, marker, n_vx, n_edge

def get_boundary_aligned_dashline(w_small, w_big, h_small, h_big, avg_len):
    triArea = avg_len ** 2

    num_width_seg = int(np.round(w_small / avg_len))
    num_height_seg = int(np.round(h_small / avg_len))

    num_width_seg_addition = int(np.round((w_big - w_small) / avg_len))
    num_height_seg_addition = int(np.round((h_big - h_small) / avg_len))

    width_points = np.linspace(0, w_small, num_width_seg)
    height_points = np.linspace(0, h_small, num_height_seg)

    width_points_addition = np.linspace(w_small, w_big, num_width_seg_addition)
    height_points_addition = np.linspace(h_small, h_big, num_height_seg_addition)

    total_width_points = np.concatenate([width_points, width_points_addition[1:]])
    total_height_points = np.concatenate([height_points, height_points_addition[1:]])

    final_points = []
    final_points.extend(list(np.column_stack([total_width_points, np.zeros(len(total_width_points))])))

    final_points.extend(list(np.column_stack([np.zeros(len(total_height_points) - 1), total_height_points[1:]])))

    final_points.extend(list(np.column_stack([total_width_points[1:], h_big * np.ones(len(total_width_points) - 1)])))

    final_points.extend(list(np.column_stack([w_big * np.ones(len(total_height_points) - 2), total_height_points[1:-1]])))

    final_points.extend(list(np.column_stack([width_points[1:], h_small * np.ones(num_width_seg - 1)])))

    final_points.extend(list(np.column_stack([w_small * np.ones(num_height_seg - 2), height_points[1:-1]])))

    final_edges = []

    # First two boundary edges
    for i in range(len(total_width_points) - 1):
        final_edges.append([i, i+1])
        
    end_width = len(total_width_points) - 1

    prev_index = 0
    for i in range(len(total_height_points) - 1):
        next_edge_index = end_width + i + 1
        final_edges.append([prev_index, next_edge_index])
        prev_index = next_edge_index

    end_height = prev_index

    # Copying the periodic boundary edges
    prev_index = end_height
    for i in range(len(total_width_points) - 1):
        next_edge_index = end_height + i + 1
        final_edges.append([prev_index, next_edge_index])
        prev_index = next_edge_index
        
    end_second_width = prev_index

    prev_index = end_width
    for i in range(len(total_height_points) - 2):
        next_edge_index = end_second_width + i + 1
        final_edges.append([prev_index, next_edge_index])
        prev_index = next_edge_index
        
    final_edges.append([prev_index, end_second_width])
    end_second_height = prev_index + 1

    # Draw the internal edges

    prev_index = end_width + num_height_seg - 1
    for i in range(len(width_points) - 1):
        next_edge_index = end_second_height + i 
        final_edges.append([prev_index, next_edge_index])
        prev_index = next_edge_index
        
    end_third_width = prev_index

    prev_index = num_width_seg - 1
    for i in range(len(height_points) - 2):
        next_edge_index = end_third_width + i + 1
        final_edges.append([prev_index, next_edge_index])
        prev_index = next_edge_index
        
    final_edges.append([prev_index, end_third_width])
    end_third_height = prev_index + 1


    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(final_points, final_edges, triArea, flags="Y")

    wall_box = get_scaled_box(np.array([[0, 0], [w_small, h_small]]), 1)

    def point_in_periodic_box(pt):
        if np.abs(pt[0] - w_big) < 1e-5 and pt[1] <= h_small:
            return True 
        if np.abs(pt[1] - h_big) < 1e-5 and pt[0] <= w_small:
            return True
        
        if np.abs(pt[0] - w_big) < 1e-5 and np.abs(pt[1] - h_big) < 1e-5:
            return True
        return False

    marker = [point_in_box(pt, wall_box) or point_in_periodic_box(pt) for pt in m.vertices()]


    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    return ipu, final_points, final_edges, m, marker



def embed_square_pipe_in_box(pts, edges, pts2, edges2, avg_len, scale = 1):
    bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x = get_discretized_box(pts, scale, avg_len)

    n_vx =  list(bbox_vx) + list(pts)
    n_edge = bbox_edges + list(np.array(edges) + len(bbox_vx))
    nn_vx =  list(n_vx) + list(pts2)
    nn_edge = n_edge + list(np.array(edges2) + len(n_vx))

    return nn_vx, nn_edge, top_y, right_x, bot_y, left_x


def get_mesh_from_embedded_pipe(pts, edges, pts2, edges2, avg_len, scale, triArea):
    n_vx, n_edge, top_y, right_x, bot_y, left_x = embed_square_pipe_in_box(pts, edges, pts2, edges2, avg_len, scale)
    wall_box_1 = get_scaled_box(np.array(pts), 1)
    wall_box_2 = get_scaled_box(np.array(pts2), 1)

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(n_vx, n_edge, triArea, flags="Y")
    marker = [(point_in_box(pt, wall_box_2) or not point_in_box(pt, wall_box_1, False)) for pt in m.vertices()]
    return m, marker, n_vx, n_edge

def get_square_pipe(h, w, h2, w2, avg_len):    
    triArea = avg_len ** 2
    pts, edges, _, _, _, _ = get_discretized_box_from_w_h(w, h, 1, avg_len)
    pts2, edges2, _, _, _, _ = get_discretized_box_from_w_h(w2, h2, 1, avg_len, shift = np.array([h - h2, w - w2]) / 2)
    m, marker, n_vx, n_edge = get_mesh_from_embedded_pipe(pts, edges, pts2, edges2, avg_len, 5 / h, triArea)
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    return ipu, n_vx, n_edge, m, marker

def get_zero_area_parallel_tube(h, w, avg_len):
    pts, edges, _, _, _, _ = get_discretized_box_from_w_h(w, h, 1, avg_len)
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(pts, edges, avg_len**2, flags="Y")
    bbox = igl.bounding_box(m.vertices())
    fusedVtx = np.zeros(len(m.vertices()), dtype=bool)
    max_x = max(bbox[0][:, 0])
    min_x = min(bbox[0][:, 0])
    max_y = max(bbox[0][:, 1])
    min_y = min(bbox[0][:, 1])

    vxs = m.vertices()
    for i, vx in enumerate(m.vertices()):
        if np.abs(vx[0] - max_x) < 1e-6:
            fusedVtx[i] = True
        if np.abs(vx[0] - min_x) < 1e-6:
            fusedVtx[i] = True    
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, fusedVtx, epsilon = 1e-5)
    return ipu, pts, edges, m, fusedVtx

def get_parallel_tube_periodic(h, w, wall_w, avg_len):
    gap = (w - wall_w) / 2
    pts = [[0, 0], [0, gap], [0, gap + wall_w], [0, w], [h, 0], [h, gap], [h, gap + wall_w], [h, w]]
    edges = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    edge_points = []
    for edge in edges:
        edge_points.append(np.linspace(pts[edge[0]], pts[edge[1]], int(np.round(la.norm(np.array(pts[edge[1]]) - np.array(pts[edge[0]])) / avg_len))))

    points = np.concatenate((edge_points[0], edge_points[1][1:], edge_points[2][1:], 
                            edge_points[3], edge_points[4][1:], edge_points[5][1:], 
                            edge_points[6][1:-1], edge_points[7][1:-1], edge_points[8][1:-1], edge_points[9][1:-1]))

    segment_edges = []

    offsets = [0]

    curr_offset = offsets[-1]
    for ei in range(3):
        for i in range(len(edge_points[ei]) - 1):
            segment_edges.append([i + curr_offset ,  i + curr_offset + 1])
        offsets.append(curr_offset + len(edge_points[ei]) - 1)
        curr_offset = offsets[-1]

    offsets.append(curr_offset + 1)

    curr_offset = offsets[-1]
    for ei in range(3):
        for i in range(len(edge_points[ei + 3]) - 1):
            segment_edges.append([i + curr_offset ,  i + curr_offset + 1])
        offsets.append(curr_offset + len(edge_points[ei + 3]) - 1)
        curr_offset = offsets[-1]

    #################################
    for ei in range(4):
        for i in range(len(edge_points[ei + 6]))[1:-2]:
            segment_edges.append([i+ curr_offset, i + curr_offset +1])
        segment_edges.append([offsets[ei], curr_offset + 1])
        segment_edges.append([curr_offset + len(edge_points[ei + 6]) - 2, offsets[ei + 4]])
        offsets.append(curr_offset + len(edge_points[ei + 6]) - 2)
        curr_offset = offsets[-1]

    def inWall(pt):
        if pt[1] >= gap and pt[1] <= gap + wall_w:
            return True
        return False

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(points, segment_edges, avg_len ** 2, flags="Y")

    marker = [inWall(pt) for pt in m.vertices()]
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    return ipu, m, marker

def get_cosine_curve(h, avg_len, amplitude):
    w = h
    pts, edges, _, _, _, _ = get_discretized_box_from_w_h(w, h, 1, avg_len, use_even = True)
    num_edges = int(np.around(h / avg_len))
    base_factor = 1 / (2 * np.pi) * h
    def curve_function(x):
        return amplitude * base_factor * np.cos(x) + h / 2
    curve = []
    for i in range(num_edges):
        angle = i / num_edges * 2 * np.pi
        curve.append([i / num_edges * h, curve_function(angle)])

    curve = np.array(curve[1:])
    new_pts = list(pts) + curve.tolist()
    curve_edges = [[i + len(pts), i + 1 + len(pts)] for i in range(len(curve) - 1)] + [[len(pts), np.argmin([np.linalg.norm(curve[0] - pt) for pt in pts])], [len(pts) + len(curve) - 1, np.argmin([np.linalg.norm(curve[-1] - pt) for pt in pts])] ]
    new_edges = list(edges) + curve_edges
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(new_pts, new_edges, avg_len ** 2, flags="Y")
    marker = [np.abs(pt[1] - curve_function(pt[0] / h * 2  * np.pi)) < 1e-2 for pt in m.vertices()]
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)

    return ipu, new_pts, new_edges, m, marker

def get_zigzag_tube(h, w, avg_len):
    gap = (h - w) / 2

    pts = [[0, 0], [0, gap], [0, gap + w], [0, h], [h, 0], [h, gap], [h, gap + w], [h, h]]

    edges = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    edge_points = []
    for edge in edges:
        edge_points.append(np.linspace(pts[edge[0]], pts[edge[1]], int(np.round(la.norm(np.array(pts[edge[1]]) - np.array(pts[edge[0]])) / avg_len))))

    zigzag_w = 1

    start = np.array(pts[1])
    end = np.array(pts[5])
    # refactor all the subtraction to use numpy arrays
    zig = start + np.array([h / 3, - zigzag_w / 2])
    zag = start + np.array([2 * h / 3, zigzag_w / 2])

    zigzag_points_1 = np.linspace(start, zig, int(np.round(la.norm(start - zig) / avg_len)))
    zigzag_points_2 = np.linspace(zig, zag, int(np.round(la.norm(zig - zag) / avg_len)))
    zigzag_points_3 = np.linspace(zag, end, int(np.round(la.norm(zag - end) / avg_len)))

    edge_points[7] = np.concatenate((zigzag_points_1, zigzag_points_2[1:], zigzag_points_3[1:]))

    start = pts[2]
    end = pts[6]
    zig = start + np.array([h / 3, - zigzag_w / 2])
    zag = start + np.array([2 * h / 3, zigzag_w / 2])

    zigzag_points_1 = np.linspace(start, zig, int(np.round(la.norm(start - zig) / avg_len)))
    zigzag_points_2 = np.linspace(zig, zag, int(np.round(la.norm(zig - zag) / avg_len)))
    zigzag_points_3 = np.linspace(zag, end, int(np.round(la.norm(zag - end) / avg_len)))

    edge_points[8] = np.concatenate((zigzag_points_1, zigzag_points_2[1:], zigzag_points_3[1:]))


    points = np.concatenate((edge_points[0], edge_points[1][1:], edge_points[2][1:], 
                            edge_points[3], edge_points[4][1:], edge_points[5][1:], 
                            edge_points[6][1:-1], edge_points[7][1:-1], edge_points[8][1:-1], edge_points[9][1:-1]))

    segment_edges = []

    offsets = [0]

    curr_offset = offsets[-1]
    for ei in range(3):
        for i in range(len(edge_points[ei]) - 1):
            segment_edges.append([i + curr_offset ,  i + curr_offset + 1])
        offsets.append(curr_offset + len(edge_points[ei]) - 1)
        curr_offset = offsets[-1]

    offsets.append(curr_offset + 1)

    curr_offset = offsets[-1]
    for ei in range(3):
        for i in range(len(edge_points[ei + 3]) - 1):
            segment_edges.append([i + curr_offset ,  i + curr_offset + 1])
        offsets.append(curr_offset + len(edge_points[ei + 3]) - 1)
        curr_offset = offsets[-1]

    #################################
    for ei in range(4):
        for i in range(len(edge_points[ei + 6]))[1:-2]:
            segment_edges.append([i+ curr_offset, i + curr_offset +1])
        segment_edges.append([offsets[ei], curr_offset + 1])
        segment_edges.append([curr_offset + len(edge_points[ei + 6]) - 2, offsets[ei + 4]])
        offsets.append(curr_offset + len(edge_points[ei + 6]) - 2)
        curr_offset = offsets[-1]

    scale = 1
    points[:, 1] *= scale

    def inWall(pt):
        if (pt[0] <= h / 3):
            if pt[1] >= (gap - pt[0] / (h / 3) * (zigzag_w / 2)) * scale and pt[1] <= (gap - pt[0] / (h / 3) * (zigzag_w / 2) + w) * scale:
                return True

        if (pt[0] > h / 3 and pt[0] < 2 * h / 3):
            if (pt[1] - (gap - zigzag_w / 2 + (pt[0] - h/3) / (h / 3) * (zigzag_w)) * scale) >= -1e-3 and pt[1] <= (gap - zigzag_w / 2 + (pt[0] - h/3) / (h / 3) * (zigzag_w) + w) * scale:
                return True

        if (pt[0] >= 2 * h / 3):
            if (pt[1] - (gap - pt[0] / (h / 3) * (zigzag_w / 2) + zigzag_w * 1.5) * scale) >= -1e-3 and pt[1] <= (gap - pt[0] / (h / 3) * (zigzag_w / 2) + w + zigzag_w * 1.5) * scale:
                return True

        return False

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(points, segment_edges, avg_len ** 2, flags="Y")

    marker = [inWall(pt) for pt in m.vertices()]
    ipu = homogenized_inflation.InflatablePeriodicUnit(m, marker, epsilon = 1e-5)
    # ipu = 0
    return ipu, points, segment_edges, m, marker



def get_circle_in_box(pts, edges, avg_len, scale):
    triArea = avg_len**2
    # bbox_vx, bbox_edges, top_y, right_x, bot_y, left_x = get_discretized_box(pts, avg_len, scale)

    bbox_vx, bbox_edges = get_scaled_box(pts, scale = scale)

    bbox_vx, bbox_edges = get_scaled_square_box(pts, scale = scale)

    box_width = np.abs(bbox_vx[0][0] - bbox_vx[2][0])
    box_height = np.abs(bbox_vx[0][1] - bbox_vx[1][1])

    num_width_seg = int(np.round(box_width / avg_len))
    num_height_seg = int(np.round(box_height / avg_len))

    top_y = bbox_vx[0][1]
    right_x = bbox_vx[0][0]
    bot_y = bbox_vx[3][1]
    left_x = bbox_vx[3][0]

    top_y,right_x, bot_y, left_x

    num_width_seg, num_height_seg

    bbox_vx = np.concatenate((np.linspace(bbox_vx[0], bbox_vx[1], num_height_seg),  np.linspace(bbox_vx[1], bbox_vx[3], num_width_seg)[1:], np.linspace(bbox_vx[3], bbox_vx[2], num_height_seg)[1:], np.linspace(bbox_vx[2], bbox_vx[0], num_width_seg)[1:-1]))

    bbox_edges = [[i, i + 1] for i in np.arange(len(bbox_vx) - 1)] + [[len(bbox_vx) - 1, 0]]

    n_vx = pts + list(bbox_vx)
    n_edge = edges + list(np.array(bbox_edges) + len(pts))



    n_vx = pts + list(bbox_vx)
    n_edge = edges + list(np.array(bbox_edges) + len(pts))
    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(n_vx, n_edge, triArea, flags="Y")

    def is_bbox(point):
        if (point[0] == left_x or point[0] == right_x or point[1] == top_y or point[1] == bot_y):
            return True
        return False
    marker = np.logical_and(np.array(fuseMarkers) == 1, [(not is_bbox(pt)) for pt in m.vertices()])

    ipu = homogenized_inflation.InflatablePeriodicUnit(m, np.logical_and(np.array(fuseMarkers) == 1, [(not is_bbox(pt)) for pt in m.vertices()]), epsilon = 1e-5)

    return ipu, n_vx, n_edge, m, marker


import mesh

# Take a periodic mesh and shift it by one periodic along the input axis and merge with the original mesh.
def shift_and_merge_two_periodic_mesh(m1, marker1, m2, marker2, axis = 0, epsilon = 1e-6, flip_orientation = -1):
    input_vxs = m1.vertices()
    min_val = min(igl.bounding_box(input_vxs)[0][:, axis])
    max_val = max(igl.bounding_box(input_vxs)[0][:, axis])

    x_max_val = max(igl.bounding_box(input_vxs)[0][:, 0])
    y_max_val = max(igl.bounding_box(input_vxs)[0][:, 1])
    # Flip the copy if requested.
    flipped_vxs = m2.vertices()
    flipped_faces = m2.elements()
    if flip_orientation >= 0:
        flipped_vxs[:, flip_orientation] *= -1
        flipped_faces = flipped_faces[:, [0, 2, 1]]
    flipped_x_max_val = max(igl.bounding_box(flipped_vxs)[0][:, 0])
    flipped_y_max_val = max(igl.bounding_box(flipped_vxs)[0][:, 1])

    shift = np.array([x_max_val - flipped_x_max_val, y_max_val - flipped_y_max_val, 0.0])

    # Apply shift to copy.
    shift[axis] += max_val - min_val

    V = np.concatenate([m1.vertices(), flipped_vxs + shift])
    F = np.concatenate([m1.elements(), flipped_faces + len(m1.vertices())])
    markers = np.concatenate([marker1, marker2 + len(m1.vertices())])
    SV, SVI, SVJ, SF = igl.remove_duplicate_vertices(V, F, epsilon)
    new_markers = SVJ[markers]
    m = MeshFEM.Mesh(SV, SF)
    return m, new_markers

# Take a periodic mesh and shift it by one periodic along the input axis and merge with the original mesh.
def shift_and_merge_2D_periodic_mesh(imesh, marker, axis = 0, epsilon = 1e-6, flip_orientation = -1):
    return shift_and_merge_two_periodic_mesh(imesh, marker, imesh, marker, axis, epsilon, flip_orientation);

# Debug patterns
def get_shifted_dash_line_with_same_mesh():
    w = 3
    h = 3
    avg_len = 0.07
    pts, edges, _, _, _, _ = get_discretized_box_from_w_h(w, h, 1, avg_len)
    additional_pts = [[0, 2], [2, 2], [2, 0]]
    additional_edges = [[0, 1], [1, 2]]
    edge_points = []
    for edge in additional_edges:
        edge_points.append(np.linspace(additional_pts[edge[0]], additional_pts[edge[1]], int(np.round(la.norm(np.array(additional_pts[edge[1]]) - np.array(additional_pts[edge[0]])) / avg_len))))

    new_points = np.concatenate((edge_points[0][1:], edge_points[1][1:-1]))
    new_edges = [[i, i + 1] for i in range(len(new_points) - 1)]

    n_vx =  list(pts) + list(new_points)
    n_edge = edges + list(np.array(new_edges) + len(pts))
    start = int(len(pts) / 6) * 4
    end = int(len(pts) / 6) * 2
    n_edge += [[start, len(pts)]]
    n_edge += [[end, len(n_vx) - 1]]

    outer_pts, outer_edges, _, _, _, _ = get_discretized_box_from_w_h(5, 5, 1, avg_len, (-1, -1))

    nn_vx =  list(n_vx) + list(outer_pts)
    nn_edge = n_edge + list(np.array(outer_edges) + len(n_vx))

    additional_pts = [[-1, 2], [0, 2], [2, 0], [2, -1]]
    additional_edges = [[0, 1], [2, 3]]
    edge_points = []
    for edge in additional_edges:
        edge_points.append(np.linspace(additional_pts[edge[0]], additional_pts[edge[1]], int(np.round(la.norm(np.array(additional_pts[edge[1]]) - np.array(additional_pts[edge[0]])) / avg_len))))

    new_points = np.concatenate((edge_points[0][1:-1], edge_points[1][1:-1]))
    first_length = len(edge_points[0][1:-1])
    new_edges = [[i, i + 1] for i in range(first_length - 1)]
    new_edges += [[i + first_length , i + first_length + 1] for i in range(len(edge_points[1][1:-1]) - 1)]

    nnn_vx =  list(nn_vx) + list(new_points)
    nnn_edge = nn_edge + list(np.array(new_edges) + len(nn_vx))
    # start = int(len(pts) / 6) * 4
    # end = int(len(pts) / 6) * 2
    nnn_edge += [[start, len(nn_vx) + first_length - 1]]
    nnn_edge += [[end, len(nn_vx) + first_length]]

    n_start = int(len(outer_pts) / 15) * 10 + 2 + len(n_vx)
    n_end = int(len(outer_pts) / 15) * 5 + 8 + len(n_vx)

    nnn_edge += [[n_start, len(nn_vx)]]
    nnn_edge += [[n_end, len(nnn_vx) - 1]]


    triArea = avg_len ** 2

    m, fuseMarkers, fuseSegments = wall_generation.triangulate_channel_walls(nnn_vx, nnn_edge, triArea, flags="Y")

    wall_box_1 = get_scaled_box(np.array(pts), 1)
    wall_box_2 = get_scaled_box(np.array(pts) - [1, 1], 1)
    wall_box_3 = get_scaled_box(np.array(pts) - [1, -4], 1)
    wall_box_4 = get_scaled_box(np.array(pts) - [-4, 1], 1)
    wall_box_5 = get_scaled_box(np.array(pts) - [-4, -4], 1)

    marker1 = [point_in_box(pt, wall_box_1) for pt in m.vertices()]

    marker2 = [point_in_box(pt, wall_box_2) or point_in_box(pt, wall_box_3) or point_in_box(pt, wall_box_4) or point_in_box(pt, wall_box_5) for pt in m.vertices()]

    return m, marker1, marker2


def export_top_bottom_mesh(isheet, export_path, shape_name, pattern_name):
    mesh_3d = isheet.visualizationMesh(True)
    mesh_2d = isheet.mesh()
    vx_3d = mesh_3d.vertices()
    elements_3d = mesh_3d.elements()

    new_mesh_3d = MeshFEM.Mesh(vx_3d[:mesh_2d.numVertices()], elements_3d[:mesh_2d.numElements()])

    new_mesh_3d.save(export_path + '{}_{}_mesh_3d_top.obj'.format(shape_name, pattern_name))

    new_mesh_3d = MeshFEM.Mesh(vx_3d[mesh_2d.numVertices():], elements_3d[mesh_2d.numElements():] - mesh_2d.numVertices())
    new_mesh_3d.save(export_path + '{}_{}_mesh_3d_bottom.obj'.format(shape_name, pattern_name))
