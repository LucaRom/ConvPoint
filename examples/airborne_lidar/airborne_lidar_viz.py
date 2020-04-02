import numpy as np
import colorsys
from plyfile import PlyData, PlyElement


def prediction2ply(filename, xyz, prediction, info_class):
    """write a ply with colors for each class"""
    color = np.zeros(xyz.shape)
    for key in info_class.keys():
        color[np.where(prediction == info_class[key]['mode']), :] = info_class[key]['color']
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


def error2ply(filename, xyz, labels, prediction, info_class):
    """write a ply with green for correct classifcation and red for error"""
    color_rgb = np.zeros(xyz.shape)

    if labels[0] == prediction[0]:
        color_rgb[np.where(prediction == 0 and labels == 0), :] = [0, 255, 0]
    unique_classes = [0]
    for key in info_class.keys():
        if info_class[key]['mode'] not in unique_classes:
            unique_classes.append(info_class[key]['mode'])

    for cla in unique_classes:
        if labels[cla] == prediction[cla]:
            color_rgb[np.where(prediction == cla), :] = [0, 255, 0]  # Green
        else:
            color_rgb[np.where(prediction == cla), :] = [255, 0, 0]  # Red
    color_rgb[np.equal(prediction, labels), :] = [0, 255, 0]  # Green
    color_rgb[np.not_equal(prediction, labels), :] = [255, 0, 0]  # Red

    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color_rgb[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


    for i_ver in range(0, len(labels)):

        color_hsv = list(colorsys.rgb_to_hsv(color_rgb[i_ver, 0], color_rgb[i_ver, 1], color_rgb[i_ver, 2]))
        if (labels[i_ver] == prediction[i_ver]) or (labels[i_ver] == 0):
            color_hsv[0] = 0.333333
        else:
            color_hsv[0] = 0
        color_hsv[1] = min(1, color_hsv[1] + 0.3)
        color_hsv[2] = min(1, color_hsv[2] + 0.1)
        color_rgb[i_ver, :] = list(colorsys.hsv_to_rgb(color_hsv[0], color_hsv[1], color_hsv[2]))
    color_rgb = np.array(color_rgb * 255, dtype='u1')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color_rgb[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
