#!/usr/bin/env python

import sys
import argparse

import numpy as np
import numpy.linalg as LA
import bezier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import PIL

PDF_GS_OPS = {
    'g': 'setgray (nonstroke)',
    'G': 'setgray (stroke)',
    'gs': 'setgraphicsstate',
    'j': 'setlinejoin',
    'M': 'setmiterlimit',
    'rg': 'setrgbcolor (nonstroke)',
    'RG': 'setrgbcolor (stroke)',
    'q': 'gsave',
    'Q': 'grestore',
    'w': 'setlinewidth',
    'W': 'clip',
    'W*': 'eoclip',
}

MAP_IMG = "../resources/TZA_map.png"

def readFile(fp):
    with open(fp) as f:
        lines = [line.rstrip() for line in f]
    return lines

def appendCurve(start, controls):
    nodes = np.concatenate((start, controls))
    nodes = nodes.reshape(len(nodes)//2,2).transpose()
    #print(nodes)
    curve = bezier.Curve.from_nodes(nodes)
    return curve

def getCentroid(vertices):
    #print(vertices)
    if len(vertices):
        v = np.array(vertices)
        return np.mean(v, axis = 0)

def processBlock(lines):
    #print(list(line.rstrip() for line in lines))
    path_is_open = False
    start_xy = []
    current = []
    next_xy = []
    controls = []
    vertices = []
    line_collection = []
    draw_filled_area = False
    for line in lines:
        s = line.split()
        if not len(s):
            continue
        #print(s[-1])
        op = s[-1]
        if op == "m":
            path_is_open = True
            if (len(s) > 3):
                s = s[len(s)-3:]
            start_xy = current = np.array(s[:-1], dtype = float)
            vertices.append(current)
            print("[PATH] start point:", start_xy)
        elif op == "c":
            if path_is_open:
                #print("append bezier curve")
                controls = np.array(s[:-1], dtype = float)
                print("[PATH] bezier curve, control points:", controls)
                curve = appendCurve(current, controls)
                line_collection.append(curve)
                current = controls[-2:]
                vertices.append(current)
            else:
                print("[PATH] current path is not open to append bezier curve")
        elif op == "l":
            if path_is_open:
                print("[PATH] append line segment")
                next_xy = np.array(s[:-1], dtype = float)
                curve = appendCurve(current, next_xy)
                line_collection.append(curve)
                current = next_xy
                vertices.append(current)
            else:
                print("[PATH] current path is not open to append line segment")
        elif op == "f*":
            print("[PATH] fill region")
            path_is_open = False
            if not draw_filled_area:
                del line_collection[:]
            break
        elif op == "S":
            print("[PATH] stroke region")
            path_is_open = False
            break
        elif op == "n":
            print("[PATH] end path without fill or stroke")
            path_is_open = False
            del line_collection[:]
            #break
        elif op == "h":
            print("[PATH] close path")
            if path_is_open:
                if (current - start_xy).any():
                    print("[PATH] append line segment to close path")
                    line = appendCurve(current, start_xy)
                    line_collection.append(line)
                    current = start_xy
                    vertices.append(current)
                    path_is_open = False
            else:
                print("[PATH] current path is not open to close path")
            break
        else:
            if op in PDF_GS_OPS.keys():
                print("[PATH] got operator: " + op + " = " + PDF_GS_OPS[op])
            else:
                print("[PATH] unknown operator: " + op)
    centroid = getCentroid(vertices)
    return {'contour': line_collection, 'centroid': centroid}

def processColour(line):
    colour = { 'stroke': { 'type': None, 'val': '' },
               'nonstroke': { 'type': None, 'val': '' } }
    s = line.split()
    if not len(s):
        return
    op = s[-1]
    if op.lower() == "rg":
        print("[COLOUR] got set RGB colour operator", op)
        print(s)
        if (len(s) > 4):
            s = s[len(s)-4:]
        otype = 'stroke' if op == "RG" else 'nonstroke'
        colour[otype]['type'] = 'rgb'
        colour[otype]['val'] = np.array(s[:-1], dtype = float)

    elif op.lower() == "g":
        print("[PATH] got set gray operator", op)
        print(s)
        if (len(s) > 2):
            s = s[len(s)-2:]
        otype = 'stroke' if op == "G" else 'nonstroke'
        colour[otype]['type'] = 'gs'
        colour[otype]['val'] = np.array(s[:-1], dtype = float)

    return colour

def processImage(lines):
    #print(list(line.rstrip() for line in lines))
    img_collection = []
    rect = []
    ctm = []
    name = ""
    for line in lines:
        s = line.split()
        if not len(s):
            continue
        #print(s[-1])
        op = s[-1]
        if op == "q":
            print("[IMG] start image")
        elif op == "re":
            rect = np.array(s[:-1], dtype = float)
        elif op == "cm":
            ctm = np.array(s[:-1], dtype = float)
            print("[IMG] ctm:", ctm)
        elif op == "Q":
            if s[-2] == "Do":
                name = s[-3]
                img_collection.append({'name': name, 'clip': rect, 'ctm': ctm})
        elif op == "n":
            print("[IMG] end path")
        else:
            if op in PDF_GS_OPS.keys():
                print("[IMG] got operator: " + op + " = " + PDF_GS_OPS[op])
            else:
                print("[IMG] unknown operator: " + op)
    return img_collection

def createPlot(images, contours):
    fig, ax = plt.subplots()  # Create a figure containing a single axis
    n_col = len(plt.rcParams['axes.prop_cycle'])
    for i in range(len(contours)):
        for curve in contours[i]['contour']:
            _ = curve.plot(num_pts = 256, color = "C" + str(i%n_col), ax = ax)
        # plot centroid i
        cx, cy = contours[i]['centroid']
        plt.plot(cx, cy, "o")

    for i in range(len(images)):
        for img in images[i]:
            print("Image:", img['name'])

            ## x y w h re
            # xy = img[1][:2]
            # wh = img[1][2:]
            # w, h = img[1][2:]
            # print(xy)
            # print(wh)

            ## w 0 0 h x y cm
            ctm = img['ctm'].reshape(2,3)
            scale = [img['ctm'][0], img['ctm'][3]]
            position = img['ctm'][4:]
            w, h = scale
            xy = position
            print("Position:", xy)
            print("Size:", w, "x", h)
            
            pos_check = xy[1] + h < 450
            size_check = w > 120

            if pos_check & size_check:
                rect = mpatches.Rectangle(tuple(xy), w, h, 
                                          fc="none", ec="green")
                ax.add_patch(rect)
                arr_img = plt.imread(MAP_IMG, format='png')
                ax.imshow(arr_img, interpolation='none',
                          origin='lower',
                          extent=[xy[0], xy[0]+w, xy[1]+h, xy[1]], 
                          clip_on=True)

    _ = ax.set_xlim(0, 842)
    _ = ax.set_ylim(0, 595)
    #_ = ax.set_xlim(150, 650)
    #_ = ax.set_ylim(200, 400)
    _ = ax.set_aspect(1)
    plt.show()

def getMapGroups(images, graphics):

    map_groups = []
    for i in range(len(images)):
        for img in images[i]:
            #print("Image:", img['name'])

            # w 0 0 h x y cm
            ctm = img['ctm'].reshape(2,3)
            w, h = [img['ctm'][0], img['ctm'][3]]
            x, y = img['ctm'][4:]
            #print("Position: ", x, ",", y)
            #print("Size: ", w, "x", h)

            # Identify map images by location/size
            pos_check = y + h < 450
            size_check = w > 120

            if pos_check & size_check:
                # Get graphics within map boundary
                graphics_list = []
                for gfx in graphics:
                    ix, iy = gfx['path']['centroid']
                    #print("Centroid: ", ix, ",", iy)
                    x_check = (x < ix) & (ix < x + w)
                    y_check = (y < iy) & (iy < y + h)
                    if x_check & y_check:
                        graphics_list.append(gfx)
                #print("Got graphics:", len(graphics_list))
                map_groups.append((img, graphics_list))

            def getXPos(mg):
                return mg[0]['ctm'][4]

            map_groups.sort(key = getXPos)
    return map_groups

def plotMapGroup(map_group, ax):
    img, graphics = map_group

    print("Image:", img['name'])

    # Construct transformation matrix
    #   a b 0 
    #   c d 0
    #   e f 1
    m = np.hstack((img['ctm'].reshape(3,2), np.array([[0],[0],[1]])))
    try:
        m_inv = LA.inv(m)
    except LinAlgError:
        sys.exit("Could not invert transformation matrix")

    # Create new transformation matrix for coordinate system that
    # displays canonical map using the size of the original image and
    # position at origin
    dim = PIL.Image.open(MAP_IMG).size
    m2 = np.array([[dim[0],0,0],[0,dim[1],0],[0,0,1]])
    # FIXME transform from canonical image coords to PlateCarree projection:
    # m3 = np.array([[13.07, 0,     0],
    #                [0,     11.255,0],
    #                [28.405,-12,   1]])

    n_col = len(plt.rcParams['axes.prop_cycle'])
    col = "C" + str(i%n_col)
    print("Processing graphics:", len(graphics))
    for i in range(len(graphics)):
        if graphics[i]['colour'] is not None:
            #print("got colour state:", graphics[i]['colour'])
            # Get stroke colour specification
            col_spec = graphics[i]['colour']['stroke']
            if col_spec['type'] is None:
                fill_col = graphics[i]['colour']['nonstroke']
                if fill_col['type'] == 'rgb':
                    # Use fill colour specification
                    col_spec = fill_col
            if col_spec['type'] == 'rgb':
                # Set rgb colour
                col = tuple(col_spec['val'])
            elif col_spec['type'] == 'gs':
                # Set grayscale colour
                col = tuple(col_spec['val']) * 3
            else:
                col = (0.,0.,0.)
        for curve in graphics[i]['path']['contour']:
            ## Relocate curve according to new coordinate system
            nodes = curve.nodes
            nodes_new = []
            for i in range(len(nodes.T)):
                v = np.append(nodes.T[i], 1)
                v1 = np.matmul(v, m_inv)
                v2 = np.matmul(m2, v1.T).flatten()
                nodes_new.append(v2[:-1])
            nodes = np.array(nodes_new).T
            curve = bezier.Curve.from_nodes(nodes)
            _ = curve.plot(num_pts = 256, color = col, ax = ax)
        # plot centroid i
        # cx, cy = graphics[i]['path']['centroid']
        # plt.plot(cx, cy, "o")

    # Get position and scaling for display on rescaled coordinate system
    #   w 0 0 
    #   0 h 0
    #   x y 1
    ctm = m2
    w, h = ctm[:2,:2].diagonal()
    x, y  = ctm[2,:2]
    print("Position:", x, ",", y)
    print("Size:", w, "x", h)

    arr_img = plt.imread(MAP_IMG, format='png')
    ax.imshow(arr_img, interpolation='none',
              origin='upper',
                      extent=[x, x+w, y, y+h], 
                      clip_on=True)

    _ = ax.set_xlim(x, x+w)
    _ = ax.set_ylim(y, y+h)
    _ = ax.set_aspect(1)
    ## end of plotMapGroup()

# Main
def main():

    parser = argparse.ArgumentParser(description='Process TMA PDF contents')
    parser.add_argument('filepath', nargs=1, type=str)
    args = parser.parse_args()

    try:
        # read lines from input file
        lines = readFile(args.filepath[0])
    except: 
        # IOError
        print("Input file not found:", args.filepath[0])
        sys.exit(4)

    path_ops = {'m', 'c', 'l'}
    term_ops = {'f*', 'S', 'n', 'h'}
    col_ops = {'rg', 'RG', 'g', 'G'}
    block = []
    graphics = []
    images = []
    col = None
    # Iterate over the lines
    for line in lines:
        if line.endswith(tuple(path_ops)):
            #print("got path operator")
            block.append(line)
        elif line.endswith(tuple(term_ops)):
            #print("got terminating path operator")
            block.append(line)
            path = processBlock(list(block))
            if len(path['contour']):
                graphics.append({'path': path, 'colour': col})
            del block[:]
        elif line.endswith(tuple(col_ops)):
            block.append(line)
            col = processColour(line)
        elif "Do" in line:
            #print("got image operator")
            block.append(line)
            image = processImage(list(block))
            if len(image):
                images.append(image)
            del block[:]
        else:
            block.append(line)

    print(len(graphics))
    print(len(graphics[0]['path']))
    print(len(images))

    mgroups = getMapGroups(images, graphics)

    #createPlot(images, graphics)
    fig, axs = plt.subplots(1, 4)
    for mg, ax in zip(mgroups, axs.flat):
        plotMapGroup(mg, ax)
    plt.show()
    ## end of main()

# Run main
if __name__ == "__main__":
    main()
