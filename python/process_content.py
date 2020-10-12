#!/usr/bin/env python

import sys
import argparse

import numpy as np
import numpy.linalg as LA
import bezier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import skimage.draw
import xarray as xr
import pandas as pd

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

# Extent of original map image when matched to PlateCarree projection
extent_MAP_IMG = [28.405, 41.475, -12., -0.745]

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
                centroids = []
                for gfx in graphics:
                    ix, iy = gfx['path']['centroid']
                    #print("Centroid: ", ix, ",", iy)
                    x_check = (x < ix) & (ix < x + w)
                    y_check = (y < iy) & (iy < y + h)
                    if x_check & y_check:
                        print("Centroid: ", ix, ",", iy)
                        if (ix, iy) not in centroids:
                            graphics_list.append(gfx)
                            centroids.append((ix, iy))
                        else:
                            print("Already added graphics with matching centroid")
                #print("Got graphics:", len(graphics_list))
                map_groups.append((img, graphics_list))

            def getXPos(mg):
                return mg[0]['ctm'][4]

            map_groups.sort(key = getXPos)
    return map_groups

def transformMapGroup(map_group):
    img, graphics = map_group

    print("Image:", img['name'])

    # Construct current transformation matrix for image
    #   a b 0
    #   c d 0
    #   e f 1
    m1 = np.hstack((img['ctm'].reshape(3,2), np.array([[0],[0],[1]])))
    try:
        m1_inv = LA.inv(m1)
    except LinAlgError:
        sys.exit("Could not invert transformation matrix")

    # Create transformation matrix to map from canonical image coords
    # to extent of original map image matched to PlateCarree projection
    lon_min, lon_max, lat_min, lat_max = extent_MAP_IMG
    tm = np.array([lon_max - lon_min, 0, 0, lat_max - lat_min, lon_min, lat_min])
    m2 = np.hstack((tm.reshape(3,2), np.array([[0],[0],[1]])))

    # Pre-multiply transformation matrices
    m = np.matmul(m1_inv, m2)

    graphics_new = []
    print("Processing graphics:", len(graphics))
    for i in range(len(graphics)):
        col = None
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
        contour = []
        for curve in graphics[i]['path']['contour']:
            ## Relocate curve according to new coordinate system
            nodes = curve.nodes
            nodes_new = []
            for i in range(len(nodes.T)):
                # Multiply node by combined transformation matrix m to
                # get coordinates with respect to image space and
                # transform from canonical image coords to PlateCarree
                # map projection
                v = np.matmul(np.append(nodes.T[i], 1), m)
                nodes_new.append(v[:-1])
            nodes = np.array(nodes_new).T
            curve_new = bezier.Curve.from_nodes(nodes)
            contour.append(curve_new)
        # Relocate centroid i
        centroid = np.matmul(np.append(graphics[i]['path']['centroid'], 1), m)[:-1]
        path = { 'colour': col,
                 'contour': contour,
                 'centroid': centroid }
        graphics_new.append({ 'path': path})

    return (img, graphics_new)
    ## end of transformMapGroup()

def plotMapGroup(map_group, ax):
    _, graphics = map_group
    n_col = len(plt.rcParams['axes.prop_cycle'])
    print("Processing graphics:", len(graphics))
    for i in range(len(graphics)):
        col = "C" + str(i%n_col)
        if graphics[i]['path']['colour'] is not None:
            col = graphics[i]['path']['colour']
        for curve in graphics[i]['path']['contour']:
            _ = curve.plot(num_pts = 256, color = col, ax = ax)
        # plot centroid i
        cx, cy = graphics[i]['path']['centroid']
        ax.plot(cx, cy, "o")

    ## end of plotMapGroup()

def getAlertMasks(map_group):
    _, graphics = map_group

    # mask will have shape defined by the image map extent divided
    # into 0.1 degree grid
    res = 0.1
    # lon_min, lon_max, lat_min, lat_max = [round(x, 1) for x in extent_MAP_IMG]
    # nx, ny = np.array([lon_max - lon_min, lat_max - lat_min])/res
    # img_shape = (round(nx), round(ny), 3)
    lon_min, lon_max, lat_min, lat_max = extent_MAP_IMG
    x = np.arange(lon_min, lon_max, res) # [round(x,1) for x in x]
    y = np.arange(lat_min, lat_max, res) # [round(y,1) for y in y]
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack((xx.ravel(), yy.ravel())).T

    # Create transformation matrix
    tm = np.array([res, 0, 0, res, lon_min, lat_min])
    m = np.hstack((tm.reshape(3,2), np.array([[0],[0],[1]])))
    try:
        m_inv = LA.inv(m)
    except LinAlgError:
        sys.exit("Could not invert transformation matrix")

    mask_list = []
    for i in range(len(graphics)):
        col = graphics[i]['path']['colour']
        if col is not None and col.count(col[0]) != 3:
            # got a contour with RGB colour
            print(col)
            alert_val = 0
            r, g, b = col
            if col == (0.0, 0.0, 0.0):
                print("colour: black")
            elif col == (1.0, 1.0, 0.0):
                print("colour: yellow")
                alert_val = 1
            elif g > 0.33 and g < 0.66:
                # (0.89, 0.424, 0.0392)
                # (0.969, 0.588, 0.275)
                print("colour: orange")
                alert_val = 2
            elif g < 0.33:
                print("colour: red")
                alert_val = 3
            else:
                print("colour: other")

            #img = np.zeros(img_shape, dtype = np.double)
            img2 = np.array([alert_val]*xx.size).reshape(xx.shape)
            img = np.zeros(xx.shape, dtype = np.double)

            # nodes_new = []
            # for curve in graphics[i]['path']['contour']:
            #     # transform curve to grid coords
            #     nodes = curve.nodes
            #     for i in range(len(nodes.T)):
            #         # Multiply node by transformation matrix m to
            #         # get grid coordinates
            #         v = np.matmul(np.append(nodes.T[i], 1), m_inv)
            #         nodes_new.append(v[:-1])
            # nodes = np.array([node.round() for node in nodes_new])
            # mask = skimage.draw.polygon2mask(img_shape[:-1], nodes)
            # img[mask] = col
            #mask_list.append(img)

            ## alternative approach
            vv = np.vstack([curve.nodes.T for curve in graphics[i]['path']['contour']])
            # construct a Path from the vertices
            pth = Path(vv, closed=False)

            # test which pixels fall within the path
            mask = pth.contains_points(xy)

            # reshape to the same size as the grid
            mask = mask.reshape(xx.shape)

            # create a masked array
            masked = np.ma.masked_array(img2, ~mask)

            # or simply set values for masked pixels
            img[mask] = alert_val

            # combine with coords...
            am = np.stack((xx, yy, img))

            mask_list.append(am)
    return mask_list
    ## end

def createGriddedData(map_groups, alert_data, file_path=None):
    # container for gridded data layers
    vars = {}

    # data will have shape defined by the image map extent divided
    # into 0.1 degree grid
    res = 0.1
    lon_min, lon_max, lat_min, lat_max = extent_MAP_IMG
    x = np.arange(lon_min, lon_max, res) # [round(x,1) for x in x]
    y = np.arange(lat_min, lat_max, res) # [round(y,1) for y in y]
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack((xx.ravel(), yy.ravel())).T

    # Create transformation matrix
    tm = np.array([res, 0, 0, res, lon_min, lat_min])
    m = np.hstack((tm.reshape(3,2), np.array([[0],[0],[1]])))
    try:
        m_inv = LA.inv(m)
    except LinAlgError:
        sys.exit("Could not invert transformation matrix")

    for i, mg in enumerate(map_groups):
        _, graphics = mg
        print(i)

        # count arrays added for this group
        n = 0

        for j, gfx in enumerate(graphics):
            col = gfx['path']['colour']
            #breakpoint()
            if col is not None and col.count(col[0]) != 3:
                # got a contour with RGB colour
                print(col)
                alert_val = 0
                r, g, b = col
                if col == (0.0, 0.0, 0.0):
                    print("colour: black")
                elif col == (1.0, 1.0, 0.0):
                    print("colour: yellow")
                    alert_val = 1
                elif g > 0.33 and g < 0.66:
                    # (0.89, 0.424, 0.0392)
                    # (0.969, 0.588, 0.275)
                    print("colour: orange")
                    alert_val = 2
                elif g < 0.33:
                    print("colour: red")
                    alert_val = 3
                else:
                    print("colour: other")

                img = np.zeros(xx.shape, dtype = np.double)

                # get nodes for the alert area
                vv = np.vstack([curve.nodes.T for curve in gfx['path']['contour']])

                # construct a Path from the vertices
                pth = Path(vv, closed=False)

                # test which pixels fall within the path
                mask = pth.contains_points(xy)

                # reshape to the same size as the grid
                mask = mask.reshape(xx.shape)

                # set values for masked pixels
                img[mask] = alert_val

                da = xr.DataArray(data=img, dims=["lat", "lon"], coords=[y, x])

                da.attrs = {
                    'issue_date': alert_data.loc[i,'issue_date'],
                    'alert_date': alert_data.loc[i,'date'],
                    'alert_day': alert_data.loc[i,'day'],
                    'alert_weekday': alert_data.loc[i,'weekday'],
                    'alert_id': n+1,
                    'alert_type': '',
                    'alert_text': alert_data.loc[i,'alert_text'],
                }

                var_name = '_'.join(['alert', 'day'+str(da.attrs['alert_day']),
                                     str(da.attrs['alert_id'])])
                vars[var_name] = da
                n += 1

    # combine data arrays into data set
    issue_date = alert_data.loc[0, 'issue_date']
    ds = xr.Dataset(data_vars=vars,
                    attrs={
                        'title': 'TMA weather warnings for ' + issue_date,
                        'issue_date': issue_date,
                    })
    if file_path is None:
        file_path = 'TMA_weather_warning_'+issue_date+'.nc'
    ds.to_netcdf(file_path)
    ## end

# Main
def main():

    parser = argparse.ArgumentParser(description='Process TMA PDF contents')
    parser.add_argument('filepath', nargs=1, type=str)
    parser.add_argument('metadata', nargs=1, type=str)
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
    mgroups = [transformMapGroup(mg) for mg in mgroups]

    try:
        # Get associated data - one row per forecast date
        alert_data = pd.read_csv(args.metadata[0])
    except FileNotFoundError:
        print("Couldn't read metadata file:", args.metadata[0])
    else:
        fn = args.metadata[0].split(".")[0].split("/")[-1] + ".nc"
        createGriddedData(mgroups, alert_data, fn)

    # am = getAlertMasks(mgroups[0])
    # #fig, axs = plt.subplots(1, len(am))
    # fig, ax = plt.subplots()
    # #for m, ax in zip(am, axs.flat):
    # for m in am:
    #     print(m.shape)
    #     ax.imshow(m, origin = 'lower', alpha=.3, vmin=0, vmax=1)
    # plt.show()

    # # Downloaded from https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_TZA_shp.zip
    # shp_fname = '../../mapping/gadm/shapefile/gadm36_TZA_1.shp'
    # adm1_shapes = list(shpreader.Reader(shp_fname).geometries())

    # #createPlot(images, graphics)
    # fig, axs = plt.subplots(1, 4, subplot_kw={'projection': ccrs.PlateCarree()})
    # for mg, ax in zip(mgroups, axs.flat):

    #     ax.set_extent(extent_MAP_IMG, ccrs.PlateCarree())
    #     ax.coastlines(resolution='10m')
    #     ax.add_feature(cfeature.LAND)
    #     ax.add_feature(cfeature.LAKES, alpha=0.5)
    #     ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
    #                       edgecolor='black', facecolor='gray', alpha=0.5)
    #     arr_img = plt.imread(MAP_IMG, format='png')
    #     ax.imshow(arr_img, interpolation='none',
    #               origin='upper',
    #               extent=extent_MAP_IMG,
    #               clip_on=True)

    #     plotMapGroup(mg, ax)
    # plt.show()
    ## end of main()

# Run main
if __name__ == "__main__":
    main()
