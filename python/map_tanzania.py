import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt

MAP_IMG = "../resources/TZA_map.png"

# Downloaded from https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_TZA_shp.zip
fname = '../resources/gadm/shapefile/gadm36_TZA_1.shp'

adm1_shapes = list(shpreader.Reader(fname).geometries())

ax = plt.axes(projection=ccrs.PlateCarree())

plt.title('Tanzania')
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.LAKES, alpha=0.5)

ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
                  edgecolor='black', facecolor='gray', alpha=0.5)

# To determine extent_MAP_IMG, started by plotting axes with
# extent_TZA then adjusted the extent until the canonical map image
# matched boundaries from the Tanzania shape file plotted in Plate
# Carree projection.
extent_TZA = [29., 41., -12., 0.]

# extent_img = np.array(extent_TZA)
# extent_img[0] -= 0.595
# extent_img[1] += 0.475
# extent_img[3] = -0.745

extent_MAP_IMG = [28.405, 41.475, -12., -0.745]

ax.set_extent(extent_MAP_IMG, ccrs.PlateCarree())

arr_img = plt.imread(MAP_IMG, format='png')
ax.imshow(arr_img, interpolation='none',
          origin='upper',
          extent=extent_MAP_IMG,
          clip_on=True)

plt.show()
