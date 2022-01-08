import rasterio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from scipy.special import erfc
import pyproj
import rioxarray

def timedil(t, origin, min, max):
    def f(t, origin):
        return (t-origin)**3 + (t-origin)/30
    results = f(t, origin)*max/f(max, origin)
    return results

# a = np.linspace(0, 1, 1000)
# plt.plot(a, (a-0.5)**3 + (a-0.5)**1/10)
# plt.plot(a, timedil(a, 0.5, 0., 1.))
# plt.show()
# pause()
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_raster(in_path, out_path):

    """
    """
    crs = "EPSG:3857"
    # reproject raster to project crs
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest)
    return(out_path)

# Reprojects the .tiff in mercator coordinates
# reproject_raster("data.tiff", "data_new.tiff")
# pause()

fp = r'ETOPO1_Bed_g_geotiff.tif'

# Cylindric data, get limits
fp = r'data.tiff'
dataset = rasterio.open(fp, crs='EPSG:3857')
print(dataset.crs)
print(dataset)
bounds = dataset.bounds

# Mercator data, get data
fp = r'data_new.tiff'
dataset = rasterio.open(fp)
print(dataset.crs)
print(dataset)
data = dataset.read(1)

print(np.min(data), np.max(data))
print(bounds.left, bounds.right, bounds.bottom, bounds.top)
data_masked = np.ma.masked_where((data > 1e20), data)

fig = plt.figure(figsize=(8,8))
plt.axis('off')

m = Basemap(projection='merc', llcrnrlon=bounds.left, llcrnrlat=bounds.bottom, urcrnrlon=bounds.right, urcrnrlat=bounds.top, resolution='h')
# m.drawcountries(linewidth=0.10)
# m.drawcoastlines(linewidth=0.10)


xlow, ylow = m(bounds.left, bounds.bottom)
xup, yup = m(bounds.right, bounds.top)
extent = [xlow, xup, ylow, yup]

#ax1 = plt.imshow(data, extent=extent, origin="upper", cmap='viridis')
ax1 = m.bluemarble()
ax2 = plt.imshow(data_masked, extent=extent, cmap='Blues')
txt = plt.text(2e6, 0.95*yup, f"SEA LEVEL {round(-6920)}m")

Nframes = 1000

#
# ax1 = plt.imshow(data, cmap="afmhot_r")
# ax2 = plt.imshow(data_masked, cmap="Blues")
# plt.show()



def animate(i):
    print(i, "/", Nframes)
    dt = 5390+6920
    origin = -6920
    thres = origin + i * dt / Nframes

    thres = timedil(origin + i * dt / Nframes, origin+dt/2, origin, origin+dt)

    #thres = -10 + i * (10+10) / Nframes

    data_masked = np.ma.masked_where((data > thres), data)
    ax2.set_array(data_masked)
    txt.set_text(f"SEA LEVEL {round(thres)}m")
    return ax2, txt

ani = animation.FuncAnimation(fig, animate, frames=list(range(Nframes)), blit=True, interval=30000/Nframes, repeat=True)
ani.save("video_6.mp4")
print("Saved")

plt.show()

