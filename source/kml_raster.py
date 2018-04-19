# -*- coding: utf-8 -*-

import os

import gdal
import matplotlib.pyplot as plt
import numpy as np
from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY, AltitudeMode, Camera)


def make_kml(location, period, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, figs, colorbar=None, **kw):

    """TODO: LatLon bbox, list of figs, optional colorbar figure, and several simplekml kw..."""

    kml = Kml()
    altitude = kw.pop('altitude', 2e5)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)

    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)

    kml.document.camera = camera
    draworder = 0

    loc_names = {'BOG': 'Bogota D.C. y Alrededores',
                 'COL': 'Republica de Colombia'}

    str_period = kw['name']

    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        ground.visibility = kw.pop('visibility', 1)
        ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'jchavarro')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Densidad de Descargas Electricas en {} <br />'
                                                   'Periodo: {} <br />'
                                                   'Red Linet/Keraunos Suministrado IDEAM <br />'
                                                   'Elaborado por: OSPA - IDEAM <br /><br />'
                                                   '<img src="http://bart.ideam.gov.co/portal/prono_fin_semana/ospa/logo/logos.png" alt="picture" width="151" height="25" align="left" />'
                                                   '<br /><br />'.format(loc_names[location], str_period))

        ground.gxaltitudemode = kw.pop('gxaltitudemode', 'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = llcrnrlon
        ground.latlonbox.south = llcrnrlat
        ground.latlonbox.north = urcrnrlat
        ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).

        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')

    kml.savekmz(kmzfile)


def raster2kml(timestep, location, pixels=1024, interactive=False):

    """Return a Matplotlib 'fig' and 'ax' handles for a Google-Earth Image."""

    path_tifs = os.listdir('../results/rasters/density/{}/{}'.format(timestep, location))

    for path_tif in path_tifs:
        raster = gdal.Open('../results/rasters/density/{}/{}/{}'.format(timestep, location, path_tif))
        band = raster.GetRasterBand(1)
        grid = band.ReadAsArray()

        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]

        cols = raster.RasterXSize
        rows = raster.RasterYSize

        lat_min = originY + (pixelHeight * rows)
        lat_max = originY
        lon_min = originX
        lon_max = originX + (cols * pixelWidth)

        aspect = np.cos(np.mean([lat_min, lat_max]) * np.pi / 180.0)
        xsize = np.ptp([lon_max, lon_min]) * aspect
        ysize = np.ptp([lat_max, lat_min])
        aspect = ysize / xsize

        if aspect > 1.0:
            figsize = (10.0 / aspect, 10.0)
        else:
            figsize = (10.0, 10.0 * aspect)

        if not interactive:
            plt.ioff()

        fig = plt.figure(figsize=figsize, frameon=False, dpi=pixels//10)

        # KML friendly image.  If using basemap try: 'fix_aspect=False'.
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # cmap = colorbrewer.get_map('RdYlGn', 'diverging', 11, reverse=True).mpl_colormap
        cs = ax.imshow(grid, extent=(lon_min, lon_max, lat_min, lat_max), interpolation='bilinear', cmap=plt.get_cmap('hot'))
        ax.set_axis_off()

        name_fig = '../results/png/kml_raster/{}/{}/{}.png'.format(timestep, location, path_tif.split('.')[0])
        name_leg = '../results/png/kml_raster/{}/{}/leg_{}.png'.format(timestep, location, path_tif.split('.')[0])
        name_kmz = '../results/kml/raster/{}/{}/{}.kmz'.format(timestep, location, path_tif.split('.')[0])

        if timestep is 'Y':
            name = path_tif.split('.')[0].split('_')[2]
        elif timestep is 'M':
            name = path_tif[8:15]
        else:
            name = path_tif[:7]

        fig.savefig(name_fig, format='png', transparent=True)
        plt.close()

        leg_fig(cs=cs, name_leg=name_leg)
        make_kml(location=location, period=path_tif, llcrnrlon=lon_min,
                 llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                 figs=[name_fig], colorbar=name_leg, kmzfile=name_kmz,
                 name=name)


def leg_fig(cs, name_leg):
    fig = plt.figure(figsize=(1.0, 4.0), facecolor=None, frameon=False)
    ax = fig.add_axes([0.10, 0.05, 0.2, 0.9])
    cb = fig.colorbar(cs, cax=ax, format='%.1f')
    cb.set_label('DDT [rayos/km2]', rotation=-270, color='k', labelpad=5)
    fig.savefig(name_leg, transparent=False, format='png')  # Change transparent to True if your colorbar is not on space :)
    plt.close()


if __name__ == '__main__':

    res = ['Y']
    loc = ['BOG']

    for i in res:
        for j in loc:
            raster2kml(timestep=i, location=j)
