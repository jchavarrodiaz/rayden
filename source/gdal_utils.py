import os
import sys

import numpy as np
import osr
import pandas as pd
from osgeo import gdal, gdalnumeric
from pyproj import Proj, transform

platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }

if sys.platform not in platforms:
    print sys.platform

if platforms[sys.platform] is not 'Windows':
    exec_prefix = sys.exec_prefix
    gdal_data = '{}/share/gdal/'.format(exec_prefix)
    os.environ['GDAL_DATA'] = gdal_data
if platforms[sys.platform] is 'Windows':
    exec_prefix = sys.exec_prefix
    gdal_data = '{}/Library/share/gdal/'.format(exec_prefix)
    os.environ['GDAL_DATA'] = gdal_data


def coor2pix(trans_data, x, y):
    """
    Returns geo-matrix pixel index from coordinates x, y.
    :param trans_data: geo-transformation data (GetTransform())
    :param x: x
    :param y: y
    :return: geo-matrix index
    """
    px = int((x - trans_data[0]) / trans_data[1])  # x pixel
    py = int((y - trans_data[3]) / trans_data[5])  # y pixel

    return px, py


def array2raster(new_raster_fn, geo_transform, array, no_data_value=0., epsg=4326, metadata=None):
    """
    Saves an array as an raster file.
    :param new_raster_fn: New raster filename
    :param geo_transform: Geo-Transformation matrix
    :param array: array input
    :param no_data_value: no data value
    :param epsg: epsg code for output raster
    :param metadata:
    :return:
    """

    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(new_raster_fn, cols, rows, 1, gdal.GDT_Float32)

    if metadata:
        out_raster.SetMetadata(metadata)

    out_raster.SetGeoTransform(geo_transform)
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(no_data_value)
    out_raster_srs = osr.SpatialReference()
    out_raster_srs.ImportFromEPSG(epsg)
    out_raster.SetProjection(out_raster_srs.ExportToWkt())
    out_band.FlushCache()


def calculate_pixel_size(x_max, x_min, y_max, y_min, nx, ny):
    """
    Returns pixel size from extents and number of pixels.
    :param x_max:
    :param x_min:
    :param y_max:
    :param y_min:
    :param nx:
    :param ny:
    :return:
    """

    pixel_width = abs(x_min - x_max) / nx
    pixel_height = abs(y_min - y_max) / (ny - 1)

    return pixel_width, pixel_height


def get_value_from_raster(raster_filename, x, y):
    """
    Gets values from a raster file based on given points (x, y).
    :param raster_filename:
    :param y:
    :param x:
    :return:
    """
    points = np.vstack((x, y)).T
    raster = gdal.Open(raster_filename)
    trans_data = raster.GetGeoTransform()
    band = raster.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()
    values = []

    for point in points:
        mx = point[0]
        my = point[1]
        px = int((mx - trans_data[0]) / trans_data[1])
        py = int((my - trans_data[3]) / trans_data[5])

        try:
            raster_value = band.ReadAsArray(px, py, 1, 1)[0][0]

        except TypeError, e:
            print("Point not found ({}, {}). {}".format(mx, my, e))
            raster_value = no_data_value

        if raster_value == no_data_value:
            raster_value = float('nan')

        values.append(raster_value)

    return values


def get_elev_from_dem():
    """
    Returns elevation based on a defined dem.
    :return:
    """
    xls_input = pd.ExcelFile('../data/catalogos_ideam.xlsx')
    xls_elevations = pd.ExcelWriter('../results/elevations.xlsx')
    databases = ['postgresql', 'cassandra', 'sshm']

    for database in databases:
        df_catalog = xls_input.parse(database, index_col='codigo')
        x = df_catalog['lng']
        y = df_catalog['lat']

        z = get_value_from_raster('../data/srtm_colombia.tif', x=x, y=y)
        df_catalog['Elevation'] = z
        df_catalog['Elevation'][df_catalog['Elevation'] < 0] = 0
        df_catalog.to_excel(xls_elevations, database)

    xls_elevations.save()


def transform_coordinates(x, y, epsg_in, epsg_out):
    """
    Transforms from an input src (epsg_in) to another one (epsg_out)
    :param x:
    :param y:
    :param epsg_in:
    :param epsg_out:
    :return:
    """
    in_proj = Proj(init='epsg:{}'.format(epsg_in))
    out_proj = Proj(init='epsg:{}'.format(epsg_out))
    xi, yi = transform(in_proj, out_proj, x, y)
    return xi, yi


def isothermal():
    alpha = -0.0055182680606
    beta = 28.2088488402
    raster_filename = '../data/SRTM_Bog_DEM.tif'
    raster = gdal.Open(raster_filename)
    trans_data = raster.GetGeoTransform()
    band = raster.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()

    output = gdalnumeric.BandReadAsArray(band) * alpha + beta
    pd.DataFrame(output).to_clipboard()
    pass


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }

    if sys.platform not in platforms:
        return sys.platform

    if platforms[sys.platform] is not 'Windows':
        exec_prefix = sys.exec_prefix
        gdal_data = '{}/share/gdal/'.format(exec_prefix)
        os.environ['GDAL_DATA'] = gdal_data


if __name__ == '__main__':
    # get_elev_from_dem()
    pass
