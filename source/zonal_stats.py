# -*- coding: latin-1 -*-

import os
import sys

import gdal
import numpy as np
import ogr
import osr
import pandas as pd

'''
Este codigo estima el promedio (si se requiere otro estadistico se debe cambiar
en la funcion zonal stats) de un conjunto de raster para zonas
definidas por un shapefile. El script devuelve un archivo excel con el valor
estadistico para cada area en forma de serie. Si el analisis es para mas de
una variable, el archivo de excel tendra tantas hojas como variables con los
resultados. Se deben organizar en sub-carpetas los distintos rasters de acuerdo
a cada variable.

El archivo shapefile debera tener dentro de sus atributos la columna "Name" con
el nombre de cada poligono.
'''

__author__ = 'jchavarro'


def zonal_stats(feat, input_zone_polygon, input_value_raster, FID):
    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR, targetSR)
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if geom.GetGeometryName() == 'MULTIPOLYGON':
        count = 0
        pointsX = []
        pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            count += 1
    elif geom.GetGeometryName() == 'POLYGON':
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []
        pointsY = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # print pixelWidth

    # Specify offset and rows and columns to read
    xoff = int(np.round((xmin - xOrigin) / pixelWidth, decimals=0))
    yoff = int(np.round((yOrigin - ymax) / pixelWidth, decimals=0))
    xcount = int(np.round((xmax - xmin) / pixelWidth, decimals=0))
    ycount = int(np.round((ymax - ymin) / pixelWidth, decimals=0))

    if xcount == 0 or ycount == 0:
        xcount = 1
        ycount = 1
    elif xoff < 0:
        xcount = xcount + xoff
        xoff = 0.
    elif yoff < 0.:
        ycount = ycount + yoff
        yoff = 0.

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, gdal.GDT_Byte)
    target_ds.SetGeoTransform((xmin, pixelWidth, 0, ymax, 0, pixelHeight,))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Create a layer for Rasterize (only feature selected)
    outShapefile = "../temp/shape_temp.shp"
    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)

    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(outShapefile)
    # print type(outDataSource)
    outLayer = outDataSource.CreateLayer("../temp/shape_temp", sourceSR, geom_type=ogr.wkbPolygon)

    # Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    outLayer.CreateField(idField)

    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(geom)
    feature.SetField("id", 1)
    outLayer.CreateFeature(feature)

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], outLayer, burn_values=[1])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)

    try:
        dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float64)
    except Exception as e:

        if xoff + xcount > raster.RasterXSize:
            adj_xcount = raster.RasterXSize - xoff
        else:
            adj_ycount = raster.RasterYSize - yoff

        dataraster = banddataraster.ReadAsArray(xoff, yoff, adj_xcount, adj_ycount).astype(np.float64)

    dataraster = np.where(dataraster == banddataraster.GetNoDataValue(), 0., dataraster)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float64)

    # Mask zone of raster
    zoneraster = np.ma.masked_array(dataraster, np.logical_not(datamask))

    valor = np.ma.mean(zoneraster)
    if valor is np.ma.masked:
        valor = np.mean(dataraster)
    else:
        pass
    return valor


def loop_zonal_stats(stat, input_zone_polygon, input_value_raster):
    TimeStep = input_value_raster.split('/')[-1]
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    feature_list = range(lyr.GetFeatureCount())
    for FID in feature_list:
        feat = lyr.GetFeature(FID)
        stat = stat.append({'COD_DEPART': feat.GetField("COD_DEPART"), 'Count': zonal_stats(feat, input_zone_polygon, input_value_raster, FID), 'TimeStep': TimeStep}, ignore_index=True)
    return stat


def fn_setup(stat, input_zone_polygon, input_value_raster):

    return loop_zonal_stats(stat, input_zone_polygon, input_value_raster)


if __name__ == "__main__":

    resolution = 'M'
    stats_var = 'density'
    loc = 'COL'

    fromfile = False
    project = 'LightStats_{}_{}_{}'.format(resolution, stats_var, loc)
    raster_files = os.listdir('../rasters/{}/{}'.format(stats_var, resolution))

    date_start = {'density': '1/1/2018'}

    if loc is 'COL':
        shapes = 'Colombia_Continental.shp'
        col = ['COD_DEPART', 'Count', 'TimeStep']
    else:
        shapes = 'Bog_localidades.shp'
        col = ['LocCodigo', 'Count', 'TimeStep']

    xls_writer = pd.ExcelWriter('../xlsx/{}_2018.xlsx'.format(project))

    statistics = pd.DataFrame(data=None, columns=col)

    for f in raster_files:

        statistics = fn_setup(stat=statistics, input_zone_polygon='../gis/{}'.format(shapes), input_value_raster='../rasters/{}/{}/{}'.format(stats_var, resolution, f))

    df_statistics = statistics.pivot(index='TimeStep', columns='COD_DEPART', values='Count')
    df_index = pd.date_range(start=date_start[stats_var], periods=len(df_statistics), freq=resolution)
    df_statistics.loc[:, 'Date'] = df_index
    df_statistics.set_index(keys='Date', inplace=True)

    df_statistics.to_excel(xls_writer, sheet_name=stats_var, merge_cells=False)

    xls_writer.save()
