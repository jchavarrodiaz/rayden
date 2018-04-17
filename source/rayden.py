import itertools
import os
import sys

import gdal
import numpy as np
import ogr
import osr
import pandas as pd
from geopy.distance import vincenty

from config_utils import get_pars_from_ini
from web_utils import get_all_files
from web_utils import get_all_names

platforms = {
    'linux1': 'Linux',
    'linux2': 'Linux',
    'darwin': 'OS X',
    'win32': 'Windows'
}


# if sys.platform not in platforms:
#     print sys.platform
#
# if platforms[sys.platform] is not 'Windows':
#     exec_prefix = sys.exec_prefix
#     gdal_data = '{}/share/gdal/'.format(exec_prefix)
#     os.environ['GDAL_DATA'] = gdal_data
# elif platforms[sys.platform] is 'Windows':
#     exec_prefix = sys.exec_prefix


def fn_make_summary_light(files, outputfile):
    cols_headers = ['YMD', 'Hour', 'Latitude', 'Longitude', 'Height', 'Type', 'amperage', 'Error']
    df_summary = pd.DataFrame()

    for f in files:
        df_data = pd.read_csv('../data/rawdata/{}'.format(f), delim_whitespace=True, header=None)
        df_summary = pd.concat([df_summary, df_data], axis=0)
        print f

    df_summary.columns = cols_headers
    df_summary['str_time'] = df_summary['YMD'].astype(str) + ' ' + df_summary['Hour'].astype(str)
    df_summary['Date'] = pd.to_datetime(df_summary['str_time'], infer_datetime_format=True, format='%Y/%m/%d %H:%M:%S')
    df_summary.set_index('Date', inplace=True)
    df_summary.index.name = 'Date'
    df_summary.drop(['YMD', 'Hour', 'str_time'], axis=1, inplace=True)

    df_summary.to_csv(path_or_buf=outputfile, sep=',', header=True)


def fn_read_summary_csv(path_file):
    return pd.read_csv(path_file, index_col='Date')


def fn_add_flashes(strokes):
    strok = strokes.strokes
    count = pd.Series(data=0, index=strok.index, name='Count')
    clean_count = pd.Series(data=0, index=strok.index, name='Clean_Count')

    for i in range(1, len(strok[1:]) + 1):
        if strok[i] == strok[i - 1]:
            count[i] = count[i - 1]
        else:
            count[i] = count[i - 1] + strok[i]

    clean_count = clean_count.where(strok == 0, count)
    count_lag = clean_count.shift(-1)
    count_lag[-1] = 0
    flashes = pd.Series(data=np.where(clean_count == count_lag, count_lag, clean_count + count_lag), index=count.index, name='flashes').astype(int)
    strokes.flashes = flashes

    flash_group = flashes.drop_duplicates()[1:].astype(int)
    idx = pd.Series(index=flash_group.values)

    for i in flash_group:
        idx[i] = strokes[strokes.flashes == i]['abs_amperage'].idxmax()

    rem_strokes = strokes[strokes.flashes == 0]
    add_strokes = strokes.loc[idx, :]

    return pd.concat(objs=[rem_strokes, add_strokes], axis=0)


def fn_filter_light(raw, max_amp=10, max_error=1, delta_strokes=1, delta_x=1):
    """
    Esta funcion filtra los datos de descargas electricas por los criterios de
    voltaje, error espacial y agrega los eventos a nivel de flash.
    :param raw: son los datos consolidados en una sola matriz desde los txt unitarios
    almacenados en http://172.16.1.237/almacen/externo/varios/Rayos/TXT/Consolidado24H/
    NO deben ser tenidos en cuenta
    :param max_amp: es el umbral minimo de voltaje, eventos con menor magnitud (valor absoluto)
    NO deben ser tenidos en cuenta
    :param max_error: es el grado de precision de la estimacion, valores de error por encima de 1km
    para el distrito capital deben ser eliminados. Para el resto del Pais es aceptable un umbral de
    error mayor.
    :param delta_strokes: es el intervalo minimo de tiempo en seg entre dos registros para que estos sean considerados
    como el mismo flash. Eventos que suceden consecutivamente en un intervalo menor a 1 seg son consideros STROKES.
    este debe ser un entero entre 1 y 60 seg
    :param delta_x: es el umbral en el espacio en km minimo para que dos eventos consecutivos sean considerados
    como del mismo flash
    :return: DataFrame con los eventos a tener en cuenta para la agregacion e interpolacion
    """

    raw.index = pd.to_datetime(raw.index)

    # Primer Filtro magnitud voltage

    abs_amp = pd.Series(data=raw.loc[:, 'amperage'].abs().values, name='abs_amperage', index=raw.index)
    raw = pd.concat(objs=[raw, abs_amp], axis=1)
    data = raw[raw.abs_amperage > max_amp]

    # Segundo Filtro
    # Por Error

    clean_data = data[data.Error < max_error]

    if len(clean_data) >= 2:

        # Tercer Filtro Flash y Strokes
        # Intervalo Tiempo

        clean_data = pd.concat(objs=[clean_data, pd.DataFrame(index=clean_data.index, columns=['delta_time', 'time_eval'])], axis=1)
        clean_data.delta_time = pd.Series(clean_data.index).diff().values
        clean_data.time_eval = clean_data.delta_time > pd.to_timedelta('0 days 00:00:{:02}'.format(delta_strokes))

        # Intervalo Espacio
        strokes = clean_data.copy()
        strokes = pd.concat(objs=[strokes, pd.DataFrame(data=0, index=strokes.index, columns=['latlong', 'distance', 'strokes', 'flashes'])], axis=1)

        strokes.latlong = list(zip(strokes.Latitude, strokes.Longitude))
        ls_dist = [vincenty(x, y).kilometers for x, y in zip(strokes.latlong[1:], strokes.latlong[:-1])]
        ls_dist.insert(0, np.NaN)
        strokes.distance = ls_dist
        strokes = strokes[~strokes.index.duplicated(keep='first')]

        idx_strokes = strokes[(strokes.time_eval == False) & (strokes.distance < [delta_x])].index
        strokes.loc[idx_strokes, 'strokes'] = 1
        df_exit = fn_add_flashes(strokes)
    else:
        df_exit = pd.DataFrame(columns=[u'Latitude', u'Longitude', u'Height', u'Type', u'amperage', u'Error', u'abs_amperage', u'delta_time', u'time_eval', u'latlong', u'distance', u'strokes', u'flashes'])

    return df_exit


def fn_get_shape_extent(path_shape):
    shp = ogr.Open(path_shape)
    lyr = shp.GetLayer()
    dc_ext = {'xmin': lyr.GetExtent()[0], 'xmax': lyr.GetExtent()[1], 'ymin': lyr.GetExtent()[2], 'ymax': lyr.GetExtent()[3]}
    return dc_ext


def fn_mask_data(raw, ext, bound=0.5):
    return raw[(raw.Longitude > ext['xmin'] - bound) & (raw.Longitude < ext['xmax'] + bound) & (raw.Latitude > ext['ymin'] - bound) & (raw.Latitude < ext['ymax'] + bound)]


def fn_array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, gdtype):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def fn_mask(vec, rast):
    shp = ogr.Open(vec)
    lyr = shp.GetLayer()
    feat = lyr.GetFeature(0)
    # Open data
    raster = gdal.Open(rast)
    shp = ogr.Open(vec)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]

    # project vector geometry to same projection as raster
    source_sr = lyr.GetSpatialRef()
    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(source_sr, target_sr)
    # feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coord_trans)

    # Get extent of feat
    geom = feat.GetGeometryRef()
    if geom.GetGeometryName() == 'MULTIPOLYGON':
        feat_count = 0
        points_x = []
        points_y = []
        for polygon in geom:
            geom_inner = geom.GetGeometryRef(feat_count)
            ring = geom_inner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                points_x.append(lon)
                points_y.append(lat)
            feat_count += 1
    elif geom.GetGeometryName() == 'POLYGON':
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        points_x = []
        points_y = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            points_x.append(lon)
            points_y.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(points_x)
    xmax = max(points_x)
    ymin = min(points_y)
    ymax = max(points_y)

    # Specify offset and rows and columns to read
    xoff = int((xmin - x_origin) / pixel_width)
    yoff = int((y_origin - ymax) / pixel_width)
    xcount = int((xmax - xmin) / pixel_width) + 1
    ycount = int((ymax - ymin) / pixel_width) + 1

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, gdal.GDT_Byte)
    target_ds.SetGeoTransform((xmin, pixel_width, 0, ymax, 0, pixel_height,))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    band_data_raster = raster.GetRasterBand(1)
    data_raster = band_data_raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)

    band_mask = target_ds.GetRasterBand(1)
    data_mask = band_mask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

    # Mask zone of raster
    zone_raster = np.ma.masked_array(data_raster, np.logical_not(data_mask))

    return zone_raster


def get_txt_files(all_files, request=None):
    # Load txt data into rawdata folder
    if all_files:
        dt_config = get_pars_from_ini('../config/config.ini')
        url_files = get_all_names(folder=dt_config['Paths']['path_url_light'], ext='txt')
        local_files = os.listdir('../data/rawdata/')
        new_files = list(set(url_files) - set(local_files))
        get_all_files(new_files, dt_config['Paths']['path_url_light'], dt_config['Paths']['path_txt_light_out'])
    else:
        dt_config = get_pars_from_ini('../config/config.ini')
        new_files = request
        get_all_files(new_files, dt_config['Paths']['path_url_light'], dt_config['Paths']['path_txt_light_out'])


def fn_density_maps(years, months, resolution, loc):
    dt_config = get_pars_from_ini('../config/config.ini')

    if resolution is 'Y':
        for year in years:
            all_data = fn_read_summary_csv(path_file='../data/summary/year/lightning_{}.csv'.format(str(year)))
            if loc is 'BOG':
                mask_shape = '../gis/Bog_Localidades.shp'
                max_error = dt_config['LightFilters']['error_bog']
            elif loc is 'COL':
                mask_shape = '../gis/Colombia_Continental.shp'
                max_error = dt_config['LightFilters']['error_col']
            else:
                mask_shape = None
            data = fn_mask_data(raw=all_data, ext=fn_get_shape_extent(mask_shape))
            data_filter = fn_filter_light(raw=data)
            fn_maps(data_filter, loc, resolution, year)

    elif resolution is 'M':
        months = ['{:02}'.format(i) for i in months]
        comb_days = ['_'.join(x) for x in list((itertools.product(map(str, years), months)))]
        for month in comb_days:
            all_data = fn_read_summary_csv(path_file='../data/summary/month/raw_lightning_{}.csv'.format(month))
            if loc is 'BOG':
                mask_shape = '../gis/Bog_Localidades.shp'
                max_error = dt_config['LightFilters']['error_bog']
            elif loc is 'COL':
                mask_shape = '../gis/Colombia_Continental.shp'
                max_error = dt_config['LightFilters']['error_col']
            else:
                mask_shape = None
                max_error = None
            data = fn_mask_data(raw=all_data, ext=fn_get_shape_extent(mask_shape))
            data_filter = fn_filter_light(raw=data, max_amp=dt_config['LightFilters']['amperage'], max_error=max_error)
            fn_maps(data_filter, loc, resolution, month)

    elif resolution is 'D':
        days = ['{:02d}'.format(i) for i in np.arange(1, 32)]
        months = ['{:02}'.format(i) for i in months]
        comb_days = [''.join(x) for x in list((itertools.product(map(str, years), months, days)))]
        for day in comb_days:
            try:
                all_data = fn_read_summary_csv(path_file='../data/summary/day/raw_lightning_{}1200.csv'.format(day))

                if loc is 'BOG':
                    mask_shape = '../gis/Bog_Localidades.shp'
                    max_error = dt_config['LightFilters']['error_bog']
                elif loc is 'COL':
                    mask_shape = '../gis/Colombia_Continental.shp'
                    max_error = dt_config['LightFilters']['error_col']
                else:
                    mask_shape = None
                    max_error = None

                data = fn_mask_data(raw=all_data, ext=fn_get_shape_extent(mask_shape))
                data_filter = fn_filter_light(raw=data, max_amp=dt_config['LightFilters']['amperage'], max_error=max_error)
                fn_maps(data_filter, loc, resolution, day)
            except IOError:
                print 'cannot open :', day


def fn_maps(df_data, loc, resolution, name):
    dt_config = get_pars_from_ini('../config/config.ini')

    if loc is 'BOG':
        ext = fn_get_shape_extent('../gis/Bog_Localidades.shp')
        pixel = dt_config['QuerySetUp']['bog_res']

    elif loc is 'COL':
        ext = fn_get_shape_extent('../gis/Colombia_Continental.shp')
        pixel = dt_config['QuerySetUp']['col_res']

    else:
        ext = None
        pixel = None

    xmin = ext['xmin'] - 0.1
    xmax = ext['xmax'] + 0.1
    ymin = ext['ymin'] - 0.1
    ymax = ext['ymax'] + 0.1

    raster_origin = (xmin - (pixel / 2), ymax + (pixel / 2))
    ncols = int((xmax - xmin) / pixel) + 1
    nrows = int((ymax - ymin) / pixel) + 1

    counts, y, x = np.histogram2d(df_data.Latitude, df_data.Longitude, bins=(nrows, ncols), range=([ymin, ymax], [xmin, xmax]))
    # density = counts / (pixel ** 2)
    density = counts / 25.

    # Dado que np.histogram2d devuelve ordenado al reves la matriz de conteo, es necesario reversar la matriz solo en las filas
    fn_array2raster(newRasterfn='../rasters/count/{}/CDT_{}_{}.tif'.format(resolution, loc, name), rasterOrigin=raster_origin, pixelWidth=pixel, pixelHeight=-pixel, array=np.flip(counts, axis=0), gdtype=gdal.GDT_Int16)
    fn_array2raster(newRasterfn='../rasters/density/{}/DDT_{}_{}.tif'.format(resolution, loc, name), rasterOrigin=raster_origin, pixelWidth=pixel, pixelHeight=-pixel, array=np.flip(density, axis=0), gdtype=gdal.GDT_Float32)


def fn_make_summary(years, months, resolution):
    if resolution is 'Y':

        for year in years:
            dt_config = get_pars_from_ini('../config/config.ini')
            year_files = os.listdir(dt_config['Paths']['path_txt_light_out'])

            ls_files = [x.split('_')[1][:4] for x in year_files]
            ls_index = [i for i, j in enumerate(ls_files) if j == str(year)]

            fn_make_summary_light(files=[os.listdir('../data/rawdata/')[i] for i in ls_index], outputfile='../data/summary/year/raw_light_{}.csv'.format(str(year)))

    elif resolution is 'M':

        for year in years:

            dt_config = get_pars_from_ini('../config/config.ini')
            year_files = os.listdir(dt_config['Paths']['path_txt_light_out'])
            ls_files = [x.split('_')[1][:6] for x in year_files]

            for m in months:
                ls_month_index = [i for i, j in enumerate(ls_files) if j == '{}{:02d}'.format(str(year), m)]
                fn_make_summary_light(files=[os.listdir('../data/rawdata/')[i] for i in ls_month_index], outputfile='../data/summary/month/raw_lightning_{}_{:02d}.csv'.format(str(year), m))

    elif resolution is 'D':

        for year in years:

            dt_config = get_pars_from_ini('../config/config.ini')
            year_files = os.listdir(dt_config['Paths']['path_txt_light_out'])
            ls_files = [x.split('_')[1][:6] for x in year_files]

            for m in months:

                ls_month_index = [i for i, j in enumerate(ls_files) if j == '{}{:02d}'.format(str(year), m)]

                for d in ls_month_index:

                    fn_make_summary_light(files=[os.listdir('../data/rawdata/')[d]], outputfile='../data/summary/day/raw_lightning_{}.csv'.format(os.listdir('../data/rawdata/')[d].split('_')[1]))


def main():
    """
    This script accesses the 'Almacen/externo' path and downloads only the files that are not found in
    'data/rawdata'. The user must enter the temporary resolution of the query, 'Y' for annuals,
    'M' for monthly and 'D' for days. You must also indicate if the query is for Colombia ('COL')
    or Bogota ('BOG')
    :return: CSV Compressed with the summary info
    """
    # get_txt_files(all_files=True)
    # fn_make_summary(years=[2018], months=[1, 2, 3, 4], resolution='M')
    # fn_make_summary(years=[2017], months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], resolution='M')
    # fn_make_summary(years=[2017], months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], resolution='D')
    # fn_make_summary(years=[2018], months=[1, 2, 3, 4], resolution='D')
    # fn_density_maps(years=[2018], months=[1, 2, 3, 4], resolution='M', loc='BOG')
    # fn_density_maps(years=[2017], months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], resolution='M', loc='BOG')
    # fn_density_maps(years=[2018], months=[1, 2, 3, 4], resolution='D', loc='BOG')
    # fn_density_maps(years=[2017], months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], resolution='D', loc='BOG')

    fn_density_maps(years=[2018], months=[4], resolution='M', loc='COL')


if __name__ == '__main__':
    main()
    pass
