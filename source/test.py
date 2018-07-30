from __future__ import print_function
import math
import os
import xml.etree.ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_files(name):
    filename, ext = os.path.splitext(name)
    ext = ext[1:]
    base_path = '../data/cromero/{}'.format(name)
    if ext == 'csv':
        pre_data = pd.read_csv(base_path, sep=';', skiprows=[0, 1], header=None, parse_dates=[[0, 1]], dayfirst=True, index_col=['0_1'])
        if type(pre_data[2].sum()) is str:
            df_data = pd.read_csv(base_path, sep=';', skiprows=[0, 1], header=None, parse_dates=[[0, 1]], dayfirst=True, index_col=['0_1'], decimal=',')[2]
        else:
            df_data = pre_data[2]
    else:
        xls_data = pd.ExcelFile(base_path)
        sh_names = xls_data.sheet_names[0]
        df_data = xls_data.parse(sheet_name=sh_names, skiprows=[0, 1])

    df_data.index.name = 'Date'
    df_data.name = filename
    # df_data.columns = ['Fecha', 'Hour', filename]
    # df_data['DateStr'] = df_data.Fecha.astype(str) + ' ' + df_data.Hour.astype(str)
    # df_data['Date'] = pd.to_datetime(df_data['DateStr'], dayfirst=True, infer_datetime_format=True, format='%d/%m/%Y %HH:%MM:%SS')
    # df_data.set_index('Date', inplace=True)
    # df_data.drop(labels=['Fecha', 'Hour', 'DateStr'], inplace=True, axis=1)

    df_resample = df_data.resample('5T').sum()
    # df_resample.to_clipboard()

    return df_resample


def agg_meteo_day():
    df_results = pd.DataFrame(data=None, index=pd.date_range(start='1/1/1995', end='3/13/2018', freq='D'))
    df_data = pd.read_csv('../results/PT_5T_Summary.csv', index_col='Date')
    stations = df_data.columns
    # df_data.index.name = 'Date'
    df_data.index = pd.to_datetime(df_data.index)
    df_data['MDate'] = df_data.index - pd.Timedelta('7 hours')
    df_data.set_index(keys='MDate', drop=False, inplace=True)
    df_data['day'] = df_data.index.day
    df_data['month'] = df_data.index.month
    df_data['year'] = df_data.index.year

    for sta in stations:
        df_count_data = df_data[['year', 'month', 'day', sta]].groupby(['year', 'month', 'day']).count()
        df_data_agg = df_data[['year', 'month', 'day', sta]].groupby(['year', 'month', 'day']).sum()
        df_sta = pd.DataFrame(np.where(df_count_data < 230, np.nan, df_data_agg), index=df_results.index, columns=[sta])
        df_results = pd.concat(objs=[df_results, df_sta], axis=1)
    
    df_results.index.name = 'Date'
    df_results.to_excel('../results/Corpocaldas_Series_Resample_1D_Summary.xlsx', merge_cells=False)


def main():

    df_all_data = pd.DataFrame(data=None, index=pd.date_range(start='1/1/1995', end='3/13/2018', freq='5min'))
    df_all_data.index.name = 'Date'
    files_list = os.listdir('../data/cromero/')
    # files_list = ['00_TEST_M001.csv', '01_TEST_M002.csv']

    for nfile in files_list:
        print(nfile)
        df_all_data = pd.concat(objs=[df_all_data, read_files(nfile)], axis=1)

    df_all_data.to_csv('../results/PT_5T_Summary.csv')


def xml_read():
    e = xml.etree.ElementTree.parse('C:/Users/jchavarro/Downloads/Municipios.xml').getroot()
    df_data = pd.DataFrame(data=None)

    for atype in e._children:
        for btype in atype._children:
            df_data = df_data.append(pd.DataFrame.from_dict(btype.attrib, orient='index').T)
    df_data.to_clipboard()
    # df_data.to_csv('C:/Users/jchavarro/Downloads/Municipios.csv')


def make_hist_raw_light_data():
    folders = os.listdir('../data/historic_raw_data')
    for folder in folders:
        files = os.listdir('../data/historic_raw_data/{}'.format(folder))
        for f in files:
            df_data = pd.read_csv('../data/')
            print(f)


def date_extract():
    stations = os.listdir('E:/jchavarro/OSPA/cromero2/stations')
    df_date_ext = pd.ExcelFile('E:/jchavarro/OSPA/cromero2/FECHAS_MUESTRA.xlsx').parse(sheet_name='Hoy', index_col='ID_Date')
    df_index = pd.MultiIndex.from_arrays([df_date_ext['Year'].values, df_date_ext['Month'].values, df_date_ext['Day'].values], names=('Year', 'Month', 'Day'))
    df_all = pd.DataFrame(data=None, index=df_index, columns=[x.split('.')[0] for x in stations])
    for sta in stations:
        # sta = '11020010.csv'
        print(sta)
        db_data = pd.read_csv('E:/jchavarro/OSPA/cromero2/stations/{}'.format(sta), header=None, names=['Year', 'Day', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], sep=';', na_values=99999, decimal=',')
        db_data.set_index(['Year', 'Day'], inplace=True)
        df_data = db_data.stack().reset_index()
        df_data.columns = ['Year', 'Day', 'Month', sta[:-4]]
        df_data.set_index(['Year', 'Month', 'Day'], inplace=True)
        df_data_extract = df_data.ix[[tuple(df_date_ext.ix[x, :].values) for x in df_date_ext.index], ]
        df_clean = df_data_extract[~df_data_extract.index.duplicated(keep='first')]
        df_all[sta[:-4]] = df_clean.values
    df_all.to_excel('E:/jchavarro/OSPA/cromero2/datos_muestra_actual.xlsx', sheet_name='results', merge_cells=True)


if __name__ == '__main__':
    # main()
    # xml_read()
    # agg_meteo_day()
    # xplore_data()
    # make_hist_raw_light_data()
    date_extract()
    print('Done')

