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
        df_data = pd.read_csv(base_path, sep=';', skiprows=[0, 1])
    else:
        xls_data = pd.ExcelFile(base_path)
        sh_names = xls_data.sheet_names[0]
        df_data = xls_data.parse(sheet_name=sh_names, skiprows=[0, 1])

    df_data.columns = ['Fecha', 'Hour', filename]
    df_data['DateStr'] = df_data.Fecha.astype(str) + ' ' + df_data.Hour.astype(str)
    df_data['Date'] = pd.to_datetime(df_data['DateStr'], dayfirst=True, infer_datetime_format=True, format='%d/%m/%Y %HH:%MM:%SS')
    df_data.set_index('Date', inplace=True)
    df_data.drop(labels=['Fecha', 'Hour', 'DateStr'], inplace=True, axis=1)

    df_resample = df_data.resample('5T').sum()
    # df_resample.to_clipboard()

    return df_resample


def agg_meteo_day():
    df_data = pd.read_csv('../results/Corpocaldas_Series_Resample_5T_Summary.csv', index_col=[0], infer_datetime_format=True)
    df_data.index.name = 'Date'
    df_data.index = pd.to_datetime(df_data.index)
    df_data['MDate'] = df_data.index - pd.Timedelta('7 hours')

    df_data.set_index(keys='MDate', drop=False, inplace=True)
    df_data['day'] = df_data.index.day
    df_data['month'] = df_data.index.month
    df_data['year'] = df_data.index.year

    df_count_data = df_data.groupby(['year', 'month', 'day']).count()
    df_data_agg = df_data.groupby(['year', 'month', 'day']).sum()

    df = df_data_agg.where(df_count_data > 230, np.nan)
    df.to_excel('../results/Corpocaldas_Series_Resample_1D_Summary.xlsx', merge_cells=False)


def main():

    df_all_data = pd.DataFrame(data=None, index=pd.date_range(start='1/1/1995', end='3/13/2018', freq='5min'))

    files_list = os.listdir('../data/cromero/')
    # files_list = ['M001.csv', 'M002.csv']

    for nfile in files_list:
        print nfile
        df_all_data = pd.concat(objs=[df_all_data, read_files(nfile)], axis=1)

    df_all_data.to_csv('../results/PT_5T_Summary.csv')


def xplore_data():

    plt.style.use('classic')

    xls_data = pd.ExcelFile('../test/help_me/9TablaBalances.xlsx')
    # table1 = xls_data.parse(sheet_name='table1', index_col='Year')
    # table2 = xls_data.parse(sheet_name='table2', index_col='Year')

    table1 = xls_data.parse(sheet_name='dataset1', index_col='Year')

    # # Punto 1
    # df_point1 = table1.describe()
    # df_point1.index.name = 'stats'
    # df_point1.ix['median', :] = table1.median()
    # df_point1.ix['trimedia', :] = (df_point1.loc[u'25%', :] + (2 * df_point1.loc[u'50%', :]) + df_point1.loc[u'75%', :]) / 4
    # # Punto 2
    # df_point1.ix['mad', :] = robust.mad(table1)
    # df_point1.ix['IQR', :] = df_point1.loc[u'75%', :] - df_point1.loc[u'25%', :]
    #
    # # Punto 4
    # markerline, stemlines, baseline = plt.stem(table1['Temperature'], '-.')
    # plt.setp(baseline, color='r', linewidth=2)
    # plt.savefig('../test/help_me/stemgraphic.png', dpi=400)
    # plt.close()
    #
    # # Punto 3
    # df_point1.ix['YKI', :] = ((df_point1.loc[u'75%', :] - df_point1.loc[u'50%', :]) - (df_point1.loc[u'50%', :] - df_point1.loc[u'25%', :])) / df_point1.loc['IQR', :]
    # df_point1.ix['Skewness', :] = ss.skew(table1)
    #
    # df_point1.to_excel('../test/help_me/point1.xlsx')
    #
    # # Punto 5
    # fig, ax = plt.subplots(figsize=(8, 4))
    # # plot the cumulative histogram
    # n, bins, patches = ax.hist(table1['Presion'], 10, histtype='step', cumulative=True, label='Empirical')
    # # tidy up the figure
    # ax.grid(True)
    # ax.legend(loc='best')
    # ax.set_title('Cumulative step histograms')
    # ax.set_xlabel('Presion (mb)')
    # ax.set_ylabel('Likelihood of occurrence')
    # plt.savefig('../test/help_me/cum_hist.png', dpi=400)
    # plt.close()
    #
    # # the histogram of the data
    # n, bins, patches = plt.hist(table1['Presion'], 10, density=False, facecolor='g', alpha=0.75)
    # plt.xlabel('Presion')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Presion')
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.savefig('../test/help_me/hist.png', dpi=400)
    # plt.close()
    #
    # # Punto 6
    # # BoxPlots
    # plt.boxplot(table1['Rainfall'].dropna(), 0, '')
    # plt.ylabel('mm per year')
    # plt.title('Histogram of Rainfall')
    # plt.savefig('../test/help_me/boxplots.png', dpi=400)
    # plt.close()
    # # Schematic Plots
    # plt.boxplot(x=table1['Rainfall'].dropna(), whis=[1.5, 3])
    # plt.ylabel('mm per year')
    # plt.title('Schematic Plot of Rainfall')
    # plt.savefig('../test/help_me/schematic.png', dpi=400)
    # plt.close()
    #
    # # punto 7
    # # Power Transformation
    # landa = np.arange(-3, 3, 0.5)
    # df_hinkley = pd.DataFrame(data=None, index=landa, columns=['Median', 'Mean', 'IQR', 'Hinkley Stats'])
    # df_hinkley.index.name = 'Lamda'
    # df_reexpresion = pd.DataFrame(data=None, index=table2.index, columns=landa)
    #
    # for l in landa:
    #     if l > 0:
    #         t1 = table2['Rainfall Ithaca'] ** l
    #     elif l == 0:
    #         t1 = np.log(table2['Rainfall Ithaca'])
    #     else:
    #         t1 = -(table2['Rainfall Ithaca'] ** l)
    #
    #     df_reexpresion.ix[:, l] = t1.copy()
    #     df_stats2 = t1.describe()
    #     df_stats2.index.name = 'stats'
    #     df_stats2.ix['median'] = t1.median()
    #     df_stats2.ix['IQR'] = df_stats2.ix[u'75%'] - df_stats2.ix[u'25%']
    #     hinkley = np.abs(df_stats2.ix['mean'] - df_stats2.ix['median']) / df_stats2.ix['IQR']
    #     df_hinkley.ix[l, :] = [df_stats2.ix['median'], df_stats2.ix['mean'], df_stats2.ix['IQR'], hinkley]
    # df_hinkley.to_excel('../test/help_me/hinkley.xlsx')
    # df_reexpresion.to_excel('../test/help_me/ithaca_reexpresion.xlsx')
    #
    # # Punto 8
    # # Schematic Plots
    # df_reexpresion.ix[:, [-0.5, 1.0, 0.0, 0.5]].boxplot()
    # plt.savefig('../test/help_me/schematic_Ithaca.png', dpi=400)
    # plt.close()
    #
    # # Punto 9
    # # Standardized Anomalies
    # df_std = (table1.Temperature - table1.Temperature.mean()) / table1.Temperature.std()
    # df_std.to_excel('../test/help_me/standardized_Anomalies_temp.xlsx')
    # plt.plot(list(df_std.index), df_std.values)
    # plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    # plt.ylabel(u'Standardized Difference')
    # plt.savefig('../test/help_me/standardized_Anomalies_Temp.png', dpi=400)
    #
    # # Punto 10
    # # Correlation Function
    # df_corr = pd.Series(data=None, index=[1, 2, 3], name='Correlation')
    # for lag in df_corr.index:
    #     df_corr.ix[lag] = table2['Rainfall Ithaca'].autocorr(lag=lag)
    # df_corr.plot.bar()
    # plt.axhline(0, color='k')
    # plt.ylabel(u'AutoCorrelation Function')
    # plt.xlabel(u'Lag')
    # plt.savefig('../test/help_me/AutoCorrelation.png', dpi=400)
    #
    # # Punto 11
    # # Scatter Plot
    # table1.plot.scatter(x='Rainfall', y='Temperature')
    # plt.ylabel(u'Temperature')
    # plt.xlabel(u'Rainfall')
    # plt.savefig('../test/help_me/ScatterPlot.png', dpi=400)
    #
    # # Punto 12
    # # Pearson and Spearman
    #
    # variables = table1.columns
    # data = table1.dropna()
    # df_pearson = pd.DataFrame(data=None, columns=variables, index=variables)
    # for i in variables:
    #     for j in variables:
    #         df_pearson.ix[i, j] = ss.pearsonr(data[i], data[j])[0]
    # df_pearson.to_excel('../test/help_me/Pearson.xlsx')
    # pd.DataFrame(ss.spearmanr(data)[0], columns=variables, index=variables).to_excel('../test/help_me/Spearman.xlsx')
    #
    # # Punto 13
    # # Star plot (Diagrama de Telaranas)
    # # number of variable
    # data = table1.dropna()
    # categories = list(data)
    # N = len(categories)
    #
    # for year in data.ix[1965:1969].index:
    #     values = data.ix[year].values.flatten().tolist()
    #     values += values[:1]
    #     values
    #
    #     # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    #     angles = [n / float(N) * 2 * math.pi for n in range(N)]
    #     angles += angles[:1]
    #
    #     # Initialise the spider plot
    #     ax = plt.subplot(111, polar=True)
    #
    #     # Draw one axe per variable + add labels labels yet
    #     plt.xticks(angles[:-1], categories, color='grey', size=8)
    #
    #     # Draw ylabels
    #     ax.set_rlabel_position(0)
    #     # plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    #     plt.ylim(0, max(values))
    #
    #     # Plot data
    #     ax.plot(angles, values, linewidth=1, linestyle='solid')
    #
    #     # Fill area
    #     ax.fill(angles, values, 'b', alpha=0.1)
    #     plt.savefig('../test/help_me/StarPlot{}.png'.format(str(year)), dpi=400)
    #     plt.close()

    # Punto 1

    # df_point1 = table1.describe()
    # df_point1.index.name = 'stats'
    # df_point1.ix['median', :] = table1.median()
    # df_point1.ix['trimedia', :] = (df_point1.loc[u'25%', :] + (2 * df_point1.loc[u'50%', :]) + df_point1.loc[u'75%', :]) / 4
    # # Punto 2
    # df_point1.ix['mad', :] = robust.mad(table1)
    # df_point1.ix['IQR', :] = df_point1.loc[u'75%', :] - df_point1.loc[u'25%', :]
    #
    # # Punto 4
    # markerline, stemlines, baseline = plt.stem(table1['PET'], '-.')
    # plt.setp(baseline, color='r', linewidth=2)
    # plt.savefig('../test/help_me/Point14/stemgraphic.png', dpi=400)
    # plt.close()
    #
    # # Punto 3
    # df_point1.ix['YKI', :] = ((df_point1.loc[u'75%', :] - df_point1.loc[u'50%', :]) - (df_point1.loc[u'50%', :] - df_point1.loc[u'25%', :])) / df_point1.loc['IQR', :]
    # df_point1.ix['Skewness', :] = ss.skew(table1)
    #
    # df_point1.to_excel('../test/help_me/Point14/point1.xlsx')
    #
    # # Punto 5
    # fig, ax = plt.subplots(figsize=(8, 4))
    # # plot the cumulative histogram
    # n, bins, patches = ax.hist(table1['Rainfall'], 10, histtype='step', cumulative=True, label='Empirical')
    # # tidy up the figure
    # ax.grid(True)
    # ax.legend(loc='best')
    # ax.set_title('Cumulative step histograms')
    # ax.set_xlabel('Precipitacion (mm/year)')
    # ax.set_ylabel('Likelihood of occurrence')
    # plt.savefig('../test/help_me/Point14/cum_hist.png', dpi=400)
    # plt.close()
    #
    # # the histogram of the data
    # n, bins, patches = plt.hist(table1['Rainfall'], 10, density=False, facecolor='g', alpha=0.75)
    # plt.xlabel('Rainfall')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Rainfall')
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.savefig('../test/help_me/Point14/hist.png', dpi=400)
    # plt.close()
    #
    # # Punto 6
    # # BoxPlots
    # plt.boxplot(table1['Rainfall'].dropna(), 0, '')
    # plt.ylabel('mm per year')
    # plt.title('Histogram of Rainfall')
    # plt.savefig('../test/help_me/Point14/boxplots.png', dpi=400)
    # plt.close()
    # # Schematic Plots
    # plt.boxplot(x=table1['Rainfall'].dropna(), whis=[1.5, 3])
    # plt.ylabel('mm per year')
    # plt.title('Schematic Plot of Rainfall')
    # plt.savefig('../test/help_me/Point14/schematic.png', dpi=400)
    # plt.close()
    #
    # # Punto 9
    # # Standardized Anomalies
    # df_std = (table1.Runoff - table1.Runoff.mean()) / table1.Runoff.std()
    # df_std.to_excel('../test/help_me/Point14/standardized_Anomalies_Runoff.xlsx')
    # plt.plot(list(df_std.index), df_std.values)
    # plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    # plt.ylabel(u'Standardized Difference')
    # plt.savefig('../test/help_me/Point14/standardized_Anomalies_Temp.png', dpi=400)
    #
    # # Punto 11
    # # Scatter Plot
    # table1.plot.scatter(x='Rainfall', y='PET')
    # plt.ylabel(u'PET')
    # plt.xlabel(u'Rainfall')
    # plt.savefig('../test/help_me/Point14/ScatterPlot.png', dpi=400)
    #
    # # Punto 12
    # # Pearson and Spearman
    #
    # variables = table1.columns
    # data = table1.dropna()
    # df_pearson = pd.DataFrame(data=None, columns=variables, index=variables)
    # for i in variables:
    #     for j in variables:
    #         df_pearson.ix[i, j] = ss.pearsonr(data[i], data[j])[0]
    # df_pearson.to_excel('../test/help_me/Point14/Pearson.xlsx')
    # pd.DataFrame(ss.spearmanr(data)[0], columns=variables, index=variables).to_excel('../test/help_me/Point14/Spearman.xlsx')
    #
    # Punto 13
    # Star plot (Diagrama de Telaranas)
    # number of variable
    data = table1.dropna().drop('Balance', axis=1)

    categories = list(data)
    N = len(categories)

    for year in data.ix[1998:2001].index:
        values = data.ix[year].values.flatten().tolist()
        values += values[:1]
        values

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        # plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
        plt.ylim(0, max(values))

        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')

        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.savefig('../test/help_me/Point14/StarPlot{}.png'.format(str(year)), dpi=400)
        plt.close()


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
            print f


if __name__ == '__main__':
    # main()
    # xml_read()
    # agg_meteo_day()
    # ponchito(data='../test/data_misr.xlsx')
    xplore_data()
    # make_hist_raw_light_data()
    print 'Done'
