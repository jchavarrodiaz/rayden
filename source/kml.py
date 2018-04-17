# -*- coding: utf-8 -*-

import urllib2

import pandas as pd
import simplekml

from config_utils import get_pars_from_ini


def gen_kml(points, other_info, icon, name, light):
    kml = simplekml.Kml(open=1)
    logo_path = get_pars_from_ini('../config/config.ini')['Paths']['logo_path']

    if light:
        print len(points.index)
        for event in points.index:
            pnt = kml.newpoint()
            pnt.name = str(event + 1)

            pnt.description = 'Presentado: {} <br />' \
                              'Localidad: {} <br />Periodo Observado: {} <br />' \
                              'Red Linet/Keraunos Suministrado IDEAM <br /><br />' \
                              '<img src="{}" alt="picture" width="151" height="25" align="left" />' \
                              '<br /><br />'.format(points.loc[event, 'TIEMPO'], points.loc[event, 'LOCALIDAD'], other_info, logo_path)

            pnt.coords = [(points.loc[event, 'LONGITUD'], points.loc[event, 'LATITUD'])]
            pnt.style.labelstyle.scale = 1
            pnt.style.iconstyle.scale = 1
            pnt.style.iconstyle.icon.href = icon

        # Save the KML
        kml.save('../results/{}.kml'.format(name))
    else:
        pnt = kml.newpoint()
        pnt.name = str('Sin Eventos')

        pnt.description = 'Novedad: {} <br />' \
                          'Red Linet/Keraunos Suministrado IDEAM <br /><br />' \
                          '<img src="{}" alt="picture" width="151" height="25" align="left" />' \
                          '<br /><br />'.format(other_info, logo_path)

        pnt.coords = [(points.loc[0, 'LONGITUD'], points.loc[0, 'LATITUD'])]
        pnt.style.labelstyle.scale = 1
        pnt.style.iconstyle.scale = 1
        pnt.style.iconstyle.icon.href = icon

        # Save the KML
        kml.save('../results/{}.kml'.format(name))
        print 'Done'


def main(kml_date):
    year = kml_date[:4]
    month = kml_date[5:7]
    day = kml_date[8:10]

    dt_config = get_pars_from_ini('../config/config.ini')
    path_url = dt_config['Paths']['path_url_kml']

    dict_files = {'acum': 'EVENTOS_LOCALIDADES_BOGOTA.xlsx',
                  '24H': 'EVENTOS_LOCALIDADES_BOGOTA24.xlsx'
                  }

    dict_nameout = {'acum': '{}{}{}_Current'.format(year, month, day),
                    '24H': '{}{}{}_Yesterday'.format(year, month, day)
                    }

    for i in dict_files:

        path = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, dict_files[i])

        if i == 'acum':
            path_txt = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, 'LapsoCurrent.txt')
            for line in urllib2.urlopen(path_txt):
                per_obs = line
        else:
            path_txt = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, 'Lapso24Hrs.txt')
            for line in urllib2.urlopen(path_txt):
                per_obs = line

        try:
            df_data = pd.ExcelFile(path).parse(sheet_name='Sheet1')
            df_data['LOCALIDAD'] = df_data['LOCALIDAD'].str.replace(u'\xd1', 'N')
            df_data.sort_values(by='TIEMPO', inplace=True)
            df_data.reset_index(drop=True, inplace=True)

            gen_kml(points=df_data, other_info=per_obs, icon=dt_config['Paths']['icon'], name=dict_nameout[i], light=True)

        except IOError as e:

            print "I/O error ({0}) --->>> Date: {1}".format(e.filename.split('/')[-1], kml_date)

            dict_none = {'DESCRIPCION': '{} {}'.format(year, month),
                         'LATITUD': 4.6748,
                         'LONGITUD': -74.1135,
                         }

            df_data = pd.DataFrame.from_dict(data=dict_none, orient='index').T

            per_obs_desc = ''

            if i == 'acum':
                path_txt = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, 'LapsoCurrent.txt')
                for line in urllib2.urlopen(path_txt):
                    per_obs = line
                path_txt_desc = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, 'BOGOTA_T3.txt')
                for line_desc in urllib2.urlopen(path_txt_desc):
                    per_obs_desc = line_desc
                text_info = '{} {}'.format(per_obs, per_obs_desc).replace(u'\xe9'.encode('utf-8'), 'e')
            else:
                path_txt = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, 'Lapso24Hrs.txt')
                for line in urllib2.urlopen(path_txt):
                    per_obs = line
                path_txt_desc = '{}{}/{}/{}/Bogota/{}'.format(path_url, year, month, day, 'BOGOTA_24.txt')
                for line_desc in urllib2.urlopen(path_txt_desc):
                    per_obs_desc = per_obs_desc + line_desc
                text_info = '{} {}'.format(per_obs, per_obs_desc).replace(u'\xe9'.encode('utf-8'), 'e')

            gen_kml(points=df_data, other_info=text_info, icon=dt_config['Paths']['icon'], name=dict_nameout[i], light=False)


if __name__ == '__main__':
    dates = ['2018-03-27', '2018-03-28', '2018-03-29', '2018-03-30', '2018-03-31', '2018-04-01', '2018-04-02', '2018-04-03']
    for date in dates:
        main(kml_date=date)
