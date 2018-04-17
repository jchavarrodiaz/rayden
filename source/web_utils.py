import urllib

import requests
from bs4 import BeautifulSoup


def get_all_names(folder, ext):
    """
    Esta funcion crea una lista con el nombre de los archivos con una extension
    especifica contenidos dentro de la tabla de una direcion URL
    :param folder: es la ruta a la carpeta donde estan los archivos
    :param ext: es la extension del tipo de archivos a listar
    :return: una lista con el nombre de los archivos
    """
    soup = BeautifulSoup(requests.get(folder).text, 'xml')
    return [link.get('href') for link in soup.find_all('a') if ext in link.get('href')]


def get_all_files(ls_names, path_src, path_out):
    """
    esta funcion descarga desde una carpeta en una
    direccion URL los archivos indicados en la lista
    a otra carpeta. Por el momento solo descarga de archivos
    de texto '*.txt'
    :param ls_names: es la lista con el nombre de los archivos a descargar
    :param path_src: es la carpeta fuente URL
    :param path_out: es la carpeta destino
    :return: None
    """
    for filename in ls_names:
        urllib.urlretrieve('{}/{}'.format(path_src, filename), '{}/{}'.format(path_out, filename))
        print filename


if __name__ == '__main__':
    # path_url = "http://172.16.1.237/almacen/externo/varios/Rayos/TXT/Consolidado24H/"
    # path_out = 'rawdata/'
    # get_all_files(get_all_names(folder=path_url, ext='txt'), path_url, path_out)
    pass
