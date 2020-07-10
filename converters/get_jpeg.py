from osgeo import gdal, osr
import pyproj
import numpy as np
import math
from math import cos, pi


def get_jpg(src_name: str, lon_lat: tuple, distance: int, pix: int):
    """

    :param pix: length of the output JPEG in pix (pix * pix)
    :param src_name: name of the source file
    :param lon_lat: (longitude, latitude) in degrees
    :param distance: length of the output JPEG in km (distance * distance)
    :return: src_name.jpeg (one band - no RGB)
    """

    def get_coordinates(lon_lat: tuple, distance: int):
        """

        :param distance: length of the output JPEG in km (distance * distance)
        :param lon_lat: (longitude, latitude) in degrees
        :return: lon lat coordinates for bbox
        """
        distance = distance / 2
        r_earth = 6378

        lrx = lon_lat[1] + (distance / r_earth) * (180 / pi)
        lry = lon_lat[0] - (distance / r_earth) * (180 / pi) / cos(lon_lat[1] * pi / 180)
        ulx = lon_lat[1] - (distance / r_earth) * (180 / pi)
        uly = lon_lat[0] + (distance / r_earth) * (180 / pi) / cos(lon_lat[1] * pi / 180)
        out = [ulx, uly, lrx, lry]
        return tuple(out)

    box = get_coordinates(lon_lat, distance)
    str_projwin = '-projwin'
    for x in range(4):
        str_projwin = str_projwin + ' ' + str(round(box[x], 5))
    print(str_projwin)
    dst_name = src_name.strip('.TIF') + '.jpg'
    outsize = '-outsize ' + str(pix) + ' ' + str(pix)
    print(outsize)

    warp_list = [
        '-s_srs EPSG:4326',
        '-t_srs EPSG:4326',
        '-overwrite'
    ]
    warp_string = " ".join(warp_list)
    gdal.Warp(src_name,
              src_name,
              options=warp_string)

    options_list = [
        '-ot Byte',
        '-of JPEG',
        '-b 1',
        outsize,
        str_projwin,
        '-projwin_srs EPSG:4326',
        '-a_srs EPSG:4326',
        '-scale'
    ]
    options_string = " ".join(options_list)
    gdal.Translate(dst_name,
                   src_name,
                   options=options_string)
