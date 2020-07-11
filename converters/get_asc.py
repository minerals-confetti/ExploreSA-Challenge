from osgeo import gdal

def get_asc(src_name):
    """
    Converts .TIF (src_name) to .asc
    """

    dst_name = src_name.strip('.TIF') + '.asc'
    options_list = [
            '-of AAIGrid',
            '-b 1'
        ]
    options_string = " ".join(options_list)
    gdal.Translate(dst_name,
                  src_name,
                  options=options_string)

