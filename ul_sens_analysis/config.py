
import os


class ConfigContainer(object):
    pass


def get_conf():

    conf = ConfigContainer()

    conf.tr_s = 2.0

    conf.st_vols_to_ignore = 16

    conf.slice_timing_path = os.path.join(
        "/sci/study/ul_sens/code/",
        "ul_sens_analysis/ul_sens_analysis",
        "ul_sens_slice_timing_pattern.1D"
    )

    return conf
