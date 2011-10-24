### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import logging

VERSION = '2.0.0'

usage = 'foo'


# SIEMENS stuff
siemens_cas = 14  # coefficient array size
siemens_fov_min = -.30  # fov min in meters
siemens_fov_max = .31  # fov max in meters
siemens_resolution = .01
# max jacobian determinant for siemens
siemens_max_det = 10.


# GE stuff
ge_cas = 6  # coefficient array size
ge_fov_min = -0.5
ge_fov_max = 0.5
ge_resolution = 0.0075


def get_logger():
    log = logging.getLogger('gradunwarp')
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s-%(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
