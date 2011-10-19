### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import logging

VERSION = '2.0.0'

usage = 'foo'


# the coefficient array size for Siemens & GE .coef files
siemens_cas = 14
ge_cas = 6


def get_logger():
    log = logging.getLogger('gradunwarp')
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s-%(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
