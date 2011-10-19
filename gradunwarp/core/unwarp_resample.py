### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import numpy as np


class Unwarper(object):
    '''
    '''
    def __init__(vol, m_rcs2ras, vendor, coeffs):
        '''
        '''
        self.vol = vol
        self.m_rcs2ras = m_rcs2ras
        self.vendor = vendor
        self.coeffs = coeffs
        self.warp = False
        self.nojac = False

    def run():
        '''
        '''
        # transform RAS-coordinates into LAI-coordinates
        m_ras2lai = np.array([-1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0], dtype=np.float)
        m_rcs2lai = m_ras2lai * self.m_rcs2ras

        # indices of image volume
        nr, nc, ns = self.vol.shape[:3]
        vr, vc, vs = utils.ndgrid(np.arange(nr), np.arange(nc), np.arange(ns))

        # account for half-voxel shift in R and C directions
        halfvox = np.zeros((4, 4))
        halfvox[0, 3] = m_rcs2lai[0, 0] / 2.0
        halfvox[1, 3] = m_rcs2lai[1, 1] / 2.0
        m_rcs2lai = m_rcs2lai + halfvox

        # convert image vol coordinates from RAS to LAI
        [vx, vy, vz] = utils.transform_coordinates(vr, vc, vs, m_rcs2lai)

        # extract rotational and scaling parts of the transformation matrix
        # ignore the translation part
        r_rcs2lai = np.eye(4, 4)
        r_rcs2lai[:3, :3] = m_rcs2lai[:3, :3]

        # Jon Polimeni:
        # Since partial derivatives in Jacobian Matrix are differences
        # that depend on the ordering of the elements of the 3D array, the
        # coordinates may increase in the opposite direction from the array
        # indices, in which case the differential element should be negative.
        # The differentials can be determined by mapping a vector of 1s
        # through rotation and scaling, where any mirror
        # will impose a negation
        [dx, dy, dz] = utils.transform_coordinates(1., 1., 1., r_rcs2lai)
