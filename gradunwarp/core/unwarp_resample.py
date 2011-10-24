### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import numpy as np
import utils
from utils import CoordsVector as CV
from utils import factorial
import globals
from globals import siemens_max_det
import nibabel as nib
import pdb
import scipy


log = globals.get_logger()


class Unwarper(object):
    '''
    '''
    def __init__(self, vol, m_rcs2ras, vendor, coeffs):
        '''
        '''
        self.vol = vol
        self.m_rcs2ras = m_rcs2ras
        self.vendor = vendor
        self.coeffs = coeffs
        self.warp = False
        self.nojac = False

    def run(self):
        '''
        '''
        #pdb.set_trace()
        # define polarity based on the warp requested
        self.polarity = 1.
        if self.warp:
            self.polarity = -1.

        # transform RAS-coordinates into LAI-coordinates
        m_ras2lai = np.array([[-1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float)
        m_rcs2lai = np.dot(m_ras2lai, self.m_rcs2ras)

        # indices of image volume
        nr, nc, ns = self.vol.shape[:3]
        vr, vc, vs = utils.ndgrid(np.arange(nr), np.arange(nc), np.arange(ns))
        vrcs = CV(x=vr, y=vc, z=vs)

        # account for half-voxel shift in R and C directions
        halfvox = np.zeros((4, 4))
        halfvox[0, 3] = m_rcs2lai[0, 0] / 2.0
        halfvox[1, 3] = m_rcs2lai[1, 1] / 2.0
        m_rcs2lai = m_rcs2lai + halfvox

        # convert image vol coordinates from RAS to LAI
        vxyz = utils.transform_coordinates(vrcs, m_rcs2lai)

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
        ones = CV(1., 1., 1.)
        dxyz = utils.transform_coordinates(ones, r_rcs2lai)

        # compute new coordinates and the jacobian determinant
        # TODO still not clear about what to return
        self.out, self.vjacmult_lps = self.non_linear_unwarp(vxyz, dxyz,
                                                             m_rcs2lai)

        # return image is contained in self.imgout

    def write(self):
        img = nib.Nifti1Image(self.out, self.m_rcs2ras)
        img.to_filename('testsonata.nii.gz')

    def non_linear_unwarp(self, vxyz, dxyz, m_rcs2lai):
        ''' Performs the crux of the unwarping.
        It's agnostic to Siemens or GE and uses more functions to
        do the processing separately.

        Needs self.vendor, self.coeffs, self.warp, self.nojac to be set

        Parameters
        ----------
        vxyz : CoordsVector (namedtuple) contains np.array
            has 3 elements x,y and z each representing the grid coordinates
        dxyz : CoordsVector (namedtuple)
           differential coords vector

        Returns
        -------
        TODO still vague what to return
        vwxyz : CoordsVector (namedtuple) contains np.array
            x,y and z coordinates of the unwarped coordinates
        vjacmult_lps : np.array
            the jacobian multiplier (determinant)
        '''
        # compute the displacements for the voxel positions in LAI
        dv, modv = eval_spherical_harmonics(self.coeffs, self.vendor, vxyz)
        log.info('finished spherical harmonics evaluation')

        # Jacobian multiplier is unitless but calculated in terms of
        # displacements and coordinates in LPS orientation
        # (right handed form of p.o.v of patient )
        if self.vendor == 'siemens':
            if dxyz == 0:
                vjacdet_lps = 1
            else:
                vjacdet_lps = eval_siemens_jacobian_mult(dv, dxyz)

            # new locations of the image voxels in XYZ ( LAI ) coords
            vxyzw = CV(x=vxyz.x + self.polarity * dv.x,
                       y=vxyz.y + self.polarity * dv.y,
                       z=vxyz.z + self.polarity * dv.z)

            # if polarity is negative, the jacobian is also inversed
            vjacdet_lps = 1. / vjacdet_lps

            # convert the locations got into RCS indices
            vrcsw = utils.transform_coordinates(vxyzw,
                                                np.linalg.inv(self.m_rcs2lai))

            # resample the image
            log.info('Interpolating the image')
            if vol.ndim == 3:
                # note that out is always in float32
                out = utils.interp3(vol, vrcsw.x, vrcsw.y, vrcsw.z)
                out = out.reshape(vol.shape)
            if vol.ndim == 4:
                nframes = vol.shape[3]
                out = np.zeros(vol.shape)
                for f in nframes:
                    _out = utils.interp3(vol[:, :, :, f],
                                         vrcsw.x, vrcsw.y, vrcsw.z)
                    out[..., f] = _out.reshape(vol.shape[:3])

            # resample the jacobian determinant image
            vjacdet_lps = vjacdet_lps.reshape(vol.shape[:3])
            vjacdet_lpsw = utils.interp3(vjacdet_lps, vrcsw.x,
                                         vrcsw.y, vrcsw.z)

            # find NaN voxels, report them and set them to 0
            out[np.where(np.isnan(out))] = 0.
            vjacdet_lpsw[np.where(np.isnan(out))] = 0.

            # Multiply the intensity with the Jacobian det, if needed
            if not self.nojac:
                if out.ndim == 3:
                    out = out * vjacdet_lpsw
                elif out.ndim == 4:
                    for f in out.shape[3]:
                        out[..., f] = out[..., f] * vjacdet_lpsw

            # return image and the jacobian
            return out, vjacdet_lpsw

        if self.vendor == 'ge':
            pass  # for now


def eval_siemens_jacobian_mult(F, dxyz):
    '''
    '''
    d0, d1, d2 = dxyz[0], dxyz[1], dxyz[2]

    if d0 == 0 or d1 == 0 or d2 == 0:
        raise ValueError('weirdness found in Jacobian calculation')

    dFxdx, dFxdy, dFxdz = np.gradient(F.x, d0, d1, d2)
    dFydx, dFydy, dFydz = np.gradient(F.y, d0, d1, d2)
    dFzdx, dFzdy, dFzdz = np.gradient(F.z, d0, d1, d2)

    jacdet = (1. + dFxdx) * (1. + dFydy) * (1. + dFzdz) \
           - (1. + dFxdx) * dFydz * dFzdy \
           - dFxdy * dFydx * (1. + dFzdz) \
           + dFxdy * dFydz * dFzdx \
           + dFxdz * dFydx * dFzdy \
           - dFxdz * (1. + dFydy) * dFzdx
    jacdet = np.abs(jacdet)
    jacdet[np.where(jacdet > siemens_max_det)] = siemens_max_det

    return jacdet


def eval_spherical_harmonics(coeffs, vendor, vxyz, resolution=None):
    ''' Evaluate spherical harmonics

    Parameters
    ----------
    coeffs : Coeffs (namedtuple)
        the sph. harmonics coefficients got by parsing
    vxyz : CoordsVector (namedtuple). Could be a scalar or a 6-element list
        the x, y, z coordinates
        in case of scalar or 3-element list, the coordinates are eval
        in the function
    resolution : float
        (optional) useful in case vxyz is scalar
    '''
    # convert radius into mm
    R0 = coeffs.R0_m * 1000

    # in case vxyz is float
    if type(vxyz) is float:
        if not resolution:
            raise ValueError('eval_spherical_harmonics needs a resolution. '
                             'argument if vxyz is scalar or 3-element vector')
        coords = np.linspace(-vxyz, vxyz, resolution)
        x, y, z = utils.ndgrid(coords, coords, coords)

    # in case vxyz is a 3-element vector where each element represents the
    # edges of the x, y and z direction respectively
    if len(vxyz) == 3 and not type(vxyz) == CV:
        if not resolution:
            raise ValueError('eval_spherical_harmonics needs a resolution '
                             'argument if vxyz is scalar or 3-element vector')
        cx = np.linspace(-vxyz[0], vxyz[0], resolution)
        cy = np.linspace(-vxyz[1], vxyz[1], resolution)
        cz = np.linspace(-vxyz[2], vxyz[2], resolution)
        x, y, z = utils.ndgrid(cx, cy, cz)

    # at this point vxyz is an instance of CoordsVector
    x, y, z = vxyz

    # make them a 1d array
    x1d = np.ravel(x, order='F')
    y1d = np.ravel(y, order='F')
    z1d = np.ravel(z, order='F')

    log.info('calculating displacements (mm) '
            'using spherical harmonics coeffcients...')
    if vendor == 'siemens':
        log.info('along x...')
        bx = siemens_B(coeffs.alpha_x, coeffs.beta_x, x1d, y1d, z1d, R0)
        log.info('along y...')
        by = siemens_B(coeffs.alpha_y, coeffs.beta_y, x1d, y1d, z1d, R0)
        log.info('along z...')
        bz = siemens_B(coeffs.alpha_z, coeffs.beta_z, x1d, y1d, z1d, R0)
    else:
        # GE
        log.info('along x...')
        bx = ge_D(coeffs.alpha_x, coeffs.beta_x, x1d, y1d, z1d)
        log.info('along y...')
        by = ge_D(coeffs.alpha_y, coeffs.beta_y, x1d, y1d, z1d)
        log.info('along z...')
        bz = ge_D(coeffs.alpha_z, coeffs.beta_z, x1d, y1d, z1d)

    Bx = np.reshape(bx, x.shape, order='F')
    By = np.reshape(by, y.shape, order='F')
    Bz = np.reshape(bz, z.shape, order='F')
    return CV(Bx * R0, By * R0, Bz * R0), CV(x, y, z)


def siemens_B(alpha, beta, x1, y1, z1, R0):
    ''' Calculate displacement field from Siemens coefficients
    '''
    nmax = alpha.shape[0] - 1
    x1 = x1 + 0.0001  # hack to avoid singularities at R=0

    # convert to spherical coordinates
    r = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    theta = np.arccos(z1 / r)
    phi = np.arctan2(y1 / r, x1 / r)

    b = np.zeros(x1.shape)
    for n in xrange(0, nmax + 1):
        f = np.power(r / R0, n)
        for m in xrange(0, n + 1):
            f2 = alpha[n, m] * np.cos(m * phi) + beta[n, m] * np.sin(m * phi)
            #_ptemp = utils.legendre(n, m, np.cos(theta))
            _ptemp = scipy.special.lpmv(m, n, np.cos(theta))
            normfact = 1
            # this is Siemens normalization
            if m > 0:
                normfact = math.pow(-1, m) * \
                math.sqrt((2 * n + 1) * factorial(n - m) \
                          / (2 * factorial(n + m)))
            _p = normfact * _ptemp
            b = b + f * _p * f2
    return b


def ge_D(alpha, beta, x1, y1, z1):
    ''' GE Gradwarp coeffs define the error rather than the total
    gradient field'''

    nmax = alpha.shape[0] - 1
    x1 = x1 + 0.0001  # hack to avoid singularities
    r = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    # For consistency with GE papers, use theta & phi -> phi & theta
    phi = np.arccos(z1 / r)
    theta = np.arctan2(y1 / r, x1 / r)

    r = r * 100.0  # GE wants cm, so meters -> cm
    d = np.zeros(x1.shape)

    for n in xrange(0, nmax + 1):
        # So GE uses the usual unnormalized legendre polys.
        f = np.power(r, n)
        for m in xrange(0, n + 1):
            f2 = alpha[n, m] * np.cos(m * theta) + beta[n, m] \
            * np.sin(m * theta)
            _p = utils.legendre(n, m, np.cos(phi))
            d = d + f * _p * f2
    d = d / 100.0  # cm back to meters
    return d
