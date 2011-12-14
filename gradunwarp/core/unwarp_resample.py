### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import numpy as np
import sys
import pdb
import math
import logging
from scipy import ndimage
import utils
from utils import CoordsVector as CV
from utils import factorial
import globals
from globals import siemens_max_det
import nibabel as nib

#np.seterr(all='raise')

log = logging.getLogger('gradunwarp')


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
        self.m_rcs2lai = None

        # grid is uninit by default
        self.fovmin = None
        self.fovmax = None
        self.numpoints = None

        # interpolation order ( 1 = linear)
        self.order = 1

    def eval_spharm_grid(self, vendor, coeffs):
        ''' 
        We evaluate the spherical harmonics on a less sampled grid.
        This is a spacetime vs accuracy tradeoff.
        '''
        # init the grid first
        if not self.fovmin:
            fovmin = globals.siemens_fovmin
        else:
            fovmin = self.fovmin
        if not self.fovmax:
            fovmax = globals.siemens_fovmax
        else:
            fovmax = self.fovmax
        if not self.numpoints:
            numpoints = globals.siemens_numpoints
        else:
            numpoints = self.numpoints

        # convert to mm
        fovmin = fovmin * 1000.
        fovmax = fovmax * 1000.
        # the grid in meters. this is needed for spherical harmonics
        vec = np.linspace(fovmin, fovmax, numpoints)
        gvx, gvy, gvz = utils.meshgrid(vec, vec, vec)
        # mm
        cf = (fovmax - fovmin) / numpoints
        
        # deduce the transformation from rcs to grid
        g_rcs2xyz = np.array( [[0, cf, 0, fovmin],
                               [cf, 0, 0, fovmin],
                               [0, 0, cf, fovmin],
                               [0, 0, 0, 1]], dtype=np.float32 )

        # get the grid to rcs transformation also
        g_xyz2rcs = np.linalg.inv(g_rcs2xyz)

        # indices into the gradient displacement vol
        gr, gc, gs = utils.meshgrid(np.arange(numpoints), np.arange(numpoints),
                                 np.arange(numpoints), dtype=np.float32)

        log.info('Evaluating spherical harmonics')
        log.info('on a ' + str(numpoints) + '^3 grid')
        log.info('with extents ' + str(fovmin) + 'mm to ' + str(fovmax) + 'mm')
        gvxyz = CV(gvx, gvy, gvz)
        _dv, _dxyz = eval_spherical_harmonics(coeffs, vendor, gvxyz)
            
        return CV(_dv.x, _dv.y, _dv.z), CV(gr, gc, gs), g_xyz2rcs


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
        vc3, vr3, vs3 = utils.meshgrid(np.arange(nr), np.arange(nc), np.arange(ns), dtype=np.float32)
        vrcs = CV(x=vr3, y=vc3, z=vs3)
        vxyz = utils.transform_coordinates(vrcs, m_rcs2lai)

        # account for half-voxel shift in R and C directions
        halfvox = np.zeros((4, 4))
        halfvox[0, 3] = m_rcs2lai[0, 0] / 2.0
        halfvox[1, 3] = m_rcs2lai[1, 1] / 2.0
        m_rcs2lai = m_rcs2lai + halfvox

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

        # # convert image vol coordinates from RAS to LAI
        # vxyz = utils.transform_coordinates(vrcs, m_rcs2lai)

        # # compute new coordinates and the jacobian determinant
        # # TODO still not clear about what to return
        # self.out, self.vjacmult_lps = self.non_linear_unwarp(vxyz, dxyz,
        #                                                     m_rcs2lai)


        # for each slice
        '''
        for slice in xrange(ns):
            sys.stdout.flush()
            if (slice + 1) % 10 == 0:
                print(slice + 1),
            else:
                print('.'),

            # we are doing it slice by slice
            vs = np.ones(vr.shape) * slice

            vrcs2d = CV(vr, vc, slice)
            # rcs2lai
            vxyz2d = utils.transform_coordinates(vrcs2d, m_rcs2lai)

            # compute new coordinates and the jacobian determinant
            moddv, modxyz = eval_spherical_harmonics(self.coeffs, self.vendor, vxyz2d)
            dvx[..., slice] = moddv.x
            dvy[..., slice] = moddv.y
            dvz[..., slice] = moddv.z
            '''

        print
        # Evaluate spherical harmonics on a smaller grid 
        dv, grcs, g_xyz2rcs = self.eval_spharm_grid(self.vendor, self.coeffs)
        # do the nonlinear unwarp
        self.out, self.vjacmult_lps = self.non_linear_unwarp(vxyz, grcs, dv, dxyz,
                                                                 m_rcs2lai, g_xyz2rcs)

    def write(self, outfile):
        log.info('Writing output to ' + outfile)
        if outfile.endswith('.nii') or outfile.endswith('.nii.gz'):
            img = nib.Nifti1Image(self.out, self.m_rcs2ras)
        if outfile.endswith('.mgh') or outfile.endswith('.mgz'):
            self.out = self.out.astype(np.float32)
            img = nib.MGHImage(self.out, self.m_rcs2ras)
        nib.save(img, outfile)

    def non_linear_unwarp(self, vxyz, grcs, dv, dxyz, m_rcs2lai, g_xyz2rcs):
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
        # Jacobian multiplier is unitless but calculated in terms of
        # displacements and coordinates in LPS orientation
        # (right handed form of p.o.v of patient )
        if self.vendor == 'siemens':

            log.info('Interpolating the spherical harmonics grid')
            vrcsg = utils.transform_coordinates(vxyz, g_xyz2rcs)
            vrcsg_m = CV(vrcsg.y, vrcsg.x, vrcsg.z)
            dvx = ndimage.interpolation.map_coordinates(dv.y,
                                                        vrcsg_m,
                                                        order=self.order)
            dvy = ndimage.interpolation.map_coordinates(dv.x,
                                                        vrcsg_m,
                                                        order=self.order)
            dvz = ndimage.interpolation.map_coordinates(dv.z,
                                                        vrcsg_m,
                                                        order=self.order)

            log.info('Calculating the new locations of voxels')
            # new locations of the image voxels in XYZ ( LAI ) coords
            vxyzw = CV(x=vxyz.x + self.polarity * dvx,
                       y=vxyz.y + self.polarity * dvy,
                       z=vxyz.z + self.polarity * dvz)

            # if polarity is negative, the jacobian is also inversed
            if self.polarity == -1:
                vjacdet_lps = 1. / vjacdet_lps

            # hopefully, free memory
            del dvx, dvy, dvz, vxyz
            # convert the locations got into RCS indices
            vrcsw = utils.transform_coordinates(vxyzw,
                                                np.linalg.inv(m_rcs2lai))

            # hopefully, free memory
            del vxyzw
            # resample the image
            log.info('Interpolating the image')
            if self.vol.ndim == 3:
                #out = utils.interp3(self.vol, vrcsw.x, vrcsw.x, vrcsw.z)
                out = ndimage.interpolation.map_coordinates(self.vol,
                                                            vrcsw,
                                                            order=self.order)
            if self.vol.ndim == 4:
                nframes = self.vol.shape[3]
                out = np.zeros(self.vol.shape)
                for f in nframes:
                    _out = ndimage.interpolation.map_coordinates(self.vol[..., f],
                                                                vrcsw,
                                                                order=self.order)

            # find NaN voxels, and set them to 0
            out[np.where(np.isnan(out))] = 0.
            out[np.where(np.isinf(out))] = 0.

            vjacdet_lpsw = None
            # Multiply the intensity with the Jacobian det, if needed
            if not self.nojac:
                log.info('Evaluating the Jacobian multiplier')
                if dxyz == 0:
                    vjacdet_lps = 1
                else:
                    vjacdet_lps = eval_siemens_jacobian_mult(dv, dxyz)
                log.info('Interpolating the Jacobian multiplier')
                vjacdet_lpsw = ndimage.interpolation.map_coordinates(vjacdet_lps,
                                                            vrcsg_m,
                                                            order=self.order)
                vjacdet_lpsw[np.where(np.isnan(out))] = 0.
                vjacdet_lpsw[np.where(np.isinf(out))] = 0.

                log.info('Performing Jacobian multiplication')
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
    d0, d1, d2 = dxyz.x, dxyz.y, dxyz.z
    #print F.x.shape, d0, d1, d2

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


def eval_spherical_harmonics(coeffs, vendor, vxyz):
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
    R0 = coeffs.R0_m  * 1000

    x, y, z = vxyz

    #pdb.set_trace()
    # log.info('calculating displacements (mm) '
    #        'using spherical harmonics coeffcients...')
    if vendor == 'siemens':
        log.info('along x...')
        bx = siemens_B(coeffs.alpha_x, coeffs.beta_x, x, y, z, R0)
        log.info('along y...')
        by = siemens_B(coeffs.alpha_y, coeffs.beta_y, x, y, z, R0)
        log.info('along z...')
        bz = siemens_B(coeffs.alpha_z, coeffs.beta_z, x, y, z, R0)
    else:
        # GE
        log.info('along x...')
        bx = ge_D(coeffs.alpha_x, coeffs.beta_x, x, y, z)
        log.info('along y...')
        by = ge_D(coeffs.alpha_y, coeffs.beta_y, x, y, z)
        log.info('along z...')
        bz = siemens_B(coeffs.alpha_z, coeffs.beta_z, x, y, z, R0)
        bz = ge_D(coeffs.alpha_z, coeffs.beta_z, x, y, z)

    return CV(bx * R0, by * R0, bz * R0), CV(x, y, z)


#@profile
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
            _ptemp = utils.legendre(n, m, np.cos(theta))
            #_ptemp = scipy.special.lpmv(m, n, np.cos(theta))
            normfact = 1
            # this is Siemens normalization
            if m > 0:
                normfact = math.pow(-1, m) * \
                math.sqrt(float((2 * n + 1) * factorial(n - m)) \
                          / float(2 * factorial(n + m)))
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
