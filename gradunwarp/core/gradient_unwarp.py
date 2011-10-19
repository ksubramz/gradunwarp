### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import argparse as arg
import os
import logging
import globals
import coeffs

log = globals.get_logger()


def argument_parse_gradunwarp():
    '''Arguments parser from the command line
    '''
    # initiate
    p = arg.ArgumentParser(version=globals.VERSION, usage=globals.usage)

    # required arguments
    p.add_argument('infile', action='store')
    p.add_argument('outfile', action='store')
    p.add_argument('vendor', action='store', choices=['siemens', 'ge'])

    coef_grp = p.add_mutually_exclusive_group(required=True)
    coef_grp.add_argument('-g', '--gradfile', dest='gradfile')
    coef_grp.add_argument('-c', '--coeffile', dest='coeffile')

    # optional arguments
    p.add_argument('-w', '--warp', action='store_true', default=False)
    p.add_argument('-n', '--nojacobian', dest='nojac', action='store_true',
                  default=False)
    p.add_argument('--verbose', action='store_true', default=False)

    args = p.parse_args()

    # do some validation
    if not os.path.exists(args.infile):
        raise IOError(args.infile + ' not found')
    if not os.path.exists(args.outfile):
        raise IOError(args.outfile + ' not found')
    if args.gradfile:
        if not os.path.exists(args.gradfile):
            raise IOError(args.gradfile + ' not found')
    if args.coeffile:
        if not os.path.exists(args.coeffile):
            raise IOError(args.coeffile + ' not found')

    return args


class GradientUnwarpRunner(object):
    ''' Takes the option datastructure after parsing the commandline.
    run() method performs the actual unwarping
    write() method performs the writing of the unwarped volume
    '''
    def __init__(self, args):
        ''' constructor takes the option datastructure which is the
        result of (options, args) = parser.parse_args()
        '''
        self.args = args

        log.setLevel(logging.INFO)
        if hasattr(self.args, 'verbose'):
            log.setLevel(logging.DEBUG)

    def run(self):
        ''' run the unwarp resample
        '''
        # get the spherical harmonics coefficients from parsing
        # the given .coeff file xor .grad file
        if hasattr(self.args, 'gradfile'):
            self.coeffs = coeffs.get_coefficients(self.args.vendor,
                                                 self.args.gradfile)
        else:
            self.coeffs = coeffs.get_coefficients(self.args.vendor,
                                                 self.args.coeffile)

        self.vol, self.m_rcs2ras = utils.get_vol_affine(self.args‥infile)

        unwarper = Unwarper(self.vol, self.m_rcs2ras, self.args.vendor,
                            self.coeffs)
        if hasattr(self.args, 'warp') and self.args.warp:
            unwarper.warp = True
        if hasattr(self.args, 'nojac') and self.args.nojac:
            unwarper.nojac = True

    def write(self):
        pass


if __name__ == '__main__':
    args = argument_parse_gradunwarp()

    grad_unwarp = GradientUnwarpRunner(args)

    grad_unwarp.run()

    grad_unwarp.write()
