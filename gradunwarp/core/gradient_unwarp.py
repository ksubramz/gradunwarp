### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import optparse as opt
import logging
import constants

log = constants.getlogger()


def options_parse_gradunwarp():
    # initiate the option parser
    p = opt.OptionParser(version=constants.VERSION, usage=constants.usage)


class GradientUnwarpRunner(object):
    ''' Takes the option datastructure after parsing the commandline.
    run() method performs the actual unwarping
    write() method performs the writing of the unwarped volume
    '''
    def __init__(self, options):
        ''' constructor takes the option datastructure which is the
        result of (options, args) = parser.parse_args()
        '''
        self.opts = options

        log.setLevel(logging.DEBUG if self.opts.verbose else logging.INFO)

    def run(self):
        pass

    def write(self):
        pass


if __name__ == '__main__':
    opts = options_parse_gradunwarp()

    grad_unwarp = GradientUnwarpRunner(opts)

    grad_unwarp.run()

    grad_unwarp.write()
