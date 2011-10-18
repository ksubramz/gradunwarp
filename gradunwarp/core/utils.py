### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


def transform_coordinates(A1, A2, A3, M):
    ''' 4x4 matrix M operates on orthogonal coordinates arrays A1, A2, A3 to give
    B1, B2, B3
    '''
    B1 = A1 * M[0, 0] + A2 * M[0, 1] + A3 * M[0, 2] + M[0, 3]
    B2 = A1 * M[1, 0] + A2 * M[1, 1] + A3 * M[1, 2] + M[1, 3]
    B3 = A1 * M[2, 0] + A2 * M[2, 1] + A3 * M[2, 2] + M[2, 3]
    return B1, B2, B3
