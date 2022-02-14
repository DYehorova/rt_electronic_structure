import numpy as np
import scipy.linalg as la


def diagonalize(H, S=None):
    if S is None:
        S = np.identity(len(H))

    E, C = la.eigh(H, S)

    return E, C
##################################


def commutator(Mat1, Mat2):
    return np.dot(Mat1, Mat2) - np.dot(Mat2, Mat1)
##################################


def printarray(array, filename='array.dat', long_fmt=False):
    # subroutine to print out an ndarry of
    # 2,3 or 4 dimensions to be read by humans

    dim = len(array.shape)
    filehandle = open(filename, 'w')
    comp_log = np.iscomplexobj(array)

    if(comp_log):

        if(long_fmt):
            fmt_str = '%25.14e%+.14ej'
        else:
            fmt_str = '%15.4f%+.4fj'
    else:

        if(long_fmt):
            fmt_str = '%25.14e'
        else:
            fmt_str = '%15.4f'

    if (dim == 1):

        Ncol = 1
        np.savetxt(filehandle, array, fmt_str*Ncol)

    elif (dim == 2):

        Ncol = array.shape[1]
        np.savetxt(filehandle, array, fmt_str*Ncol)

    elif (dim == 3):

        for dataslice in array:
            Ncol = dataslice.shape[1]
            np.savetxt(filehandle, dataslice, fmt_str*Ncol)
            filehandle.write('\n')

    elif (dim == 4):

        for i in range(array.shape[0]):
            for dataslice in array[i, :, :, :]:
                Ncol = dataslice.shape[1]
                np.savetxt(filehandle, dataslice, fmt_str*Ncol)
                filehandle.write('\n')
        filehandle.write('\n')

    else:
        print('ERROR: Input array for printing is not of dimension 2, 3, or 4')
        exit()

    filehandle.close()
##################################


def rot1el(h_orig, rotmat):
    # rotate one electron integrals

    tmp = np.dot(h_orig, rotmat)
    if(np.iscomplexobj(rotmat)):
        h_rot = np.dot(rotmat.conjugate().transpose(), tmp)
    else:
        h_rot = np.dot(rotmat.transpose(), tmp)

    return h_rot
##################################


def adjoint(Mat):
    # subroutine to calculate the conjugate
    # transpose (ie adjoint) of a matrix
    return np.conjugate(np.transpose(Mat))
##################################


def rot2el_chem(V_orig, rotmat):
    # subroutine to rotate two electron integrals,
    # V_orig must be in chemist notation
    # V_orig starts as Nb x Nb x Nb x Nb and rotmat is Nb x Ns

    if(np.iscomplexobj(rotmat)):
        rotmat_conj = rotmat.conjugate().transpose()
    else:
        rotmat_conj = rotmat.transpose()

    V_new = np.einsum('trus,sy -> truy', V_orig, rotmat)
    # V_new now Nb x Nb x Nb x Ns

    V_new = np.einsum('vu,truy -> trvy', rotmat_conj, V_new)
    # V_new now Nb x Nb x Ns x Ns

    V_new = np.einsum('trvy,rx -> txvy', V_new, rotmat)
    # V_new now Nb x Ns x Ns x Ns

    V_new = np.einsum('wt,txvy -> wxvy', rotmat_conj, V_new)
    # V_new now Ns x Ns x Ns x Ns

    return V_new
##################################
