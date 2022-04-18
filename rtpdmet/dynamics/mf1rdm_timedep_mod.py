# Mod that contatins subroutines necessary to calculate the analytical form
# of the first time-derivative of the mean-field 1RDM
# this is necessary when integrating the MF 1RDM and CI coefficients explicitly
# while diagonalizing the MF1RDM at each time-step to obtain embedding orbitals

import rt_electronic_structure.scripts.utils as utils
import numpy as np
import multiprocessing as multproc

import time
#####################################################################


def get_ddt_mf_NOs(system, G_site):

    ddt_mf1RDM = (-1j * (np.dot(G_site, system.mf1RDM)
                         - np.dot(system.mf1RDM, G_site)))
    ddt_NOevecs = (-1j * np.dot(G_site, system.NOevecs))

    return ddt_mf1RDM, ddt_NOevecs
#####################################################################


def get_ddt_glob(dG, system):
    iddt_corr_time = time.time()
    system.get_frag_iddt_corr1RDM()
    print("corr derivative", time.time()-iddt_corr_time)
    # Calculate i times time-derivative of global 1RDM
    iddt_glob_time = time.time()
    iddt_glob1RDM = calc_iddt_glob1RDM(system)
    print("glob deriv", time.time()-iddt_glob_time)

    # Calculate G-matrix governing time-dependence of natural orbitals
    # This G is in the natural orbital basis
    Gmat_time = time.time()
    G_site = calc_Gmat(dG, system, iddt_glob1RDM)
    print("gmat time ", time.time()-Gmat_time)
    ddt_glob1RDM = -1j * iddt_glob1RDM

    return ddt_glob1RDM, G_site
#####################################################################


def get_ddt_mf1rdm_serial(dG, system, Nocc):

    # Subroutine to solve for the time-dependence of the MF 1RDM
    # this returns the time-derivative NOT i times the time-derivative

    # NOTE: prior to this routine being called, necessary to have
    # the rotation matrices, 1RDM, and 2RDM for each fragment
    # as well as the natural orbitals and eigenvalues of the global
    # 1RDM previously calculated

    # Calculate the Hamiltonian commutator portion
    # of the time-dependence of correlated 1RDM for each fragment

    # ie i\tilde{ \dot{ correlated 1RDM } } using notation from notes

    iddt_corr_time = time.time()
    system.get_frag_iddt_corr1RDM()
    print("corr derivative", time.time()-iddt_corr_time)

    # Calculate i times time-derivative of global 1RDM
    iddt_glob_time = time.time()
    iddt_glob1RDM = calc_iddt_glob1RDM(system)
    print("glob deriv", time.time()-iddt_glob_time)

    # Calculate G-matrix governing time-dependence of natural orbitals
    # This G is in the natural orbital basis
    Gmat_time = time.time()
    G_site = calc_Gmat(dG, system, iddt_glob1RDM)
    print("gmat time ", time.time()-Gmat_time)

    ddt_mf1RDM = (-1j * (np.dot(G_site, system.mf1RDM)
                         - np.dot(system.mf1RDM, G_site)))
    ddt_NOevecs = (-1j * np.dot(G_site, system.NOevecs))

    ddt_glob1RDM = -1j * iddt_glob1RDM

    # Calculate time-derivative of MF 1RDM
    short_NOcc = np.copy(system.NOevecs[:, :round(system.Nele/2)])
    short_ddtNOcc = np.copy(ddt_NOevecs[:, :round(system.Nele/2)])
    chk = 2 * (
        np.dot(short_ddtNOcc, short_NOcc.conj().T)
        + np.dot(short_NOcc, short_ddtNOcc.conj().T))

    print("#########################")
    print("check for derivative in NOs with comm",
          np.allclose(chk, ddt_mf1RDM, rtol=0, atol=1e-5))
    print("#########################")
    return ddt_glob1RDM, ddt_NOevecs, ddt_mf1RDM, G_site
#####################################################################


def calc_iddt_glob1RDM(system):
    # Subroutine to calculate i times
    # time dependence of global 1RDM forcing anti-hermiticity

    # unpack necessary stuff
    rotmat_unpck = np.zeros(
        [system.Nsites, system.Nsites, system.Nsites], dtype=complex)
    iddt_corr1RDM_unpck = np.zeros(
        [system.Nsites, system.Nsites], dtype=complex)
    for q in range(system.Nsites):

        # fragment for site q
        frag = system.frag_list[system.site_to_frag_list[q]]

        # index within fragment corresponding to site q -
        # note that q is an impurity orbital
        qimp = system.site_to_impindx[q]

        # unpack rotation matrix
        rotmat_unpck[:, :, q] = np.copy(frag.rotmat)

        # unpack necessary portion of iddt_corr1RDM
        iddt_corr1RDM_unpck[:, q] = np.copy(frag.iddt_corr1RDM[:, qimp])

    # calculate intermediate matrix
    tmp = np.einsum('paq,aq->pq', rotmat_unpck, iddt_corr1RDM_unpck)

    return 0.5*(tmp - tmp.conj().T)
#####################################################################


def calc_Gmat(dG, system, iddt_glob1RDM):
    # Subroutine to calculate matrix that
    # governs time-dependence of natural orbitals

    # Matrix of one over the difference in global 1RDM eigenvalues
    # Set diagonal terms and terms where eigenvalues are almost equal to zero
    evals = np.copy(system.NOevals)
    G2_fast_time = time.time()
    G2_fast = utils.rot1el(iddt_glob1RDM, system.NOevecs)
    for a in range(system.Nsites):
        for b in range(system.Nsites):
            if(a != b and np.abs(evals[a] - evals[b]) > dG):
                G2_fast[a, b] /= (evals[b] - evals[a])
            else:
                G2_fast[a, b] = 0
    G2_fast = (np.triu(G2_fast) +
               np.triu(G2_fast, 1).conjugate().transpose())
    G2_site = (utils.rot1el(G2_fast, utils.adjoint(system.NOevecs)))
    print("G2 fast time", time.time()-G2_fast_time)

    return G2_site
#####################################################################
