# Mod that contains subroutines associated with pyscf FCI calculations

import numpy as np
import feb_8_update.scripts.utils as utils
import feb_8_update.scripts.applyham_pyscf as applyham_pyscf
import pyscf.fci
from pyscf import gto, scf, ao2mo
#####################################################################


def FCI_GS(h, V, Ecore, Norbs, Nele):

    # Subroutine to perform groundstate FCI calculation using pyscf

    if(isinstance(Nele, tuple)):
        Nele = sum(Nele)

    # Define pyscf molecule
    mol = gto.M()
    mol.nelectron = Nele
    # this call is necessary to use user defined hamiltonian in fci step
    mol.incore_anyway = True

    # First perform HF calculation
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.get_ovlp = lambda *args: np.eye(Norbs)
    mf._eri = ao2mo.restore(8, V, Norbs)
    mf.kernel()

    # Perform FCI calculation using HF MOs
    cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
    E_FCI, CIcoeffs = cisolver.kernel()

    # Rotate CI coefficients back to basis used in DMET calculations
    CIcoeffs = (
        pyscf.fci.addons.transform_ci_for_orbital_rotation(
            CIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff)))

    return CIcoeffs
#####################################################################


def get_corr1RDM(CIcoeffs, Norbs, Nele):

    # Subroutine to get the FCI 1RDM, notation is rho_pq = < c_q^dag c_p >

    if(np.iscomplexobj(CIcoeffs)):

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        corr1RDM = (
            1j * pyscf.fci.direct_spin1.trans_rdm1(
                Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele))

        corr1RDM -= (
            1j * pyscf.fci.direct_spin1.trans_rdm1(
                Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele))

        corr1RDM += (
            pyscf.fci.direct_spin1.make_rdm1(
                Re_CIcoeffs, Norbs, Nele))

        corr1RDM += pyscf.fci.direct_spin1.make_rdm1(Im_CIcoeffs, Norbs, Nele)

    else:

        corr1RDM = pyscf.fci.direct_spin1.make_rdm1(CIcoeffs, Norbs, Nele)

    return corr1RDM
#####################################################################


def get_corr12RDM(CIcoeffs, Norbs, Nele):

    # Subroutine to get the FCI 1 & 2 RDMs together
    # Notation for 1RDM is rho_pq = < c_q^dag c_p >
    # Notation for 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

    if(np.iscomplexobj(CIcoeffs)):

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        corr1RDM, corr2RDM = (
            pyscf.fci.direct_spin1.trans_rdm12(
                Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele))

        corr1RDM = corr1RDM*1j
        corr2RDM = corr2RDM*1j

        tmp1, tmp2 = (
            pyscf.fci.direct_spin1.trans_rdm12(
                Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele))

        corr1RDM -= 1j * tmp1
        corr2RDM -= 1j * tmp2

        tmp1, tmp2 = (
            pyscf.fci.direct_spin1.make_rdm12(
                Re_CIcoeffs, Norbs, Nele))

        corr1RDM += tmp1
        corr2RDM += tmp2

        tmp1, tmp2 = (
            pyscf.fci.direct_spin1.make_rdm12(
                Im_CIcoeffs, Norbs, Nele))

        corr1RDM += tmp1
        corr2RDM += tmp2

    else:

        corr1RDM, corr2RDM = (
            pyscf.fci.direct_spin1.make_rdm12(
                CIcoeffs, Norbs, Nele))

    return corr1RDM, corr2RDM
#####################################################################


def get_trans1RDM(CIcoeffs_1, CIcoeffs_2, Norbs, Nele):

    # Subroutine to get the transition 1RDM between two CI vectors
    # notation is rho_pq = < psi_1 | c_q^dag c_p | psi_2 >

    if(np.iscomplexobj(CIcoeffs_1) or np.iscomplexobj(CIcoeffs_2)):

        Re_CIcoeffs_1 = np.copy(CIcoeffs_1.real)
        Im_CIcoeffs_1 = np.copy(CIcoeffs_1.imag)

        Re_CIcoeffs_2 = np.copy(CIcoeffs_2.real)
        Im_CIcoeffs_2 = np.copy(CIcoeffs_2.imag)

        corr1RDM = (
            1j * pyscf.fci.direct_spin1.trans_rdm1(
                Re_CIcoeffs_1, Im_CIcoeffs_2, Norbs, Nele))

        corr1RDM -= (
            1j * pyscf.fci.direct_spin1.trans_rdm1(
                Im_CIcoeffs_1, Re_CIcoeffs_2, Norbs, Nele))

        corr1RDM += (
            pyscf.fci.direct_spin1.trans_rdm1(
                Re_CIcoeffs_1, Re_CIcoeffs_2, Norbs, Nele))

        corr1RDM += (
            pyscf.fci.direct_spin1.trans_rdm1(
                Im_CIcoeffs_1, Im_CIcoeffs_2, Norbs, Nele))

    else:

        corr1RDM = (pyscf.fci.direct_spin1.trans_rdm1(
            CIcoeffs_1, CIcoeffs_2, Norbs, Nele))

    return corr1RDM
#####################################################################


def get_FCI_E(h, V, Econst, CIcoeffs, Norbs, Nalpha, Nbeta):

    # Subroutine to calculate the FCI electronic energy
    # for given Hamiltonian and FCI vector
    # Works with complex Hamitlonian and FCI vector

    Hpsi = applyham_pyscf.apply_ham_pyscf_fully_complex(
        CIcoeffs, h, V, Nalpha, Nbeta, Norbs, Econst)

    Re_Hpsi = np.copy(Hpsi.real)
    Im_Hpsi = np.copy(Hpsi.imag)

    Re_CIcoeffs = np.copy(CIcoeffs.real)
    Im_CIcoeffs = np.copy(CIcoeffs.imag)

    FCI_E = pyscf.fci.addons.overlap(
        Re_CIcoeffs, Re_Hpsi, Norbs, (Nalpha, Nbeta))
    FCI_E += pyscf.fci.addons.overlap(
        Im_CIcoeffs, Im_Hpsi, Norbs, (Nalpha, Nbeta))
    FCI_E += 1j * pyscf.fci.addons.overlap(
        Re_CIcoeffs, Im_Hpsi, Norbs, (Nalpha, Nbeta))
    FCI_E -= 1j * pyscf.fci.addons.overlap(
        Im_CIcoeffs, Re_Hpsi, Norbs, (Nalpha, Nbeta))

    return FCI_E.real
#####################################################################
