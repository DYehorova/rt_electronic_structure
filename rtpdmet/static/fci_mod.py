import numpy as np
import rt_electronic_structure.rtpdmet.static.codes as codes
import pyscf.fci
from pyscf import gto, scf, ao2mo
import sys


def RHF(h, V, Norbs, Nele):
    if(isinstance(Nele, tuple)):
        Nele = sum(Nele)

    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.get_ovlp = lambda *args: np.eye(Norbs)

    mf._eri = ao2mo.restore(8, V, Norbs)
    mf.kernel()
    RDM = mf.make_rdm1()
    return RDM
###########################################################


def FCI_GS(h, V, U, Norbs, Nele):
    if(isinstance(Nele, tuple)):
        Nele = sum(Nele)
    # Define PySCF molecule
    mol = gto.M()
    print("made mol")
    mol.nelectron = Nele
    mol.imncore_anyway = True
    # necessary to use because want to use
    # customized  hamiltonian after HF

    # First need an HF calculation
    # (to get a better guess for the starting point )
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h

    mf.get_ovlp = lambda *args: np.eye(Norbs)

    mf._eri = ao2mo.restore(8, V, Norbs)
    # taking advantage of symmetry in 2e term
    # (assuming orbitals are real - 8fold symmetry)
    # if orbitals are complex - 4 fold symmetry
    # has to do with  8-fold permutation symmetry of the integrals
    # and 2e integrals (notation symmetries of 2e integrals)
    mf.kernel()
    sys.stdout.flush()

    # FCI calculation using HF molecular orbitals

    # might be useful to use direct_uhf.FCI() instead for the cisolver
    # Second - FCI calculation using HF molecular orbitals

    cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
    E_FCI, CIcoeffs = cisolver.kernel()

    # Need to rotate CI coefficients back to embeding basis
    # used in DMET (because now they are in orbital basis)

    CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
        CIcoeffs, Norbs, Nele, codes.adjoint(mf.mo_coeff))
    return CIcoeffs, E_FCI
###########################################################


def get_corr1RDM(CIcoeffs, Norbs, Nele):

    # subroutine to get the FCI 1RDM, (rho_pq = < c_q^dag c_p >)
    # C = RC +i IC => can rewrite:
    # PySCF uses only dencity amtricies for real numbers,
    # so broken it into complex/real parts
    # <psi|a+a|psi> = <Rpsi|~|Rpsi> + <Ipsi|~|Ipsi>
    # + i<Rpsi|~|Ipsi> - i<Ipsi|~|Rpsi>
    # transition density matrix  = any <a | ~ | b>

    if(np.iscomplexobj(CIcoeffs)):
        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        corr1RDM = 1j * pyscf.fci.direct_spin1.trans_rdm1(
            Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele)

        corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1(
            Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele)

        corr1RDM += pyscf.fci.direct_spin1.make_rdm1(
            Re_CIcoeffs, Norbs, Nele)
        corr1RDM += pyscf.fci.direct_spin1.make_rdm1(
            Im_CIcoeffs, Norbs, Nele)
    else:
        corr1RDM = pyscf.fci.direct_spin1.make_rdm1(
            CIcoeffs, Norbs, Nele)

    return corr1RDM
###########################################################


def get_corr12RDM(CIcoeffs, Norbs, Nele):
    # Subroutine to get the FCI 1 & 2 RDMs together
    # Notation for 1RDM is rho_pq = < c_q^dag c_p >
    # Notation for 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

    if(np.iscomplexobj(CIcoeffs)):

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        corr1RDM, corr2RDM = pyscf.fci.direct_spin1.trans_rdm12(
            Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele)

        corr1RDM = corr1RDM*1j
        corr2RDM = corr2RDM*1j

        tmp1, tmp2 = pyscf.fci.direct_spin1.trans_rdm12(
            Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele)

        corr1RDM -= 1j * tmp1
        corr2RDM -= 1j * tmp2

        tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12(
            Re_CIcoeffs, Norbs, Nele)

        corr1RDM += tmp1
        corr2RDM += tmp2

        tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12(
            Im_CIcoeffs, Norbs, Nele)

        corr1RDM += tmp1
        corr2RDM += tmp2

    else:

        corr1RDM, corr2RDM = pyscf.fci.direct_spin1.make_rdm12(
            CIcoeffs, Norbs, Nele)

    return corr1RDM, corr2RDM
