import numpy as np
import scipy.linalg as la
import sys
import os
from real_time_elec_structureGN.scripts.utils import *
from pyscf import gto, scf, ao2mo

#####################################################################
def interactive_RHF(Nele, h_site, V_site):
    Norbs = Nele
    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h_site
    mf.get_ovlp = lambda *args: np.eye(Norbs)
    mf._eri = ao2mo.restore(8, V_site, Norbs)
    mf.kernel()
    mf1RDM = mf.make_rdm1()

    return mf1RDM

#####################################################################
def rhf_calc_hubbard(Nelec,Hcore):

    #simplified subroutine to perform a mean-field (ie U=0) calculation for Hubbard model

    #Diagonalize hopping-hamiltonian
    evals,orbs = diagonalize(Hcore)

    #Form the 1RDM
    P = rdm_1el(orbs,int(Nelec/2))

    return P

#####################################################################

def rdm_1el(C,Ne):
    #subroutine that calculates and returns the one-electron density matrix in original site basis

    Cocc = C[:,:Ne]
    P = 2*np.dot( Cocc,np.transpose(np.conjugate(Cocc)) )

    return P

#####################################################################

