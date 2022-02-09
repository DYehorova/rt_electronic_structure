import numpy as np
import sys
import os
#import research.codes as codes
import codes
import pyscf.fci 
from pyscf import gto, scf, ao2mo
#import applyham_pyscf

def RHF(h,V, Norbs, Nele):
    if( isinstance(Nele,tuple)  ):
        Nele = sum(Nele)

    mol = gto.M()
    mol.nelectron = Nele
    mol.imncore_anyway = True

    mf = scf.RHF(mol)
#    mf = scf.UHF(mol)# after fixed modify the initial guess to break the symmetry to converge
    mf.get_hcore = lambda *args: h
    mf.get_ovlp = lambda *args: np.eye(Norbs)

    mf._eri = ao2mo.restore(8, V, Norbs)
   # evals, h = np.linalg.eigh(h_site) 
   # mf.init_guess = h 
    mf.kernel()
    RDM = mf.make_rdm1()
    # add noise:
    #for i in range(Nele):
     #   if i % 2 ==0:
      #      RDM[i][i] += 0.001
       # else:
        #    RDM[i][i] -= 0.001
    #RDM[0][0] += 0.002
    #RDM[1][1] -= 0.002
    #RDM[2][2] -= 0.003
    #RDM[3][3] += 0.001
    #RDM[4][4] += 0.003
    #RDM[5][5] -= 0.001
    return RDM



###########################################################

def FCI_GS( h, V, U, Norbs, Nele ):
    if( isinstance(Nele,tuple)  ):
        Nele = sum(Nele)
   #print('FCI nele', Nele)
    #print('h')
    #print(h)
    #print('U')
    #print(U)
    #print('V')
    #print(V)
    #print('norb',Norbs)
#Define PySCF molecule
    mol = gto.M()
    print("made mol")
    mol.nelectron = Nele
    mol.imncore_anyway = True
    #necessary to use because want to use customized  hamiltonian after HF

    #First need an HF calculation (to get a better guess for the starting point )
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h #what is lambda ?
    
    mf.get_ovlp = lambda *args: np.eye(Norbs)

    # why pyscf uses Nele and Josh uses Norb?
    #(wouldn't there be a differenece of 2?)
    #args - goes though the list and gives each member to the function
    #lambda function - way to define a function

    mf._eri = ao2mo.restore(8, V, Norbs)
    print('about to run HF with lernel command')
    #taking advantage of symmetry in 2e term (assuming orbitals are real - 8fold symmetry)
    #if orbitals are complex - 4 fold symmetry
    # has to do with  8-fold permutation symmetry of the integrals
    #and 2e integrals (notation symmetries of 2e integrals)
    mf.kernel()
    sys.stdout.flush()
    print("done with HF")
    #Second - FCI calculation using HF molecular orbitals

    # check HF  for hubbard model, V = 0.
    if U == 0:
        evals, evecs = codes.diagonalize(h)

        E_check = (sum(evals [:int(Norbs/2)]))
        print('E_check',2*E_check)

    #might be useful to use direct_uhf.FCI() instead for the cisolver
    #Second - FCI calculation using HF molecular orbitals

    cisolver = pyscf.fci.FCI(mf,mf.mo_coeff)
    print("assigned cisolver")
    E_FCI, CIcoeffs = cisolver.kernel()
    print("calculated energy")
#    print('E_FCI:',E_FCI)

    #Check full CI

    #print('U=',U)

  #  Test_H = [[2*h[0][0]+U, h[0][1], h[0][1], 0],
   #          [h[1][0], h[0][0]+h[1][1], 0, h[0][1]],
    #         [h[1][0], 0, h[0][0]+h[1][1], h[0][1]],
     #        [0, h[1][0], h[1][0], 2*h[1][1] + V[1,1,1,1]]]

    #print('Test H', Test_H)

    #evals, evecs = codes.diagonalize(Test_H)

    #print('evals_fci', evals)

#    E_FCI_check = (sum(evals [:int(Norbs/2)]))
 #   print('E_FCI_check',E_FCI_check)


    #Need to rotate CI coefficients back to embeding basis used in DMET (because now they are in orbital basis)
    CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(CIcoeffs, Norbs, Nele, codes.adjoint(mf.mo_coeff))
    print('done rotating CI_coeficients:')
    return CIcoeffs, E_FCI
###########################################################

def get_corr1RDM( CIcoeffs, Norbs, Nele ):

    #subroutine to get the FCI 1RDM, (rho_pq = < c_q^dag c_p >)
    # C = RC +i IC => can rewrite:
    #PySCF uses only dencity amtricies for real numbers, so broken it into complex/real parts
    #<psi|a+a|psi> = <Rpsi|~|Rpsi> + <Ipsi|~|Ipsi> + i<Rpsi|~|Ipsi> - i<Ipsi|~|Rpsi>
    #transition density matrix  = any <a | ~ | b>

    if( np.iscomplexobj(CIcoeffs) ):
        Re_CIoeffs  = np.copy( CIcoeffs.real  )
        Im_CIcoeffs = np.copy( CIcoeffs.imag  )

        corr1RDM = 1j * pyscf.fci.direct_spin1.trans_rdm1( Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele )

        corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1( Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele )

        corr1RDM += pyscf.fci.direct_spin1.make_rdm1( Re_CIcoeffs, Norbs, Nele )
        orr1RDM += pyscf.fci.direct_spin1.make_rdm1( Im_CIcoeffs, Norbs, Nele )

    else:

       corr1RDM  = pyscf.fci.direct_spin1.make_rdm1( CIcoeffs, Norbs, Nele )

    return corr1RDM

###########################################################


def get_corr12RDM(CIcoeffs, Norbs, Nele):
    # Subroutine to get the FCI 1 & 2 RDMs together
    # Notation for 1RDM is rho_pq = < c_q^dag c_p >
    # Notation for 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

    if( np.iscomplexobj(CIcoeffs) ):

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        corr1RDM, corr2RDM  = pyscf.fci.direct_spin1.trans_rdm12(Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele)

        corr1RDM = corr1RDM*1j
        corr2RDM = corr2RDM*1j

        tmp1, tmp2 = pyscf.fci.direct_spin1.trans_rdm12( Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele )

        corr1RDM -= 1j * tmp1
        corr2RDM -= 1j * tmp2

        tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12( Re_CIcoeffs, Norbs, Nele )

        corr1RDM += tmp1
        corr2RDM += tmp2

        tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12( Im_CIcoeffs, Norbs, Nele )

        corr1RDM += tmp1
        corr2RDM += tmp2


    else:

        corr1RDM, corr2RDM  = pyscf.fci.direct_spin1.make_rdm12( CIcoeffs, Norbs, Nele )

    return corr1RDM, corr2RDM

