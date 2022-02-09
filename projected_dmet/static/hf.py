import numpy as np
import sys
#from research.codes import diagonalize
from codes import diagonalize
 #1 electron density matrix, C =spatial  orbitals, Ne = # of el

def rdm_1el(C, Nocc):
    Coc = C[:,:Nocc] #occupied orbitals
    P = 2*np.dot(Coc, np.transpose(np.conjugate(Coc)))
    return P

###########################################################
#mean-field (U=0) calculation for hubbard model

def hubbard_1RDM(Nelec, Hcore):
    #diagonalize hopping hamiltonian
    evals,orbs = diagonalize(Hcore)
    #form 1RDM
    RDM = rdm_1el(orbs, int(Nelec/2))

#    for i in range(Nelec):
 #        if i % 2 ==0:
  #           RDM[i][i] += 0.01
   #      else:
    #         RDM[i][i] -= 0.01
    return RDM
