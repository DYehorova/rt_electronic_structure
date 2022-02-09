#Mod that contatins subroutines necessary to calculate the analytical form
#of the first time-derivative of the mean-field 1RDM
#this is necessary when integrating the MF 1RDM and CI coefficients explicitly
#while diagonalizing the MF 1RDM at each time-step to obtain embedding orbitals

import real_time_elec_structureGN.scripts.utils as utils
import numpy as np
import multiprocessing as multproc

import time

#####################################################################

def get_ddt_mf1rdm_serial(dG, system, Nocc ):

    #Subroutine to solve for the time-dependence of the MF 1RDM
    #this returns the time-derivative NOT i times the time-derivative

    #NOTE: prior to this routine being called, necessary to have the rotation matrices, 1RDM, and 2RDM for each fragment
    #as well as the natural orbitals and eigenvalues of the global 1RDM previously calculated

    #Calculate the Hamiltonian commutator portion of the time-dependence of correlated 1RDM for each fragment
    #ie i\tilde{ \dot{ correlated 1RDM } } using notation from notes
    system.get_frag_iddt_corr1RDM()

    #Calculate i times time-derivative of global 1RDM
    iddt_glob1RDM = calc_iddt_glob1RDM( system )

    #Calculate G-matrix governing time-dependence of natural orbitals
    #This G is in the natural orbital basis
    G = calc_Gmat(dG, system, iddt_glob1RDM )
   # print("G", G)
    tmp  = np.linalg.multi_dot( [system.NOevecs, G[:,:Nocc], np.conjugate( np.transpose(system.NOevecs[:,:Nocc] ) )] )
    ddt_mf1RDM_test = -1j * 2.0 * ( tmp - tmp.conj().T )

   #G in site basis

    G_site = utils.rot1el( G, utils.adjoint(system.NOevecs) )
    #calculate derivative of natural orbitals
    ddt_mf1RDM = -1j * ( np.dot( G_site, system.mf1RDM ) - np.dot( system.mf1RDM, G_site ) )
    ddt_NOevecs = -1j * np.dot(G_site, system.NOevecs)

    #Calculate time-derivative of MF 1RDM
    print("G in emb is hermitian:", np.allclose(G, G.conjugate().transpose(), rtol=0.0, atol=1e-12 ),"G in site is hermitian", np.allclose(G_site, G_site.conjugate().transpose(), rtol=0.0, atol=1e-12 ) )
    #print("two ddt mf methods are the same:", np.allclose(ddt_mf1RDM, ddt_mf1RDM_test, rtol=0.0, atol=1e-12 ) )
    #print(np.allclose(G, system.h_site, rtol=0.0, atol=1e-12 ) )
    #print("difference between ddt mf:")
    #print(ddt_mf1RDM - ddt_mf1RDM_test)
    short_NOcc = np.copy(system.NOevecs[:, :round(system.Nele/2)])
    short_ddtNOcc = np.copy(ddt_NOevecs[:, :round(system.Nele/2)])
    chk = 2 * ( np.dot(short_ddtNOcc, short_NOcc.conj().T) + np.dot(short_NOcc, short_ddtNOcc.conj().T))

    ddt_glob1RDM = -1j * iddt_glob1RDM
    #print("#########################")
    #print("check for derivative in NOs", np.allclose(chk, ddt_mf1RDM_test, 1e-12))
    #print("#########################")
    print("#########################")
    print("check for derivative in NOs with comm", np.allclose(chk, ddt_mf1RDM, rtol=0, atol=1e-12))
    print("#########################")
    if (np.allclose(chk, ddt_mf1RDM, rtol=0, atol=1e-12))==False:

        print("difference", ddt_mf1RDM_test - ddt_mf1RDM)
    
    return ddt_glob1RDM, ddt_NOevecs, ddt_mf1RDM

#####################################################################

def calc_iddt_glob1RDM( system ):
    #Subroutine to calculate i times time dependence of global 1RDM forcing anti-hermiticity

    #unpack necessary stuff
    rotmat_unpck = np.zeros( [system.Nsites,system.Nsites,system.Nsites], dtype=complex )
    iddt_corr1RDM_unpck = np.zeros( [system.Nsites,system.Nsites], dtype=complex )
    for q in range(system.Nsites):

        #fragment for site q
        frag = system.frag_list[ system.site_to_frag_list[q] ]

        #index within fragment corresponding to site q - note that q is an impurity orbital
        qimp = system.site_to_impindx[q]

        #unpack rotation matrix
        rotmat_unpck[:,:,q] = np.copy( frag.rotmat )

        #unpack necessary portion of iddt_corr1RDM
        iddt_corr1RDM_unpck[:,q] = np.copy( frag.iddt_corr1RDM[:,qimp] )

    #calculate intermediate matrix
    tmp  = np.einsum( 'paq,aq->pq', rotmat_unpck, iddt_corr1RDM_unpck )
   
    #ERRR    

    #return iddt_glob1RDM
    #print("iddt Glob1RDM", 0.5*( tmp - tmp.conj().T ))
    return 0.5*( tmp - tmp.conj().T )

#####################################################################

def calc_Gmat( dG, system, iddt_glob1RDM ):
    #Subroutine to calculate matrix that governs time-dependence of natural orbitals

    #Matrix of one over the difference in global 1RDM eigenvalues
    #Set diagonal terms and terms where eigenvalues are almost equal to zero
    evals    = np.copy( system.NOevals )
    eval_dif = np.zeros( [ system.Nsites, system.Nsites ] )
    for a in range(system.Nsites):
        for b in range(system.Nsites):
           # if( a != b ):
            if( a != b and np.abs( evals[a] - evals[b] ) > dG):
                eval_dif[a,b] = 1.0 / ( evals[b] - evals[a] )

  #  print("1/eval_dif", eval_dif)
    #Rotate iddt_glob1RDM
    #Gmat = np.zeros( [ system.Nsites, system.Nsites ] )
    Gmat = utils.rot1el( iddt_glob1RDM, system.NOevecs )
#    print("Gmat", Gmat)
    #mrar
    #print( np.allclose(Gmat, -Gmat.conjugate().transpose(), rtol=0.0, atol=1e-12 ) )

    #Multiply difference in eigenvalues and rotated time-derivative matrix
    Gmat1 = np.multiply( eval_dif, Gmat )
 #   print("Gmat times evals", Gmat1)
    #Force Hermiticity (this is all inefficient as of now)
    Gmat2 = np.triu(Gmat1) + np.triu(Gmat1,1).conjugate().transpose()
  #  print("Gmat hermitian", Gmat2)
    #mrar
    #print( np.allclose(eval_dif, -eval_dif.conjugate().transpose(), rtol=0.0, atol=1e-12 ) )
    #print( np.allclose(Gmat, Gmat.conjugate().transpose(), rtol=0.0, atol=1e-12 ) )
    ##print(eval_dif)
    #print()

    #mrar
    #chk = np.zeros( [system.Nsites,system.Nsites], dtype=complex )
    #for mu in range(system.Nsites):
    #    for nu in range(system.Nsites):
    #        for p in range(system.Nsites):
    #            for q in range(system.Nsites):
    #                if( mu != nu and np.abs(evals[nu]-evals[mu]) > 1e-10 ):
    #                    chk[mu,nu] += np.conjugate(system.NOevecs[p,mu]) * iddt_glob1RDM[p,q] * system.NOevecs[q,nu] / ( evals[nu]-evals[mu] )
    #                #chk[mu,nu] += np.conjugate(system.NOevecs[p,mu]) * iddt_glob1RDM[p,q] * system.NOevecs[q,nu]

    #chk2 = np.zeros( [system.Nsites,system.Nsites], dtype=complex )
    #for mu in range(system.Nsites):
    #    for nu in range(system.Nsites):
    #        for p in range(system.Nsites):
    #            for q in range(system.Nsites):
    #                    chk2[mu,nu] += np.conjugate(system.NOevecs[p,mu]) * iddt_glob1RDM[p,q] * system.NOevecs[q,nu]
    #        if( mu != nu and np.abs(evals[nu]-evals[mu]) > 1e-10 ):
    #            chk2[mu,nu] = chk2[mu,nu] / ( evals[nu]-evals[mu] )
    #        else:
    #            chk2[mu,nu] = 0.0

    #mrar
    #print( np.allclose( Gmat, chk, rtol=1e-10, atol=1e-13 ) )
    #print( np.allclose( chk, chk2, rtol=1e-10, atol=1e-13 ) )
    #print( np.allclose( Gmat, chk2, rtol=1e-10, atol=1e-13 ) )
    #print(eval_dif)
    #print()

    return Gmat2

#####################################################################




