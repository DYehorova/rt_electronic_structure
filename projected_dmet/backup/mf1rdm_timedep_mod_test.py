#Mod that contatins subroutines necessary to calculate the analytical form
#of the first time-derivative of the mean-field 1RDM
#this is necessary when integrating the MF 1RDM and CI coefficients explicitly
#while diagonalizing the MF 1RDM at each time-step to obtain embedding orbitals

import real_time_elec_structureGN.scripts.utils as utils
import numpy as np
import multiprocessing as multproc

import time

#####################################################################

def get_ddt_mf1rdm_serial(old_G_site_max, dG, system, Nocc ):

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
    #ERRR
    G_site = calc_Gmat(old_G_site_max, dG, system, iddt_glob1RDM )
   # print("G", G)
  #  tmp  = np.linalg.multi_dot( [system.NOevecs, G[:,:Nocc], np.conjugate( np.transpose(system.NOevecs[:,:Nocc] ) )] )
  #  ddt_mf1RDM_test = -1j * 2.0 * ( tmp - tmp.conj().T )

   #G in site basis

    #G_site = utils.rot1el( G, utils.adjoint(system.NOevecs) )
    #calculate derivative of natural orbitals
    #ERR
    
  #  ddt_mf1RDM =  (np.real(old_G_site_max) * 1000) * (-1j * ( np.dot( G_site, system.mf1RDM ) - np.dot( system.mf1RDM, G_site ) ))
  #  ddt_NOevecs = (np.real(old_G_site_max) * 1000) * (-1j * np.dot(G_site, system.NOevecs))
    iddt_mf1RDM =  (( np.dot( G_site, system.mf1RDM ) - np.dot( system.mf1RDM, G_site ) ))
    iddt_NOevecs = np.dot(G_site, system.NOevecs)
    #if abs(np.real(old_G_site_max)) > 1 :
 #   G_site *= ((old_G_site_max))
        
    

    #Calculate time-derivative of MF 1RDM
    #print("G in emb is hermitian:", np.allclose(G, G.conjugate().transpose(), rtol=0.0, atol=1e-12 ),"G in site is hermitian", np.allclose(G_site, G_site.conjugate().transpose(), rtol=0.0, atol=1e-12 ) )
    #print("two ddt mf methods are the same:", np.allclose(ddt_mf1RDM, ddt_mf1RDM_test, rtol=0.0, atol=1e-12 ) )
    #print(np.allclose(G, system.h_site, rtol=0.0, atol=1e-12 ) )
    #print("difference between ddt mf:")
    #print(ddt_mf1RDM - ddt_mf1RDM_test)
    #short_NOcc = np.copy(system.NOevecs[:, :round(system.Nele/2)])
    #short_ddtNOcc = np.copy(ddt_NOevecs[:, :round(system.Nele/2)])
    #chk = 2 * ( np.dot(short_ddtNOcc, short_NOcc.conj().T) + np.dot(short_NOcc, short_ddtNOcc.conj().T))

    ddt_glob1RDM = -1j * iddt_glob1RDM
    #print("#########################")
    #print("check for derivative in NOs", np.allclose(chk, ddt_mf1RDM_test, 1e-12))
    #print("#########################")
    print("#########################")
   # print("check for derivative in NOs with comm", np.allclose(chk, ddt_mf1RDM, rtol=0, atol=1e-5))
    print("#########################")
    #print("difference", ddt_mf1RDM_test - ddt_mf1RDM)
    return ddt_glob1RDM, iddt_NOevecs, iddt_mf1RDM, G_site

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
   



    #return iddt_glob1RDM
    #print("iddt Glob1RDM", 0.5*( tmp - tmp.conj().T ))
    return 0.5*( tmp - tmp.conj().T )

#####################################################################
def return_max_value(system, array):
    largest = 0
    for x in range(0, len(array)):
        for y in range(0, len(array)):
            if (abs(array[x,y] > largest)):
                largest = array[x,y]
    return largest

#####################################################################

def calc_Gmat( old_G_site_max, dG, system, iddt_glob1RDM ):
    #Subroutine to calculate matrix that governs time-dependence of natural orbitals

    #Matrix of one over the difference in global 1RDM eigenvalues
    #Set diagonal terms and terms where eigenvalues are almost equal to zero
 #   Gt = utils.rot1el( iddt_glob1RDM, system.NOevecs )
 #   Gt_site = utils.rot1el( Gt, utils.adjoint(system.NOevecs) )
    #Gmat1 = utils.rot1el( iddt_glob1RDM, system.NOevecs )
    evals    = np.copy( system.NOevals )
    eval_dif = np.zeros( [ system.Nsites, system.Nsites ] )
    #ERRR
    G1 = np.zeros( [ system.Nsites, system.Nsites ], dtype = complex)
    G2 = np.zeros( [ system.Nsites, system.Nsites ], dtype = complex )
    G3 = np.zeros( [ system.Nsites, system.Nsites ], dtype = complex )
####### METHOD 1 the original

    for a in range(system.Nsites):
        for b in range(system.Nsites):
            if( a != b and np.abs( evals[a] - evals[b] ) > 1e-10 ):
                eval_dif[a,b] = 1.0 / ( evals[b] - evals[a] )

    G1 = utils.rot1el( iddt_glob1RDM, system.NOevecs )     
    G1 = np.multiply( eval_dif, G1 )
    G1 = np.triu(G1) + np.triu(G1,1).conjugate().transpose()
    G1_site = utils.rot1el( G1, utils.adjoint(system.NOevecs) )

######################

####### METHOD 2 No 1/dif term
    G2 = utils.rot1el( iddt_glob1RDM, system.NOevecs )
    for mu in range(system.Nsites):
        for nu in range(system.Nsites):
           # for p in range(system.Nsites):
            #    for q in range(system.Nsites):
            if(mu != nu and np.abs( evals[mu] - evals[nu] ) > dG):
                        #G2[mu, nu] += (np.conjugate(system.NOevecs[p, mu]) * iddt_glob1RDM[p,q] * system.NOevecs[q, nu])
                           # print(G2)
                     #   G2[mu, nu] /= (evals[nu]-evals[mu])
                #G2[mu, nu] += ((np.conjugate(system.NOevecs[p, mu]) * iddt_glob1RDM[p,q] * system.NOevecs[q, nu])/(evals[nu]-evals[mu]))
                G2[mu, nu] /=(evals[nu]-evals[mu])
            else:
                G2[mu, nu] = 0    
    G2 = np.triu(G2) + np.triu(G2,1).conjugate().transpose() 
    G2_site = utils.rot1el( G2, utils.adjoint(system.NOevecs) )
    #G2_site = np.triu(G2) + np.triu(G2,1).conjugate().transpose()

    #if abs(np.real(old_G_site_max)) > 1 :
   # G2_site *= 1/(np.real(old_G_site_max) * 10000)
    # G2_site /= old_G_site_max
   # G2_site_max = return_max_value(system,G2_site)    
   # G2_site /= G2_site_max 
########################

######## Method 3 Full site basis

   # for r in range(system.Nsites):
    #    for s in range(system.Nsites):
     #       for mu in range(system.Nsites):
      #          for nu in range(system.Nsites):
       #             for p in range(system.Nsites):
        #                for q in range(system.Nsites):
         #                   if(mu != nu and np.abs( evals[mu] - evals[nu] ) > 1e-10):
          #                      G3[r,s] += (system.NOevecs[r, mu]*np.conjugate(system.NOevecs[p, mu]) * iddt_glob1RDM[p,q] * system.NOevecs[q, nu] * np.conjugate(system.NOevecs[s, nu]))/(evals[nu]-evals[mu])
  #  G3_site = np.triu(G3) + np.triu(G3,1).conjugate().transpose()
##########################
   # print("Metod 1 and 2", np.allclose(G1_site, G2_site, rtol=0, atol=1e-12))
   # print("1 and 2 differnce", G1_site-G2_site)
   # print("Method 2 and 3", np.allclose(G2_site, G3_site, rtol=0, atol=1e-12))
   # print("2 and 3 difference", G2_site-G3_site)
   # print("Method 1 and 3", np.allclose(G1_site, G3_site, rtol=0, atol=1e-12))
   # print("1 and 3 difference", G1_site-G3_site)



#for a in range(system.Nsites):
 #       for b in range(system.Nsites):
           # if( a != b and np.abs( evals[a] - evals[b] ) > dG):
   #         eval_dif_sum += ( evals[b] - evals[a] )                
   # Gt_site /= eval_dif_sum  
   # Gmat_test_herm = np.triu(Gt_site) + np.triu(Gt_site,1).conjugate().transpose()  
    
#  print("1/eval_dif", eval_dif)
    #Rotate iddt_glob1RDM
    #Gmat = np.zeros( [ system.Nsites, system.Nsites ] )
    #Gmat = utils.rot1el( iddt_glob1RDM, system.NOevecs )
    #for a in range(system.Nsites):
     #   for b in range(system.Nsites):
    #        if( a != b and np.abs( evals[a] - evals[b] ) > dG):
     #           eval_dif[a,b] = 1/( evals[b] - evals[a] )
#    print("Gmat", Gmat)
    #mrar
    #print( np.allclose(Gmat, -Gmat.conjugate().transpose(), rtol=0.0, atol=1e-12 ) )

    #Multiply difference in eigenvalues and rotated time-derivative matrix
   # Gmat1 = np.multiply( eval_dif, Gmat )
 #   print("Gmat times evals", Gmat1)
    #Force Hermiticity (this is all inefficient as of now)
   # Gmat2 = np.triu(Gmat1) + np.triu(Gmat1,1).conjugate().transpose()
   # Gmat_site = utils.rot1el( Gmat2, utils.adjoint(system.NOevecs) )
    
   # print("GMAT IN SITE BASIS", np.allclose(Gmat_test_herm, Gmat_site, rtol=0.0, atol=1e-10))
   # print("differnce", Gmat_test_herm -Gmat_site)
   # G_test = utils.rot1el( iddt_glob1RDM, system.NOevecs )
   # for a in range(system.Nsites):
    #    for b in range(system.Nsites):
     #       if( a != b and np.abs( evals[a] - evals[b] ) > dG):
      #          G_test[a,b] /= ( evals[b] - evals[a] )

   # G_2 = np.triu(G_test) + np.triu(G_test,1).conjugate().transpose()
   # G_site = utils.rot1el( G_2, utils.adjoint(system.NOevecs) )
   # print("GMAT IN SITE BASIS", np.allclose(G_site, Gmat_site, rtol=0.0, atol=1e-10))
   # print("difference", G_site-Gmat_site) 
    
    #quit()
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

    return G2_site
#    return Gmat_site
#####################################################################




