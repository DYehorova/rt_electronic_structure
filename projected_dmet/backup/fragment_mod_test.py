#Define a class for a fragment, including all quantities specific to a given fragment

import numpy as np
import josh_script.utils as utils

import static_pdmet.fci_mod as fci_mod
import josh_script.applyham_pyscf as applyham_pyscf
import time

######## FRAGMENT CLASS #######

class fragment():

    #####################################################################

    def __init__( self, impindx, Nsites, Nele ):
        self.impindx = impindx #array defining index of impurity orbitals in site basis
        self.Nimp    = impindx.shape[0] #number of impurity orbitals in fragment
        self.Nsites  = Nsites #total number of sites (or basis functions) in total system
        self.Nele    = Nele #total number of electrons in total system

        self.Ncore = int(Nele/2) - self.Nimp #Number of core orbitals in fragment
        self.Nvirt = Nsites - 2*self.Nimp - self.Ncore #Number of virtual orbitals in fragment

        #range of orbitals in embedding basis, embedding basis always indexed as impurity, virtual, bath, core
        self.imprange  = np.arange(0, self.Nimp)
        self.virtrange = np.arange(self.Nimp, self.Nimp+self.Nvirt)
        self.bathrange = np.arange(self.Nimp+self.Nvirt, 2*self.Nimp+self.Nvirt)
        self.corerange = np.arange(2*self.Nimp+self.Nvirt, self.Nsites)

    #####################################################################

    def get_rotmat( self, mf1RDM ):
        #Subroutine to generate rotation matrix from site to embedding basis
        #PING currently impurities have to be listed in ascending order (though dont have to be sequential)

        #remove rows/columns corresponding to impurity sites from mf 1RDM
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 0 )
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 1 )

        #diagonalize environment part of 1RDM to obtain embedding (virtual, bath, core) orbitals
        evals, evecs = np.linalg.eigh( mf1RDM )

        #form rotation matrix consisting of unit vectors for impurity and the evecs for embedding
        #rotation matrix is ordered as impurity, virtual, bath, core
        self.rotmat = np.zeros( [ self.Nsites, self.Nimp ] )
        for imp in range(self.Nimp):
            indx                     = self.impindx[imp]
            self.rotmat[ indx, imp ] = 1.0
            evecs                    = np.insert( evecs, indx, 0.0, axis=0 )

        self.rotmat = np.concatenate( (self.rotmat,evecs), axis=1 )
        #print(self.rotmat, "self.rotmat")
    #####################################################################

    def get_Hemb( self, h_site, V_site, hamtype=0, hubsite_indx=None ):
        #Subroutine to the get the 1 and 2 e- terms of the Hamiltonian in the embedding basis
        #Transformation accounts for interaction with the core
        #Also calculates 1 e- term with only 1/2 interaction with the core - this is used in calculation of DMET energy

        #remove the virtual states from the rotation matrix
        #the rotation matrix is of form ( site basis fcns ) x ( impurities, virtual, bath, core )
        #print("V_site", V_site)
        rotmat_small = np.delete( self.rotmat, np.s_[self.Nimp:self.Nimp+self.Nvirt], 1 )
        #print("rotmat_small", rotmat_small)
        #rotate the 1 e- terms, h_emb currently ( impurities, bath, core ) x ( impurities, bath, core )
        h_emb = utils.rot1el( h_site, rotmat_small )
        self.h_site = h_site

        #define 1 e- term of size ( impurities, bath ) x ( impurities, bath ) that will only have 1/2 interaction with the core
        self.h_emb_halfcore = np.copy( h_emb[ :2*self.Nimp, :2*self.Nimp ] )

        #rotate the 2 e- terms
        if( hamtype == 0 ):
            #General hamiltonian, V_emb currently ( impurities, bath, core ) ^ 4
            V_emb = utils.rot2el_chem( V_site, rotmat_small )
        elif( hamtype == 1 ):
            #Hubbard hamiltonian
            rotmat_vsmall = np.copy(rotmat_small[ hubsite_indx, :2*self.Nimp ]) #remove core states from rotation matrix
            #print("rotmat_vsmall", rotmat_vsmall)
            self.V_emb = V_site*np.einsum( 'ap,cp,pb,pd->abcd', utils.adjoint( rotmat_vsmall ), utils.adjoint( rotmat_vsmall ), rotmat_vsmall, rotmat_vsmall )
        #augment the impurity/bath 1e- terms from contribution of coulomb and exchange terms btwn impurity/bath and core
        #and augment the 1 e- term with only half the contribution from the core to be used in DMET energy calculation
        if( hamtype == 0 ):
            #General hamiltonian
            for core in range( 2*self.Nimp, 2*self.Nimp+self.Ncore ):
                h_emb[ :2*self.Nimp, :2*self.Nimp ] = h_emb[ :2*self.Nimp, :2*self.Nimp ] + 2*V_emb[ :2*self.Nimp, :2*self.Nimp, core, core ] - V_emb[ :2*self.Nimp, core, core, :2*self.Nimp ]
                self.h_emb_halfcore += V_emb[ :2*self.Nimp, :2*self.Nimp, core, core ] - 0.5*V_emb[ :2*self.Nimp, core, core, :2*self.Nimp ]
        elif( hamtype == 1):
            #Hubbard hamiltonian
            core_int = V_site * np.einsum( 'ap,pb,p->ab', utils.adjoint( rotmat_vsmall ), rotmat_vsmall, np.einsum( 'pe,ep->p',rotmat_small[hubsite_indx,2*self.Nimp:], utils.adjoint( rotmat_small[hubsite_indx,2*self.Nimp:] ) ) )
            h_emb[ :2*self.Nimp, :2*self.Nimp ] += core_int
            self.h_emb_halfcore += 0.5*core_int


        #calculate the energy associated with core-core interactions, setting it numerically to a real number since it always will be
        Ecore = 0
        for core1 in range( 2*self.Nimp, 2*self.Nimp+self.Ncore ):

            Ecore += 2*h_emb[ core1, core1 ]

            if( hamtype == 0 ):
                #General hamiltonian
                for core2 in range( 2*self.Nimp, 2*self.Nimp+self.Ncore ):
                    Ecore += 2*V_emb[ core1, core1, core2, core2 ] - V_emb[ core1, core2, core2, core1 ]
        if( hamtype == 1):
            #Hubbard hamiltonian
            vec = np.einsum( 'pe,ep->p',rotmat_small[hubsite_indx,2*self.Nimp:],utils.adjoint( rotmat_small[hubsite_indx,2*self.Nimp:] ) )
            Ecore += V_site * np.einsum( 'p,p', vec, vec )

        self.Ecore = Ecore.real

        #shrink h_emb and V_emb arrays to only include the impurity and bath
        self.h_emb = h_emb[ :2*self.Nimp, :2*self.Nimp ]
        if( hamtype == 0 ):
            #General hamiltonian
            self.V_emb = V_emb[ :2*self.Nimp, :2*self.Nimp, :2*self.Nimp, :2*self.Nimp ]
    #####################################################################

    def add_mu_Hemb( self, mu ):

        #Subroutine to add a chemical potential, mu, to only the impurity sites of embedding Hamiltonian
        for i in range( self.Nimp ):
            self.h_emb[i,i] += mu

    #####################################################################

    def solve_GS( self ):
        #Use the embedding hamiltonian to solve for the FCI ground-state

        self.CIcoeffs = fci_mod.FCI_GS( self.h_emb, self.V_emb, self.Ecore, 2*self.Nimp, (self.Nimp,self.Nimp) )

    #####################################################################

    def get_corr1RDM( self ):
        #Subroutine to get the FCI 1RDM

        self.corr1RDM = fci_mod.get_corr1RDM( self.CIcoeffs, 2*self.Nimp, (self.Nimp,self.Nimp) )

    #####################################################################

    def get_corr12RDM( self ):
        #Subroutine to get the FCI 1RDM and 2RDM

        self.corr1RDM, self.corr2RDM = fci_mod.get_corr12RDM( self.CIcoeffs, 2*self.Nimp, (self.Nimp,self.Nimp) )
        #print("self.corr1RDM", self.corr1RDM)
        self.full_corr1RDM = np.zeros( [ self.Nsites, self.Nsites ] )
        self.full_corr1RDM = self.full_corr1RDM.astype(complex)
        for c in self.corerange:
            self.full_corr1RDM[c][c] = 2
        corr1RDM_virt = np.insert(self.corr1RDM, self.Nimp, np.zeros((self.Nvirt, self.corr1RDM.shape[0])), 0)
        corr1RDM_virt = np.insert(corr1RDM_virt, self.Nimp, np.zeros((self.Nvirt, corr1RDM_virt.shape[0])), 1)

        self.full_corr1RDM[0:0+corr1RDM_virt.shape[0], 0:0+corr1RDM_virt.shape[1]] += corr1RDM_virt

        #print("self.full_corr1RDM", self.full_corr1RDM)

    #####################################################################

    def eigvec_MF_check( self, mf1RDM ):
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 0 )
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 1 )
        rotmat = np.delete(self.rotmat, self.impindx, axis = 0)
        diag = utils.rot1el(mf1RDM, rotmat)
        #print("diagonalized MF:")
        #print(diag)

    #####################################################################
    def static_corr_calc( self, mf1RDM, mu, h_site, V_site, hamtype=0, hubsite_indx=None ):
        #Subroutine to perform all steps of the static correlated calculation
        #ERRR
        self.get_rotmat( mf1RDM ) #1) get rotation matrix to embedding basis
        self.get_Hemb( h_site, V_site, hamtype, hubsite_indx ) #2) use rotation matrix to compute embedding hamiltonian
        self.add_mu_Hemb( mu ) #3) add chemical potential to only impurity sites of embedding hamiltonian
        self.solve_GS() #4) perform corrleated calculation using embedding hamiltonian
        self.get_corr1RDM() #5) calculate correlated 1RDM

    #####################################################################

    def get_frag_E( self ):
        #Subroutine to calculate contribution to DMET energy from fragment
        #Need to calculate embedding hamiltonian and 1/2 rdms prior to calling this routine
        #Using democratic partitioning using Eq. 28 from  Wouters JCTC 2016
        #This equation uses 1 e- part that only includes half the interaction with the core
        #Notation for 1RDM is rho_pq = < c_q^dag c_p >
        #Notation for 2RDM is gamma_pqrs = < c_p^dag c_r^dag c_s c_q >
        #Notation for 1 body terms h1[p,q] = <p|h|q>
        #Notation for 2 body terms V[p,q,r,s] = (pq|rs)

        #Calculate fragment energy using democratic partitioning
        self.Efrag = 0.0
        for orb1 in range(self.Nimp):
            for orb2 in range(2*self.Nimp):
                self.Efrag += self.h_emb_halfcore[ orb1, orb2 ] * self.corr1RDM[ orb2, orb1 ]
                for orb3 in range(2*self.Nimp):
                    for orb4 in range(2*self.Nimp):
                        self.Efrag += 0.5 * self.V_emb[ orb1, orb2, orb3, orb4 ] * self.corr2RDM[ orb1, orb2, orb3, orb4 ]

    #####################################################################

    def get_iddt_corr1RDM( self, h_site, V_site, hamtype=0, hubsite_indx=None ):

        #Calculate the Hamiltonian commutator portion of the time-dependence of correlated 1RDM for each fragment
        #ie i\tilde{ \dot{ correlated 1RDM } } using notation from notes
        #indexing in the embedding basis goes as ( impurities, virtual, bath, core )

        #NOTE: Should be able to make this routine more efficient and it probably double calculates stuff from emb hamiltonian routine

        #rotate the 2 e- terms into embedding basis - use notation MO (for molecular orbital) to distinguish
        #from normal embedding Hamiltonian above which focuses just on impurity/bath region
        if( hamtype == 0 ):
            #General hamiltonian
            V_MO = utils.rot2el_chem( V_site, self.rotmat )
        elif( hamtype == 1 ):
            #Hubbard hamiltonian
            rotmat_Hub = self.rotmat[ hubsite_indx, : ]

        #Form inactive Fock matrix
        if( hamtype == 0 ):
            #General hamiltonian
            IFmat =  utils.rot1el( h_site, self.rotmat )
            IFmat += 2.0*np.einsum( 'abcc->ab', V_MO[:,:,self.corerange[:,None],self.corerange] )
            IFmat -= np.einsum( 'accb->ab', V_MO[:,self.corerange[:,None],self.corerange,:] )
        elif( hamtype == 1 ):
            #Hubbard hamiltonian
            IFmat  = utils.rot1el( h_site, self.rotmat )
            tmp    = np.einsum( 'pc,cp->p', rotmat_Hub[:,self.corerange], utils.adjoint(rotmat_Hub[:,self.corerange]) )
            IFmat += V_site*np.einsum( 'ap,pb,p->ab', utils.adjoint(rotmat_Hub), rotmat_Hub, tmp )

        #Form active Fock matrix
        actrange = np.concatenate( (self.imprange,self.bathrange) )
        if( hamtype == 0 ):
            #General hamiltonian
            tmp = V_MO[:,:,actrange[:,None],actrange] - 0.5*np.einsum( 'acdb->abdc', V_MO[:,actrange[:,None],actrange,:] )
            AFmat = np.einsum( 'cd,abdc->ab', self.corr1RDM, tmp )
        elif( hamtype == 1 ):
            #Hubbard hamiltonian
            tmp = np.einsum( 'pc,cd,dp->p', rotmat_Hub[:,actrange], self.corr1RDM, utils.adjoint(rotmat_Hub[:,actrange]) )
            AFmat = 0.5 * V_site * np.einsum( 'ap,pb,p->ab', utils.adjoint(rotmat_Hub), rotmat_Hub, tmp )

        #Form generalized Fock matrix from inactive and active ones
        if( hamtype == 0 ):
            #General hamiltonian
            genFmat = np.zeros( [self.Nsites, self.Nsites], dtype=complex )
            genFmat[ self.corerange, : ] = np.transpose( 2*( IFmat[:,self.corerange] + AFmat[:,self.corerange] ) )
            genFmat[ actrange, : ] =  np.transpose( np.dot( IFmat[:,actrange], self.corr1RDM ) )
            genFmat[ actrange, : ] += np.einsum( 'acde,bcde->ba', V_MO[ :, actrange[:,None,None], actrange[:,None], actrange ], self.corr2RDM )
        elif( hamtype == 1 ):
            #Hubbard hamiltonian
            genFmat = np.zeros( [self.Nsites, self.Nsites], dtype=complex )
            genFmat[ self.corerange, : ] = np.transpose( 2*( IFmat[:,self.corerange] + AFmat[:,self.corerange] ) )
            genFmat[ actrange, : ] =  np.transpose( np.dot( IFmat[:,actrange], self.corr1RDM ) )
            tmp = np.einsum( 'dp,pc,pe,bcde->pb', utils.adjoint( rotmat_Hub[:,actrange] ), rotmat_Hub[:,actrange], rotmat_Hub[:,actrange], self.corr2RDM )
            genFmat[ actrange, : ] += V_site * np.transpose( np.dot( utils.adjoint(rotmat_Hub), tmp ) )

        #Calculate i times H commutator portion of time-dependence of corr1RDM
        self.iddt_corr1RDM = np.transpose(genFmat) - np.conjugate(genFmat)

    #####################################################################

    def get_Xmat( self, mf1RDM, ddt_mf1RDM ):
        #Subroutine to calculate the X-matrix to propagate embedding orbitals

        #Initialize X-matrix
        self.Xmat = np.zeros( [self.Nsites, self.Nsites], dtype=complex )
        self.Xmat_test = np.zeros( [self.Nsites, self.Nsites], dtype=complex )
        envindx = np.setdiff1d( np.arange(self.Nsites), self.impindx)

        self.Xmat_test2 = np.zeros( [self.Nsites, self.Nsites], dtype=complex )
        self.Xmat_test2[self.Nimp:, self.Nimp:] = utils.rot1el( 1j*ddt_mf1RDM[ envindx[:,None], envindx ], self.rotmat[ envindx, self.Nimp: ] )
        #Index of orbitals in the site-basis corresponding to the environment
        #envindx = np.setdiff1d( np.arange(self.Nsites), self.impindx)

        #Eigenvalues of environment part of mf1RDM
        env1RDM_evals = np.diag( np.real( utils.rot1el( mf1RDM[ envindx[:,None], envindx ], self.rotmat[ envindx, self.Nimp: ] ) ) )
        evals = np.zeros([self.Nsites, self.Nsites], dtype=complex )

       # print("env evals size", env1RDM_evals.shape)
       # print(env1RDM_evals)
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 0 )
        mf1RDM = np.delete( mf1RDM, self.impindx, axis = 1 )

        #diagonalize environment part of 1RDM to obtain embedding (virtual, bath, core) orbitals
        evals, evecs = np.linalg.eigh( mf1RDM )
      #  print("evals", evals.shape)
       # print(evals)



        self.mfevals = np.copy(env1RDM_evals)#mrar

        #Calculate X-matrix setting diagonal, core-core, and virtual virtual terms to zero

        #Matrix of one over the difference in embedding orbital eigenvalues
        #Set redundant terms to zero, ie diagonal, core-core, and virtual-virtual
        eval_dif = np.zeros( [ self.Nsites-self.Nimp, self.Nsites-self.Nimp ] )
        #core-bath and core-virt
        for b in self.corerange:
            for a in np.concatenate( (self.bathrange, self.virtrange) ):
                if( a != b and np.abs( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] ) > 1e-7 ):
                    eval_dif[ b-self.Nimp, a-self.Nimp ] = 1.0 / ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )
                    self.Xmat_test2[b, a] /= ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )

        #bath-virt
        #print("after core range", eval_dif)
        for b in self.bathrange:
            for a in self.virtrange:
                if( a != b and np.abs( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] ) > 1e-7 ):
                    eval_dif[ b-self.Nimp, a-self.Nimp ] = 1.0 / ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )
                    self.Xmat_test2[b, a] /= ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )
        # print("after bath range", eval_dif)
        #anti-symmetrize prior to calculating bath-bath portion


        eval_dif =  (eval_dif - eval_dif.T)
        #other triangular mat fro xtest2
        for a in self.bathrange:
            for b in self.virtrange:
                if( a != b and np.abs( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] ) > 1e-7 ):
                    self.Xmat_test2[b, a] /= ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )

        for a in self.corerange:
            for b in np.concatenate( (self.bathrange, self.virtrange) ):
                if( a != b and np.abs( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] ) > 1e-7 ):
                    eval_dif[ b-self.Nimp, a-self.Nimp ] = 1.0 / ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )
                    self.Xmat_test2[b, a] /= ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )


        #bath-bath
        for b in self.bathrange:
            for a in self.bathrange:
                if( a != b and np.abs(env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] ) > 1e-7 ):
                    eval_dif[ b-self.Nimp, a-self.Nimp ] = 1.0 / ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )
                    self.Xmat_test2[b, a] /= ( env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] )
        #Rotate time-derivative of mean-field 1RDM
        self.Xmat[self.Nimp:, self.Nimp:] = utils.rot1el( 1j*ddt_mf1RDM[ envindx[:,None], envindx ], self.rotmat[ envindx, self.Nimp: ] )

        #Multiply difference in eigenalues and rotated time-derivative matrix
        self.Xmat[self.Nimp:, self.Nimp:] = np.multiply( eval_dif, self.Xmat[self.Nimp:, self.Nimp:] )
        self.Xmat_test2 = np.triu(self.Xmat_test2,1) + np.triu(self.Xmat_test2,1).conjugate().transpose()
        self.Xmat = np.triu(self.Xmat,1) + np.triu(self.Xmat,1).conjugate().transpose()
        ############### method 2 ##############
        #need fragment dependent range for rotmat to get environment oirbitals
        self.Xmat_test[self.Nimp:, self.Nimp:] = utils.rot1el( 1j*ddt_mf1RDM[ envindx[:,None], envindx ], self.rotmat[ envindx, self.Nimp: ] )
        for a in range(self.Nimp, self.Nsites):
            for b in range(self.Nimp, self.Nsites):
               # for p in range(self.Nimp, self.Nsites):
                #    for q in range(self.Nimp, self.Nsites):
                if( a != b and np.abs(env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp] ) > 1e-7 ):
                            #self.Xmat_test[b-self.Nimp, a-self.Nimp] += (np.conjugate(self.rotmat[p,b]) * 1j*ddt_mf1RDM[p,q] * self.rotmat[q, a])/(env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp])
                    self.Xmat_test[b, a] /= (env1RDM_evals[a-self.Nimp] - env1RDM_evals[b-self.Nimp])
                else:
                    self.Xmat_test[b, a] = 0
       # coreindx = self.Nvirt+self.Nimp
        #print("shape", self.Xmat_test[ coreindx:, coreindx:].shape)
        self.Xmat_test = np.triu(self.Xmat_test,1) + np.triu(self.Xmat_test,1).conjugate().transpose()

     #   if abs(np.real(self.old_Xmat_max)) >1e-7:
      #      self.Xmat_test /= (np.real(self.old_Xmat_max) * 1000)
    #  self.Xmat_test[:self.Nvirt, :self.Nvirt] -= self.Xmat_test[:self.Nvirt, :self.Nvirt]
      #  self.Xmat_test[ coreindx:, coreindx:] -= self.Xmat_test[ coreindx:, coreindx:]
        #self.Xmat = np.copy(self.Xmat_test)
       # Imps = np.zeros((self.Nimp, self.Nsites-self.Nimp), dtype=complex)
       # self.Xmat_test = np.concatenate((Imps,self.Xmat_test), axis = 0)
       # Imps = np.zeros((self.Nsites, self.Nimp), dtype=complex)
       # self.Xmat_test = np.concatenate((Imps,self.Xmat_test), axis = 1)
        self.Xmat = np.copy(self.Xmat_test)

        #print(self.Xmat_test.shape)

        #print("test Xmat")
        #print(self.Xmat_test)
        #print("Xmat")
        #print(self.Xmat)
        #print(self.Xmat_test)
        #print(np.allclose(self.Xmat_test2, self.Xmat, rtol=0, atol=1e-8))
        #print("differnecei X2 and X ", self.Xmat-self.Xmat_test2)

       #quit()

        #print(np.allclose(self.Xmat, self.Xmat_test, rtol=0.0, atol=1e-8))
        #print("differnece Xtest and  X", self.Xmat-self.Xmat_test)
        #self.Xmat = np.copy(self.Xmat_test)



        #quit()
        #print("self.Xmat", self.Xmat)
    #####################################################################
