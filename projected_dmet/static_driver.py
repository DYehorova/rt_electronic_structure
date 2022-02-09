#Routines to run a static projected-DMET calculation

import numpy as np
from scipy.optimize import brentq
import real_time_elec_structureGN.projected_dmet.system_mod as system_mod
import real_time_elec_structureGN.projected_dmet.hartreefock as hf
import real_time_elec_structureGN.scripts.utils as utils

import pyscf.fci

############ CLASS TO RUN STATIC DMET CALCULATION #########

class static_driver():

    #####################################################################

    def __init__( self, Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype=0, mubool=True, nproc=1, hubsite_indx=None, periodic=False, Maxitr=10000, tol=1e-12 ):

        #Nsites  - total number of sites (or basis functions) in total system
        #Nele    - total number of electrons
        #Nfrag   - total number of fragments for DMET calculation
        #impindx - a list of numpy arrays containing the impurity indices for each fragment
        #h_site  - 1 e- hamiltonian in site-basis for total system
        #V_site  - 2 e- hamiltonian in site-basis for total system
        #hamtype - integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity
        #mubool  - boolean switching between using a global chemical potential to optimize DMET # of electrons or not
        #nproc   - number of processors for calculation - careful, there is no check that this matches the pbs script
        #periodic - boolean which states whether system is periodic or not and thus only need to solve for one impurity as they're all the same
        #Maxitr  - max number of DMET iterations
        #tol     - tolerance for difference in 1RDM during DMET cycle

        print()
        print('********************************************')
        print('     INITIALIZING STATIC DMET CALCULATION       ')
        print('********************************************')
        print()

        self.mubool = mubool
        self.Maxitr = Maxitr
        self.tol    = tol
        self.nproc  = nproc

        #Check for input errors
        self.check_for_error( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, periodic )

        #Begin by calculating initial mean-field Hamiltonian
        print('Calculating initial mean-field 1RDM for total system')
        if( hamtype == 0 ):
            mf1RDM = hf.rhf_calc_hubbard( Nele, h_site ) #PING need to change this to general HF call
        elif( hamtype == 1 ):
            mf1RDM = hf.rhf_calc_hubbard( Nele, h_site )

        #Initialize the total system including the mf 1RDM and fragment information
        print('Initialize fragment information')
        self.tot_system = system_mod.system( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mf1RDM, hubsite_indx, periodic )

    #####################################################################

    def kernel( self ):

        print()
        print('********************************************')
        print('     BEGIN STATIC DMET CALCULATION       ')
        print('********************************************')
        print()

        ####DMET outer-loop####
        convg = False
        for itr in range(self.Maxitr):

            #Perform correlated embedding calculations
            if( self.mubool ):
                #PING need to change this to the algorithm used by boxiao/zhihao or to Newton-Rhapson
                #Self-consistently update global chemical potential for embedding calculations
                #to obtain correct DMET number of electrons
                lint = -1.0
                rint = 1.0
                brentq( self.Nele_cost_function, lint, rint )
            else:
                #Single impurity calculation not using a global chemical potential for embedding calculations
                self.tot_system.corr_emb_calc(self.nproc)

            #Form the global density matrix from all impurities
            self.tot_system.get_glob1RDM()
   #         self.tot_system.glob1RDM = np.load("fci1rdm.npy")
            #Form new mean-field 1RDM from the first N-occupied natural orbitals of global 1RDM
            self.tot_system.get_nat_orbs()
            self.tot_system.get_new_mf1RDM( int(self.tot_system.Nele/2) )

    #        dif = 0
     #       break
            #Check if difference between previous and current 1RDM is less than tolerance
            if( itr > 0 ):
                dif = np.linalg.norm( old_glob1RDM - self.tot_system.glob1RDM )
                if( dif < self.tol ):
                    convg = True
                    break

            #Copy over old global 1RDM
            old_glob1RDM = np.copy( self.tot_system.glob1RDM )

            if( np.mod( itr, self.Maxitr/10 ) == 0 and itr > 0 ):
                print('Finished DMET Iteration',itr)
                print('Current difference in global 1RDM =',dif)
                print()

        ####END DMET outer-loop####

        if( convg ):
            print('DMET calculation succesfully converged in',itr,'iterations')
            print('Final difference in global 1RDM =',dif)
            print()
        else:
            print('WARNING: DMET calculation finished, but did not converge in', self.Maxitr, 'iterations')
            print('Final difference in global 1RDM =',dif)
            print()

        ##### Calculate final DMET energy ####
        self.tot_system.get_DMET_E( self.nproc )
        print('Final DMET energy =',self.tot_system.DMET_E)
        #np.save("GlobalHubMF0to0", self.tot_system.glob1RDM)
        #print()
        #ERR
        #self.tot_system.mf1RDM = np.load("RMGNmfRDMU1.npy")
        #CI_list = np.load("RMGNciU1.npy")
        #rotmat_list = np.load("RMGNrotmatU1.npy")
        #for cnt, frag in enumerate(self.tot_system.frag_list):
        #    frag.CIcoeffs = CI_list[cnt]
         #   frag.rotmat   = rotmat_list[cnt]

#####################################################################

    def Nele_cost_function( self, mu ):

        self.tot_system.mu = mu
        self.tot_system.corr_emb_calc(self.nproc)
        self.tot_system.get_DMET_Nele()

        return self.tot_system.Nele - self.tot_system.DMET_Nele

#####################################################################

    def check_for_error( self, Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, periodic ):
        #Subroutine that takes all inputs and checks for any input errors

        #Check number of indices
        if( sum([len(arr) for arr in impindx]) > Nsites ):
            print('ERROR: List of impurity indices (impindx) has more indices than sites')
            print()
            exit()
        elif( sum([len(arr) for arr in impindx]) < Nsites ):
            print('ERROR: List of impurity indices (impindx) has fewer indices than sites')
            print()
            exit()

        #Check number of fragments
        if( len(impindx) != Nfrag ):
            print('ERROR: Number of fragments specified by Nfrag does not match')
            print('       number of fragments in list of impurity indices (impindx)')
            print()
            exit()

        #Check that impurities defined using unique indices
        #PING: need to fix this b/c an error w/the np.unique command
        #chk = impindx[0]
        #for count, arr in enumerate(impindx):
        #    if( count != 0 ):
        #        chk = np.concatenate((chk,arr))

        #unqchk, cnt = np.unique(chk, return_counts=True)

        #if( len(chk) != len(unqchk) ):
        #    print('ERROR: The following indices were repeated in the definition of the impurities:')
        #    print(unqchk[cnt>1])
        #    print()
        #    exit()

        #Check that for each fragment, impurities are assigned in ascending order (does not have to be sequential)
        for count, arr in enumerate(impindx):
            if( not np.all( np.diff(arr) > 0 ) ):
                print('ERROR: Fragment number',count,'does not have impurity indices in ascending order')
                print()
                exit()

        #Check that all fragments same size if system is periodic
        if( periodic ):
            Nimp = len(impindx[0])
            for arr in impindx:
                if( len(arr) != Nimp ):
                    print('ERROR: System is periodic, but all fragments are not of the same size')
                    print()
                    exit()

    #####################################################################

