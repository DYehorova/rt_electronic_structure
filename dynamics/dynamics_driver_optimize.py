# Routines to run a real-time projected-DMET calculation
import numpy as np
import sys
import feb_8_update.dynamics.system_mod_paral as system_mod
import feb_8_update.dynamics.mf1rdm_timedep_mod_G2 as mf1rdm_timedep_mod
import feb_8_update.scripts.applyham_pyscf as applyham_pyscf
import feb_8_update.scripts.utils as utils
import multiprocessing as multproc
import pickle
import scipy
import math
import time

# ########### CLASS TO RUN REAL-TIME DMET CALCULATION #########


class dynamics_driver():

    #####################################################################

    def __init__(self, h_site, V_site, hamtype, tot_system, delt, dG, Nstep, nproc, Nprint=100, integ='rk1', hubsite_indx=None, init_time=0.0):

        #h_site     - 1 e- hamiltonian in site-basis for total system to run dynamics
        #V_site     - 2 e- hamiltonian in site-basis for total system to run dynamics
        #hamtype    - integer defining if using a special Hamiltonian like Hubbard or Anderson Impurity
        #tot_system - a previously defined DMET total system including all fragment information
        #delt       - time step
        #Nstep      - total number of time-steps
        #Nprint     - number of time-steps between printing
        #init_time  - the starting time for the calculation
        #integ      - the type of integrator used
        #nproc      - number of processors for calculation - careful, there is no check that this matches the pbs script

        self.tot_system = tot_system
        self.delt       = delt
        self.Nstep      = Nstep
        self.Nprint     = Nprint
        self.init_time  = init_time
        self.integ      = integ
        self.nproc      = nproc
        self.dG         = dG
        print()
        print('********************************************')
        print('     SET-UP REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()
        #Input error checks
        #print("MF RDM", self.tot_system.mf1RDM)
        #old_MF = np.copy(self.tot_system.mf1RDM)

        #self.tot_system.get_frag_corr1RDM()
        #print("dynamic global", self.tot_system.glob1RDM)
        #print("dynamic mf1rdm", self.tot_system.mf1RDM)
        #quit()
        #old_global = np.copy(self.tot_system.glob1RDM)
        #self.tot_system.get_frag_corr1RDM()
        #self.tot_system.get_glob1RDM()
        #print("global rdm")
        #print(np.allclose(old_global, self.tot_system.glob1RDM, 10e-10))
        #print(old_global - self.tot_system.glob1RDM)
        #np.save("GlobU1to0A", self.tot_system.glob1RDM)
        #self.tot_system.get_frag_corr12RDM()
        #self.tot_system.get_glob1RDM()
        #self.tot_system.get_nat_orbs()
        #self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        #print("MF RDM", self.tot_system.mf1RDM)
        #print("import check", np.allclose(old_MF, self.tot_system.mf1RDM, 10e-10))
        #print(old_MF - self.tot_system.mf1RDM)
        #old_MF = np.copy(self.tot_system.mf1RDM)
        #self.tot_system.get_frag_corr1RDM()
        #self.tot_system.get_glob1RDM()
        #self.tot_system.get_nat_orbs()
        #self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        #print("mean field rdm")
        #print(np.allclose(old_MF, self.tot_system.mf1RDM, 10e-10))
        #print(old_MF - self.tot_system.mf1RDM)


        #Convert rotation matrices, CI coefficients, and MF 1RDM to complex arrays if they're not already
        for frag in self.tot_system.frag_list:
            if( not np.iscomplexobj( frag.rotmat ) ):
                frag.rotmat = frag.rotmat.astype(complex)
            if( not np.iscomplexobj( frag.CIcoeffs ) ):
                frag.CIcoeffs = frag.CIcoeffs.astype(complex)

        if( not np.iscomplexobj( self.tot_system.mf1RDM ) ):
            self.tot_system.mf1RDM = self.tot_system.mf1RDM.astype(complex)

        if( not np.iscomplexobj( self.tot_system.glob1RDM ) ):
            self.tot_system.glob1RDM = self.tot_system.glob1RDM.astype(complex)

        if( not np.iscomplexobj( self.tot_system.NOevecs ) ):
            self.tot_system.NOevecs = self.tot_system.NOevecs.astype(complex)
        #Set-up Hamiltonian for dynamics calculation
        self.tot_system.h_site = h_site
        self.tot_system.V_site = V_site
        self.tot_system.hamtype = hamtype
        #If running Hubbard-like model, need an array containing index of all sites that have hubbard U term
        self.tot_system.hubsite_indx = hubsite_indx
        if( self.tot_system.hamtype == 1 and self.tot_system.hubsite_indx is None ):
            print('ERROR: Did not specify an array of sites that contain Hubbard U term')
            print()
            exit()

        #Define output files
        self.file_output   = open( 'outputJ.dat', 'w' )
        self.file_corrdens = open( 'corr_densityJ.dat', 'w' )
       # self.file_GMat = open('GMat.dat', 'w')
        self.file_current = open('current.dat', 'w')
        self.max_diagonalG = 0
        self.corrdens_old = np.zeros((self.tot_system.Nsites))
        #self.corrdens_old += 1
        #start_pool = time.time()
        #self.frag_pool = multproc.Pool(nproc)
        #print("time to from pool", time.time()-start_pool)
    #####################################################################

    def kernel( self ):
        start_time = time.time()
        print()
        print('********************************************')
        print('     BEGIN REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()

        #fraddg_pool = multproc.Pool(self.nproc)
        #DYNAMICS LOOP
        print("time", self.init_time)
        current_time = self.init_time
        self.curr_time = current_time
        for step in range(self.Nstep):
            print("step", step)
            self.step = np.copy(step)
            #Print data before taking time-step, this always prints out data at initial time step
            if( np.mod( step, self.Nprint ) == 0 ) and step > 1:

                print('Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation')
                self.print_data( current_time )
                sys.stdout.flush()
            if current_time != 0 and step ==0:
                print('Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation')
                fmt_str = '%20.8e'

                self.tot_system.get_DMET_E( self.nproc )

                self.tot_system.get_DMET_Nele()

                cnt = 0
                corrdens = np.zeros(self.tot_system.Nsites)
                for frag in self.tot_system.frag_list:
                    corrdens[cnt:cnt+frag.Nimp] = np.copy( np.diag( np.real( frag.corr1RDM[:frag.Nimp] ) ) )
                    cnt += frag.Nimp
                corrdens_short =  np.copy(corrdens)
                corrdens = np.insert( corrdens, 0, current_time )
                np.savetxt( self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str )
                self.file_corrdens.flush()

                sys.stdout.flush()
            #Integrate FCI coefficients and rotation matrix for all fragments
            print("####################")
            print("STEP", step)
            print("####################")
            self.integrate(self.nproc)

            #Increase current_time
            current_time = self.init_time + (step+1)*self.delt
            self.curr_time = current_time
            sys.stdout.flush()
        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_corrdens.close()
        self.file_current.close()
#        self.frag_pool.close()
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END REAL-TIME DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
        print("--- %s seconds ---" % (time.time() - start_time))
    #####################################################################

    def integrate( self, nproc ):
        #Subroutine to integrate equations of motion
        if( self.integ == 'rk4' ):
            #Use 4th order runge-kutta to integrate EOMs

            #Copy MF 1RDM, CI coefficients, and embedding orbs at time t
            init_NOevecs = np.copy( self.tot_system.NOevecs )
            init_glob1RDM = np.copy( self.tot_system.glob1RDM )
            init_mf1RDM = np.copy(self.tot_system.mf1RDM)

            init_CIcoeffs_list = []
            init_rotmat_list   = []
            for frag in self.tot_system.frag_list:
                init_rotmat_list.append( np.copy(frag.rotmat) )
                init_CIcoeffs_list.append( np.copy(frag.CIcoeffs) )

            #Calculate appropriate changes in MF 1RDM, embedding orbitals, and CI coefficients
            #Note that l, k and m terms are for MF 1RDM, emb orbs, and CI coefficients respectively

            print("######################")
            print("GETTING 1ST SUBSTEP DT")
            print("######################")

            self.substep = 1
            l1, k1_list, m1_list, n1, p1 = self.one_rk_step(nproc)

            self.tot_system.NOevecs = init_NOevecs + 0.5*l1
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n1
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p1
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k1_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m1_list[cnt]

            print("######################")
            print("GETTING 2ST SUBSTEP DT")
            print("######################")
            self.substep = 2
            l2, k2_list, m2_list, n2, p2 = self.one_rk_step(nproc)

            self.tot_system.NOevecs = init_NOevecs + 0.5*l2
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n2
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p2

            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k2_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]

            print("######################")
            print("GETTING 3ST SUBSTEP DT")
            print("######################")

            l3, k3_list, m3_list, n3, p3 = self.one_rk_step(nproc)
            self.tot_system.NOevecs = init_NOevecs + 1.0*l3
            self.tot_system.glob1RDM = init_glob1RDM + 1.0*n3
            self.tot_system.mf1RDM = init_mf1RDM + 1.0*p3
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0*k3_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*m3_list[cnt]

            print("######################")
            print("GETTING 4ST SUBSTEP DT")
            print("######################")

            l4, k4_list, m4_list, n4, p4 = self.one_rk_step(nproc)
            self.tot_system.NOevecs = init_NOevecs + 1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
            self.tot_system.glob1RDM = init_glob1RDM  + 1.0/6.0 * ( n1 + 2.0*n2 + 2.0*n3 + n4 )
            self.tot_system.mf1RDM =  init_mf1RDM +1.0/6.0 * ( p1 + 2.0*p2 + 2.0*p3 + p4 )
            for cnt, frag in enumerate( self.tot_system.frag_list ):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0/6.0 * ( k1_list[cnt] + 2.0*k2_list[cnt] + 2.0*k3_list[cnt] + k4_list[cnt] )
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0/6.0 * ( m1_list[cnt] + 2.0*m2_list[cnt] + 2.0*m3_list[cnt] + m4_list[cnt] )

            print("#############################")
            print("FULL RK STEP DONE, QUANTITIES:")
            print("#############################")

            # check how well natural orbitals diagonalize global rdm
            eve = (utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ))
            evals = np.diag(np.real(utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs )))

            print("FINISHED STEP:", self.step)
            zero = np.zeros((self.tot_system.Nsites, self.tot_system.Nsites))
            eve_diag = np.diag( eve)
            self.off_diag_glob_diag = np.copy(eve - np.diag(eve_diag))
            self.max_diagonalG = self.return_max_value(self.off_diag_glob_diag)

            if np.allclose(self.off_diag_glob_diag, zero,rtol=0, atol= 1e-12) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-12")
            print("diagonalized Global RDM", eve)
            if np.allclose(self.off_diag_glob_diag, zero,rtol=0, atol= 1e-5) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-5")
            if np.isnan(np.sum(self.off_diag_glob_diag)) == True:
                quit()
        else:
            print('ERROR: A proper integrator was not specified')
            exit()

            quit()
    #####################################################################

    def return_max_value(self, array):
        largest = 0
        for x in range(0, len(array)):
            for y in range(0, len(array)):
                if (abs(array[x,y]) > largest):
                    largest = array[x,y]
        return largest
    #####################################################################

    def one_rk_step( self, nproc ):
        #Subroutine to calculate one change in a runge-kutta step of any order
        #Using EOM that integrates CI coefficients, rotmat, and MF 1RDM
        #Prior to calling this routine need to update MF 1RDM, rotmat and CI coefficients

        #calculate the terms needed for time-derivative of mf-1rdm
        self.tot_system.get_frag_corr12RDM()

        self.tot_system.NOevals = np.diag( np.real( utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ) ) )

        #calculate embedding hamiltonian
        make_ham = time.time()
        self.tot_system.get_frag_Hemb()
        print("making hamiltonian:", time.time()-make_ham)
        #Make sure Ecore for each fragment is 0 for dynamics
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        #Calculate change in mf1RDM
        make_derivs = time.time()
        ddt_glob1RDM, ddt_NOevec, ddt_mf1RDM, G_site = mf1rdm_timedep_mod.get_ddt_mf1rdm_serial( self.dG, self.tot_system, round(self.tot_system.Nele/2) )
        print("making derivatives:", time.time()-make_derivs)

        #Use change in mf1RDM to calculate X-matrix for each fragment
        make_xmat = time.time()
        self.tot_system.get_frag_Xmat( ddt_mf1RDM )
        print("making xmat:", time.time()-make_xmat)

        change_glob1RDM = ddt_glob1RDM * self.delt
        change_NOevecs = ddt_NOevec * self.delt
        change_mf1RDM = ddt_mf1RDM * self.delt

        #Calculate change in embedding orbitals
        change_rotmat_list = []
        for frag in self.tot_system.frag_list:
            change_rotmat_list.append( -1j * self.delt * np.dot( frag.rotmat, frag.Xmat ) )

        #Calculate change in CI coefficients in parallel

        no_paralel_start = time.time()
        change_CIcoeffs_list = []

        for ifrag, frag in enumerate(self.tot_system.frag_list):
            change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )
        print("time without paralelization:", "--- %s seconds ---" % (time.time()-no_paralel_start))
        return change_NOevecs, change_rotmat_list, change_CIcoeffs_list, change_glob1RDM, change_mf1RDM

    #####################################################################
    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ######## CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate DMET energy, which also includes calculation of 1 & 2 RDMs and embedding hamiltonian for each fragment
        energy_calc = time.time()
        self.tot_system.get_DMET_E( self.nproc )
        print("time for energy calc", time.time()-energy_calc)
        #Calculate total number of electrons
        nele_calc = time.time()
        self.tot_system.get_DMET_Nele()
        print("time for nele calc", time.time()-nele_calc)
        ######## PRINT OUT EVERYTHING #######

        #Print correlated density in the site basis
        elec_dens=time.time()
        cnt = 0
        corrdens = np.zeros(self.tot_system.Nsites)
        for frag in self.tot_system.frag_list:
            corrdens[cnt:cnt+frag.Nimp] = np.copy( np.diag( np.real( frag.corr1RDM[:frag.Nimp] ) ) )
            cnt += frag.Nimp
        corrdens_short =  np.copy(corrdens)
        corrdens = np.insert( corrdens, 0, current_time )
        np.savetxt( self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str )
        self.file_corrdens.flush()
        print("time to calc and write elec data", time.time()-elec_dens)

        #Print output data
        writing_outfile = time.time()
        Nimp      = self.tot_system.frag_list[0].Nimp
        Nsites    = self.tot_system.Nsites
        output    = np.zeros((16+5*self.tot_system.Nsites))
        output[0] = current_time
        output[1] = self.tot_system.DMET_E
        output[2] = self.tot_system.DMET_Nele
        output[3] = np.trace( self.tot_system.mf1RDM )
        output[4] = np.trace( self.tot_system.frag_list[0].corr1RDM )
        output[5] = np.einsum( 'ppqq', self.tot_system.frag_list[0].corr2RDM )
        output[6] = np.linalg.norm( self.tot_system.frag_list[0].CIcoeffs )**2
        output[7] = np.linalg.norm( self.tot_system.frag_list[0].rotmat[:,4] )**2
       # self.tot_system.get_nat_orbs()
        if(np.allclose( self.tot_system.glob1RDM, utils.adjoint(self.tot_system.glob1RDM), rtol=0.0, atol=1e-14 )):
            output[8] = 1
        else:
            output[8] = 0
        output[9:9+self.tot_system.Nsites] = np.copy(self.tot_system.NOevals)
        current=np.zeros((self.tot_system.Nsites))
        current_glob=np.zeros((self.tot_system.Nsites))
        current_num =np.zeros((self.tot_system.Nsites))

        output[10+2*self.tot_system.Nsites] = np.copy(np.real(self.max_diagonalG))
        output[11+2*self.tot_system.Nsites] = np.copy(np.imag(self.max_diagonalG))

        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        self.corrdens_old = np.copy(corrdens_short)
       #current output file
        current    = np.zeros((8+3*self.tot_system.Nsites))
        current[0] = current_time
        JL = 1j *self.tot_system.h_site[0,1] * ( self.tot_system.glob1RDM[0,1] - self.tot_system.glob1RDM[1,0] )
        JR = 1j * self.tot_system.h_site[2,0] * ( self.tot_system.glob1RDM[2,0] - self.tot_system.glob1RDM[0,2] )
        J = (1/(0.001))*((JL+JR)/2)
        J_2pi = (1/(0.001))*((JL+JR)/2)*2*math.pi
        current_rot=np.zeros((self.tot_system.Nsites), dtype=complex)
        current_glob=np.zeros((self.tot_system.Nsites), dtype=complex)
       #for q in range(1, self.tot_system.Nsites-1):
       #     rfrag = self.tot_system.frag_list[self.tot_system.site_to_frag_list[q+1]]
       #     qfrag = self.tot_system.frag_list[self.tot_system.site_to_frag_list[q]]
       #     pfrag = self.tot_system.frag_list[self.tot_system.site_to_frag_list[q-1]]

       #     rfrag.last_virt = rfrag.Nimp + rfrag.Nvirt
       #     pfrag.last_virt = pfrag.Nimp + pfrag.Nvirt
       #     qfrag.last_virt = qfrag.Nimp + qfrag.Nvirt

       #     rfrag.last_bath = 2*rfrag.Nimp + rfrag.Nvirt
       #     pfrag.last_bath = 2*pfrag.Nimp + pfrag.Nvirt
       #     qfrag.last_bath = 2*qfrag.Nimp + qfrag.Nvirt

       #     rindx = np.r_[ :rfrag.Nimp, rfrag.last_virt : rfrag.last_bath ]
       #     pindx = np.r_[ :pfrag.Nimp, pfrag.last_virt : pfrag.last_bath ]
       #     qindx = np.r_[ :qfrag.Nimp, qfrag.last_virt : qfrag.last_bath ]
       #     p = q-1
       #     r = q+1
       #     current_rot[q] = 0.5 *1j* self.tot_system.h_site[q, p]*\
       #          (np.linalg.multi_dot([qfrag.rotmat[p,qindx], qfrag.corr1RDM, qfrag.rotmat[q,qindx].conj().T]) - \
       #             (np.linalg.multi_dot([pfrag.rotmat[p,pindx], pfrag.corr1RDM, pfrag.rotmat[q,pindx].conj().T])).conj().T)

       #     current_rot[q] += 0.5 * 1j* self.tot_system.h_site[r, q]*\
       #          ((np.linalg.multi_dot([rfrag.rotmat[r,rindx], rfrag.corr1RDM, rfrag.rotmat[q,rindx].conj().T])).conj().T - \
       #             (np.linalg.multi_dot([qfrag.rotmat[r,qindx], qfrag.corr1RDM, qfrag.rotmat[q,qindx].conj().T])))
       #     current_rot[q] *=-(1/(-0.001))
       #     current_glob[q] = -(1/(-0.001)) *1j * self.tot_system.h_site[q, p]*(self.tot_system.glob1RDM[p,q] - self.tot_system.glob1RDM[q, p])
        left_frag =\
            self.tot_system.frag_list[self.tot_system.site_to_frag_list[1]]
        dot_frag =\
            self.tot_system.frag_list[self.tot_system.site_to_frag_list[0]]
        right_frag =\
            self.tot_system.frag_list[self.tot_system.site_to_frag_list[2]]

        left_frag.last_virt = left_frag.Nimp + left_frag.Nvirt
        dot_frag.last_virt = dot_frag.Nimp + dot_frag.Nvirt
        right_frag.last_virt = right_frag.Nimp + right_frag.Nvirt

        left_frag.last_bath = 2*left_frag.Nimp + left_frag.Nvirt
        dot_frag.last_bath = 2*dot_frag.Nimp + dot_frag.Nvirt
        right_frag.last_bath = 2*right_frag.Nimp + right_frag.Nvirt


        left_indx =\
            np.r_[:left_frag.Nimp, left_frag.last_virt:left_frag.last_bath]
        dot_indx =\
            np.r_[:left_frag.Nimp, left_frag.last_virt:left_frag.last_bath]
        right_indx =\
            np.r_[:left_frag.Nimp, left_frag.last_virt:left_frag.last_bath]

        current_rot = 0.5 * 1j * self.tot_system.h_site[0, 1] *\
            (np.linalg.multi_dot(
                [dot_frag.rotmat[0, dot_indx], dot_frag.corr1RDM,
                 dot_frag.rotmat[1, dot_indx].conj().T])

             - (np.linalg.multi_dot(
                 [left_frag.rotmat[1, left_indx], left_frag.corr1RDM,
                  left_frag.rotmat[0, left_indx].conj().T])).conj().T)

        current_rot += 0.5 * 1j * self.tot_system.h_site[2, 0] *\
            (np.linalg.multi_dot(
                [right_frag.rotmat[0, right_indx], right_frag.corr1RDM,
                 right_frag.rotmat[2, right_indx].conj().T])

             - (np.linalg.multi_dot(
                 [dot_frag.rotmat[2, dot_indx], dot_frag.corr1RDM,
                  dot_frag.rotmat[0, dot_indx].conj().T])).conj().T)

        current_rot *= -1/0.001

        current[1]=np.real(J)
        current[2]=np.real(J_2pi)
        current[3]=np.imag(J)
        current[4]=np.imag(J_2pi)

        current[5]=(1/(-0.001))*(corrdens_short[0]-self.corrdens_old[0])/self.delt
        current[6]=(1/(-0.001))*(corrdens_short[1]-self.corrdens_old[1])/self.delt
        current[7]=(1/(-0.001))*(corrdens_short[2]-self.corrdens_old[2])/self.delt

        current[8]=np.real(current_rot)
        print("rot real", np.real(current_rot))
        current[9]=np.imag(current_rot)
        print("rot imag", np.imag(current_rot))
        current[10]=np.real(current_rot * 2 * math.pi)

        #current[3:(3+self.tot_system.Nsites)]=np.real(current_glob)
        #current[(4+self.tot_system.Nsites):(4+2*self.tot_system.Nsites)]=np.real(current_rot)
        #current[(5+(2*self.tot_system.Nsites)):(5+(3*self.tot_system.Nsites))]= (1/(-0.005))*(corrdens_short-self.corrdens_old)/self.delt
        #current[(6+(3*self.tot_system.Nsites))] = np.imag(J)
        #current[(7+(3*self.tot_system.Nsites))] = np.imag(J_2pi)
        np.savetxt( self.file_current, current.reshape(1, current.shape[0]), fmt_str )
        self.file_current.flush()

        #Save total system to file for restart purposes using pickle
        file_system = open( 'restart_systemJ.dat', 'wb' )
        pickle.dump( self.tot_system, file_system )
        file_system.close()
        print("time to write output file", time.time()-writing_outfile)
#####################################################################

def applyham_wrapper( frag, delt ):

    #Subroutine to call pyscf to apply FCI hamiltonian onto FCI vector in dynamics
    #Includes the -1j*timestep term and the addition of bath-bath terms of X-matrix to embedding Hamiltonian
    #The wrapper is necessary to parallelize using Pool and must be separate from
    #the class because the class includes IO file types (annoying and ugly but it works)

    Xmat_sml = np.zeros( [ 2*frag.Nimp, 2*frag.Nimp ], dtype = complex )
    Xmat_sml[ frag.Nimp:, frag.Nimp: ] = np.copy(frag.Xmat[ frag.bathrange[:,None], frag.bathrange ])
    return -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex( frag.CIcoeffs, frag.h_emb-Xmat_sml, frag.V_emb, frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore )

#####################################################################

