#Routines to run a real-time projected-DMET calculation
#Need pyscf installed

import numpy as np
import real_time_elec_structureGN.projected_dmet.system_mod_paral as system_mod
import real_time_elec_structureGN.projected_dmet.mf1rdm_timedep_mod_G2 as mf1rdm_timedep_mod
import real_time_elec_structureGN.scripts.applyham_pyscf as applyham_pyscf
import real_time_elec_structureGN.scripts.utils as utils
import multiprocessing as multproc
import pickle
import scipy
import sys
import os

import time
############ CLASS TO RUN REAL-TIME DMET CALCULATION #########

class dynamics_driver():

    #####################################################################

    def __init__( self, h_site, V_site, hamtype, tot_system, delt, min_step, dG, Nstep, nproc, Nprint=100, integ='rk1', hubsite_indx=None, init_time=0.0):

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
        self.min_step   =min_step 
        print()
        print('********************************************')
        print('     SET-UP REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()
        #Input error checks
        #print("MF RDM", self.tot_system.mf1RDM)
        #old_MF = np.copy(self.tot_system.mf1RDM)

        #self.tot_system.get_frag_corr1RDM()
        #self.tot_system.get_glob1RDM()
        #np.save("GlobU1to0A", self.tot_system.glob1RDM)
        #self.tot_system.get_frag_corr12RDM()
        #self.tot_system.get_glob1RDM()
        #self.tot_system.get_nat_orbs()
        #self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        #print("MF RDM", self.tot_system.mf1RDM)
        #print("import check", np.allclose(old_MF, self.tot_system.mf1RDM, 10e-10))
        #print(old_MF - self.tot_system.mf1RDM)
        #old_MF = np.copy(self.tot_system.mf1RDM)
        #   self.tot_system.get_frag_corr1RDM()
    #    self.tot_system.get_glob1RDM()
   #     self.tot_system.get_nat_orbs()
    #    self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
     #   print("MF RDM", self.tot_system.mf1RDM)
       # print(np.allclose(old_MF, self.tot_system.mf1RDM, 10e-10))
      #  print(old_MF - self.tot_system.mf1RDM)

#        quit()

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
        self.file_GMat = open('GMat.dat', 'w')
        self.max_diagonalG = 0
        self.corrdens_old = np.zeros((self.tot_system.Nsites))
        self.corrdens_old += 1
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
        
        #frag_pool = multproc.Pool(self.nproc)
        #DYNAMICS LOOP
        print("time", self.init_time)
        current_time = self.init_time
        self.curr_time = current_time
        for step in range(self.Nstep):
            self.step = np.copy(step)
            eval_min_dif = self.min_dif(self.tot_system.NOevals)
            if self.min_step is not None and (eval_min_dif <= self.dG):
                self.minimization(self.nproc)
                print("minimization step", step, "is done")
            else:
                self.integrate(self.nproc)                 
            print("step", step)
            print("Nstep", self.Nstep)
            self.step = np.copy(step)
            #Print data before taking time-step, this always prints out data at initial time step
            if( np.mod( step, self.Nprint ) == 0 ) and step > 0:
                print('Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation')
                self.print_data( current_time )
                sys.stdout.flush()
            if current_time != 0 and step ==0:
                print('Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation')
                self.print_data( current_time )
                sys.stdout.flush()
            #Integrate FCI coefficients and rotation matrix for all fragments
            print("####################")
            print("STEP", step)
            print("####################")

            #Increase current_time
            current_time = self.init_time + (step+1)*self.delt
            self.curr_time = current_time
            

        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_corrdens.close()
       # self.frag_pool.close()
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

#            print("INIT NOevecs before the 1st substep")
 #           print(self.tot_system.NOevecs)
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
            #k1_list, m1_list,  p1 = self.one_rk_step(nproc)
            #print("INIT NOevecs after calc dt", self.tot_system.NOevecs)
   #         print("dt for NOs", l1)
  #          print("dt for rotmat", k1_list)
    #        print("dt for CIs", m1_list)

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
            #k2_list, m2_list, p2 = self.one_rk_step(nproc)
            #exit()

            self.tot_system.NOevecs = init_NOevecs + 0.5*l2
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n2
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p2

            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k2_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]
               # print("frag.rotmat", frag.rotmat)
                #print("frag.CIcoeffs", frag.CIcoeffs)


            print("######################")
            print("GETTING 3ST SUBSTEP DT")
            print("######################")

            l3, k3_list, m3_list, n3, p3 = self.one_rk_step(nproc)
            #k3_list, m3_list, p3 = self.one_rk_step(nproc)
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
            #k4_list, m4_list, p4 = self.one_rk_step(nproc)
            #Update MF 1RDM, emb orbs and CI coefficients by full time-step
            self.tot_system.NOevecs = init_NOevecs + 1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
            self.tot_system.glob1RDM = init_glob1RDM  + 1.0/6.0 * ( n1 + 2.0*n2 + 2.0*n3 + n4 )
            self.tot_system.mf1RDM =  init_mf1RDM +1.0/6.0 * ( p1 + 2.0*p2 + 2.0*p3 + p4 )
            for cnt, frag in enumerate( self.tot_system.frag_list ):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0/6.0 * ( k1_list[cnt] + 2.0*k2_list[cnt] + 2.0*k3_list[cnt] + k4_list[cnt] )
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0/6.0 * ( m1_list[cnt] + 2.0*m2_list[cnt] + 2.0*m3_list[cnt] + m4_list[cnt] )

            print("#############################")
            print("FULL RK STEP DONE, QUANTITIES:")
            print("#############################")



           # print("MF after rk4", self.tot_system.mf1RDM)
           # print("GlobalRDM", self.tot_system.glob1RDM)
           # print("NO evecs", self.tot_system.NOevecs)
            eve = (utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ))
       #     print("diagonalized global rdm", eve)
      #      print("evals")
            evals = np.diag(np.real(utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs )))
            print(evals)
            print("FINISHED STEP:", self.step)
            zero = np.zeros((self.tot_system.Nsites, self.tot_system.Nsites))
            eve_diag = np.diag( eve)
            eve_no_diag = eve - np.diag(eve_diag)
            self.max_diagonalG = self.return_max_value(eve_no_diag) 
            if np.allclose(eve_no_diag, zero,rtol=0, atol= 1e-12) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-12")
            print("diagonalized Global RDM", eve)
            if np.allclose(eve_no_diag, zero,rtol=0, atol= 1e-5) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-5")
               # quit()
            if np.isnan(np.sum(eve_no_diag)) == True:
                quit()
            #########quit()
        else:
            print('ERROR: A proper integrator was not specified')
            print()
            exit()

            quit()
    
   ####################################################################### 
    def min_dif(self, array):
        n = len(array)
        diff = 10**20
        for i in range(n-1):
            for j in range(i+1, n):
                if(abs(array[i]-array[j])) < diff:
                    diff = abs(array[i]-array[j])
        return diff


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

        #print( self.tot_system.frag_list[0].rotmat[2:,2:] ) #mrar

        #calculate the terms needed for time-derivative of mf-1rdm
        self.tot_system.get_frag_corr12RDM()

       #if self.substep == 3:
        print("############### GLOB CHECK #####################")
      #  propagatedGlob =  np.copy(self.tot_system.glob1RDM)
      #  self.tot_system.get_glob1RDM()
      #  print("formed vs propagated Glob RDM:",np.allclose(propagatedGlob, self.tot_system.glob1RDM, rtol=0, atol=1e-5))
        #print("Difference of Glob", propagatedGlob-self.tot_system.glob1RDM)
      #  self.tot_system.glob1RDM = np.copy(propagatedGlob)
        #print("##############################################")

      #  eve1 = (np.real(utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs )))
     #   print("diagonalized global rdm", eve1)
        self.tot_system.NOevals = np.diag( np.real( utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ) ) )
        #self.tot_system.NOevals = np.diag( np.real( self.tot_system.NOevecs.conjugate().transpose() @ self.tot_system.glob1RDM @ self.tot_system.NOevecs ))
      #  print("no evals", self.tot_system.NOevals)

        print("############### MFCHECK #####################")
#        projected_MF = np.copy(self.tot_system.mf1RDM)
 #       self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
  #      print(np.allclose(projected_MF, self.tot_system.mf1RDM,rtol=0, atol=10e-10))
       # print("difference")
        #print( projected_MF - self.tot_system.mf1RDM )
   #     self.tot_system.mf1RDM = np.copy(projected_MF)
        print("##############################################")


   #     print("Diagonalizing env of mf RDM with R after making new rdm")
    #    self.tot_system.eigvec_frag_MF_check()

        #print("NO evals", self.tot_system.NOevals)#mrar
        #print( utils.rot1el( self.tot_system.mf1RDM, self.tot_system.frag_list[0].rotmat )[2:,2:] )
        #print( self.tot_system.NOevecs )
        #print( self.tot_system.frag_list[0].rotmat[2:,2:] )
        #garb, chk = np.linalg.eigh( self.tot_system.mf1RDM[2:,2:] )
        #print()
        #print( chk )
        #print()

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
        self.G_site = np.copy(G_site)
    # print("ddt_mf1RDM", ddt_mf1RDM)
        #Use change in mf1RDM to calculate X-matrix for each fragment
        make_xmat = time.time()
        self.tot_system.get_frag_Xmat( ddt_mf1RDM )
        print("making xmat:", time.time()-make_xmat)
        #ERR
        G_site_max = self.return_max_value(G_site)        
        G_MF_max = self.return_max_value( np.dot( G_site, self.tot_system.mf1RDM ))
        G_NO_max = self.return_max_value(np.dot(G_site, self.tot_system.NOevecs))
        ddt_mf_max = self.return_max_value(ddt_mf1RDM)
        ddt_NO_max = self.return_max_value(ddt_NOevec)
        fmt_str = '%20.8e'
        G_max = 0.0 
        output    = np.zeros(8)
        output[0] = self.curr_time
        output[1] = G_max
        output[2] = G_site_max
        output[3] = G_MF_max
        output[4] = G_NO_max
        output[5] = ddt_mf_max
        output[6] = ddt_NO_max
        output[7] = self.max_diagonalG
        print("printing G parameters")

        np.savetxt( self.file_GMat, output.reshape(1, output.shape[0]), fmt_str )
        self.file_GMat.flush()


        change_glob1RDM = ddt_glob1RDM * self.delt
        change_NOevecs = ddt_NOevec * self.delt
        change_mf1RDM = ddt_mf1RDM * self.delt

        #print("change_NOevecs", change_NOevecs)
        #Calculate change in embedding orbitals
        change_rotmat_list = []
        for frag in self.tot_system.frag_list:
            change_rotmat_list.append( -1j * self.delt * np.dot( frag.rotmat, frag.Xmat ) )

        #Calculate change in CI coefficients in parallel
        #print(self.delt)
#        if( nproc == 1 ):
#            change_CIcoeffs_list = []
#
#            for ifrag, frag in enumerate(self.tot_system.frag_list):
#                change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )
#        else:
#            frag_pool = multproc.Pool(processes=nproc)
#            #old_pool = np.copy(self.frag_pool)
#            change_CIcoeffs_list = frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
#            frag_pool.close()
#            frag_pool.join()
        print("nproc", nproc)
        no_paralel_start = time.time() 
        change_CIcoeffs_list = []

        for ifrag, frag in enumerate(self.tot_system.frag_list):
            change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )
        print("time without paralelization:", "--- %s seconds ---" % (time.time()-no_paralel_start))
       # frag_pool_time = time.time()
        #frag_pool = multproc.Pool(processes=nproc)
        #print("fragpool is formed in","--- %s seconds ---" % (time.time()-frag_pool_time))
            #old_pool = np.copy(self.frag_pool)
        
      #  apply_pool_time = time.time()
      #  change_CIcoeffs_list = self.frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
        #frag_pool.close()
        #frag_pool.join()
        # self.frag_pool.join()
       # print("time of applying pool with starmap","--- %s seconds ---" % (time.time()-apply_pool_time))
        #self.frag_pool.close()
        #return  change_rotmat_list, change_CIcoeffs_list, change_mf1RDM
        return change_NOevecs, change_rotmat_list, change_CIcoeffs_list, change_glob1RDM, change_mf1RDM

    #####################################################################
    def minimization(self,nproc ):
        #Subroutine to integrate equations of motion
        if( self.integ == 'rk4' ):
            #Use 4th order runge-kutta to integrate EOMs
            half_Nele=int(np.copy(self.tot_system.Nsites)/2)
            
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
            print(" MINIMIZE 1ST SUBSTEP DT")
            print("######################")

            
            #make a dictionary for the rk4 substeps
            self.ddt = {1: 'rk1', 2: 'rk2', 3: 'rk3', 4:'rk4'}
            print("######################")
            print("GETTING 1ST SUBSTEP DT")
            print("######################")

            self.substep = 1
            l1, k1_list, m1_list, n1, p1 = self.one_rk_step(nproc)
            new_ddt = {self.substep:{'G':n1, 'N':l1, 'M':p1,\
                                    'R':k1_list, 'C':m1_list}}
 
            self.ddt.update(new_ddt)

#k1_list, m1_list,  p1 = self.one_rk_step(nproc)
            #print("INIT NOevecs after calc dt", self.tot_system.NOevecs)
   #         print("dt for NOs", l1)
  #          print("dt for rotmat", k1_list)
    #        print("dt for CIs", m1_list)

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
            new_ddt = {self.substep:{'G':n2, 'N':l2, 'M':p2,\
                                    'R':k2_list, 'C':m2_list}}
            self.ddt.update(new_ddt)
            #k2_list, m2_list, p2 = self.one_rk_step(nproc)
            #exit()

            self.tot_system.NOevecs = init_NOevecs + 0.5*l2
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n2
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p2

            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k2_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]
               # print("frag.rotmat", frag.rotmat)
                #print("frag.CIcoeffs", frag.CIcoeffs)


            print("######################")
            print("GETTING 3ST SUBSTEP DT")
            print("######################")
            self.substep = 3
            l3, k3_list, m3_list, n3, p3 = self.one_rk_step(nproc)
            #k3_list, m3_list, p3 = self.one_rk_step(nproc)
            new_ddt = {self.substep:{'G':n3, 'N':l3, 'M':p3,\
                                    'R':k3_list, 'C':m3_list}}
            self.ddt.update(new_ddt)

            self.tot_system.NOevecs = init_NOevecs + 1.0*l3
            self.tot_system.glob1RDM = init_glob1RDM + 1.0*n3
            self.tot_system.mf1RDM = init_mf1RDM + 1.0*p3
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat   = init_rotmat_list[cnt]   + 1.0*k3_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*m3_list[cnt]
            

          #  substep = 1
          #  G_bad_real, ddt_glob1RDM = self.initialize_G(half_Nele)
          #  scipy.optimize.leastsq(self.cost_function_G_substep, G_bad_real, (half_Nele, ddt_glob1RDM, init_NOevecs, init_glob1RDM, init_mf1RDM, init_CIcoeffs_list, init_rotmat_list, substep)) 
          #  print('done with supstep 1')
          #  
          #  substep = 2
          #  G_bad_real, ddt_glob1RDM = self.initialize_G(half_Nele)
          #  scipy.optimize.leastsq(self.cost_function_G_substep, G_bad_real, (half_Nele, ddt_glob1RDM, init_NOevecs, init_glob1RDM, init_mf1RDM, init_CIcoeffs_list, init_rotmat_list, substep))
          #  print('done with supstep 2')
          #  
          #  substep = 3
          #  G_bad_real, ddt_glob1RDM = self.initialize_G(half_Nele)
          #  scipy.optimize.leastsq(self.cost_function_G_substep, G_bad_real, (half_Nele, ddt_glob1RDM, init_NOevecs, init_glob1RDM, init_mf1RDM, init_CIcoeffs_list, init_rotmat_list, substep)) 
          #  print('done with supstep 3')
          #  
            substep = 4
            G_bad_real, ddt_glob1RDM = self.initialize_G(half_Nele)
            scipy.optimize.leastsq(self.cost_function_G_substep, G_bad_real, (half_Nele, ddt_glob1RDM, init_NOevecs, init_glob1RDM, init_mf1RDM, init_CIcoeffs_list, init_rotmat_list, substep)) 
            print('done with supstep 4')
      
            eve = (utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ))
            evals = np.diag(utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ))
            print(evals)
            zero = np.zeros((self.tot_system.Nsites, self.tot_system.Nsites))
            eve_diag = np.diag( eve)
            eve_no_diag = eve - np.diag(eve_diag)
            self.max_diagonalG = self.return_max_value(eve_no_diag) 
            if np.allclose(eve_no_diag, zero,rtol=0, atol= 1e-12) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-12")
            print("diagonalized Global RDM", eve_no_diag)
            if np.allclose(eve_no_diag, zero,rtol=0, atol= 1e-5) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-5")


 #####################################################################         
    def initialize_G(self, half_Nele):   

            self.tot_system.get_frag_corr12RDM()
            self.tot_system.NOevals = np.diag( np.real( utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ) ) )

            self.tot_system.get_frag_Hemb()
            for frag in self.tot_system.frag_list:
                frag.Ecore = 0.0

            #Calculate change in mf1RDM
            ddt_glob1RDM, G_site = mf1rdm_timedep_mod.get_ddt_glob( self.dG, self.tot_system )
            self.G_site = np.copy(G_site)

            #form G_bad

            top_G_bad = self.G_site[:half_Nele, :half_Nele].flatten()
            bottom_G_bad = self.G_site[half_Nele:, half_Nele:].flatten()
            G_bad = np.concatenate((top_G_bad, bottom_G_bad))
            G_bad_real = np.concatenate((np.real(G_bad), np.imag(G_bad))) 

            return G_bad_real, ddt_glob1RDM

 #####################################################################

    def cost_function_G_substep( self, G_bad_real, half_Nele, ddt_glob1RDM, init_NOevecs, init_glob1RDM, init_mf1RDM, init_CIcoeffs_list, init_rotmat_list, substep):
            
            if (substep==1) or (substep==2):
                k_int = 0.5 
            elif (substep==3):
                k_int = 1.0
            else:
                k_int = 1.0/6.0

            G_bad = G_bad_real[:len(G_bad_real)//2] + 1j * G_bad_real[len(G_bad_real)//2:]
            new_G_bad = np.array_split(G_bad,2)
            new_top_G = np.reshape(new_G_bad[0], (half_Nele, half_Nele))
            new_bottom_G = np.reshape(new_G_bad[1], (half_Nele, half_Nele))
            new_G = np.copy(self.G_site)
            new_G[:half_Nele, :half_Nele]=np.copy(new_top_G)
            new_G[half_Nele:, half_Nele:]=np.copy(new_bottom_G)
       
            #form new gamma mf dt
            ddt_mf1RDM, ddt_NOevec =mf1rdm_timedep_mod.get_ddt_mf_NOs(self.tot_system, new_G)     
            self.tot_system.get_frag_Xmat( ddt_mf1RDM )

            change_glob1RDM = ddt_glob1RDM * self.delt
            change_NOevecs = ddt_NOevec * self.delt
            change_mf1RDM = ddt_mf1RDM * self.delt

            #Calculate change in embedding orbitals
            change_rotmat_list = []
            for frag in self.tot_system.frag_list:
                change_rotmat_list.append( -1j * self.delt * np.dot( frag.rotmat, frag.Xmat ) )

            change_CIcoeffs_list = []
            for ifrag, frag in enumerate(self.tot_system.frag_list):
                change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )
            
            new_ddt = {substep:{'G':change_glob1RDM, 'N':change_NOevecs, 'M':change_mf1RDM,\
                                    'R':change_rotmat_list, 'C':change_CIcoeffs_list}}
            self.ddt.update(new_ddt)
            
           # print("(1)", self.ddt[1]['N'])
           # print("(2)", 2.0*self.ddt[2]['N'])
          #  print("(3)", 2.0*self.ddt[3]['N'])
           # print("(4)", self.ddt[4]['N'])

 
            if (substep in (1,2,3)):
                self.tot_system.NOevecs = init_NOevecs + k_int*change_NOevecs
                self.tot_system.glob1RDM = init_glob1RDM + k_int*change_glob1RDM
                self.tot_system.mf1RDM = init_mf1RDM + k_int*change_mf1RDM
                for cnt, frag in enumerate(self.tot_system.frag_list):
                    frag.rotmat   = init_rotmat_list[cnt]   + k_int*change_rotmat_list[cnt]
                    frag.CIcoeffs = init_CIcoeffs_list[cnt] + k_int*change_CIcoeffs_list[cnt]
            
            else:
                self.tot_system.NOevecs = init_NOevecs + k_int * ( self.ddt[1]['N'] + 2.0*self.ddt[2]['N'] + 2.0*self.ddt[3]['N'] + self.ddt[4]['N'] )
                self.tot_system.glob1RDM = init_glob1RDM  + k_int * ( self.ddt[1]['G'] + 2.0*self.ddt[2]['G'] + 2.0*self.ddt[3]['G'] + self.ddt[4]['G'] )
                self.tot_system.mf1RDM =  init_mf1RDM + k_int* ( self.ddt[1]['M'] + 2.0*self.ddt[2]['M'] + 2.0*self.ddt[3]['M'] + self.ddt[4]['M'] )
                for cnt, frag in enumerate( self.tot_system.frag_list ):
                    frag.rotmat   = init_rotmat_list[cnt]   + k_int * ( self.ddt[1]['R'][cnt] + 2.0*self.ddt[2]['R'][cnt] + 2.0*self.ddt[3]['R'][cnt] + self.ddt[4]['R'][cnt] )
                    frag.CIcoeffs = init_CIcoeffs_list[cnt] + k_int * ( self.ddt[1]['C'][cnt] + 2.0*self.ddt[2]['C'][cnt] + 2.0*self.ddt[3]['C'][cnt] + self.ddt[4]['C'][cnt] )
           
            eve = (utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ))
            
            eve_diag = np.diag( eve)
        
            eve_no_diag = eve - np.diag(eve_diag)
            eve_no_diag_array = eve_no_diag.flatten()

 
            #return eve_no_diag_array
            return np.concatenate((np.real(eve_no_diag_array), np.imag(eve_no_diag_array)))
    #####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ######## CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate DMET energy, which also includes calculation of 1 & 2 RDMs and embedding hamiltonian for each fragment
        self.tot_system.get_DMET_E( self.nproc )

        #Calculate total number of electrons
        self.tot_system.get_DMET_Nele()

        ######## PRINT OUT EVERYTHING #######

        #Print correlated density in the site basis
        cnt = 0
        corrdens = np.zeros(self.tot_system.Nsites)
        for frag in self.tot_system.frag_list:
            corrdens[cnt:cnt+frag.Nimp] = np.copy( np.diag( np.real( frag.corr1RDM[:frag.Nimp] ) ) )
            cnt += frag.Nimp
        corrdens_short =  np.copy(corrdens)
        corrdens = np.insert( corrdens, 0, current_time )
        np.savetxt( self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str )
        self.file_corrdens.flush()
        

        #Print output data
        Nimp      = self.tot_system.frag_list[0].Nimp
        Nsites    = self.tot_system.Nsites
        output    = np.zeros(13+3*Nsites-Nimp+2*Nimp)
        output[0] = current_time
        output[1] = self.tot_system.DMET_E
        output[2] = self.tot_system.DMET_Nele
        #self.tot_system.get_frag_corr12RDM()
        #self.tot_system.get_glob1RDM()

        #self.tot_system.NOevals = np.diag( np.real( utils.rot1el( self.tot_system.glob1RDM,  self.tot_system.NOevecs ) ) )
        diag = np.real( utils.rot1el( self.tot_system.glob1RDM,  self.tot_system.NOevecs ) )
        #offDiag = np.copy(diag[0][5])
        #self.tot_system.get_new_mf1RDM( round(self.tot_system.Nele/2) )
        #mrar

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
        #change_NOevecs, change_mf1RDM = mf1rdm_timedep_mod.get_ddt_mf1rdm_serial( self.tot_system, round(self.tot_system.Nele/2) )
        #self.tot_system.get_frag_Xmat( change_mf1RDM )

        output[9+Nsites:9+2*Nsites-Nimp] = self.tot_system.frag_list[0].mfevals

        #w1,garb = np.linalg.eigh( self.tot_system.mf1RDM )
        #w2,garb = np.linalg.eigh( self.tot_system.frag_list[0].corr1RDM )
        print(corrdens_short.shape, "cordense short")
        print(self.corrdens_old.shape, "corrdense")
        output[9+2*Nsites-Nimp:9+3*Nsites-Nimp ] = (corrdens_short-self.corrdens_old)/self.delt
        #output[9+3*Nsites-Nimp:9+3*Nsites-Nimp+2*Nimp ] = w2
        output[9+3*Nsites-Nimp+2*Nimp] = np.copy(diag[0][5])
        output[10+3*Nsites-Nimp+2*Nimp] = np.copy(diag[1][5])
        output[11+3*Nsites-Nimp+2*Nimp] = np.copy(diag[3][5])
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

        self.corrdens_old = np.copy(corrdens_short)
        
        #Save total system to file for restart purposes using pickle
        file_system = open( 'restart_systemJ.dat', 'wb' )
        pickle.dump( self.tot_system, file_system )
        file_system.close()

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

