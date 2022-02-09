#Routines to run a real-time projected-DMET calculation
#Need pyscf installed

import numpy as np
import real_time_elec_structureGN.projected_dmet.system_mod as system_mod
import real_time_elec_structureGN.projected_dmet.mf1rdm_timedep_mod as mf1rdm_timedep_mod
import real_time_elec_structureGN.scripts.applyham_pyscf as applyham_pyscf
import real_time_elec_structureGN.scripts.utils as utils
import real_time_elec_structure.projected_dmet.mf1rdm_timedep_mod as mf1rdm_timedep_modMF

import multiprocessing as multproc
import pickle
import scipy
import sys
import os

import time

############ CLASS TO RUN REAL-TIME DMET CALCULATION #########

class dynamics_driver():

    #####################################################################

    def __init__( self, h_site, V_site, hamtype, tot_system, delt, dG, Nstep, Nprint=100, integ='rk1', nproc=1, hubsite_indx=None, init_time=0.0):

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
        self.file_output   = open( 'outputNoPert.dat', 'w' )
        self.file_corrdens = open( 'corr_densityNoPert.dat', 'w' )

    #####################################################################

    def kernel( self ):

        print()
        print('********************************************')
        print('     BEGIN REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()
        print("CORRRECT DYNAMIC FILE")
        #DYNAMICS LOOP
        current_time = self.init_time
        #ERRR
        init_NOevals = np.copy( self.tot_system.NOevals )
        init_NOevecs = np.copy( self.tot_system.NOevecs )
        init_glob1RDM = np.copy( self.tot_system.glob1RDM )
        init_mf1RDM = np.copy(self.tot_system.mf1RDM)
        init_CIcoeffs_list = []
        init_rotmat_list   = []
        for frag in self.tot_system.frag_list:
            init_rotmat_list.append( np.copy(frag.rotmat) )
            init_CIcoeffs_list.append( np.copy(frag.CIcoeffs) )
                 
#        self.integrate(self.nproc)
 
        self.tot_system.NOevecs = np.copy(init_NOevecs)
        self.tot_system.NOevals = np.copy(init_NOevals)
        self.tot_system.mf1RDM = np.copy(init_mf1RDM)
        self.tot_system.glob1RDM = np.copy(init_glob1RDM)

        for ctr, frag in  enumerate( self.tot_system.frag_list ):
            frag.rotmat = np.copy(init_rotmat_list[ctr])
            frag.CIcoeffs = np.copy(init_CIcoeffs_list[ctr])
        

 #       self.breakDegen(init_glob1RDM,init_mf1RDM)
        
       # self.tot_system.glob1RDM = init_glob1RDM
        #self.tot_system.mf1RDM = init_mf1RDM
        #for ctr, frag in  enumerate( self.tot_system.frag_list ): 
         #   frag.rotmat = np.copy(init_rotmat_list[ctr])
          #  frag.CIcoeffs = np.copy(init_CIcoeffs_list[ctr])
        
        for step in range(self.Nstep):
            self.step = np.copy(step)
            #Print data before taking time-step, this always prints out data at initial time step
            
            if step < 80:
                print("MF STEP", step)
                self.initialize_mf(self.nproc)
            else:
                print("GN STEP", step)
                self.integrate(self.nproc)

            if( np.mod( step, self.Nprint ) == 0 ) and step > 0:
                print('Writing data at step ', step, 'and time', current_time, 'for RT-pDMET calculation')
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate FCI coefficients and rotation matrix for all fragments
            print("####################")
            print("STEP", step)
            print("####################")
           # if step == 4:
            #    quit()    
            #quit()
            #quit()
            #Increase current_time
            current_time = self.init_time + (step+1)*self.delt
            
            #if step == self.Nstep-1:
                #self.save_prop()

        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_corrdens.close()

        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END REAL-TIME DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()

    #####################################################################
#    #ERRR
    def breakDegen( self, init_glob1RDM,init_mf1RDM ):
        #n1 - dt global rdm
        #self.integrate(self.nproc) 
        #print("DMET Energy before rotation", self.tot_system.DMET_E)
            

     #   l1, k1_list, m1_list, n1, p1 = self.one_rk_step(self.nproc)
        n1 = np.copy(self.dtGlob)
        initial_NOevals, initial_NOevecs = np.linalg.eigh(init_glob1RDM)
        diag_init_glob = utils.rot1el(init_glob1RDM, initial_NOevecs)
        

        new_glob1 = utils.rot1el(n1[:3,:3],(initial_NOevecs[:3,:3])) 
#        newNOevals1, newNOevecs1 = np.linalg.eigh(new_glob1)
        new_glob2 = utils.rot1el(n1[3:,3:], (initial_NOevecs[3:,3:]))
 #       newNOevals2, newNOevecs2 = np.linalg.eigh(new_glob2)  
        
 #       newNOevecsR = np.zeros((6,6),dtype=np.complex_)
        #newNOevecsR = np.copy(newNOevecs2)
        #A = np.identity(6)
        #newNOevecsR += A
 #       newNOevecsR[:3, :3] = newNOevecs1
  #      newNOevecsR[3:, 3:] = newNOevecs2
        
        W_2_up = np.zeros((3,3), dtype=np.complex_)
        W_2_down = np.zeros((3,3), dtype=np.complex_)
        
        for i in range(3):
            for j in range(3):
                for m in range(3,6):
                    for k in range (3,6):
                        W_2_up[i,j] = initial_NOevecs[i,j].T.conj() * n1[i, j] * initial_NOevecs[m,k]\
                             * initial_NOevecs[m,k].T.conj() * n1[i, j] *initial_NOevecs[i,j]
                        W_2_up[i,j] /= 1/(initial_NOevals[i] - initial_NOevals[m])
                        
                        W_2_down[i,j] = initial_NOevecs[m,k].T.conj() * n1[m, k] * initial_NOevecs[i,j]\
                             *initial_NOevecs[i,j].T.conj() * n1[m, k] *initial_NOevecs[m,k]
                        W_2_down[i,j] /= 1/(initial_NOevals[m] - initial_NOevals[i])
         
        W_up = new_glob1 + W_2_up
        print("W_up", W_up)
        newNOevals1, newNOevecs1 = np.linalg.eigh(W_up)

        W_down = new_glob2 + W_2_down
        print("W_down", W_down)
        newNOevals2, newNOevecs2 = np.linalg.eigh(new_glob2)
        
        newNOevecsR = np.zeros((6,6),dtype=np.complex_)
        newNOevecsR[:3, :3] = newNOevecs1
        newNOevecsR[3:, 3:] = newNOevecs2

        #newNOevecsShort= np.zeros((6,3),dtype=np.complex_)
        newNOevecs = np.zeros((6,6),dtype=np.complex_)
        for a in range(6):
            newNOevecs[:,a] = newNOevecsR[0,a]* initial_NOevecs[:,0] + newNOevecsR[1,a]* initial_NOevecs[:,1] +newNOevecsR[2,a]*initial_NOevecs[:,2] + newNOevecsR[3,a]*initial_NOevecs[:,3] \
            + newNOevecsR[4,a]*initial_NOevecs[:,4] + newNOevecsR[5,a]* initial_NOevecs[:,5] 
         #   newNOevecsShort[:,a] = newNOevecsR[0,a]*initial_NOevecs[:,3]  + newNOevecsR[1,a]*initial_NOevecs[:,4] + newNOevecsR[2,a]* initial_NOevecs[:,5]  
        #newNOevecs = np.concatenate((initial_NOevecs[:, :3], newNOevecsShort), axis = 1)
        glob_init_diag = utils.rot1el( init_glob1RDM, (newNOevecs))
        newNOeval = np.diag(np.real(glob_init_diag))
        print("glob_init_diag", glob_init_diag)
        self.tot_system.NOevals = np.flip(newNOeval)
        self.tot_system.NOevecs = np.flip(newNOevecs, 1) 
        
       #self.tot_system.get_DMET_E(self.nproc)
       # print("energy after changing nos", self.tot_system.DMET_E)
       # print("initial NOevecs", initial_NOevecs) 
       # print("newNOevecs", newNOevecs)
        
        self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        print("whether mf after change of nos and before is the same", np.allclose(init_mf1RDM, self.tot_system.mf1RDM, rtol=0, atol=1e-12))
        print("mf difference", init_mf1RDM - self.tot_system.mf1RDM)
        print("self.tot_system.mf1RDM", self.tot_system.mf1RDM)

       # for ctr, frag in enumerate(self.tot_system.frag_list):
      #      old_rotmat = frag.rotmat
       #     print("old_rotmat", old_rotmat)
        #    frag.get_rotmat( self.tot_system.mf1RDM )
         #   print("new rotmat", frag.rotmat)
         #   print()
          #  print("rotmat after transformation")
           # RotmatCheck = np.allclose(old_rotmat, frag.rotmat, rtol=0, atol=1e-13) 
           # print(np.allclose(old_rotmat, frag.rotmat, rtol=0, atol=1e-13))
            #print("difference", old_rotmat - frag.rotmat)
                
           # if RotmatCheck == False:
            #    frag.rotmat[:,3] *= -1
             #   frag.rotmat[:,4] *= -1
              #  frag.rotmat[:,5] *= -1
               # print(np.allclose(old_rotmat, frag.rotmat, rtol=0, atol=1e-13))
                # newIdent = np.dot(old_rotmat, frag.rotmat.T.conj())
               # print("newIdent", newIdent)
               # new_rotmat = np.dot(newIdent, frag.rotmat)
               # print("new_rotmat", new_rotmat)
               # print(np.allclose(old_rotmat, new_rotmat, rtol=0, atol=1e-13)) 
               # frag.rotmat = np.copy(new_rotmat) 
        #quit()
        #self.tot_system.mf1RDM = np.copy(init_mf1RDM)
        #self.tot_system.NOevecs = np.copy(init_NOevecs)
        #self.tot_system.NOevals = np.copy(init_NOevals)
        #self.tot_system.glob1RDM = np.copy(new_init_glob1RDM)
        #for cnt, frag in enumerate(self.tot_system.frag_list):        
         #   frag.rotmat = np.copy(init_rotmat_list[cnt])
          #  frag.CIcoeffs = np.copy(init_CIcoeffs_list[cnt]) 
    #    for frag in self.tot_system.frag_list:
     #       if( not np.iscomplexobj( frag.rotmat ) ):
    #            frag.rotmat = frag.rotmat.astype(complex)
     #       if( not np.iscomplexobj( frag.CIcoeffs ) ):
      #          frag.CIcoeffs = frag.CIcoeffs.astype(complex)

       # if( not np.iscomplexobj( self.tot_system.mf1RDM ) ):
        #    self.tot_system.mf1RDM = self.tot_system.mf1RDM.astype(complex)
    #####################################################################
#    def initalize_mf_from_file(self, nproc):

 #       dynamics_driver_MF.integrate(nproc)


    ####################################################################

    def initialize_mf(self, nproc):
        init_mf1RDM = np.copy(self.tot_system.mf1RDM)
        init_CIcoeffs_list = []
        init_rotmat_list   = []
        for frag in self.tot_system.frag_list:
            init_rotmat_list.append( np.copy(frag.rotmat) )
            init_CIcoeffs_list.append( np.copy(frag.CIcoeffs) )
        
        self.substep = 1
        l1, k1_list, m1_list = self.one_rk_stepMF(nproc)
        self.tot_system.mf1RDM = init_mf1RDM + 0.5*l1
        for cnt, frag in enumerate(self.tot_system.frag_list):
            frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k1_list[cnt]
            frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m1_list[cnt]

        self.substep = 2
        l2, k2_list, m2_list = self.one_rk_stepMF(nproc)
        self.tot_system.mf1RDM = init_mf1RDM + 0.5*l2
        for cnt, frag in enumerate(self.tot_system.frag_list):
            frag.rotmat   = init_rotmat_list[cnt]   + 0.5*k2_list[cnt]
            frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]

        l3, k3_list, m3_list = self.one_rk_stepMF(nproc)
        self.tot_system.mf1RDM = init_mf1RDM + 1.0*l3
        for cnt, frag in enumerate(self.tot_system.frag_list):
            frag.rotmat   = init_rotmat_list[cnt]   + 1.0*k3_list[cnt]
            frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*m3_list[cnt]

        l4, k4_list, m4_list = self.one_rk_stepMF(nproc)
        self.tot_system.mf1RDM =  init_mf1RDM +1.0/6.0 * ( l1 + 2.0*l2 + 2.0*l3 + l4 )
        for cnt, frag in enumerate( self.tot_system.frag_list ):
            frag.rotmat   = init_rotmat_list[cnt]   + 1.0/6.0 * ( k1_list[cnt] + 2.0*k2_list[cnt] + 2.0*k3_list[cnt] + k4_list[cnt] )
            frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0/6.0 * ( m1_list[cnt] + 2.0*m2_list[cnt] + 2.0*m3_list[cnt] + m4_list[cnt] )
        print("self.tot_system.NOevals", self.tot_system.NOevals)
    
    #####################################################################
    def one_rk_stepMF(self, nproc):
        self.tot_system.get_frag_corr12RDM()
        self.tot_system.get_glob1RDM()
        self.tot_system.get_nat_orbs()
        self.tot_system.NOevals = np.diag( np.real( utils.rot1el( self.tot_system.glob1RDM, self.tot_system. NOevecs ) ) )
        self.tot_system.get_frag_Hemb()
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        change_mf1RDM = mf1rdm_timedep_modMF.get_ddt_mf1rdm_serial( self.tot_system, round(self.tot_system.Nele/2) )
        self.tot_system.get_frag_Xmat( change_mf1RDM )
        change_mf1RDM *= self.delt

        change_rotmat_list = []
        for frag in self.tot_system.frag_list:
            change_rotmat_list.append( -1j * self.delt * np.dot( frag.rotmat, frag.Xmat ) )
        if( nproc == 1 ):
            change_CIcoeffs_list = []
            for ifrag, frag in enumerate(self.tot_system.frag_list):
                change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )
        else:
            frag_pool = multproc.Pool(nproc)
            change_CIcoeffs_list = frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
            frag_pool.close()
        return change_mf1RDM, change_rotmat_list, change_CIcoeffs_list
    #####################################################################



    def integrate( self, nproc ):
        #Subroutine to integrate equations of motion
        if( self.integ == 'rk4' ):
            #Use 4th order runge-kutta to integrate EOMs

            #Copy MF 1RDM, CI coefficients, and embedding orbs at time t
            init_NOevecs = np.copy( self.tot_system.NOevecs )
            init_glob1RDM = np.copy( self.tot_system.glob1RDM )
            init_mf1RDM = np.copy(self.tot_system.mf1RDM)

          #  print("INIT NOevecs before the 1st substep")
           # print(self.tot_system.NOevecs)
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
        #    print("dt for NOs", l1)
         #   print("dt for rotmat", k1_list)
          #  print("dt for CIs", m1_list)
            
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
            
            #ERRR
            self.dtGlob =  1.0/6.0 * ( n1 + 2.0*n2 + 2.0*n3 + n4 )            

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
            #ERRR
            self.n4 = n4
            eve = (utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ))
           # print("diagonalized global rdm", eve)
            #print("evals")
            evals = np.diag(np.real(utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs )))
            print(evals)
        #    print("FINISHED STEP:", self.step)
            zero = np.zeros((self.tot_system.Nsites, self.tot_system.Nsites))
            eve_diag = np.diag( eve)
            eve_no_diag = eve - np.diag(eve_diag)
            if np.allclose(eve_no_diag, zero, rtol=0, atol=1e-12) == False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-12")
            print("diagonalized Global RDM", eve)
            if np.allclose(eve_no_diag, zero, rtol=0, atol=1e-7) == False:
                print("GLOBAL DIAGONALIZED LESS THEN 10e-7")
                quit()
            #quit()
            #########quit()
        else:
            print('ERROR: A proper integrator was not specified')
            print()
            exit()

        #    quit()

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
        propagatedGlob =  np.copy(self.tot_system.glob1RDM)
        self.tot_system.get_glob1RDM()
        print("formed vs propagated Glob RDM:",np.allclose(propagatedGlob, self.tot_system.glob1RDM, 1e-10))
      #  print("Difference of Glob", propagatedGlob-self.tot_system.glob1RDM)
        self.tot_system.glob1RDM = np.copy(propagatedGlob)
        #print("##############################################")

        eve1 = ((utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs )))
        print("diagonalized global rdm", eve1)
        self.tot_system.NOevals = np.diag(np.real( utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ) ) )
       # self.tot_system.NOevals = np.diag(( utils.rot1el( self.tot_system.glob1RDM, self.tot_system.NOevecs ) ) )
        #self.tot_system.NOevals = np.diag( np.real( self.tot_system.NOevecs.conjugate().transpose() @ self.tot_system.glob1RDM @ self.tot_system.NOevecs ))
       # print("no evals", self.tot_system.NOevals)

        print("############### MFCHECK #####################")
        projected_MF = np.copy(self.tot_system.mf1RDM)
        self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        print(np.allclose(projected_MF, self.tot_system.mf1RDM, 10e-10))
       # print("difference")
        #print( projected_MF - self.tot_system.mf1RDM )
        self.tot_system.mf1RDM = np.copy(projected_MF)
        print("##############################################")


        print("Diagonalizing env of mf RDM with R after making new rdm")
        self.tot_system.eigvec_frag_MF_check()

        #print("NO evals", self.tot_system.NOevals)#mrar
        #print( utils.rot1el( self.tot_system.mf1RDM, self.tot_system.frag_list[0].rotmat )[2:,2:] )
        #print( self.tot_system.NOevecs )
        #print( self.tot_system.frag_list[0].rotmat[2:,2:] )
        #garb, chk = np.linalg.eigh( self.tot_system.mf1RDM[2:,2:] )
        #print()
        #print( chk )
        #print()

        #calculate embedding hamiltonian
        self.tot_system.get_frag_Hemb()

        #Make sure Ecore for each fragment is 0 for dynamics
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        #Calculate change in mf1RDM
        #ERR
        ddt_glob1RDM, ddt_NOevec, ddt_mf1RDM = mf1rdm_timedep_mod.get_ddt_mf1rdm_serial( self.dG, self.tot_system, round(self.tot_system.Nele/2) )
       # print("ddt_mf1RDM", ddt_mf1RDM)
        #Use change in mf1RDM to calculate X-matrix for each fragment
        self.tot_system.get_frag_Xmat( ddt_mf1RDM )
        #Multiply change in mf1RDM by time-step
        #change_NOevecs = -1j * self.delt * np.dot(G_site, self.tot_system.NOevecs)
        #change_NOevecs = -1j * self.delt * np.dot(self.tot_system.NOevecs, G_emb)
       # print("self delta", self.delt)

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
        if( nproc == 1 ):
            change_CIcoeffs_list = []

            for ifrag, frag in enumerate(self.tot_system.frag_list):
                change_CIcoeffs_list.append( applyham_wrapper( frag, self.delt ) )
        else:
            frag_pool = multproc.Pool(nproc)
            change_CIcoeffs_list = frag_pool.starmap( applyham_wrapper, [(frag,self.delt) for frag in self.tot_system.frag_list] )
            frag_pool.close()


        #return  change_rotmat_list, change_CIcoeffs_list, change_mf1RDM
        return change_NOevecs, change_rotmat_list, change_CIcoeffs_list, change_glob1RDM, change_mf1RDM

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
        #output[9+2*Nsites-Nimp:9+3*Nsites-Nimp ] = w1
        #output[9+3*Nsites-Nimp:9+3*Nsites-Nimp+2*Nimp ] = w2
        output[9+3*Nsites-Nimp+2*Nimp] = np.copy(diag[0][5])
        output[10+3*Nsites-Nimp+2*Nimp] = np.copy(diag[1][5])
        output[11+3*Nsites-Nimp+2*Nimp] = np.copy(diag[3][5])
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

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


