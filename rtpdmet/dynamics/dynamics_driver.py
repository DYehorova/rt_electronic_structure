# Routines to run a real-time projected-DMET calculation
import numpy as np
import sys
import rt_electronic_structure.rtpdmet.dynamics.mf1rdm_timedep_mod as mf1rdm_timedep_mod
import rt_electronic_structure.scripts.applyham_pyscf as applyham_pyscf
import rt_electronic_structure.scripts.utils as utils
import pickle
import time

# ########### CLASS TO RUN REAL-TIME DMET CALCULATION #########


class dynamics_driver():

    #####################################################################

    def __init__(self, h_site, V_site, hamtype, tot_system, delt, dG,
                 Nstep, nproc, Nprint=100, integ='rk1', hubsite_indx=None,
                 init_time=0.0):

        # h_site -
        # 1 e- hamiltonian in site-basis for total system to run dynamics
        # V_site -
        # 2 e- hamiltonian in site-basis for total system to run dynamics
        # hamtype -
        # integer defining if using a special Hamiltonian (Hubbard or Anderson)
        # tot_system -
        # previously defined DMET total system including fragment information
        # delt - time step
        # Nstep - total number of time-steps
        # Nprint - number of time-steps between printing
        # init_time - the starting time for the calculation
        # integ - the type of integrator used
        # nproc - number of processors for calculation
        # - careful, there is no check that this matches the pbs script

        self.tot_system = tot_system
        self.delt = delt
        self.Nstep = Nstep
        self.Nprint = Nprint
        self.init_time = init_time
        self.integ = integ
        self.nproc = nproc
        self.dG = dG
        print()
        print('********************************************')
        print('     SET-UP REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()
        # Input error checks
        self.import_check(1e-7)

        # Convert rotation matrices, CI coefficients,
        # and MF 1RDM to complex arrays if they're not already
        for frag in self.tot_system.frag_list:
            if(not np.iscomplexobj(frag.rotmat)):
                frag.rotmat = frag.rotmat.astype(complex)
            if(not np.iscomplexobj(frag.CIcoeffs)):
                frag.CIcoeffs = frag.CIcoeffs.astype(complex)

        if(not np.iscomplexobj(self.tot_system.mf1RDM)):
            self.tot_system.mf1RDM = self.tot_system.mf1RDM.astype(complex)

        if(not np.iscomplexobj(self.tot_system.glob1RDM)):
            self.tot_system.glob1RDM = self.tot_system.glob1RDM.astype(complex)

        if(not np.iscomplexobj(self.tot_system.NOevecs)):
            self.tot_system.NOevecs = self.tot_system.NOevecs.astype(complex)

        # Set-up Hamiltonian for dynamics calculation
        self.tot_system.h_site = h_site
        self.tot_system.V_site = V_site
        self.tot_system.hamtype = hamtype

        # If running Hubbard-like model, need an array
        # containing index of all sites that have hubbard U term
        self.tot_system.hubsite_indx = hubsite_indx
        if(self.tot_system.hamtype == 1
           and self.tot_system.hubsite_indx is None):
            print('ERROR: Did not specify an array of sites that have U term')
            print()
            exit()

        # Define output files
        self.file_output = open('outputJ.dat', 'w')
        self.file_corrdens = open('corr_densityJ.dat', 'w')
        self.file_current = open('current.dat', 'w')
        self.max_diagonalG = 0
        self.corrdens_old = np.zeros((self.tot_system.Nsites))
        self.corrdens_old += 1
        # start_pool = time.time()
        # self.frag_pool = multproc.Pool(nproc)
        # print("time to from pool", time.time()-start_pool)
    #####################################################################

    def kernel(self):
        start_time = time.time()
        print()
        print('********************************************')
        print('     BEGIN REAL-TIME DMET CALCULATION       ')
        print('********************************************')
        print()

        # fraddg_pool = multproc.Pool(self.nproc)

        # DYNAMICS LOOP
        print("time", self.init_time)
        current_time = self.init_time
        self.curr_time = current_time
        for step in range(self.Nstep):
            print("step", step)
            self.step = np.copy(step)

            # Print data
            if(np.mod(step, self.Nprint) == 0) and step > 1:
                print('Writing data at step ', step, 'and time',
                      current_time, 'for RT-pDMET calculation')
                self.print_data(current_time)
                sys.stdout.flush()

            # if a trajectory restarted, record data before a 1st step
            if current_time != 0 and step == 0:
                print('Writing data at step ', step, 'and time',
                      current_time, 'for RT-pDMET calculation')
                self.print_data(current_time)
                sys.stdout.flush()

            # Integrate FCI coefficients and rotation matrix for all fragments
            print("####################")
            print("STEP", step)
            print("####################")
            self.integrate(self.nproc)

            # Increase current_time
            current_time = self.init_time + (step+1)*self.delt
            self.curr_time = current_time
            sys.stdout.flush()

        # Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time',
              current_time, 'for RT-pDMET calculation')
        self.print_data(current_time)
        sys.stdout.flush()

        # Close output files
        self.file_output.close()
        self.file_corrdens.close()
        self.file_current.close()
        # self.frag_pool.close()
        print()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("END REAL-TIME DMET CALCULATION")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
        print("--- %s seconds ---" % (time.time() - start_time))
    #####################################################################

    def integrate(self, nproc):
        # Subroutine to integrate equations of motion
        if(self.integ == 'rk4'):

            # Use 4th order runge-kutta to integrate EOMs

            # Copy MF 1RDM, global RDM, CI coefficients,
            # natural and embedding orbs at time t
            init_NOevecs = np.copy(self.tot_system.NOevecs)
            init_glob1RDM = np.copy(self.tot_system.glob1RDM)
            init_mf1RDM = np.copy(self.tot_system.mf1RDM)

            init_CIcoeffs_list = []
            init_rotmat_list = []
            for frag in self.tot_system.frag_list:
                init_rotmat_list.append(np.copy(frag.rotmat))
                init_CIcoeffs_list.append(np.copy(frag.CIcoeffs))

            # Calculate appropriate changes in MF 1RDM,
            # embedding orbitals, and CI coefficients
            # l, k, m, n, p =
            # change in NOevecs, rotmat_list, CIcoeffs_list, glob1RDM, mf1RDM

            print("######################")
            print("GETTING 1ST SUBSTEP DT")
            print("######################")

            self.substep = 1
            l1, k1_list, m1_list, n1, p1 = self.one_rk_step(nproc)

            self.tot_system.NOevecs = init_NOevecs + 0.5*l1
            self.tot_system.glob1RDM = init_glob1RDM + 0.5*n1
            self.tot_system.mf1RDM = init_mf1RDM + 0.5*p1
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = init_rotmat_list[cnt] + 0.5*k1_list[cnt]
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
                frag.rotma = init_rotmat_list[cnt] + 0.5*k2_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 0.5*m2_list[cnt]

            print("######################")
            print("GETTING 3ST SUBSTEP DT")
            print("######################")

            l3, k3_list, m3_list, n3, p3 = self.one_rk_step(nproc)

            self.tot_system.NOevecs = init_NOevecs + 1.0*l3
            self.tot_system.glob1RDM = init_glob1RDM + 1.0*n3
            self.tot_system.mf1RDM = init_mf1RDM + 1.0*p3
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = init_rotmat_list[cnt] + 1.0*k3_list[cnt]
                frag.CIcoeffs = init_CIcoeffs_list[cnt] + 1.0*m3_list[cnt]

            print("######################")
            print("GETTING 4ST SUBSTEP DT")
            print("######################")

            l4, k4_list, m4_list, n4, p4 = self.one_rk_step(nproc)

            self.tot_system.NOevecs = (
                init_NOevecs + 1.0/6.0 * (l1 + 2.0*l2 + 2.0*l3 + l4))
            self.tot_system.glob1RDM = (
                init_glob1RDM + 1.0/6.0 * (n1 + 2.0*n2 + 2.0*n3 + n4))
            self.tot_system.mf1RDM = (
                init_mf1RDM + 1.0/6.0 * (p1 + 2.0*p2 + 2.0*p3 + p4))
            for cnt, frag in enumerate(self.tot_system.frag_list):
                frag.rotmat = (
                    init_rotmat_list[cnt] + 1.0/6.0 * (
                        k1_list[cnt] + 2.0*k2_list[cnt] +
                        2.0*k3_list[cnt] + k4_list[cnt]))

                frag.CIcoeffs = (
                    init_CIcoeffs_list[cnt] + 1.0/6.0 * (
                        m1_list[cnt] + 2.0*m2_list[cnt] +
                        2.0*m3_list[cnt] + m4_list[cnt]))

            print("#############################")
            print("FULL RK STEP DONE, QUANTITIES:")
            print("#############################")

            # check how well natural orbitals diagonalize global rdm
            eve = (utils.rot1el(
                self.tot_system.glob1RDM, self.tot_system.NOevecs))
            evals = np.diag(
                np.real(utils.rot1el(
                    self.tot_system.glob1RDM, self.tot_system.NOevecs)))
            print("utils rot1el matches direct diagonalization",
                  np.allclose(eve, evals, rtol=0, atol=1e-10))

            print("FINISHED STEP:", self.step)
            zero = np.zeros((self.tot_system.Nsites, self.tot_system.Nsites))
            eve_diag = np.diag(eve)
            self.off_diag_glob_diag = np.copy(eve - np.diag(eve_diag))
            self.max_diagonalG = self.return_max_value(self.off_diag_glob_diag)

            if np.allclose(self.off_diag_glob_diag,
                           zero, rtol=0, atol=1e-12) is False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-12")

            print("diagonalized Global RDM", eve)

            if np.allclose(self.off_diag_glob_diag,
                           zero, rtol=0, atol=1e-5) is False:
                print("GLOBAL DIAGOMALIZED LESS THEN 10e-5")
            if np.isnan(np.sum(self.off_diag_glob_diag)) is True:
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
                if (abs(array[x, y]) > largest):
                    largest = array[x, y]
        return largest
    #####################################################################

    def one_rk_step(self, nproc):
        # Subroutine to calculate one change in a runge-kutta step of any order
        # Using EOM that integrates CI coefficients, rotmat, and MF 1RDM

        # Prior to calling this routine need to update
        # MF 1RDM, rotmat and CI coefficients

        # Calculate the terms needed for time-derivative of mf-1rdm
        self.tot_system.get_frag_corr12RDM()

        self.tot_system.NOevals = np.diag(np.real(
            utils.rot1el(self.tot_system.glob1RDM, self.tot_system.NOevecs)))

        # Calculate embedding hamiltonian
        make_ham = time.time()
        self.tot_system.get_frag_Hemb()
        print("making hamiltonian:", time.time()-make_ham)
        # Make sure Ecore for each fragment is 0 for dynamics
        for frag in self.tot_system.frag_list:
            frag.Ecore = 0.0

        # Calculate change in propagated variables
        make_derivs = time.time()
        ddt_glob1RDM, ddt_NOevec, ddt_mf1RDM, G_site = (
            mf1rdm_timedep_mod.get_ddt_mf1rdm_serial(
                self.dG, self.tot_system, round(self.tot_system.Nele/2)))
        print("making derivatives:", time.time()-make_derivs)

        # Use change in mf1RDM to calculate X-matrix for each fragment
        make_xmat = time.time()
        self.tot_system.get_frag_Xmat(ddt_mf1RDM)
        print("making xmat:", time.time()-make_xmat)

        change_glob1RDM = ddt_glob1RDM * self.delt
        change_NOevecs = ddt_NOevec * self.delt
        change_mf1RDM = ddt_mf1RDM * self.delt

        # Calculate change in embedding orbitals
        change_rotmat_list = []
        for frag in self.tot_system.frag_list:
            change_rotmat_list.append(
                -1j * self.delt * np.dot(frag.rotmat, frag.Xmat))

        # Calculate change in CI coefficients in parallel

        no_paralel_start = time.time()
        change_CIcoeffs_list = []

        for ifrag, frag in enumerate(self.tot_system.frag_list):
            change_CIcoeffs_list.append(applyham_wrapper(frag, self.delt))
        print("time without paralelization:",
              "--- %s seconds ---" % (time.time()-no_paralel_start))
        return change_NOevecs, change_rotmat_list, \
            change_CIcoeffs_list, change_glob1RDM, change_mf1RDM
    #####################################################################

    def print_data(self, current_time):
        # Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        # ####### CALCULATE OBSERVABLES OF INTEREST #######

        # Calculate DMET energy, which also includes calculation
        # of 1 & 2 RDMs and embedding hamiltonian for each fragment
        energy_calc = time.time()
        self.tot_system.get_DMET_E(self.nproc)
        print("time for energy calc", time.time()-energy_calc)

        # Calculate total number of electrons
        nele_calc = time.time()
        self.tot_system.get_DMET_Nele()
        print("time for nele calc", time.time()-nele_calc)
        # ####### PRINT OUT EVERYTHING #######

        # Print correlated density in the site basis
        elec_dens = time.time()
        cnt = 0
        corrdens = np.zeros(self.tot_system.Nsites)
        for frag in self.tot_system.frag_list:
            corrdens[cnt:cnt+frag.Nimp] = np.copy(
                np.diag(np.real(frag.corr1RDM[:frag.Nimp])))
            cnt += frag.Nimp
        corrdens = np.insert(corrdens, 0, current_time)
        np.savetxt(
            self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]),
            fmt_str)
        self.file_corrdens.flush()
        print("time to calc and write elec data", time.time()-elec_dens)

        # Print output data
        writing_outfile = time.time()
        output = np.zeros((12+self.tot_system.Nsites))
        output[0] = current_time
        output[1] = self.tot_system.DMET_E
        output[2] = self.tot_system.DMET_Nele
        output[3] = np.trace(self.tot_system.mf1RDM)
        output[4] = np.trace(self.tot_system.frag_list[0].corr1RDM)
        output[5] = np.einsum('ppqq', self.tot_system.frag_list[0].corr2RDM)
        output[6] = np.linalg.norm(
            self.tot_system.frag_list[0].CIcoeffs)**2
        output[7] = np.linalg.norm(
            self.tot_system.frag_list[0].rotmat[:, 4])**2
        # self.tot_system.get_nat_orbs()
        if(np.allclose(self.tot_system.glob1RDM,
                       utils.adjoint(self.tot_system.glob1RDM),
                       rtol=0.0, atol=1e-14)):
            output[8] = 1
        else:
            output[8] = 0
        output[9:9+self.tot_system.Nsites] = (
            np.copy(self.tot_system.NOevals))
        output[10+self.tot_system.Nsites] = (
            np.copy(np.real(self.max_diagonalG)))
        output[11+self.tot_system.Nsites] = (
            np.copy(np.imag(self.max_diagonalG)))

        np.savetxt(
            self.file_output, output.reshape(1, output.shape[0]), fmt_str)
        self.file_output.flush()

        # Save total system to file for restart purposes using pickle
        file_system = open('restart_systemJ.dat', 'wb')
        pickle.dump(self.tot_system, file_system)
        file_system.close()
        print("time to write output file", time.time()-writing_outfile)

#####################################################################
    def import_check(self, tol):
        old_MF = np.copy(self.tot_system.mf1RDM)
        old_global = np.copy(self.tot_system.glob1RDM)
        self.tot_system.get_frag_corr1RDM()
        self.tot_system.get_glob1RDM()
        self.tot_system.get_nat_orbs()
        self.tot_system.get_new_mf1RDM(round(self.tot_system.Nele/2))
        check = print(np.allclose(old_MF, self.tot_system.mf1RDM, tol))
        check_glob = print(
            np.allclose(old_global, self.tot_system.glob1RDM, tol))
        if check is False:
            print("MF calculation doesnt agree up to the ", tol)
            print(old_MF-self.tot_system.mf1RDM)
            quit()
        if check_glob is False:
            print("Global calculation doesnt agree up to the ", tol)
            print(old_global-self.tot_system.glob1RDM)
            quit()
#####################################################################


def applyham_wrapper(frag, delt):

    # Subroutine to call pyscf to apply FCI
    # hamiltonian onto FCI vector in dynamics
    # Includes the -1j*timestep term and the
    # addition of bath-bath terms of X-matrix
    # to embedding Hamiltonian
    # The wrapper is necessary to parallelize using
    # Pool and must be separate from
    # the class because the class includes
    # IO file types (annoying and ugly but it works)

    Xmat_sml = np.zeros([2*frag.Nimp, 2*frag.Nimp], dtype=complex)
    Xmat_sml[frag.Nimp:, frag.Nimp:] = np.copy(
        frag.Xmat[frag.bathrange[:, None], frag.bathrange])
    return -1j * delt * applyham_pyscf.apply_ham_pyscf_fully_complex(
        frag.CIcoeffs, frag.h_emb-Xmat_sml, frag.V_emb,
        frag.Nimp, frag.Nimp, 2*frag.Nimp, frag.Ecore)
#####################################################################
