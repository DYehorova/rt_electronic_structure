import numpy as np
import time
import scipy.linalg as la
import feb_8_update.static.hf as hf
import feb_8_update.static.fragment_mod as fragment_mod
from pyscf import gto, scf, ao2mo
from feb_8_update.static.quad_fit import quad_fit_mu
from math import copysign
from pyscf import lib
DiisDim = 4
adiis = lib.diis.DIIS()
adiis.space = DiisDim


class static_pdmet():

    def __init__(self, Nsites, Nele, Nfrag, impindx, h_site, V_site, U, Maxitr,\
                 mf1RDM, tol, hamtype=0, mubool=False, muhistory=False, hubb_indx=None,\
                 nelecTol = 1e-5, dmu = 0.02, step = 0.05, trust_region=2.5):
        """
        Nsites    - total number of sites (or basis functions) in total system
        Nele      - total number of electrons
        Nfrag     - total number of fragments for DMET calculation
        impindx   - a list of numpy arrays containing the impurity indices for each fragment
        h_site    - 1 e- hamiltonian in site-basis for total system
        V-site    - 2 e- hamiltonian in site-basis for total system
        mubool    - boolean switching between using a global chem potential to optimize DMET # of electrons or not
        muhistory - boolean switch for chemical potential fitting
        (if true = use historic information, if false = use quadratic fit and linear regression)
        maxitr    - max number of DMET iterations#maxitr  - max number of DMET iterations
        tol       - tolerance for difference in 1 RDM during DMET cycle
        U         - Hubbard constant for electron interactions
        """
        print()
        print('***************************************')
        print('         INITIALIZING PDMET          ')
        print('***************************************')
        print()

        self.mubool = mubool
        self.muhistory = muhistory
        self.trust_region = trust_region
        self.dmu = dmu
        self.step = step
        self.Maxitr = Maxitr
        self.tol = tol
        self.nelecTol = nelecTol
        self.h_site = h_site
        self.V_site = V_site
        self.hamtype = hamtype
        self.hubb_indx = hubb_indx
        self.U = U
        self.Nsites = Nsites
        self.Nele = Nele
        self.Nfrag = Nfrag
        self.mu = 0
        self.DiisStart = 4
        self.DiisDim = 4
        self.history = []

    # Calculate an initial mean-field Hamiltonian

        print('Calculating initial mean-field Haamiltonian')
        if (hamtype == 0):
            # mf1RDM = hf.hubbard_1RDM(self.Nele, h_site)
            # mf1RDM = self.initialize_UHF(h_site, V_site)
            if mf1RDM is None:
                mf1RDM = self.initialize_RHF(h_site, V_site)
                self.old_glob1RDM = np.copy(mf1RDM)
            else:
                self.old_glob1RDM = np.copy(mf1RDM)

        else:
            # mf1RDM = hf.hubbard_1RDM(self.Nele, h_site)
            # mf1RDM = self.initialize_UHF(h_site, V_site)
            if mf1RDM is None:
                mf1RDM = self.initialize_RHF(h_site, V_site)
                self.old_glob1RDM = np.copy(mf1RDM)
            else:
                self.old_glob1RDM = np.copy(mf1RDM)

        self.V_site = self.U

        # random noise
        # noise_triag = np.triu(np.random.random([Nsites, Nsites]))
        # mf1RDM += noise_triag + noise_triag.transpose() -2*np.diag(noise_triag)

        self.mf1RDM = mf1RDM

    # Initialize the system from mf 1RDM and fragment information
        print('Initialize fragment information')

        self.frag_list = []
        for i in range(Nfrag):
            self.frag_list.append(fragment_mod.fragment(impindx[i], Nsites, Nele, hubb_indx))

    # list that takes site index and gives fragmnt index corresponding to that site
        self.site_to_frag_list = []
        self.site_to_impindx = []
        for i in range(Nsites):
            for ifrag, array in enumerate(impindx):
                if (i in array):
                    self.site_to_frag_list.append(ifrag)
                    self.site_to_impindx.append( np.argwhere(array==i)[0][0] )
    # output file
        self.file_output = open( 'output.dat', 'w' )
##########################################################

    def kernel(self):

        # Initialize the system from mf 1RDM and fragment information

        print()
        print('***************************************')
        print('         BEGIN PDMET CALCULATION       ')
        print('***************************************')

    # DMET outer loop
        start_time = time.perf_counter()
        last_dmu = 0.0
        dVcor_per_ele = None
        conv = False
        old_E = 0.0
        old_mf1RDM =  np.copy(self.mf1RDM)
        old_glob1RDM = np.copy(self.old_glob1RDM)

        for itr in range(self.Maxitr):
            print('itr', itr)

    # embedding calculation
            if (self.mubool):
                corr1RDM_col = []
                E_FCI_col = []
                # do correlation calculation and add the self.mu to the H_emb
                print("starting Nele_0 calulation")
                totalNele_0 = self.corr_calc_with_mu(self.mu)
                #totalNele_0 = self.get_Nele(self.mu)
                record = [(0.0, totalNele_0)]

                print('####DONE WITH FIRST FCI LOOP####totalNele_0:', totalNele_0)

                if abs((totalNele_0/self.Nele)-1.0) < self.nelecTol:
                    print('chemical potential fitting is unnecessary')
                    total_Nele = totalNele_0
                    self.history.append(record)

                else:
                    if (self.muhistory):
                        # predict from  historic information
                        temp_dmu = self.predict(totalNele_0, self.Nele)
                        print('temp_dmu from prediction:', temp_dmu)
                        if temp_dmu is not None:
                            self.dmu = temp_dmu
                            step = abs(self.dmu) * self.trust_region
                        else:
                            self.dmu = abs(self.dmu) * (-1 if (totalNele_0 > self.Nele) else 1)
                    else:
                        self.dmu = abs(self.dmu) * (-1 if (totalNele_0 > self.Nele) else 1)
                    print("chemical potential dmu after 1st approximation", self.dmu)

                   # totalNele_1 = self.get_Nele(self.dmu)
                    test_mu = self.mu + self.dmu
                    totalNele_1 = self.corr_calc_with_mu(test_mu)
                    record.append((self.dmu, totalNele_1))

                    if abs((totalNele_1/self.Nele)-1.0) < self.nelecTol:
                        print('chemical potential is converged with dmu:', self.dmu)
                        self.history.append(record)
                        self.mu = test_mu
                        total_Nele = totalNele_1

                    else:
                        Neleprime = (totalNele_1 - totalNele_0) /self.dmu
                        dmu1 = (self.Nele - totalNele_0) / Neleprime
                        if abs(dmu1) > self.step:
                            print('extrapolation dmu', dmu1, 'is greater then the step ', self.step)
                            dmu1_tmp = copysign(self.step, dmu1)
                            self.step = min(abs(dmu1), 0.25)
                            dmu1 = dmu1_tmp

                        test_mu = self.mu + dmu1
                        totalNele_2 = self.corr_calc_with_mu(test_mu)
                        record.append((dmu1, totalNele_2))

                        if abs((totalNele_2/self.Nele)-1.0) < self.nelecTol:
                            print('chemical potential is converged with dmu1:', dmu1)
                            self.mu = test_mu
                            self.history.append(record)
                            total_Nele = totalNele_2

                        else:
                            mus = np.array([0.0, self.dmu, dmu1])
                            Neles = np.array([totalNele_0, totalNele_1, totalNele_2])
                            dmu2 = quad_fit_mu(mus, Neles, self.Nele/2, self.step)

                            test_mu = self.mu + dmu2
                            totalNele_3 = self.corr_calc_with_mu(test_mu)
                            record.append((dmu2, totalNele_3))

                            if abs((totalNele_3/self.Nele)-1.0) < self.nelecTol:
                                print('chemical potential is converged with dmu2:', dmu2)
                                self.mu = test_mu
                                self.history.append(record)
                                total_Nele = totalNele_3
                            else:
                                mus = np.array([0.0, self.dmu, dmu1, dmu2])
                                Neles = np.array([totalNele_0, totalNele_1, totalNele_2, totalNele_3])
                                dmu3 = quad_fit_mu(mus, Neles, self.Nele/2, self.step)

                                test_mu = self.mu + dmu3
                                totalNele_4 = self.corr_calc_with_mu(test_mu)
                                print('mu didnt converge, final number of electrons:', totalNele_4 )
                                record.append((dmu3, totalNele_4))
                                total_Nele = totalNele_4
                                self.history.append(record)
                                self.mu = test_mu



            else:
                print("going no mu route")

                for frag in self.frag_list:
                    frag.corr_calc( self.mf1RDM, self.h_site, self.V_site, self.U, self.mu, self.hamtype, self.hubb_indx, self.mubool)
                    print("done with core calculation for fragment", frag)

     # constract a global dencity matrix from all impurities
            self.get_globalRDM()

     # DIIS routine


            if itr >= self.DiisStart:
                #self.mf1RDM = adiis.update(self.mf1RDM)
                self.glob1RDM = adiis.update(self.glob1RDM)
            #dif = (self.glob1RDM - old_glob1RDM)
            dif = np.linalg.norm( self.glob1RDM - old_glob1RDM)
            #dif = np.linalg.norm( self.mf1RDM - old_mf1RDM)
            dVcor_per_ele = self.max_abs(dif)

            old_glob1RDM = np.copy(self.glob1RDM)
            #old_mf1RDM = np.copy(self.mf1RDM)

    # getting natural orbitals from eigenvectors of the global rdm
            self.get_nat_orbs()
            self.get_new_mfRDM(int(self.Nele/2))
            self.get_DMET_E()
            total_Nele = self.just_Nele()

            if old_E is None:
                old_E = np.copy(self.DMET_E)

            dE = self.DMET_E - old_E
            old_E = np.copy(self.DMET_E)
         #   if dVcor_per_ele < 1.0e-5 and dif < self.tol :
          #      conv = True
           #     break

            if( np.mod( itr, self.Maxitr/100 ) == 0 and itr >0 ):
                print('Finished DMET Iteration', itr)
                print('Current difference in global 1RDM =', dif)
                print('vcore=', dVcor_per_ele)
                self.calc_data(itr, dif, total_Nele)

            print("difference", dif, "dVcor", dVcor_per_ele, "dE", dE)
            if dVcor_per_ele < self.tol and abs(dE) < 1.0e-6:
                conv = True
                break


        print( 'Final DMET energy =', self.DMET_E )
        print('Energy per site for U=', self.U, 'is:', (self.DMET_E/self.Nsites))
        if( conv ):
            print('DMET calculation succesfully converged in',itr,'iterations')
            print('Final difference in global 1RDM =',dif)
            print()

        else:
            print('WARNING: DMET calculation finished, but did not converge in', self.Maxitr, 'iterations')
            print('Final difference in global 1RDM =',dif)

        np.save("mf1RDMN12Nimp3", self.mf1RDM)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print('total_time', total_time)
        self.file_output.close()
##########################################################

    def max_abs(self, x ):
        #Equivalent to np.max(np.abs(x)), but faster.
        if np.iscomplexobj(x):
            return np.abs(x).max()
        else:
            return max(np.max(x), abs(np.min(x)))
##########################################################

    def initialize_UHF( self, h_site, V_site ):

        Norbs = self.Nele
        mol = gto.M()
        mol.nelectron = self.Nele
        mol.imncore_anyway = True
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h_site
        mf.get_ovlp = lambda *args: np.eye(Norbs)
        mf._eri = ao2mo.restore(8, V_site, Norbs)
    #    evals, h = np.linalg.eigh(h_site)
   #     mf.init_guess = h

        mf.kernel()
        mfRDM = mf.make_rdm1()

        return mfRDM
##########################################################
    def initialize_RHF( self, h_site, V_site ):

        Norbs = self.Nele
        mol = gto.M()
        mol.nelectron = self.Nele
        mol.imncore_anyway = True
        mf = scf.RHF(mol)
        print(Norbs)
        print(V_site)
        mf.get_hcore = lambda *args: h_site
        mf.get_ovlp = lambda *args: np.eye(Norbs)

        mf._eri = ao2mo.restore(8, V_site, Norbs)
    #    evals, h = np.linalg.eigh(h_site)
   #     mf.init_guess = h
        mf.kernel()
        mfRDM = mf.make_rdm1()
        print("succesfully finished rhf")

        return mfRDM

##########################################################
    def get_globalRDM( self ):
        #get RDM from all fragments
        #need to update rotation matrices and correlated 1 RDMs

        #initialize glodal 1RDM to be complex if rotation matrix or correlated 1RDM is complex
        #if(np.iscomplexobj( self.frag_list[0].rotmat  ) or  np.iscomplexobj( self.frag_lsit[0].corr1RDM  ) ):
         #   self.glob1RDM = np.zeros( [self.Nsites, self.Nsites ], dtype=complex )

        self.glob1RDM = np.zeros( [self.Nsites, self.Nsites] )

        #form the global 1RDM forcing hermiticity
        self.globalRDMtrace = 0
        for p in range(self.Nsites):
            for q in range(p, self.Nsites):

               #fragment associated with sites q and p
                pfrag = self.frag_list[self.site_to_frag_list[p] ]
                qfrag = self.frag_list[self.site_to_frag_list[q] ]

#                #index corresponding to the impurity and bath range in the rotation matrix for each fragment
#                #rotation matrix order:(sites) x (impurity, virtual, bath, core)
#
                pindx = np.r_[ :pfrag.Nimp, pfrag.last_virt : pfrag.last_bath ]
                qindx = np.r_[ :qfrag.Nimp, qfrag.last_virt : qfrag.last_bath ]

                self.glob1RDM[p,q] = 0.5 * np.linalg.multi_dot([pfrag.rotmat[p,pindx], pfrag.corr1RDM, pfrag.rotmat[q,pindx].conj().T])

                self.glob1RDM[p,q] += 0.5 * np.linalg.multi_dot([qfrag.rotmat[p,qindx], qfrag.corr1RDM, qfrag.rotmat[q,qindx].conj().T])

                if( p != q ): #forcing Hermiticity
                    self.glob1RDM[q,p] = np.conjugate( self.glob1RDM[p,q] )
        trace1RDM = self.glob1RDM.trace()
        print("trace of global RDM", trace1RDM)

        #print('global RDM',  self.glob1RDM)
#        Nimp = self.frag_list[0].Nimp
#        if( np.iscomplexobj( self.frag_list[0].rotmat ) or np.iscomplexobj( self.frag_list[0].corr1RDM ) ):
#            rotmat_unpck = np.zeros( [self.Nsites,2*Nimp,self.Nsites], dtype=complex )
#            corr1RDM_unpck = np.zeros( [2*Nimp,self.Nsites], dtype=complex )
#        else:
#            rotmat_unpck = np.zeros( [self.Nsites,2*Nimp,self.Nsites] )
#            corr1RDM_unpck = np.zeros( [2*Nimp,self.Nsites] )
#        for q in range(self.Nsites):
#            frag = self.frag_list[ self.site_to_frag_list[q] ]
#            qimp = self.site_to_impindx[q]
#            actrange = np.concatenate( (frag.imprange,frag.bathrange) )
#            rotmat_unpck[:,:,q] = np.copy( frag.rotmat[:,actrange] )
#            corr1RDM_unpck[:,q] = np.copy( frag.corr1RDM[:,qimp] )
#        tmp  = np.einsum( 'paq,aq->pq', rotmat_unpck, corr1RDM_unpck )
#        self.glob1RDM = 0.5*( tmp + tmp.conj().T )
#        #print(np.allclose(self.glob1RDM_Josh, self.glob1RDM, 10-10))
        #print(self.glob1RDM_Josh-self.glob1RDM)


##########################################################


    def predict(self, nelec, target):
        """
        assume the chemical potential landscape more
        or less the same for revious fittings.
        the simplest thing to do is predicting
        a dmu from each previous fitting, and compute
        a weighted average. The weight should prefer
        lattest runs, and prefer the fittigs that
        has points close to current and target number of electrons

        """
        from math import sqrt, exp
        vals = []
        weights = []

        # hyperparameters
        damp_factor = np.e
        sigma2, sigma3 = 0.00025, 0.0005

        for dmu, record in enumerate(self.history):
            # exponential
            weight = damp_factor ** (dmu+1-len(self.history))

            if len(record) == 1:
                val, weight = 0., 0.
                continue

            elif len(record) == 2:
                # fit a line
                (mu1, n1), (mu2, n2) = record
                slope = (n2 - n1)/(mu2 - mu1)
                val = (target - nelec) / slope
                # weight factor
                metric = min(
                        (target - n1)**2 + (nelec - n2)**2,
                        (target - n2)**2 + (nelec - n1)**2)
                # Gaussian weight
                weight *= exp(- 0.5 * metric / sigma2)

            elif len(record) == 3:
                # check to make sure that the data is monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(record)
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0., 0.
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous

                a, b, c = np.dot(la.inv(np.asarray([
                    [mu1**2, mu1, 1],
                    [mu2**2, mu2, 1],
                    [mu3**2, mu3, 1]
                ])), np.asarray([n1, n2, n3]).reshape(-1, 1)).reshape(-1)

                # if the parabola is not monotonic, use linear interpolation instead
                if mu1 < -0.5*b/a < mu3:
                    def find_mu(n):
                        if n < n2:
                            slope = (n2-n1) / (mu2-mu1)
                        else:
                            slope = (n2-n3) / (mu2-mu3)
                        return mu2 + (n-n2) / slope

                else:
                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n-n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n-n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c-n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n1)**2 + (nelec-n3)**2,
                        (target-n2)**2 + (nelec-n1)**2,
                        (target-n2)**2 + (nelec-n3)**2,
                        (target-n3)**2 + (nelec-n1)**2,
                        (target-n3)**2 + (nelec-n2)**2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            else:  # len(record) >= 4:
                # first find three most nearest points
                mus, nelecs = zip(*record)
                mus = np.asarray(mus)
                nelecs = np.asarray(nelecs)
                delta_nelecs = np.abs(nelecs - target)
                idx_dN = np.argsort(delta_nelecs, kind='mergesort')
                mus_sub = mus[idx_dN][:3]
                nelecs_sub = nelecs[idx_dN][:3]

                # we need to check data sanity: should be monotonic
                (mu1, n1), (mu2, n2), (mu3, n3) = sorted(zip(mus_sub, nelecs_sub))
                if (not n1 < n2) or (not n2 < n3):
                    val, weight = 0., 0.
                    continue

                # parabola between mu1 and mu3, linear outside the region
                # with f' continuous
                a, b, c = np.dot(la.inv(np.asarray([
                    [mu1**2, mu1, 1],
                    [mu2**2, mu2, 1],
                    [mu3**2, mu3, 1]
                ])), np.asarray([n1,n2,n3]).reshape(-1,1)).reshape(-1)

                # if the parabola is not monotonic, use linear interpolation instead
                if mu1 < -0.5*b/a < mu3:
                    def find_mu(n):
                        if n < n2:
                            slope = (n2-n1) / (mu2-mu1)
                        else:
                            slope = (n2-n3) / (mu2-mu3)
                        return mu2 + (n-n2) / slope

                else:
                    def find_mu(n):
                        if n < n1:
                            slope = 2 * a * mu1 + b
                            return mu1 + (n-n1) / slope
                        elif n > n3:
                            slope = 2 * a * mu3 + b
                            return mu3 + (n-n3) / slope
                        else:
                            return 0.5 * (-b + sqrt(b**2 - 4 * a * (c-n))) / a

                val = find_mu(target) - find_mu(nelec)
                # weight factor
                metric = min(
                        (target-n1)**2 + (nelec-n2)**2,
                        (target-n1)**2 + (nelec-n3)**2,
                        (target-n2)**2 + (nelec-n1)**2,
                        (target-n2)**2 + (nelec-n3)**2,
                        (target-n3)**2 + (nelec-n1)**2,
                        (target-n3)**2 + (nelec-n2)**2,
                )
                weight *= exp(-0.5 * metric / sigma3)

            vals.append(val)
            weights.append(weight)





        if np.sum(weights) > 1e-3:
            dmu = np.dot(vals, weights) / np.sum(weights)
            if abs(dmu) > 0.5:
                dmu = copysign(0.5, dmu)
            print("adaptive chemical potential fitting, dmu =", dmu)
            return dmu
        else:
            print("adaptive chemical potential fitting not used")
            return None
##########################################################


    def corr_calc_with_mu(self, mu):
        totalNele = 0.0
        for frag in self.frag_list:
            fragNele = frag.corr_calc( self.mf1RDM, self.h_site, self.V_site, self.U, mu, self.hamtype, self.hubb_indx, self.mubool)
            totalNele += fragNele
        print('total electrons:', totalNele)
        return totalNele
##########################################################


    def get_Nele( self, mu ):
        totalNele = 0.0
        for frag in self.frag_list:
            frag.add_mu_Hemb( mu )
            frag.solve_GS( self.U )
            frag.get_corr1RDM()
            fragNele = frag.nele_in_frag()
            print('fragNele',fragNele)
            totalNele += fragNele
            new_mu = -1*mu
            frag.add_mu_Hemb( new_mu )#to make sure Im not changing H_emb with wrong guess for dmu
        print('total electrons:', totalNele)
        return totalNele
##########################################################

    def just_Nele(self):
        totalNele = 0.0
        for frag in self.frag_list:
            fragNele = frag.nele_in_frag()
            totalNele += fragNele
        return totalNele
##########################################################

    def get_nat_orbs( self ):

        NOevals, NOevecs = np.linalg.eigh ( self.glob1RDM )

        #Re-order such that eigenvalues are in descending order
        self.NOevals = np.flip(NOevals)
        self.NOevecs = np.flip(NOevecs,1)
##########################################################


    def get_new_mfRDM( self, NOcc):

        #get mf 1RDM from the first Nocc natural orbitals of the global rdm
        #(natural orbitals with the highest occupation)

        NOcc = self.NOevecs[ :, :NOcc ]
        self.mf1RDM = 2.0 * np.dot( NOcc, NOcc.T.conj() )
##########################################################


    def get_frag_corr12RDM( self ):
        #correlated 1 RDM for each fragment
        for frag in self.frag_list:
            frag.get_corr12RDM()
##########################################################


    def get_frag_Hemb( self ):
        #Hamiltonian for each fragment

        for frag in self.frag_list:
            frag.get_Hemb( self.h_site, self.V_site, self.U, self.hamtype, self.hubb_indx )
##########################################################


    def Hemb_add_mu( self, mu ):
         #Hamiltonian for each fragment

        for frag in self.frag_list:
            frag.add_mu_Hemb( mu )
##########################################################


    def calc_data( self, itr, dif, total_Nele ):
        fmt_str = '%20.8e'
        output = np.zeros(6+self.Nsites)
        output[0] = itr
        output[1] = self.mu
        output[2] = dif
        output[3] = total_Nele
        output[4] = self.DMET_E/self.Nsites
        output[5:5+self.Nsites] = self.NOevals
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()
        #print("shape of RDM:",self.mf1RDM.shape)
        np.save("mfRDM_static", self.mf1RDM)
        np.save("globRDM_static", self.glob1RDM)
        CI = []
        rotmat = []
        for frag in self.frag_list:
            CI.append(np.copy(frag.CIcoeffs))
            rotmat.append(np.copy(frag.rotmat))

       # print("CI", CI)
       # print("rotmat", rotmat)


        np.save("CI_ststic", CI)
# self.output_mf1RDM = open( 'output_mf1RDM_1DHubb480U1Im2.dat', 'w')
        #for row in self.mf1RDM:
            #np.savetxt(self.output_mf1RDM, row)
#can be loaded back by using original_array = np.loadtxt("test.txt").reshape(row, col)
        #self.output_mf1RDM.flush()
       # self.output_mf1RDM.close()
##########################################################


    def get_DMET_E( self ):
        self.get_frag_Hemb()
        self.get_frag_corr12RDM()

        self.DMET_E = 0.0
        for frag in self.frag_list:
            frag.get_frag_E()
            # print('fragment Energy', frag.Efrag)
            self.DMET_E += np.real( frag.Efrag ) #discard what should be numerical error of imaginary part
