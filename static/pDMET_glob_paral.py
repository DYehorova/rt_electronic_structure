import numpy as np
import time
import os
import scipy.linalg as la
import sys
#import research.hf as hf
import hf
#import research.fragment_mod as fragment_mod
import fragment_mod
sys.path.append('/storage/home/hcoda1/2/dyehorova3/research/pdmetUnrestricted/libdmet_pre_release/')
#from fci_mod import RHF
from pyscf import gto, scf, ao2mo
#from research.quad_fit import quad_fit_mu
from quad_fit import quad_fit_mu
from math import copysign
from pyscf import lib
import libdmet_solid.dmet.Hubbard as dmet
from pathos.multiprocessing import ProcessingPool as Pool
#import multiprocessing as multproc
DiisDim = 4
dc = dmet.FDiisContext(DiisDim)
adiis = lib.diis.DIIS()
adiis.space = DiisDim

class static_pdmet():

    def __init__(self, Nsites, Nele, Nfrag, nproc, impindx, h_site, V_site, U, Maxitr, mf1RDM, tol, hamtype=0, mubool=False, muhistory=False, hubb_indx=None, nelecTol = 1e-5, dmu = 0.02, step = 0.05, trust_region=2.5 ):
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
        self.history =[]
        self.output_mu = open('output_staticCont.dat', 'w')
        self.nproc=nproc 
    
    # Check the input information

       # self.check_the_input(( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, periodic ))

    # Calculate an initial mean-field Hamiltonian

        print('Calculating initial mean-field Haamiltonian')
        if (hamtype == 0): 
        #    mf1RDM = hf.hubbard_1RDM(self.Nele, h_site) 
            #mf1RDM = self.initialize_UHF(h_site, V_site)  # !!! UHF is still broken!
            #mf1RDM = np.load("mfRDM_static.npy")
            if mf1RDM is None:
                mf1RDM = self.initialize_RHF(h_site, V_site)
                self.old_glob1RDM = np.copy(mf1RDM)
            else: 
                self.old_glob1RDM = None
           #     self.old_glob1RDM = np.copy(mf1RDM) 

            #mf1RDM = np.copy(mfRDM)
        else:
         #   mf1RDM = hf.hubbard_1RDM(self.Nele, h_site)
            #mf1RDM = np.load("mfRDM_static.npy")
            #mf1RDM = np.copy(mfRDM)
            #mf1RDM = self.initialize_UHF(h_site, V_site)
            if mf1RDM is None:
                mf1RDM = self.initialize_RHF(h_site, V_site)
                self.old_glob1RDM = np.copy(mf1RDM)
            else: 
                self.old_glob1RDM = np.zeros((self.Nsites,self.Nsites))
               # self.old_glob1RDM = np.copy(mf1RDM)
        self.V_site = self.U
        #random noise
       # noise_triag = np.triu(np.random.random([Nsites, Nsites]))
       # mf1RDM += noise_triag + noise_triag.transpose() -2*np.diag(noise_triag)
 

#        print("shape of new RDM:", mf1RDM.shape)
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
# self.tot_system = system.tot_system()	
    #output file
#    self.tot_system = pDMET.static_pdmet( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mf1RDM, hubsite_indx, periodic )
        self.frag_pool = Pool(self.nproc) 
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
    #    old_glob1RDM = np.copy(self.mf1RDM)
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
                        #last_dmu = self.dmu
                        #self.Hemb_add_mu(last_dmu)

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
                        #totalNele_2 = self.get_Nele(dmu1)
                        record.append((dmu1, totalNele_2))

                        if abs((totalNele_2/self.Nele)-1.0) < self.nelecTol:
                            print('chemical potential is converged with dmu1:', dmu1)
                            self.mu = test_mu
                            #last_dmu = dmu1
                            self.history.append(record)
                            #self.Hemb_add_mu(last_dmu)
                            total_Nele = totalNele_2

                        else:
                            mus = np.array([0.0, self.dmu, dmu1])
                            Neles = np.array([totalNele_0, totalNele_1, totalNele_2])
                            dmu2 = quad_fit_mu(mus, Neles, self.Nele/2, self.step)

                            test_mu = self.mu + dmu2
                            totalNele_3 = self.corr_calc_with_mu(test_mu)
                            #totalNele_3 = self.get_Nele(dmu2)
                            record.append((dmu2, totalNele_3))

                            if abs((totalNele_3/self.Nele)-1.0) < self.nelecTol:
                                print('chemical potential is converged with dmu2:', dmu2)
                                self.mu = test_mu
                                #last_dmu = dmu2
                                self.history.append(record)
                                #self.Hemb_add_mu(last_dmu)
                                total_Nele = totalNele_3
                            else:
                                mus = np.array([0.0, self.dmu, dmu1, dmu2])
                                Neles = np.array([totalNele_0, totalNele_1, totalNele_2, totalNele_3])
                                dmu3 = quad_fit_mu(mus, Neles, self.Nele/2, self.step)

                                test_mu = self.mu + dmu3
                                totalNele_4 = self.corr_calc_with_mu(test_mu)
                                #totalNele_4 = self.get_Nele(dmu3)
                                print('mu didnt converge, final number of electrons:', totalNele_4 )
                                record.append((dmu3, totalNele_4))
                                total_Nele = totalNele_4
                                #last_dmu = dmu3
                                self.history.append(record)
                                self.mu = test_mu
                                #self.Hemb_add_mu(last_dmu)
                

                #self.mu = +last_dmu

            else:
                #print("going no mu route")
               # if(self.proc == 1):
                unparalel_time = time.time()
                for frag in self.frag_list:

                    frag.corr_calc( self.mf1RDM, self.h_site, self.V_site, self.U, self.mu, self.hamtype, self.hubb_indx, self.mubool)
                print("unparalel corr calc", time.time()-unparalel_time)
                #else:
                paralel_time=time.time()
                self.frag_list = self.frag_pool.map( self.static_corr_calc_wrapper, self.frag_list )
                print("paralelized corr calc", time.time()-paralel_time)
                self.frag_pool.close()
                quit()
     #constract a global dencity matrix from all impurities
     
            self.get_globalRDM()

     #form a new mean-field RDM

           # self.get_nat_orbs() #use natural orbitals of Global RDM to get a new MF 1RDM
            #self.get_new_mfRDM(int(self.Nele/2))


     #check the difference between new and old mean field RDM

    #        if( itr > 0 ):
     #           dif = np.linalg.norm( old_glob1RDM - self.glob1RDM)
      #          if( dif < self.tol ):
       #             conv = True
        #            break

            #old_glob1RDM = np.copy( self.glob1RDM )

     #DIIS routine

            if np.allclose(old_glob1RDM, np.zeros((self.Nsites, self.Nsites)), rtol=0, atol=1e-12) is True:
                old_glob1RDM = np.copy(self.glob1RDM)
                print("old global", old_glob1RDM)
                print("global", self.glob1RDM)
                old_E = None
 
            if itr >= self.DiisStart:
                #self.mf1RDM = adiis.update(self.mf1RDM)
                self.glob1RDM = adiis.update(self.glob1RDM)
                #print("doing DIIS")
                #self.get_nat_orbs()
                #self.get_new_mfRDM(int(self.Nele/2))
            dif = np.linalg.norm( self.glob1RDM - old_glob1RDM)
            #dif = np.linalg.norm( self.mf1RDM - old_mf1RDM)
            dVcor_per_ele = self.max_abs(dif)
            

            #copy over old global 1RDM
            old_glob1RDM = np.copy( self.glob1RDM )
            #old_mf1RDM = np.copy( self.mf1RDM)
            self.get_nat_orbs()
            self.get_new_mfRDM(int(self.Nele/2))

            self.get_DMET_E()
            print( 'Final DMET energy =', self.DMET_E )
            print('Energy per site for U=', self.U, 'is:', (self.DMET_E/self.Nsites))
            total_Nele = self.just_Nele() 
            if itr >= 1:
                self.calc_data(itr, dif, total_Nele)

            if old_E is None:
                old_E = np.copy(self.DMET_E)

            dE = self.DMET_E - old_E
            old_E = np.copy(self.DMET_E)
         #   if dVcor_per_ele < 1.0e-5 and dif < self.tol :
          #      conv = True
           #     break

           # if( np.mod( itr, self.Maxitr/10 ) == 0 and itr >0 ):
            #    print('Finished DMET Iteration', itr)
             #   print('Current difference in global 1RDM =', dif)
              #  print('vcore=', dVcor_per_ele)
              #  print()
               # self.calc_data(itr, dif, total_Nele)

            print("difference", dif)
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
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print('total_time', total_time)
            self.output_mu.close()
            quit()
        self.frag_pool.close()
        np.save("mf1RDMN_12Nimp_3", self.mf1RDM)
#        CI_list = []
#        rotmat_list = []
#        for frag in self.frag_list:    
#            CI_list.append(frag.rotmat)
#            rotmat_list.append(frag.CIcoeffs)
#        np.save("CIN12Nimp3", CI_list)
#        np.save("rotmatN12Nimp3", rotmat_list)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        print('total_time', total_time)
        self.output_mu.close()
        #self.output_mf1RDM.close()
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
    
#        print ("read new line") 
        mf.kernel()
 #       print("finished UHF successfully")
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
    
#        print ("read new line") 
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

    #            print('q', q)
     #           print('self.site_to_frag_list', len(self.site_to_frag_list))
      #          print('self.frag_list', len(self.frag_list))
       #         print('self.site_to_frag_list[q]', self.site_to_frag_list[q])

                #index corresponding to the impurity and bath range in the rotation matrix for each fragment
                #rotation matrix order:(sites) x (impurity, virtual, bath, core)

                pindx = np.r_[ :pfrag.Nimp, pfrag.last_virt : pfrag.last_bath ]
                qindx = np.r_[ :qfrag.Nimp, qfrag.last_virt : qfrag.last_bath ]

                self.glob1RDM[p,q] = 0.5 * np.linalg.multi_dot([pfrag.rotmat[p,pindx], pfrag.corr1RDM, pfrag.rotmat[q,pindx].conj().T])

                self.glob1RDM[p,q] += 0.5 * np.linalg.multi_dot([qfrag.rotmat[p,pindx], qfrag.corr1RDM, qfrag.rotmat[q,pindx].conj().T])

                if( p != q ): #forcing Hermiticity
                    self.glob1RDM[q,p] = np.conjugate( self.glob1RDM[p,q] )
        trace1RDM = self.glob1RDM.trace()
#        print("trace of global RDM", trace1RDM)

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
        np.savetxt( self.output_mu, output.reshape(1, output.shape[0]), fmt_str )
        self.output_mu.flush()
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
            #print('fragment Energy', frag.Efrag)
            self.DMET_E += np.real( frag.Efrag ) #discard what should be numerical error of imaginary part
##########################################################
    def static_corr_calc_wrapper(self, frag ):
        frag.corr_calc( self.mf1RDM, self.h_site, self.V_site, self.U, self.mu, self.hamtype, self.hubb_indx, self.mubool)
        return frag

