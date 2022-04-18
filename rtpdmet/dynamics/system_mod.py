# Define a class for the total system
import numpy as np

# ####### TOTAL SYSTEM CLASS #######


class system():

    #####################################################################

    def __init__(self, Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype=0,
                 mf1RDM=None, hubsite_indx=None, periodic=False, mu=0):

        # initialize total system variables
        self.Nsites = Nsites
        # total number of sites (or basis functions) in total system
        self.Nele = Nele
        # total number of electrons
        self.Nfrag = Nfrag
        # total number of fragments
        self.mu = mu
        # global chemical potential added to only impurity sites
        # in each embedding hamiltonian
        self.hamtype = hamtype
        # integer defining if using a special Hamiltonian (Hubbard or Anderson)
        self.periodic = periodic
        # boolean stating whether system is periodic (impurities are the same)

        # If running Hubbard-like model, need an array containing
        # index of all sites that have hubbard U term
        self.hubsite_indx = hubsite_indx
        if(self.hamtype == 1 and
           (hubsite_indx is None or not isinstance(hubsite_indx, np.ndarray))):
            print('ERROR: Did not specify an array of sites with U term')
            print()
            exit()

        # initialize fragment information
        # note that here impindx should be a list of numpy arrays
        # containing the impurity indices for each fragment
        # INCLUDED IN TRANSFER FILE
        # self.frag_list = []
        # for i in range(Nfrag):
            # self.frag_list.append(
            # fragment_mod.fragment( impindx[i], Nsites, Nele ) )

        # initialize list that takes site index and
        # outputs fragment index corresponding to that site
        # and separate list that outputs the index of where that
        # impurity appears in the list of impurities for that fragment
        self.site_to_frag_list = []
        self.site_to_impindx = []
        for i in range(Nsites):
            for ifrag, arr in enumerate(impindx):
                if(i in arr):
                    self.site_to_frag_list.append(ifrag)
                    self.site_to_impindx.append(np.argwhere(arr == i)[0][0])
                    # self.site_to_frag_list.append((ifrag,
                    # np.argwhere(arr==i)[0][0]))
                    # PING combines both lists into one list of tuples

        # initialize total system hamiltonian and mean-field 1RDM
        self.h_site = h_site
        self.V_site = V_site
        self.mf1RDM = mf1RDM
    #####################################################################

    def get_glob1RDM(self):
        '''
        Subroutine to obtain global 1RDM formed from all fragments
        Need to have updated rotation matrices and correlated 1RDMs
        '''

        # form global 1RDM forcing hermiticity

        # unpack necessary stuff
        # NOTE we're assuming all fragments have same number of impurities
        Nimp = self.frag_list[0].Nimp
        if(np.iscomplexobj(self.frag_list[0].rotmat)
           or np.iscomplexobj(self.frag_list[0].corr1RDM)):
            rotmat_unpck = np.zeros(
                [self.Nsites, 2*Nimp, self.Nsites], dtype=complex)
            corr1RDM_unpck = np.zeros(
                [2*Nimp, self.Nsites], dtype=complex)
        else:
            rotmat_unpck = np.zeros([self.Nsites, 2*Nimp, self.Nsites])
            corr1RDM_unpck = np.zeros([2*Nimp, self.Nsites])
        for q in range(self.Nsites):

            # fragment for site q
            frag = self.frag_list[self.site_to_frag_list[q]]

            # index within fragment corresponding to site q -
            # note that q is an impurity orbital
            qimp = self.site_to_impindx[q]

            # unpack just impurity/bath parts of rotation matrix
            actrange = np.concatenate((frag.imprange, frag.bathrange))
            rotmat_unpck[:, :, q] = np.copy(frag.rotmat[:, actrange])

            # unpack necessary portion of corr1RDM
            corr1RDM_unpck[:, q] = np.copy(frag.corr1RDM[:, qimp])

        # calculate intermediate matrix
        tmp = np.einsum('paq,aq->pq', rotmat_unpck, corr1RDM_unpck)

        # form global 1RDM
        self.glob1RDM = 0.5*(tmp + tmp.conj().T)
    #####################################################################

    def get_nat_orbs(self):
        # Subroutine to obtain natural orbitals of global 1RDM
        NOevals, NOevecs = np.linalg.eigh(self.glob1RDM)

        # Re-order such that eigenvalues are in descending order
        self.NOevals = np.flip(NOevals)
        self.NOevecs = np.flip(NOevecs, 1)
    #####################################################################

    def get_new_mf1RDM(self, Nocc):
        # Subroutine to obtain a new idempotent (mean-field) 1RDM from the
        # First Nocc natural orbitals of the global 1RDM
        # ie natural orbitals with the highest occupation

        NOocc = self.NOevecs[:, :Nocc]
        self.mf1RDM = 2.0 * np.dot(NOocc, NOocc.T.conj())
    #####################################################################

    def static_corr_calc_wrapper(self, frag):
        # Subroutine that simply calls the static_corr_calc
        # subroutine for the given fragment class
        # The wrapper is necessary to parallelize using Pool

        frag.static_corr_calc(self.mf1RDM, self.mu, self.h_site,
                              self.V_site, self.hamtype, self.hubsite_indx)
        return frag
    #####################################################################

    def corr_emb_calc(self, nproc, frag_pool):
        # Subroutine to perform full correlated calculation on each fragment
        # including transformations to embedding basis
        if(not self.periodic):
            # non-periodic: calculate each fragment separately in parallel
            if(nproc == 1):
                for frag in self.frag_list:
                    frag.static_corr_calc(self.mf1RDM, self.mu, self.h_site,
                                          self.V_site, self.hamtype,
                                          self.hubsite_indx)
            # else:
            #    frag_pool = multproc.Pool(nproc)
            #    self.frag_list = frag_pool.map(
            #    self.static_corr_calc_wrapper, self.frag_list)
            #    frag_pool.close()
            #    frag_pool.join()
        else:
            print('ERROR: Do not currently have periodic version of the code')
            exit()
    #####################################################################

    def get_frag_corr1RDM(self):
        # Subroutine to calculate correlated 1RDM for each fragment
        for frag in self.frag_list:
            frag.get_corr1RDM()
    #####################################################################

    def get_frag_corr12RDM(self):
        # Subroutine to calculate correlated 1RDM for each fragment
        for frag in self.frag_list:
            frag.get_corr12RDM()
    #####################################################################

    def get_frag_Hemb(self):
        # Subroutine to calculate embedding Hamiltonian for each fragment
        for frag in self.frag_list:
            frag.get_Hemb(
                self.h_site, self.V_site, self.hamtype, self.hubsite_indx)
    #####################################################################

    def get_frag_rotmat(self):
        # Subroutine to calculate rotation matrix
        # (ie embedding orbs) for each fragment
        for frag in self.frag_list:
            frag.get_rotmat(self.mf1RDM)
    #####################################################################

    def get_DMET_Nele(self):
        # Subroutine to calculate the number of electrons in all impurities
        # Necessary to calculate fragment 1RDMs prior to this routine
        self.DMET_Nele = 0.0
        for frag in self.frag_list:
            self.DMET_Nele += np.real(
                np.trace(frag.corr1RDM[:frag.Nimp, :frag.Nimp]))
    #####################################################################

    def get_DMET_E(self, nproc):
        # Subroutine to calculate the DMET energy

        self.get_frag_Hemb()
        self.get_frag_corr12RDM()

        self.DMET_E = 0.0
        for frag in self.frag_list:
            frag.get_frag_E()
            # discard what should be numerical error of imaginary part
            self.DMET_E += np.real(frag.Efrag)
    #####################################################################

    def get_frag_iddt_corr1RDM(self):

        # Calculate the Hamiltonian commutator portion of the
        # time-dependence of correlated 1RDM for each fragment
        # ie i\tilde{ \dot{ correlated 1RDM } } using notation from notes
        # NOTE: should have 1RDM and 2RDM calculated prior to calling this

        for frag in self.frag_list:
            frag.get_iddt_corr1RDM(
                self.h_site, self.V_site, self.hamtype, self.hubsite_indx)
    #####################################################################

    def get_frag_Xmat(self, change_mf1RDM):
        # Solve for X-matrix of each fragment given current mean-field 1RDM
        # and the current time-derivative of the mean-field 1RDM
        for frag in self.frag_list:
            frag.get_Xmat(self.mf1RDM, change_mf1RDM)
    ######################################################################

    def eigvec_frag_MF_check(self):
        for frag in self.frag_list:
            frag.eigvec_MF_check(self.mf1RDM)
    ######################################################################
