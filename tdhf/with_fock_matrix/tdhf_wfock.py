import numpy as np
import scipy.linalg as la
import rt_electronic_structure.scripts.utils
import rt_electronic_structure.scripts.integrators

#####################################################################

class tdhf():

    #Class to perform a time-dependent hartree-fock calculation
    #Propopates under the time-dependent fock-operator for hubbard-like models

#####################################################################

    def __init__( self, Nsites, Nelec, h_site, mf1RDM, delt, Nstep, Nprint, U=None, hubsite_indx=None ):

        self.Nsites = Nsites
        self.Nelec  = Nelec
        self.h_site = h_site
        self.mf1RDM = mf1RDM
        self.delt   = delt
        self.Nstep  = Nstep
        self.Nprint = Nprint
        self.U      = U
        self.hubsite_indx = hubsite_indx

        if( U is not None and hubsite_indx is None ):
            print('ERROR: Need to specify sites containing hubbard U - hubsite_indx')
            exit()

        #Convert mean-field 1RDM to a complex array if it's not already
        if( not np.iscomplexobj( self.mf1RDM ) ):
            self.mf1RDM = self.mf1RDM.astype(complex)

        #Define output files
        self.file_output   = open( 'output.dat', 'w' )
        self.file_corrdens = open( 'corr_density.dat', 'w' )

#####################################################################

    def kernel( self ):

        current_time = 0.0

        for step in range(self.Nstep):

            #Print data before taking time-step, this always prints out data at initial time step
            if( np.mod( step, self.Nprint ) == 0 ):
                print('Writing data at step ', step, 'and time', current_time, 'for TDHF calculation')
                self.print_data( current_time )
                sys.stdout.flush()

            #Integrate the mean-field 1RDM using 4th order-runge kutta
            self.rk4()

            #Increase current_time
            current_time = (step+1)*self.delt


        #Print data at final step regardless of Nprint
        print('Writing data at step ', step+1, 'and time', current_time, 'for RT-pDMET calculation')
        self.print_data( current_time )
        sys.stdout.flush()

        #Close output files
        self.file_output.close()
        self.file_corrdens.close()

#####################################################################

    def rk4( self ):

        #4th order runge-kutta using time-dependent Fock matrix

        init_mf1RDM = np.copy( self.mf1RDM )

        k1 = self.one_step_rk4()

        self.mf1RDM = init_mf1RDM + 0.5*k1

        k2 = self.one_step_rk4()

        self.mf1RDM = init_mf1RDM + 0.5*k2

        k3 = self.one_step_rk4()

        self.mf1RDM = init_mf1RDM + 1.0*k3

        k4 = self.one_step_rk4()

        self.mf1RDM = init_mf1RDM + 1.0/6.0 * ( k1 + 2.0*k2 + 2.0*k3 + k4 )

#####################################################################

    def one_step_rk4( self ):

        #one step in runge-kutta algorithm
        #need to update mf1RDM to appropriate time-step prior to this routine

        #calculate Fock matrix
        f_site = self.calc_fock()

        #Calculate time-derivative of 1RDM
        return -1j * self.delt * utils.commutator( f_site, self.mf1RDM )

#####################################################################

    def calc_fock( self ):

        #Calculate Fock matrix, currently hard-coded for Hubbard-like models
        f_site = np.copy( self.h_site )
        if( self.hubsite_indx is not None ):
            for i in self.hubsite_indx:
                f_site[i,i] += 0.5 * self.U * self.mf1RDM[i,i]

        return f_site

#####################################################################

    def print_data( self, current_time ):
        #Subroutine to calculate and print-out observables of interest

        fmt_str = '%20.8e'

        ######## CALCULATE OBSERVABLES OF INTEREST #######

        #Calculate total energy
        Etot = np.real( 0.5 * np.einsum( 'ij,ji', ( self.h_site + self.calc_fock() ), self.mf1RDM ) )

        #Calculate total number of electrons
        Nele = np.real( np.sum( np.diag( self.mf1RDM ) ) )


        ######## PRINT OUT EVERYTHING #######

        #Print correlated density in the site basis
        cnt = 0
        corrdens = np.real( np.diag( self.mf1RDM ) )
        corrdens = np.insert( corrdens, 0, current_time )
        np.savetxt( self.file_corrdens, corrdens.reshape(1, corrdens.shape[0]), fmt_str )
        self.file_corrdens.flush()

        #Print output data
        output    = np.zeros(3)
        output[0] = current_time
        output[1] = Etot
        output[2] = Nele
        np.savetxt( self.file_output, output.reshape(1, output.shape[0]), fmt_str )
        self.file_output.flush()

#####################################################################


