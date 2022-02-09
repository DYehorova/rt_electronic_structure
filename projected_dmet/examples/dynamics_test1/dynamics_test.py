import numpy as np
import sys
#path = '/Users/dariiayehorova/real_time_elec_structure'
#sys.path.append(path)
import real_time_elec_structureNo.projected_dmet.static_driver as static_driver
import real_time_elec_structureNo.projected_dmet.dynamics_driver as dynamics_driver
import pyscf.fci
import real_time_elec_structureNo.scripts.utils as utils
import real_time_elec_structureNo.scripts.make_hams as make_hams
import time

NL     = 3
NR     = 2
Nsites = NL+NR+1
Nele   = Nsites
Nfrag  = 3

t  = 0.4
Vg = 0.0

hubsite_indx = np.arange(0,Nsites,1)

tleads  = 1.0
Full    = False

mubool  = False
if( Full ):
    hamtype = 0
else:
    hamtype = 1

nproc  = 1

delt   = 0.001
Nstep  = 5000
Nprint = 10
integ  = 'rk4'

#N=4 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]) ]
#impindx = [ np.array([0,1]), np.array([2,3]) ]

#N=6 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]) ]
impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]) ]
#impindx = [ np.array([0,1,2]), np.array([3,4,5]) ]
#impindx = [ np.array([5,4,1]), np.array([0,3,2]) ]
#impindx = [ np.array([1,4,5]), np.array([0,2,3]) ]
#impindx = [ np.array([0]), np.array([1,2,3]), np.array([4,5]) ]

#N=8 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]), np.array([6]), np.array([7]) ]
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]) ]
#impindx = [ np.array([0,1,2,3]), np.array([4,5,6,7]) ]

#N=10 tilings
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]) ]
#impindx = [ np.array([0,1,2,3,4]), np.array([5,6,7,8,9]) ]
#impindx = [ np.array([7,8,9,3,4]), np.array([0,1,2,5,6])]
#impindx  = [ np.array([0,1]), np.array([2,3,7]), np.array([4,5,9]), np.array([6,8]) ]

#N=30 tilings
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]), np.array([10,11]), np.array([12,13]), np.array([14,15]), np.array([16,17]), np.array([18,19]), np.array([20,21]), np.array([22,23]), np.array([24,25]), np.array([26,27]), np.array([28,29]) ]
#impindx = [ np.array([0,1,2,3,4]), np.array([5,6,7,8,9]), np.array([10,11,12,13,14]), np.array([15,16,17,18,19]), np.array([20,21,22,23,24]), np.array([25,26,27,28,29]) ]

#General tilings
#impindx = []
#Nimp = round(Nsites/Nfrag)
#for i in range(Nfrag):
#    impindx.append( np.arange(i*Nimp,(i+1)*Nimp) )

#Initital Static Calculation
U     = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

the_dmet = static_driver.static_driver( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mubool, nproc, hubsite_indx )
the_dmet.kernel()

##FCI Check for static calculation
#h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, True )
#cisolver = pyscf.fci.direct_spin1.FCI()
#cisolver.conv_tol = 1e-16
#cisolver.verbose = 3
#E_FCI, CIcoeffs = cisolver.kernel( h_site, V_site, Nsites, Nele )
#print('E_FCI = ',E_FCI)

#Dynamics Calculation
U     = 0.0
Vbias = -0.001
#Vbias = 0.0
h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads, Full  )

rt_dmet = dynamics_driver.dynamics_driver( h_site, V_site, hamtype, the_dmet.tot_system, delt, Nstep, Nprint, integ, nproc, hubsite_indx )
rt_dmet.kernel()

