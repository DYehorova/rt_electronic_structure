import numpy as np
import real_time_elec_structure.projected_dmet.static_driver as static_driver
import pyscf.fci
import real_time_elec_structure.scripts.utils as utils
import real_time_elec_structure.scripts.make_hams as make_hams

Nsites   = 6
Nele     = Nsites
Nfrag    = 3

U            = 0.0
boundary     = 1.0
Full         = False
hubsite_indx = np.arange(Nsites)

mubool   = False
periodic = False
nproc    = 1
hamtype  = 1

#N=4 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]) ]
#impindx = [ np.array([0,1]), np.array([2,3]) ]

#N=6 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]) ]
impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]) ]
#impindx = [ np.array([0,1,2]), np.array([3,4,5]) ]
#impindx = [ np.array([5,4,1]), np.array([0,3,2]) ]
#impindx = [ np.array([1,4,5]), np.array([0,2,3]) ]

#N=8 tilings
#impindx = [ np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5]), np.array([6]), np.array([7]) ]
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]) ]
#impindx = [ np.array([0,1,2,3]), np.array([4,5,6,7]) ]

#N=10 tilings
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]) ]
#impindx = [ np.array([0,1,2,3,4]), np.array([5,6,7,8,9]) ]
#impindx = [ np.array([7,8,9,3,4]), np.array([0,1,2,5,6])]
#impindx  = [ np.array([0,1]), np.array([2,3,7]), np.array([4,5,9]), np.array([6,8]) ]

#N=12 tilings
#impindx = [ np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]), np.array([10,11]) ]



#Initital Static Calculation
h_site, V_site = make_hams.make_1D_hubbard( Nsites, U, boundary, Full )
the_dmet = static_driver.static_driver( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mubool, nproc, hubsite_indx, periodic )
the_dmet.kernel()

#FCI Check for static calculation
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.conv_tol = 1e-16
cisolver.verbose = 3
h_site, V_site = make_hams.make_1D_hubbard( Nsites, U, boundary, True )
E_FCI, CIcoeffs = cisolver.kernel( h_site, V_site, Nsites, Nele )
print('E_FCI = ',E_FCI)

