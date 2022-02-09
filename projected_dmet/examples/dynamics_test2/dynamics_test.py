import numpy as np
import real_time_elec_structureNo.projected_dmet.static_driver as static_driver
import real_time_elec_structureNo.projected_dmet.dynamics_driver as dynamics_driver
import pyscf.fci
import real_time_elec_structureNo.scripts.utils as utils
import real_time_elec_structureNo.scripts.make_hams as make_hams
import time

NL     = 2
NR     = 2
Ndots  = 2
Nsites = NL+NR+Ndots
Nele   = Nsites
Nfrag  = 3

timp     = 1.0
timplead = 1.0
tleads   = 1.0
Vg       = 0.0
Full     = False

hubsite_indx = np.arange(NL,NL+Ndots)

mubool  = False
if( Full ):
    hamtype = 0
else:
    hamtype = 1

nproc  = 1

delt   = 0.001
Nstep  = 5000
Nprint = 1
integ  = 'rk4'

#General tilings
impindx = []
Nimp = round(Nsites/Nfrag)
for i in range(Nfrag):
    impindx.append( np.arange(i*Nimp,(i+1)*Nimp) )

#Initital Static Calculation
U     = 0.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, Full )

the_dmet = static_driver.static_driver( Nsites, Nele, Nfrag, impindx, h_site, V_site, hamtype, mubool, nproc, hubsite_indx )
the_dmet.kernel()

#Dynamics Calculation
U     = 1.0
Vbias = 0.0
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, Full )

rt_dmet = dynamics_driver.dynamics_driver( h_site, V_site, hamtype, the_dmet.tot_system, delt, Nstep, Nprint, integ, nproc, hubsite_indx )
rt_dmet.kernel()

