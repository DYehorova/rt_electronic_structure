import numpy as np
import tdhf
import sys
import os
sys.path.append('/storage/home/hcoda1/2/dyehorova3/p-jkretchmer3-0/baskedup/PaceCopy/dynamics/globNODynamics/')
import real_time_elec_structureGN.projected_dmet.hartreefock as hartreefock
sys.path.append('/storage/home/hcoda1/2/dyehorova3/research')
import make_ham
import real_time_elec_structureGN.scripts.make_hams as make_hams
boundary = False
NL     = 32
NR     = 31
Ndots  = 1

Nsites = NL+NR+Ndots
Nele   = Nsites

t  = 0.4
Vg = 0.0
Vbias = 0.0
timp     = 1.0
tleads  = 1.0
timplead = 0.4

Full    = True

delt   = 0.001
Nstep  = 50000
Nprint = 1

#Initital Static Calculation
U     = 0.0
Vbias = 0.0
noise = None
halfU = False
Vbias_multi = 0.0
Vbias_single=0.00

#h_site_single, V_site_single = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias_single, tleads, Full  )
#h_site, V_site = make_ham.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias_multi, noise, tleads, halfU, boundary, Full) 
h_site_multi, V_site_multi = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias_multi, boundary = False, Full = True)
#print("h site single:", h_site_single)
#print("h site multi:", h_site_multi)
#print("vsite single", V_site_single)
#print("vsite multi", V_site_multi)
U     = 0.0
#Vbias_single = -0.001
Vbias_multi = 0.001
Full = False
mf1RDM = hartreefock.interactive_RHF( Nele, h_site_multi, V_site_multi )

#mf1RDM_multi = hartreefock.interactive_RHF( Nele, h_site_multi, V_site_multi )

#print(np.allclose(mf1RDM_single, mf1RDM_multi, rtol=0, atol=1e-10))
#print(mf1RDM_single-mf1RDM_multi)
#print(np.allclose(V_site_single, V_site_multi, rtol=0, atol=1e-10))
#print(V_site_single-V_site_multi)
#print(np.allclose(h_site_single, h_site_multi, rtol=0, atol=1e-10))
#print(h_site_single-h_site_multi)

#h_site, V_site = make_ham.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias, noise, tleads, halfU, boundary, Full) 
#h_site_multi, V_site_multi = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias_multi, boundary = False, Full = True)
#print("h site single:", h_site_single)
#print("h site multi:", h_site_multi)
#print("vsite single", V_site_single)
#print("vsite multi", V_site_multi)
#quit()

#mf1RDM = hartreefock.interactive_RHF( Nele, h_site, V_site )

#Dynamics Calculation
#U     = 0.0
#Vbias = -0.001
#Vbias = 0.001
#Full = False
#h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias_single, tleads, Full  )
#h_site, V_site = make_ham.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias, noise, tleads, halfU, boundary, Full) 
h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace( Ndots, NL, NR, Vg, U, timp, timplead, Vbias_multi)
tdhf = tdhf.tdhf( Nsites, Nele, h_site, mf1RDM, delt, Nstep, Nprint )

tdhf.kernel()
