# test change
import numpy as np
import feb_8_update.static.pDMET_glob as static_driver
import feb_8_update.static.transition_paral as transition_driver
import feb_8_update.scripts.make_hams as make_hams
import feb_8_update.dynamics.dynamics_driver_optimize as dynamic_driver

# set up system and static pdmet parameters
boundary = False
NL = 33
NR = 32
Ndots = 1
Nimp = 3
Nsites = NL+NR+Ndots
Nele = Nsites
Nfrag = int(Nsites/Nimp)
impindx = []
for i in range(Nfrag):
    impindx.append(np.arange(i*Nimp, (i+1)*Nimp))

mubool = True
muhistory = True
Maxitr = 100000
tol = 1e-7
mf1RDM = None
# mf1RDM = np.load()

# hamiltonian parameters
Full = True

imp_type = 'Single'
# imp_type = 'Multi'
U = 0
Vg = 0.0
Vbias = 0.0
if imp_type == 'Single':
    hubb_indx = np.array([0])
    t_implead = 0.4
    tleads = 1.0
    h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace(
        NL, NR, Vg, U, t_implead, Vbias, tleads, Full)
else:
    hubb_indx = np.arange(NL, NL+Nimp)
    timp = 1.0
    tleads = 1.0
    timplead = 0.4
    h_site_multi, V_site_multi = \
        make_hams.make_ham_multi_imp_anderson_realspace(
            Ndots, NL, NR, Vg, U, timp, timplead, Vbias, boundary, Full)

# Full = True
Full = False

if Full is True:
    hamtype = 0
else:
    hamtype = 1

# run static calculation
static = static_driver.static_pdmet(
    Nsites, Nele, Nfrag, impindx, h_site,
    V_site, U, Maxitr, mf1RDM, tol,
    hamtype, mubool, muhistory, hubb_indx)

static.kernel()
# transfer variables from static code to dynamics
system = transition_driver.transition(
    static, Nsites, Nele, Nfrag, impindx,
    h_site, V_site, hamtype, hubb_indx, boundary)

# dynamics variables
delt = 0.001
Nstep = 10000
Nprint = 10
init_time = 0.0
dG = 1e-6
nproc = 1
integrator = 'rk4'
# set up dynamics hamiltonian
U = 0.0
Full = False
Vg = 0.0
Vbias = -0.001
if imp_type == 'Single':
    t_implead = 0.4
    tleads = 1.0
    h_site, V_site = make_hams.make_ham_single_imp_anderson_realspace(
        NL, NR, Vg, U, t_implead, Vbias, tleads, Full)
else:
    timp = 1.0
    tleads = 1.0
    timplead = 0.4
    h_site_multi, V_site_multi = \
        make_hams.make_ham_multi_imp_anderson_realspace(
            Nimp, NL, NR, Vg, U, timp, timplead, Vbias, boundary, Full)

# run dynamics
dynamics = dynamic_driver.dynamics_driver(
    h_site, V_site, hamtype, system, delt, dG, Nstep, nproc, Nprint,
    integrator, hubb_indx, init_time)

dynamics.kernel()
