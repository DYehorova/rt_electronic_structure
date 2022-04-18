import numpy as np
import pDMET_glob
import make_ham

Anderson = True
Hubbard_2D = False

Full = False
mubool = True
muhistory = True

hamtype = 1
Maxitr = 100000

U = 2
boundary = 1


if Anderson is True:
    boundary = True
    NL = 4
    NR = 4
    Ndots = 4
    Nsites = NL+NR+Ndots
    Nele = Nsites
    Nfrag = 6
    Nimp = round(Nsites/Nfrag)

    timp = 1.0
    timplead = 1.0
    tleads = 1.0
    Vg = 0.0
    hubb_indx = np.arange(NL, NL+Ndots)
    Vbias = 0.00

elif Hubbard_2D is True:
    Nx = 3
    Ny = 3
    Nx_imp = 1
    Ny_imp = 1
    Nsites = Nx * Ny
    Nele = Nsites
    Nimp = Nx_imp * Ny_imp
    Nfrag = int(Nsites/Nimp)

else:
    Nele = 6
    Nimp = 2
    Nsites = Nele
    Nfrag = int(Nsites/Nimp)
    hubb_indx = np.arange(Nsites)

# Define Fragment Indices

impindx = []
for i in range(Nfrag):
    impindx.append(np.arange(i*Nimp, (i+1)*Nimp))

# Initial MF calculation

if Anderson is True:
    h_site, V_site = make_ham.make_ham_multi_imp_anderson_realspace(
        Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full)

elif Hubbard_2D is True:
    h_site, V_site = make_ham.make_2D_hubbard(
        Nx, Ny, Nx_imp, Ny_imp, U, boundary, Full)

else:
    h_site, V_site = make_ham.make_1D_hubbard(Nsites, U, boundary, Full)

# pDMET call
dmet = pDMET_glob.static_pdmet(
    Nsites, Nele, Nfrag, impindx, h_site, V_site, U, Maxitr,
    hamtype, mubool, muhistory, hubb_indx)
dmet.kernel()
