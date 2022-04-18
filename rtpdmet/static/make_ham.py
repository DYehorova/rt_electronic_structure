import numpy as np
from scipy.special import erf


def make_ham_multi_imp_anderson_realspace(
        Nimp, NL, NR, Vg, U, timp, timplead, Vbias, noise,
        tleads=1.0, halfU=False, boundary=False, Full=False):
    if(np.absolute(NL-NR) != 1 and np.absolute(NL-NR) != 0):
        print('ERROR:')
        print('Difference between number of sites in the left and right leads must be 0 or 1')
        exit()
    Vg = Vg * tleads
    U = U * tleads
    timp = timp * tleads
    timplead = timplead * tleads
    Vbias = Vbias * tleads
    # Initialize
    N = NL+NR+Nimp
    hmat = np.zeros([N, N])

    # Lists for indices of left-lead, impurities, and right-lead
    left_indx = np.arange(NL)
    imp_indx = np.arange(NL, NL+Nimp)
    right_indx = np.arange(NL+Nimp, N)

    # Coupling part of the 1e- terms
    # periodic boundary
    if boundary is True:
        hmat[0, N-1] = hmat[N-1, 0] = -tleads

    if noise is not None:
        hmat += np.diag(noise)
    if halfU is True:
        for imp in imp_indx:
            hmat[imp, imp] = -U/2

    # left lead
    for lead in left_indx[:-1]:
        hmat[lead, lead+1] = -tleads
        hmat[lead+1, lead] = -tleads

    # left lead - impurity
    hmat[left_indx[-1], imp_indx[0]] = -timplead
    hmat[imp_indx[0], left_indx[-1]] = -timplead

    # impurities
    for imp in imp_indx[:-1]:
        hmat[imp, imp+1] = -timp
        hmat[imp+1, imp] = -timp

    # impurity - right lead
    hmat[imp_indx[-1], right_indx[0]] = -timplead
    hmat[right_indx[0], imp_indx[-1]] = -timplead

    # right lead
    for lead in right_indx[:-1]:
        hmat[lead, lead+1] = -tleads
        hmat[lead+1, lead] = -tleads

    # Diagonal part of the 1e- terms
    # impurities
    for imp in imp_indx:
        hmat[imp, imp] = Vg
    # left lead
    for lead in left_indx:
        hmat[lead, lead] += -Vbias/2.0
    # right lead
    for lead in right_indx:
        hmat[lead, lead] += Vbias/2.0

    # Form the trivial two electron terms
    if(Full):
        Vmat = np.zeros([N, N, N, N])
        for imp in imp_indx:
            Vmat[imp, imp, imp, imp] = U
    else:
        Vmat = U

    return hmat, Vmat
####################################


def make_1D_hubbard(Nsites, U, boundary, Vbias, Full=False):

    t = 1.0

    Tmat = np.zeros((Nsites, Nsites))
    if(Full):
        Vmat = np.zeros((Nsites, Nsites, Nsites, Nsites))
    else:
        Vmat = U
    for i in range(Nsites-1):
        Tmat[i, i+1] = Tmat[i+1, i] = -t
        if(Full):
            Vmat[i, i, i, i] = U
        if i < (Nsites)/2:
            Tmat[i, i] = -Vbias/2
        else:
            Tmat[i, i] = Vbias/2
    Tmat[(Nsites-1), (Nsites-1)] = Vbias/2
    if(Full):
        Vmat[Nsites-1, Nsites-1, Nsites-1, Nsites-1] = U
    print(Tmat[(Nsites-1), (Nsites-1)])
    Tmat[0, Nsites-1] = Tmat[Nsites - 1, 0] = -t*boundary
    print(Tmat[(Nsites-1), (Nsites-1)])

    return Tmat, Vmat

####################################


def make_2D_hubbard(Nx, Ny, Nx_imp, Ny_imp, U, boundary, Full=False):
    '''Generates hopping and 2-electron matrices for a 2D-Hubbard model
    The indices are defined such that the first Nimp correspond
    to a single impurity cluster,
    the second Nimp correspond to a second cluster, etc
    Example of four 2x2 clusters for a total of 16 sites
      0   1   4   5
      2   3   6   7
      8   9   12  13
      10  11  14  15'''

    if((Nx % Nx_imp != 0) or (Ny % Ny_imp != 0)):
        print('ERROR: Impurity dimensions dont tesselate the full lattice')
        exit()

    Nsites = int(Nx * Ny)
    Nimp = int(Nx_imp * Ny_imp)
    Nclst = int(Nsites / Nimp)
    Nx_clst = int(Nx/Nx_imp)
    Ny_clst = int(Ny/Ny_imp)

    t = 1.0

    # Initialize matrices
    Tmat = np.zeros((Nsites, Nsites))
    if(Full):
        Vmat = np.zeros((Nsites, Nsites, Nsites, Nsites))
        for i in range(Nsites):
            Vmat[i, i, i, i] = U
    else:
        Vmat = None

    # Generate Tmat associated just with impurity cluster
    Tmat_imp = np.zeros((Nimp, Nimp))
    for iy in range(Ny_imp):
        for ix in range(Nx_imp):
            i = ix + Nx_imp*iy
            if(ix > 0):  # Left
                Tmat_imp[i, i-1] = -t
            if(ix < Nx_imp-1):  # Right
                Tmat_imp[i, i+1] = -t
            if(iy > 0):  # Up
                Tmat_imp[i, i-Nx_imp] = -t
            if(iy < Ny_imp-1):  # Down
                Tmat_imp[i, i+Nx_imp] = -t

    # Copy impurity cluster to all clusters
    for cpy in range(0, Nclst):
        for orb1 in range(Nimp):
            indx1 = (orb1+Nimp*cpy) % Nsites
            for orb2 in range(Nimp):
                indx2 = (orb2+Nimp*cpy) % Nsites
                Tmat[indx1, indx2] = Tmat_imp[orb1, orb2]

    # Inter-cluster interactions

    # Left and Right
    for clsty in range(Ny_clst):
        for clstx in range(Nx_clst):
            for iy in range(Ny_imp):
                i = Nx_imp * iy + (clstx + Nx_clst*clsty)*Nimp
                if(clstx > 0):
                    j = i - (Nimp - Nx_imp + 1)
                    Tmat[i, j] = Tmat[j, i] = -t
                else:  # Periodic boundary conditions
                    j = i + (Nx_imp - 1 + Nimp * (Nx_clst-1))
                    Tmat[i, j] = Tmat[j, i] = -t*boundary

    # Up and Down
    for clsty in range(Ny_clst):
        for clstx in range(Nx_clst):
            for ix in range(Nx_imp):
                i = ix + (clstx + Nx_clst*clsty)*Nimp
                if(clsty > 0):
                    j = i - (Ny_imp * (Nx - Nx_imp) + Nx_imp)
                    Tmat[i, j] = Tmat[j, i] = -t
                else:  # Periodic boundary conditions
                    j = i + (Nsites - Nimp * (Nx_clst-1) - Nx_imp)
                    Tmat[i, j] = Tmat[j, i] = -t*boundary

    return Tmat, Vmat
####################################


def make_ham_sto3g(typ, R, ng):

    # page 18, table 3.8
    zeta_he_sq = 2.0925**2
    zeta_h_sq = 1.24**2

# first atom
    if typ == "h2":
        z1 = 1.0
        zeta_1_sq = zeta_h_sq
    else:
        z1 = 2.0
        zeta_1_sq = zeta_he_sq

# second atom
    z2 = 1.0
    zeta_2_sq = zeta_h_sq

# nuclear repulsion
    Enuc = (z1*z2)/R
# initialize  position vectors
    Rvec = np.zeros(2)
    zvec = np.zeros(2)
    Rvec[0] = 0.0
    Rvec[1] = R
    zvec[0] = z1
    zvec[1] = z2

# contraction coefficients and exponents for the gausian sto3g
# (ф_GF(а)=(2a/п)^(3/4)e^(-ar^2) --> ф_CGF(r)=sum_n(d_n*ф_n_GF(a))
    alpha = np.zeros((ng, 2))
    d_coef = np.zeros((ng, 2))
# numbers from page 185  a, d for 1S orbital
    d_coef[0, 0] = d_coef[0, 1] = 0.444635
    d_coef[1, 0] = d_coef[1, 1] = 0.535328
    d_coef[2, 0] = d_coef[2, 1] = 0.154329

    alpha[0, 0] = zeta_1_sq*0.109818
    alpha[0, 1] = zeta_2_sq*0.109818
    alpha[1, 0] = zeta_1_sq*0.405771
    alpha[1, 1] = zeta_2_sq*0.405771
    alpha[2, 0] = zeta_1_sq*2.227660
    alpha[2, 1] = zeta_2_sq*2.227660

# normalization of d coef for gaussian basis fcns
    for i in range(ng):
        for j in range(2):
            d_coef[i, j] = d_coef[i, j]*(2*alpha[i, j]/np.pi)**(3./4.)

    K = 2  # number of basis K
    S = np.zeros((K, K))
    T = np.zeros((K, K))  # kinetic E
    V = np.zeros((K, K))  # nuc-e atraction
    Hcore = np.zeros((K, K))
    G = np.zeros((K, K, K, K))

    for i in range(K):
        for j in range(K):
            S[i, j] = get_S(i, j, Rvec, ng, alpha, d_coef)
            T[i, j] = get_T(i, j, Rvec, ng, alpha, d_coef)
            V[i, j] = get_V(i, j, Rvec, ng, alpha, d_coef, zvec)
            for k in range(K):
                for l in range(K):
                    G[i, j, k, l] = get_G(i, j, k, l, Rvec, ng, alpha, d_coef)
    Hcore = V+T
    return Hcore, S, G, Enuc
####################################


def get_T(mu, nu, R, ng, alpha, d_coef):
    Rmunu = R[mu]-R[nu]
    Tmunu = 0.0
    for i in range(ng):
        for j in range(ng):
            a = alpha[i, mu]
            b = alpha[j, nu]

            T = (a*b / (a+b) * (3-2 * a * b / (a+b) * Rmunu**2) *
                 (np.pi/(a+b))**(3/2)*np.exp(-a*b/(a+b)*Rmunu**2))
            Tmunu += d_coef[i, mu]*d_coef[j, nu]*T

    return Tmunu
#####################################


def get_V(mu, nu, R, ng, alpha, d_coef, zvec):

    Natoms = len(R)
    Rmunu = R[mu]-R[nu]
    Vmunu = 0.0
    for i in range(ng):
        for j in range(ng):
            a = alpha[i, mu]
            b = alpha[j, nu]
            Rp = (a*R[mu]+b*R[nu]) / (a+b)

            for k in range(Natoms):
                # position and charge of nuclei
                Z = zvec[k]
                Rnuc = R[k]

                V = (-2*np.pi / (a+b) * Z * np.exp(-a*b/(a+b)*Rmunu**2)
                     * F((a+b) * (Rp-Rnuc)**2))
                Vmunu += d_coef[i, mu]*d_coef[j, nu]*V
    return Vmunu
#####################################


def F(x):
    if x < 1e-3:
        return 1.0
    else:
        return 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))
#####################################


def get_S(mu, nu, R, ng, alpha, d_coef):
    Smunu = 0.0
    Rmunu = R[mu]-R[nu]
    for i in range(ng):
        for j in range(ng):
            if mu == nu:
                Smunu = 1
            else:
                a = alpha[i, mu]
                b = alpha[j, nu]

                S = (np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*Rmunu**2)
                Smunu += d_coef[i, mu]*d_coef[j, nu]*S

    return Smunu

#####################################


def get_G(i, j, k, l, R, ng, alpha, d_coef):
    # p,q,r,s represent basis function indices and
    # i,j,k,l represent primitive gaussian indices

    Ri = R[i]
    Rj = R[j]
    Rk = R[k]
    Rl = R[l]
    Gijkl = 0.0
    for m in range(ng):
        a = alpha[m, i]
        d_a = d_coef[m, i]

        for n in range(ng):
            b = alpha[n, j]
            d_b = d_coef[n, j]
            Rppp = (a*Ri+b*Rj)/(a+b)

            for o in range(ng):
                c = alpha[o, k]
                d_c = d_coef[o, k]

                for p in range(ng):
                    d = alpha[p, l]
                    d_d = d_coef[p, l]
                    Rqqq = (c*Rk+d*Rl)/(c+d)
                    Gmnop = (
                        2 * np.pi**(5/2) /
                        ((a+b)*(c+d)*np.sqrt(a+b+c+d))
                        * (np.exp(-a*b/(a+b)*(Ri-Rj)**2-c*d/(c+d)*(Rk-Rl)**2)
                           * F((a+b) * (c+d) / (a+b+c+d) * (Rppp-Rqqq)**2)))
                    Gijkl += d_a*d_b*d_c*d_d*Gmnop
    return Gijkl
