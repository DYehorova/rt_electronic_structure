#!/usr/bin/python

import numpy as np
from scipy.special import erf

#####################################################################

def make_1D_hubbard( Nsites, U, boundary, Full=False ):

    t = 1.0

    Tmat = np.zeros( (Nsites,Nsites) )
    if( Full ):
        Vmat = np.zeros( (Nsites,Nsites,Nsites,Nsites) )
    else:
        Vmat = U

    for i in range(Nsites-1):
        Tmat[ i, i+1 ] = Tmat[ i+1, i ] = -t
        if( Full ): Vmat[ i, i, i, i ] = U

    if( Full ): Vmat [ Nsites-1, Nsites-1, Nsites-1, Nsites-1 ] = U

    Tmat[ 0, Nsites-1 ] = Tmat[ Nsites-1, 0 ] = -t*boundary #Allows for boundary conditions

    return Tmat, Vmat

#####################################################################

def make_1D_hubbard_arbitrary( Nsites, U, t, boundary, Full=False ):

    if( np.iscomplex(U) ):
        print("ERROR: Hubbard U is complex")
        exit()

    Tmat = np.zeros( (Nsites,Nsites), dtype=complex )
    if( Full ):
        Vmat = np.zeros( (Nsites,Nsites,Nsites,Nsites), dtype=complex )
    else:
        Vmat = None

    for i in range(Nsites-1):
        Tmat[ i, i+1 ] = -t
        Tmat[ i+1, i ] = -np.conjugate(t)
        if( Full ): Vmat[ i, i, i, i ] = U

    if( Full ): Vmat [ Nsites-1, Nsites-1, Nsites-1, Nsites-1 ] = U

    #Allows for boundary conditions
    if( Nsites > 2 ):
        Tmat[ Nsites-1, 0 ] = -t*boundary
        Tmat[ 0, Nsites-1 ] = -np.conjugate(t)*boundary

    return Tmat, Vmat


#####################################################################

def make_2D_hubbard( Nx, Ny, Nx_imp, Ny_imp, U, boundary, Full=False ):

    #Generates hopping and 2-electron matrices for a 2D-Hubbard model
    #The indices are defined such that the first Nimp correspond to a single impurity cluster, the second Nimp correspond to a second cluster, etc
    #Example of four 2x2 clusters for a total of 16 sites
    #  0   1   4   5
    #  2   3   6   7
    #  8   9   12  13
    #  10  11  14  15

    if( ( Nx % Nx_imp != 0 ) or ( Ny % Ny_imp != 0 ) ):
        print('ERROR: Impurity dimensions dont tesselate the full lattice')
        exit()

    if( ( Nx < Nx_imp != 0 ) or ( Ny < Ny_imp != 0 ) ):
        print('ERROR: Impurity dimensions larger than the full lattice')
        exit()

    Nsites  = Nx * Ny
    Nimp    = Nx_imp * Ny_imp
    Nclst   = round( Nsites / Nimp )
    Nx_clst = round( Nx/Nx_imp )
    Ny_clst = round( Ny/Ny_imp )

    t = 1.0

    #Initialize matrices
    Tmat = np.zeros( (Nsites,Nsites) )
    if( Full ):
        Vmat = np.zeros( (Nsites,Nsites,Nsites,Nsites) )
        for i in range(Nsites):
            Vmat[i,i,i,i] = U
    else:
        Vmat = None

    #Generate Tmat associated just with impurity cluster
    Tmat_imp = np.zeros( (Nimp,Nimp) )
    for iy in range(Ny_imp):
        for ix in range(Nx_imp):
            i = ix + Nx_imp*iy
            if( ix > 0 ): #Left
                Tmat_imp[ i, i-1 ] = -t
            if( ix < Nx_imp-1 ): #Right
                Tmat_imp[ i, i+1 ] = -t
            if( iy > 0 ): #Up
                Tmat_imp[ i, i-Nx_imp ] = -t
            if( iy < Ny_imp-1 ): #Down
                Tmat_imp[ i, i+Nx_imp ] = -t

    #Copy impurity cluster to all clusters
    for cpy in range(0,Nclst):
        for orb1 in range(Nimp):
            indx1 = ( orb1+Nimp*cpy ) % Nsites
            for orb2 in range(Nimp):
                indx2 = ( orb2+Nimp*cpy ) % Nsites
                Tmat[indx1,indx2] = Tmat_imp[orb1,orb2]

    #Inter-cluster interactions

    #Left and Right
    for clsty in range(Ny_clst):
        for clstx in range(Nx_clst):
            for iy in range(Ny_imp):
                i = Nx_imp * iy + ( clstx + Nx_clst*clsty )*Nimp
                if( clstx > 0 ):
                    j = i - ( Nimp - Nx_imp + 1 )
                    Tmat[ i, j ] = Tmat[ j, i ] = -t
                else: #Periodic boundary conditions
                    j = i + ( Nx_imp - 1 + Nimp * (Nx_clst-1) )
                    Tmat[ i, j ] = Tmat[ j, i ] = -t*boundary

    #Up and Down
    for clsty in range(Ny_clst):
        for clstx in range(Nx_clst):
            for ix in range(Nx_imp):
                i = ix + ( clstx + Nx_clst*clsty )*Nimp
                if( clsty > 0 ):
                    j = i - ( Ny_imp * (Nx - Nx_imp) + Nx_imp )
                    Tmat[ i, j ] = Tmat[ j, i ] = -t
                else: #Periodic boundary conditions
                    j = i + ( Nsites - Nimp * (Nx_clst-1) - Nx_imp )
                    Tmat[ i, j ] = Tmat[ j, i ] = -t*boundary

    return Tmat, Vmat


#####################################################################

def make_ham_single_imp_anderson( Nlead, E_imp, U, A, B, delE_lead, energyorder=True, Full=False ):
    #subroutine to generate one and two electron integrals for the symmetric single impurity anderson impurity model
    #spectral density is taken as the wide-band limit and is discretized following:
    #Li, Levy, Swenson, Rabani, Miller, JCP, 138, 104110, 2013

    #NOTE NEED TO ADD THE POSSIBILITY OF ADDING A BIAS BETWEEN LEADS, WHICH MEANS WILL NEED TO CHANGE HOW ORDER THE STATES NEAR THE IMPURITY

    #Input error checks
    if( Nlead % 2 != 1 ):
        print("ERROR: Number of lead states for anderson impurity model is not odd")
        exit()

    #gamma gives the energy scale
    gamma = 1.0

    #lead energy levels
    maxE_lead = (Nlead-1) * delE_lead * gamma / 2.0
    minE_lead = -maxE_lead
    E_lead = np.arange( minE_lead, maxE_lead+delE_lead*gamma, delE_lead )

    #lead spectral density
    J_lead = 0.5 * gamma / ( ( 1 + np.exp( A*(E_lead-0.5*B) ) ) * ( 1 + np.exp( -A*(E_lead+0.5*B) ) ) )

    #impurity-lead coupling
    t_lead = np.sqrt( J_lead * delE_lead * gamma / (2.0*np.pi) )


    #Form the one electron terms
    hmat = np.zeros( [ 2*Nlead+1, 2*Nlead+1 ] )
    hmat[0,0] = E_imp
    if( energyorder ):
        #the lead states are ordered in ascending energy with the left lead coming before the right lead
        for lead in range(Nlead):
            #left lead
            hmat[ lead+1, lead+1 ] = E_lead[ lead ]
            hmat[ 0, lead+1 ] = t_lead[ lead ]
            hmat[ lead+1, 0 ] = t_lead[ lead ]
            #right lead
            hmat[ lead+1+Nlead, lead+1+Nlead ] = E_lead[ lead ]
            hmat[ 0, lead+1+Nlead ] = t_lead[ lead ]
            hmat[ Nlead+lead+1, 0 ] = t_lead[ lead ]
    else:
        #the lead states are ordered in increasing absolute energy from energy=0
        #the negative energy appears before the positive energy when both have the same absolute energy
        #the left and right leads are not separated in the ordering
        #ie if originally E_lead=[-2,-1,0,1,2], now E_lead=[0,-1,1,-2,2]
        index  = np.argsort( np.absolute(E_lead) )
        E_lead = E_lead[index]
        t_lead = t_lead[index]
        for lead in range(Nlead):
            #left lead
            L_lead_indx = 2*lead+1
            hmat[ L_lead_indx, L_lead_indx ] = E_lead[ lead ]
            hmat[ 0, L_lead_indx ] = t_lead[ lead ]
            hmat[ L_lead_indx, 0 ] = t_lead[ lead ]
            #right lead
            R_lead_indx = 2*lead+2
            hmat[ R_lead_indx, R_lead_indx ] = E_lead[ lead ]
            hmat[ 0, R_lead_indx ] = t_lead[ lead ]
            hmat[ R_lead_indx, 0 ] = t_lead[ lead ]


    #Form the trivial two electron terms
    if( Full ):
        Vmat = np.zeros( [ Nlead+1, Nlead+1, Nlead+1, Nlead+1 ] )
        Vmat[0,0,0,0] = U
    else:
        Vmat = None

    return hmat, Vmat

#####################################################################

def make_ham_single_imp_anderson_realspace( NL, NR, Vg, U, t, Vbias, tleads=1.0, Full=False ):
    #subroutine to generate one and two electron integrals for the symmetric single impurity anderson impurity model in real-space
    #ie have hopping terms between the lead states, as is done in RT-DMRG (see Heidrich-Meisner PRB 2009)
    #Returns an (NL+NR+1)x(NL+NR+1) array of the 1 e- terms
    #Returns if Full=True an (NL+NR+1)^4 array of the 2e- terms or if Full=False simply returns U
    #The sites are ordered as the dot (impurity) site being first, followed in ascending order of the distance from the dot (impurity) with the left lead states coming before the right lead states
    #NL - number of sites in the left lead
    #NR - number of sites in the right lead
    #Vg - gate voltage ie the energy of the dot (impurity) site
    #U  - Hubbard repulsion on the dot (impurity)
    #t - hopping term between the dot (impurity) and the first site on the right/left lead
    #Vbias - constant energy term applied asymetrically to the leads to mimic a bias, Vbias/2 added to the right lead and subtracted from the left lead
    #tleads  - hopping term between lead sites, taken as the energy scale
    #Full - logical stating whether to print out the full 2e- integrals


    #Input error check
    if( np.absolute(NL-NR) != 1 and np.absolute(NL-NR) != 0 ):
        print('ERROR: Difference between number of sites in the left and right leads must be 0 or 1')
        exit()

    #Scale all energy values by tleads
    Vg = Vg * tleads
    U  = U * tleads
    t  = t * tleads
    Vbias = Vbias * tleads


    #Initialize
    N = NL+NR+1
    hmat = np.zeros([N,N])


    #Coupling part of the 1e- terms
    #dot-leads
    hmat[0,1] = -t
    hmat[0,2] = -t
    #left lead
    for lead in range(NL-1):
        indx1 = 2*lead+1
        indx2 = 2*(lead+1)+1
        hmat[indx1,indx2] = -tleads
    #right lead
    for lead in range(NR-1):
        if( NR > NL and lead == NR-2 ):
            indx1 = N-2
            indx2 = N-1
        else:
            indx1 = 2*(lead+1)
            indx2 = 2*(lead+2)
        hmat[indx1,indx2] = -tleads
    #complex conjugate
    hmat = hmat + hmat.conjugate().transpose()

    #Diagonal part of the 1e- terms
    #dot (impurity)
    hmat[0,0] = Vg
    #left lead
    for lead in range(NL):
        indx = 2*lead+1
        hmat[indx,indx] = -Vbias/2.0
    #right lead
    for lead in range(NR):
        if( NR > NL and lead == NR-1 ):
            indx = N-1
        else:
            indx = 2*(lead+1)
        hmat[indx,indx] = Vbias/2.0


    #Form the trivial two electron terms
    if( Full ):
        Vmat = np.zeros( [ N, N, N, N ] )
        Vmat[0,0,0,0] = U
    else:
        Vmat = U

    return hmat, Vmat

#####################################################################

def make_ham_wilson_chain( NL, NR, Vg, U, t, Vbias, tleads=1.0, Lambda=1.0, z=0.0, zonly=False, Full=False ):
    #subroutine to generate one and two electron integrals for a wilson chain for the symmetric single impurity anderson impurity model in real-space
    #ie have hopping terms between the lead states, which are damped further away from the lead (see Dias da Silva PRB 2008 and Anders PRB 2006)
    #Returns an (NL+NR+1)x(NL+NR+1) array of the 1 e- terms
    #Returns if Full=True an (NL+NR+1)^4 array of the 2e- terms or if Full=False simply returns U
    #The sites are ordered as the dot (impurity) site being first, followed in ascending order of the distance from the dot (impurity) with the left lead states coming before the right lead states
    #NL - number of sites in the left lead
    #NR - number of sites in the right lead
    #Vg - gate voltage ie the energy of the dot (impurity) site
    #U  - Hubbard repulsion on the dot (impurity)
    #t - hopping term between the dot (impurity) and the first site on the right/left lead
    #Vbias - constant energy term applied asymetrically to the leads to mimic a bias, Vbias/2 added to the right lead and subtracted from the left lead
    #tleads - un-damped hopping term between lead sites, taken as the energy scale
    #Lambda - discretization parameter
    #z - z-parameter for the z-trick (see Anders PRB 2006)
    #zonly - logical stating whether only to use the z-parameter, but dont damp the hopping as a function of distance
    #Full - logical stating whether to print out the full 2e- integrals


    #Input error check
    if( np.absolute(NL-NR) != 1 and np.absolute(NL-NR) != 0 ):
        print('ERROR: Difference between number of sites in the left and right leads must be 0 or 1')
        exit()

    #Scale all energy values by tleads
    Vg = Vg * tleads
    U  = U * tleads
    t  = t * tleads
    Vbias = Vbias * tleads


    #Initialize
    N = NL+NR+1
    hmat = np.zeros([N,N])


    #Coupling part of the 1e- terms
    #dot-leads
    hmat[0,1] = -t
    hmat[0,2] = -t
    #left lead
    for lead in range(NL-1):
        indx1 = 2*lead+1
        indx2 = 2*(lead+1)+1
        if( zonly ):
            hmat[indx1,indx2] = -tleads * Lambda**(-z)
        else:
            hmat[indx1,indx2] = -tleads * Lambda**(-z - lead/2.0)
    #right lead
    for lead in range(NR-1):
        if( NR > NL and lead == NR-2 ):
            indx1 = N-2
            indx2 = N-1
        else:
            indx1 = 2*(lead+1)
            indx2 = 2*(lead+2)
        if( zonly ):
            hmat[indx1,indx2] = -tleads * Lambda**(-z)
        else:
            hmat[indx1,indx2] = -tleads * Lambda**(-z - lead/2.0)
    #complex conjugate
    hmat = hmat + hmat.conjugate().transpose()

    #Diagonal part of the 1e- terms
    #dot (impurity)
    hmat[0,0] = Vg
    #left lead
    for lead in range(NL):
        indx = 2*lead+1
        hmat[indx,indx] = -Vbias/2.0
    #right lead
    for lead in range(NR):
        if( NR > NL and lead == NR-1 ):
            indx = N-1
        else:
            indx = 2*(lead+1)
        hmat[indx,indx] = Vbias/2.0


    #Form the trivial two electron terms
    if( Full ):
        Vmat = np.zeros( [ N, N, N, N ] )
        Vmat[0,0,0,0] = U
    else:
        Vmat = U

    return hmat, Vmat

#####################################################################

def make_ham_multi_imp_anderson_realspace( Nimp, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full ):
    #subroutine to generate one and two electron integrals for the symmetric multi-impurity anderson impurity model in real-space
    #ie have hopping terms between the lead states, as is done in RT-DMRG (see Heidrich-Meisner PRB 2009)
    #Returns an (NL+NR+Nimp)x(NL+NR+Nimp) array of the 1 e- terms
    #Returns if Full=True an (NL+NR+Nimp)^4 array of the 2e- terms or if Full=False simply returns U
    #The sites are ordered from left to right, with impurities being in the center of the two leads
    #If the number of impurities equals number of sites you get back Hubbard model with timp=timplead=tleads
    #Nimp - number of impurity sites
    #NL - number of sites in the left lead
    #NR - number of sites in the right lead
    #Vg - gate voltage ie the energy of the impurity sites
    #U  - Hubbard repulsion on the impurities
    #timplead - hopping term between the furthest left and furtherst right impurity and the first site on the left/right lead
    #timp - hopping term between the impurities
    #Vbias - constant energy term applied asymetrically to the leads to mimic a bias, Vbias/2 added to the right lead and subtracted from the left lead
    #tleads  - hopping term between lead sites, taken as the energy scale
    #Full - logical stating whether to print out the full 2e- integrals


    #Input error check
    if( np.absolute(NL-NR) != 1 and np.absolute(NL-NR) != 0 ):
        print('ERROR: Difference between number of sites in the left and right leads must be 0 or 1')
        exit()

    #Scale all energy values by tleads
    Vg = Vg * tleads
    U  = U * tleads
    timp  = timp * tleads
    timplead  = timplead * tleads
    Vbias = Vbias * tleads

    #Initialize
    N = NL+NR+Nimp
    hmat = np.zeros([N,N])

    #Lists for indices of left-lead, impurities, and right-lead
    left_indx = np.arange(NL)
    imp_indx  = np.arange(NL,NL+Nimp)
    right_indx = np.arange(NL+Nimp,N)


    #Coupling part of the 1e- terms
    #periodic boundary
    if boundary is True:
        hmat[ 0, N-1 ] = hmat[ N-1, 0 ] = -tleads*boundary

    #left lead
    for lead in left_indx[:-1]:
        hmat[ lead, lead+1 ] = -tleads
        hmat[ lead+1, lead ] = -tleads
    #left lead - impurity
    hmat[ left_indx[-1], imp_indx[0] ] = -timplead
    hmat[ imp_indx[0], left_indx[-1] ] = -timplead
    #impurities
    for imp in imp_indx[:-1]:
        hmat[ imp, imp+1 ] = -timp
        hmat[ imp+1, imp ] = -timp
    #impurity - right lead
    hmat[ imp_indx[-1], right_indx[0] ] = -timplead
    hmat[ right_indx[0], imp_indx[-1] ] = -timplead
    #right lead
    for lead in right_indx[:-1]:
        hmat[ lead, lead+1 ] = -tleads
        hmat[ lead+1, lead ] = -tleads

    #Diagonal part of the 1e- terms
    #impurities
    for imp in imp_indx:
        hmat[imp,imp] = Vg
    #left lead
    for lead in left_indx:
        hmat[lead,lead] = -Vbias/2.0
    #right lead
    for lead in right_indx:
        hmat[lead,lead] = Vbias/2.0

    #Form the trivial two electron terms
    if( Full ):
        Vmat = np.zeros( [ N, N, N, N ] )
        for imp in imp_indx:
            Vmat[imp,imp,imp,imp] = U
    else:
        Vmat = U
    return hmat,Vmat

#####################################################################

def make_ham_diatomic_sto3g(typ,R):
    #subroutine to generate the overlap matrix, S, core hamiltonian, Hcore,
    #and a 4-index matrix of the 2-e integrals, g, for the diatomics H2 or HeH+
    #in the STO-3G basis

    ng = 3  #number of gaussians, hard-coded to 3 for sto-3g

    zeta_h_sq = 1.24**2
    zeta_he_sq = 2.0925**2

    #define first atom as either hydrogen or he
    if typ=="h2":
        z1 = 1.0
        zeta_1_sq = zeta_h_sq
    elif typ=="hehp":
        z1 = 2.0
        zeta_1_sq = zeta_he_sq
    else:
        print("Eror in make_ham_diatomic_sto3g(): typ must be either 'h2' or 'hehp'")
        sys.exit(1)

    #define second atom always as hydrogen
    z2 = 1.0
    zeta_2_sq = zeta_h_sq

    #calculate nuclear repulsion
    Enuc = z1*z2/R

    #define vectors for the position of the two atoms
    Rvec = np.zeros(2)
    zvec = np.zeros(2)
    Rvec[0] = 0.0
    Rvec[1] = R
    zvec[0] = z1
    zvec[1] = z2

    #define the contraction coefficients and exponents for sto-3g for each atom
    d_coef = np.zeros( (ng,2) )
    alpha = np.zeros( (ng,2) )

    d_coef[0,0] = d_coef[0,1] = 0.444635
    d_coef[1,0] = d_coef[1,1] = 0.535328
    d_coef[2,0] = d_coef[2,1] = 0.154329

    alpha[0,0] = zeta_1_sq*0.109818
    alpha[0,1] = zeta_2_sq*0.109818
    alpha[1,0] = zeta_1_sq*0.405771
    alpha[1,1] = zeta_2_sq*0.405771
    alpha[2,0] = zeta_1_sq*2.227660
    alpha[2,1] = zeta_2_sq*2.227660

    #multiply d_coef by normalization constant from gaussian basis fcns
    for i in np.arange(0,ng):
        for j in np.arange(0,2):
            d_coef[i,j] = d_coef[i,j]*(2*alpha[i,j]/np.pi)**(3./4.)

    #initialize matrices
    Nb = 2 #number of basis functions
    S = np.zeros( (Nb,Nb) ) #overlap matrix
    T = np.zeros( (Nb,Nb) ) #kinetic energy
    Vnuc = np.zeros( (Nb,Nb) ) #e- nuclei attraction
    Hcore = np.zeros( (Nb,Nb) ) #core Hamiltonian
    g = np.zeros( (Nb,Nb,Nb,Nb) ) #4D tensor for 2e integrals

    for p in np.arange(0,Nb):
        for q in np.arange(0,Nb):
            S[p,q] = calc_S(p,q,Rvec,ng,alpha,d_coef)
            T[p,q] = calc_T(p,q,Rvec,ng,alpha,d_coef)
            Vnuc[p,q] = calc_Vnuc(p,q,Rvec,ng,alpha,d_coef,zvec)
            for r in np.arange(0,Nb):
                for s in np.arange(0,Nb):
                    g[p,q,r,s] = calc_g(p,q,r,s,Rvec,ng,alpha,d_coef)

    Hcore = T+Vnuc

    return S,Hcore,g,Enuc

#####################################################################

def calc_S(mu,nu,R,ng,alpha,d_coef):
    #subroutine to calculate the mu,nu element of the overlap matrix S
    #note that d_coef takes care of the normalization constants

    Rmunu=R[mu]-R[nu]

    smunu=0.0
    for i in np.arange(0,ng):
        for j in np.arange(0,ng):
            if mu == nu:
                smunu = 1
            else:
                a=alpha[i,mu]
                b=alpha[j,nu]

                spq = (np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*Rmunu**2)
                smunu += d_coef[i,mu]*d_coef[j,nu]*spq

    return smunu

#####################################################################

def calc_T(mu,nu,R,ng,alpha,d_coef):
    #subroutine to calculate the mu,nu element of the kinetic energy matrix T
    #note that d_coef takes care of the normalization constants

    Rmunu=R[mu]-R[nu]

    Tmunu=0.0
    for i in np.arange(0,ng):
        for j in np.arange(0,ng):
            a=alpha[i,mu]
            b=alpha[j,nu]

            Tpq = a*b/(a+b)*(3-2*a*b/(a+b)*Rmunu**2)*(np.pi/(a+b))**1.5*np.exp(-a*b/(a+b)*Rmunu**2)
            Tmunu += d_coef[i,mu]*d_coef[j,nu]*Tpq

    return Tmunu


#####################################################################

def calc_Vnuc(mu,nu,R,ng,alpha,d_coef,zvec):
    #subroutine to calculate the mu,nu element of the e-nuclei attraction matrix
    #note that d_coef takes care of the normalization constants

    natms = len(R)

    Rmunu=R[mu]-R[nu]

    Vmunu=0.0
    for i in np.arange(0,ng):
        for j in np.arange(0,ng):
            a=alpha[i,mu]
            b=alpha[j,nu]

            Rp=(a*R[mu]+b*R[nu])/(a+b)

            for k in np.arange(0,natms):

                #position and charge of the interacting nuclei
                Ratm = R[k]
                Zatm = zvec[k]

                Vpq = -2*np.pi/(a+b)*Zatm*np.exp(-a*b/(a+b)*Rmunu**2)*F0((a+b)*(Rp-Ratm)**2)
                Vmunu += d_coef[i,mu]*d_coef[j,nu]*Vpq

    return Vmunu


#####################################################################

def calc_g(p,q,r,s,R,ng,alpha,d_coef):
    #subroutine to calculate the p,q,r,s element of the two-electron tensor
    #i.e. g[p,q,r,s] = (pq|rs) in spatial chemist's notation
    #note that d_coef takes care of the normalization constants
    #note change in index notation from previous subroutines
    #now p,q,r,s represent basis function indices and i,j,k,l represent primitive gaussian indices

    Rp = R[p]
    Rq = R[q]
    Rr = R[r]
    Rs = R[s]

    gpqrs = 0.0
    for i in np.arange(0,ng): #loop over primitives, i, in basis fcn p
        a = alpha[i,p]
        da = d_coef[i,p]
        for j in np.arange(0,ng): #loop over primitives, j, in basis fcn q
            b = alpha[j,q]
            db = d_coef[j,q]

            Rppp = (a*Rp+b*Rq)/(a+b)

            for k in np.arange(0,ng): #loop over primitives, k, in basis fcn r
                c = alpha[k,r]
                dc = d_coef[k,r]
                for l in np.arange(0,ng): #loop over primitives, l, in basis fcn s
                    d = alpha[l,s]
                    dd = d_coef[l,s]

                    Rqqq = (c*Rr+d*Rs)/(c+d)

                    gijkl = 2*np.pi**(5./2.)/( (a+b)*(c+d)*np.sqrt(a+b+c+d) )*(
                            np.exp(-a*b/(a+b)*(Rp-Rq)**2-c*d/(c+d)*(Rr-Rs)**2)
                            *F0( (a+b)*(c+d)/(a+b+c+d)*(Rppp-Rqqq)**2 ) )
                    gpqrs += da*db*dc*dd*gijkl

    return gpqrs


#####################################################################


def F0(x):

    #function to calculate necessary error function calculations
    if x<1e-3:
        return 1.0
    else:
        return 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))


#####################################################################

