#!/usr/bin/python

#THIS FILE INCLUDES A VARIETY OF DIFFERENT INTEGRATORS
#EVERYTHING IS IN ATOMIC UNITS

import time
import numpy as np
import sys
import os
import feb_8_update.scripts.utils as utils
import feb_8_update.scripts.applyham_pyscf as applyham_pyscf
from scipy.linalg import expm

##########################################################################

def runge_kutta_coeff_matrix( C, H0, H1, H2, dt ):

    #subroutine to integrate the equations of motion of a coefficient
    #matrix defining a set states according to the runge-kutta scheme
    #C defines the coefficient matrix at time t, the set of H define the Hamiltonian generating the time-dependence
    #H0 is Hamiltonian at time t, H1 at time t+dt/2, H2 at time t+dt
    #returns C at time t+dt
    #dt is the time-step

    k1 = -1j * dt * np.dot( H0, C )

    k2 = -1j * dt * np.dot( H1, C+0.5*k1 )

    k3 = -1j * dt * np.dot( H1, C+0.5*k2 )

    k4 = -1j * dt * np.dot( H2, C+k3 )

    Cnew = C + 1.0/6.0 * ( k1 + 2.0*k2 + 2.0*k3 + k4 )

    return Cnew

##########################################################################

def runge_kutta_1rdm( P, H0, H1, H2, dt ):

    #subroutine to integrate the equations of motion of the 1-electron reduced density matrix according to the runge-kutta scheme
    #P defines the density matrix at time t, the set of H define the Hamiltonian generating the time-dependence
    #H0 is Hamiltonian at time t, H1 at time t+dt/2, H2 at time t+dt
    #returns the 1RDM at time t+dt
    #dt is the time-step

    k1 = -1j * dt * utils.commutator( H0, P )

    k2 = -1j * dt * utils.commutator( H1, P+0.5*k1 )

    k3 = -1j * dt * utils.commutator( H1, P+0.5*k2 )

    k4 = -1j * dt * utils.commutator( H2, P+k3 )

    Pnew = P + 1.0/6.0 * ( k1 + 2.0*k2 + 2.0*k3 + k4 )

    return Pnew

##########################################################################

def rk2_1rdm( P, H0, H1, dt ):

    #subroutine to integrate the equations of motion of the 1-electron reduced density matrix according to the 2nd order runge-kutta scheme
    #P defines the density matrix at time t, the set of H define the Hamiltonian generating the time-dependence
    #H0 is Hamiltonian at time t, H1 at time t+dt
    #returns the 1RDM at time t+dt
    #dt is the time-step

    k1 = -1j * dt * utils.commutator( H0, P )

    k2 = -1j * dt * utils.commutator( H1, P+k1 )

    Pnew = P + 0.5 * ( k1 + k2 )

    return Pnew

##########################################################################

def exact_timeindep_coeff_matrix( C, H, dt ):

    #subroutine to integrate the equations of motion of a coefficient
    #matrix, C, defining a set of single particle states exactly given a time-independent hamiltonian, H
    #dt is the time-step

    #calculate propagator e^-iHt by diagonalizing H
    #evals, evecs = utils.diagonalize( H )
    #prop = np.dot( evecs, np.dot( np.diag( np.exp( -1j*dt*evals ) ),  evecs.conjugate().transpose() ) )
    prop = expm( -1j * dt * H ) #PING should be the same as above, check timing and numerics

    #Propagate coefficient matrix
    Cnew = np.dot( prop, C )

    return Cnew

##########################################################################

def exact_timeindep_1rdm( P, H, dt ):

    #subroutine to integrate the equations of motion of the 1-electron reduced density matrix exactly given a time-independent hamiltonian, H
    #dt is the time-step

    #calculate propagator e^-iHt by diagonalizing H
    #evals, evecs = utils.diagonalize( H )
    #prop = np.dot( evecs, np.dot( np.diag( np.exp( -1j*dt*evals ) ),  evecs.conjugate().transpose() ) )
    prop = expm( -1j * dt * H ) #PING should be the same as above, check timing and numerics

    #Propagate 1rdm
    Pnew = np.dot( prop, np.dot( P, prop.conjugate().transpose() ) )

    return Pnew

##########################################################################

def runge_kutta_pyscf( CIcoeffs, Norbs, Nalpha, Nbeta, dt, hmat_0, Vmat_0, Econst_0, \
                       hmat_1=None, Vmat_1=None, Econst_1=None, hmat_2=None, Vmat_2=None, Econst_2=None ):

    #subroutine to integrate the equations of motion of the FCI coefficients using 4th order runge-kutta scheme
    #the hamiltonian is applied using pyscf
    #allows for only real but time dependent or independent hamiltonian
    #CIcoeffs is a 2d-complex array containing the CI coefficients, the rows/columns correspond to the alpha/beta strings
    #the strings are ordered in asscending binary order with a 0/1 implies that an orbital is empty/occupied
    #the 2e- integrals, Vmat_i, are given in chemistry notation
    #Econst is a constant energy contribution to the hamiltonian and an energy shift to increase reliability of the integrator
    #(see schollwock, j. phys. soc. jpn 2005 or Sato and Ishikawa Phys Rev A 2013 Eq. 40)
    #subscript 0, 1, 2 correspond to time t, t+dt/2 and t+dt
    #dt is the time step
    #returns C at time t+dt

    if( not np.iscomplexobj( CIcoeffs ) ):
        print('ERROR: CI coefficients in integrators.runge_kutta_pyscf are not a complex object')
        exit()

    if( hmat_1 == None ):
        #Assumes time-independent hamiltonian
        hmat_1   = np.copy( hmat_0 )
        Vmat_1   = np.copy( Vmat_0 )
        Econst_1 = np.copy( Econst_0 )
        hmat_2   = np.copy( hmat_0 )
        Vmat_2   = np.copy( Vmat_0 )
        Econst_2 = np.copy( Econst_0 )

    #Separate CI coefficients into real and imaginary parts
    Re_CIcoeffs = np.copy( CIcoeffs.real )
    Im_CIcoeffs = np.copy( CIcoeffs.imag )

    #Integrate according to 4th order Runge-Kutta
    k1 = -1j * dt * applyham_pyscf.apply_ham_pyscf_check( Re_CIcoeffs, hmat_0, Vmat_0, Nalpha, Nbeta, Norbs, Econst_0 ) \
             + dt * applyham_pyscf.apply_ham_pyscf_check( Im_CIcoeffs, hmat_0, Vmat_0, Nalpha, Nbeta, Norbs, Econst_0 )

    Re_temp = Re_CIcoeffs + 0.5 * np.copy( k1.real )
    Im_temp = Im_CIcoeffs + 0.5 * np.copy( k1.imag )

    k2 = -1j * dt * applyham_pyscf.apply_ham_pyscf_check( Re_temp, hmat_1, Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
             + dt * applyham_pyscf.apply_ham_pyscf_check( Im_temp, hmat_1, Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 )

    Re_temp = Re_CIcoeffs + 0.5 * np.copy( k2.real )
    Im_temp = Im_CIcoeffs + 0.5 * np.copy( k2.imag )

    k3 = -1j * dt * applyham_pyscf.apply_ham_pyscf_check( Re_temp, hmat_1, Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
             + dt * applyham_pyscf.apply_ham_pyscf_check( Im_temp, hmat_1, Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 )

    Re_temp = Re_CIcoeffs + np.copy( k3.real )
    Im_temp = Im_CIcoeffs + np.copy( k3.imag )

    k4 = -1j * dt * applyham_pyscf.apply_ham_pyscf_check( Re_temp, hmat_2, Vmat_2, Nalpha, Nbeta, Norbs, Econst_2 ) \
             + dt * applyham_pyscf.apply_ham_pyscf_check( Im_temp, hmat_2, Vmat_2, Nalpha, Nbeta, Norbs, Econst_2 )

    CIcoeffs = CIcoeffs + 1.0/3.0 * ( 0.5*k1 + k2 + k3 + 0.5*k4 )

    return CIcoeffs

##########################################################################

def runge_kutta_pyscf_nosym( CIcoeffs, Norbs, Nalpha, Nbeta, dt, hmat_0, Vmat_0, Econst_0, \
                       hmat_1=None, Vmat_1=None, Econst_1=None, hmat_2=None, Vmat_2=None, Econst_2=None ):

    #subroutine to integrate the equations of motion of the FCI coefficients using 4th order runge-kutta scheme
    #the hamiltonian is applied using pyscf assuming no symmetry about the hamiltonian
    #allows for complex or real and time dependent or independent hamiltonian
    #CIcoeffs is a 2d-complex array containing the CI coefficients, the rows/columns correspond to the alpha/beta strings
    #the strings are ordered in asscending binary order with a 0/1 implies that an orbital is empty/occupied
    #the 2e- integrals, Vmat_i, are given in chemistry notation
    #Econst is a constant energy contribution to the hamiltonian and an energy shift to increase reliability of the integrator
    #(see schollwock, j. phys. soc. jpn 2005 or Sato and Ishikawa Phys Rev A 2013 Eq. 40)
    #subscript 0, 1, 2 correspond to time t, t+dt/2 and t+dt
    #dt is the time step
    #returns C at time t+dt

    if( not np.iscomplexobj( CIcoeffs ) ):
        print('ERROR: CI coefficients in integrators.runge_kutta_pyscf_nosym are not a complex object')
        exit()

    if( hmat_1 == None ):
        #Assumes time-independent hamiltonian
        hmat_1   = np.copy( hmat_0 )
        Vmat_1   = np.copy( Vmat_0 )
        Econst_1 = np.copy( Econst_0 )
        hmat_2   = np.copy( hmat_0 )
        Vmat_2   = np.copy( Vmat_0 )
        Econst_2 = np.copy( Econst_0 )

    #Separate CI coefficients into real and imaginary parts
    Re_CIcoeffs = np.copy( CIcoeffs.real )
    Im_CIcoeffs = np.copy( CIcoeffs.imag )

    #Separate hamiltonian into real and imaginary parts
    Re_hmat_0 = np.copy( hmat_0.real )
    Im_hmat_0 = np.copy( hmat_0.imag )
    Re_hmat_1 = np.copy( hmat_1.real )
    Im_hmat_1 = np.copy( hmat_1.imag )
    Re_hmat_2 = np.copy( hmat_2.real )
    Im_hmat_2 = np.copy( hmat_2.imag )

    Re_Vmat_0 = np.copy( Vmat_0.real )
    Im_Vmat_0 = np.copy( Vmat_0.imag )
    Re_Vmat_1 = np.copy( Vmat_1.real )
    Im_Vmat_1 = np.copy( Vmat_1.imag )
    Re_Vmat_2 = np.copy( Vmat_2.real )
    Im_Vmat_2 = np.copy( Vmat_2.imag )

    #Integrate according to 4th order Runge-Kutta, splitting CI coefficients and Hamiltonian into real and imaginary parts
    Re_k1 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_CIcoeffs, Re_hmat_0, Re_Vmat_0, Nalpha, Nbeta, Norbs, Econst_0 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_CIcoeffs, Im_hmat_0, Im_Vmat_0, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k1 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_CIcoeffs, Re_hmat_0, Re_Vmat_0, Nalpha, Nbeta, Norbs, Econst_0 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_CIcoeffs, Im_hmat_0, Im_Vmat_0, Nalpha, Nbeta, Norbs, 0.0 )

    Re_temp = Re_CIcoeffs + 0.5 * Re_k1
    Im_temp = Im_CIcoeffs + 0.5 * Im_k1


    Re_k2 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Re_hmat_1, Re_Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Im_hmat_1, Im_Vmat_1, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k2 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Re_hmat_1, Re_Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Im_hmat_1, Im_Vmat_1, Nalpha, Nbeta, Norbs, 0.0 )

    Re_temp = Re_CIcoeffs + 0.5 * Re_k2
    Im_temp = Im_CIcoeffs + 0.5 * Im_k2


    Re_k3 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Re_hmat_1, Re_Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Im_hmat_1, Im_Vmat_1, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k3 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Re_hmat_1, Re_Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Im_hmat_1, Im_Vmat_1, Nalpha, Nbeta, Norbs, 0.0 )

    Re_temp = Re_CIcoeffs + Re_k3
    Im_temp = Im_CIcoeffs + Im_k3


    Re_k4 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Re_hmat_2, Re_Vmat_2, Nalpha, Nbeta, Norbs, Econst_2 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Im_hmat_2, Im_Vmat_2, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k4 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Re_hmat_2, Re_Vmat_2, Nalpha, Nbeta, Norbs, Econst_2 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Im_hmat_2, Im_Vmat_2, Nalpha, Nbeta, Norbs, 0.0 )


    k1 = Re_k1 + 1j*Im_k1
    k2 = Re_k2 + 1j*Im_k2
    k3 = Re_k3 + 1j*Im_k3
    k4 = Re_k4 + 1j*Im_k4

    CIcoeffs = CIcoeffs + 1.0/3.0 * ( 0.5*k1 + k2 + k3 + 0.5*k4 )

    return CIcoeffs


##########################################################################

def runge_kutta_pyscf_nosym_2( CIcoeffs, Norbs, Nalpha, Nbeta, dt, hmat_0, Vmat_0, Econst_0, \
                       hmat_1, Vmat_1, Econst_1, hmat_2, Vmat_2, Econst_2, hmat_3, Vmat_3, Econst_3 ):

    #subroutine to integrate the equations of motion of the FCI coefficients using 4th order runge-kutta scheme
    #the hamiltonian is applied using pyscf assuming no symmetry about the hamiltonian
    #allows for complex or real hamiltonian
    #CIcoeffs is a 2d-complex array containing the CI coefficients, the rows/columns correspond to the alpha/beta strings
    #the strings are ordered in asscending binary order with a 0/1 implies that an orbital is empty/occupied
    #the 2e- integrals, Vmat_i, are given in chemistry notation
    #Econst is a constant energy contribution to the hamiltonian and an energy shift to increase reliability of the integrator
    #(see schollwock, j. phys. soc. jpn 2005 or Sato and Ishikawa Phys Rev A 2013 Eq. 40)
    #dt is the time step
    #returns C at time t+dt
    #in comparison to above subroutine, here subscripts 0, 1, 2, and 3 correspond to hamiltonians that in principle can depend on some set of orbitals and therefore need to be defined for the orbitals corresponding to orb, orb + delt/2*k1_orb, orb + delt/2*k2_orb, orb + delt*k3_orb, where the orbitals are also being integrated by rk4

    if( not np.iscomplexobj( CIcoeffs ) ):
        print('ERROR: CI coefficients in integrators.runge_kutta_pyscf_nosym are not a complex object')
        exit()

    #Separate CI coefficients into real and imaginary parts
    Re_CIcoeffs = np.copy( CIcoeffs.real )
    Im_CIcoeffs = np.copy( CIcoeffs.imag )

    #Separate hamiltonian into real and imaginary parts
    Re_hmat_0 = np.copy( hmat_0.real )
    Im_hmat_0 = np.copy( hmat_0.imag )
    Re_hmat_1 = np.copy( hmat_1.real )
    Im_hmat_1 = np.copy( hmat_1.imag )
    Re_hmat_2 = np.copy( hmat_2.real )
    Im_hmat_2 = np.copy( hmat_2.imag )
    Re_hmat_3 = np.copy( hmat_3.real )
    Im_hmat_3 = np.copy( hmat_3.imag )

    Re_Vmat_0 = np.copy( Vmat_0.real )
    Im_Vmat_0 = np.copy( Vmat_0.imag )
    Re_Vmat_1 = np.copy( Vmat_1.real )
    Im_Vmat_1 = np.copy( Vmat_1.imag )
    Re_Vmat_2 = np.copy( Vmat_2.real )
    Im_Vmat_2 = np.copy( Vmat_2.imag )
    Re_Vmat_3 = np.copy( Vmat_3.real )
    Im_Vmat_3 = np.copy( Vmat_3.imag )

    #Integrate according to 4th order Runge-Kutta, splitting CI coefficients and Hamiltonian into real and imaginary parts
    Re_k1 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_CIcoeffs, Re_hmat_0, Re_Vmat_0, Nalpha, Nbeta, Norbs, Econst_0 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_CIcoeffs, Im_hmat_0, Im_Vmat_0, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k1 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_CIcoeffs, Re_hmat_0, Re_Vmat_0, Nalpha, Nbeta, Norbs, Econst_0 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_CIcoeffs, Im_hmat_0, Im_Vmat_0, Nalpha, Nbeta, Norbs, 0.0 )

    Re_temp = Re_CIcoeffs + 0.5 * Re_k1
    Im_temp = Im_CIcoeffs + 0.5 * Im_k1


    Re_k2 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Re_hmat_1, Re_Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Im_hmat_1, Im_Vmat_1, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k2 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Re_hmat_1, Re_Vmat_1, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Im_hmat_1, Im_Vmat_1, Nalpha, Nbeta, Norbs, 0.0 )

    Re_temp = Re_CIcoeffs + 0.5 * Re_k2
    Im_temp = Im_CIcoeffs + 0.5 * Im_k2


    Re_k3 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Re_hmat_2, Re_Vmat_2, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Im_hmat_2, Im_Vmat_2, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k3 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Re_hmat_2, Re_Vmat_2, Nalpha, Nbeta, Norbs, Econst_1 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Im_hmat_2, Im_Vmat_2, Nalpha, Nbeta, Norbs, 0.0 )

    Re_temp = Re_CIcoeffs + Re_k3
    Im_temp = Im_CIcoeffs + Im_k3


    Re_k4 =  dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Re_hmat_3, Re_Vmat_3, Nalpha, Nbeta, Norbs, Econst_2 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Im_hmat_3, Im_Vmat_3, Nalpha, Nbeta, Norbs, 0.0 )

    Im_k4 = -dt * applyham_pyscf.apply_ham_pyscf_nosym( Re_temp, Re_hmat_3, Re_Vmat_3, Nalpha, Nbeta, Norbs, Econst_2 ) \
            +dt * applyham_pyscf.apply_ham_pyscf_nosym( Im_temp, Im_hmat_3, Im_Vmat_3, Nalpha, Nbeta, Norbs, 0.0 )


    k1 = Re_k1 + 1j*Im_k1
    k2 = Re_k2 + 1j*Im_k2
    k3 = Re_k3 + 1j*Im_k3
    k4 = Re_k4 + 1j*Im_k4

    CIcoeffs = CIcoeffs + 1.0/3.0 * ( 0.5*k1 + k2 + k3 + 0.5*k4 )

    return CIcoeffs


##########################################################################

