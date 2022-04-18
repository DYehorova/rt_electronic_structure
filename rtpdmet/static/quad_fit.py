import math
import cmath
import numpy as np
from scipy import stats

"""
    All is from from Zhihao Cui's code
    fishjojo/libmet_solid/dmet/quad_fit.py

"""

##################################################################


def get_parabola_vertex(x, y, tol=1e-12):
    """
    Give x = [x1, x2, x3], y = [y1, y2, y3]
    return a, b, c that y = ax^2 + bx + c.
    Args:
        x: [x1, x2, x3]
        y: [y1, y2, y3]
    Returns:
        a, b, c
        status: True if sucess.
    """
    x1, x2, x3 = x
    y1, y2, y3 = y
    denomenator = float((x1 - x2) * (x1 - x3) * (x2 - x3))
    if abs(denomenator) < tol:
        a = b = c = 0
        status = False
    else:
        a = (
            x3 * (y2 - y1) + x2 * (y1 - y3)
            + x1 * (y3 - y2)) / denomenator
        b = (
            x3 * x3 * (y1 - y2) + x2 * x2 *
            (y3 - y1) + x1 * x1 * (y2 - y3)) / denomenator
        c = (
            x2 * x3 * (x2 - x3) * y1 + x3 *
            x1 * (x3 - x1) * y2 + x1 * x2 *
            (x1 - x2) * y3) / denomenator
        status = True
    return a, b, c, status

##################################################################


def get_roots(a, b, c, tol=1e-12):
    """
    Find roots for quadratic equation.
    Args:
        a, b, c
        tol: tolerance for zero
    Returns:
        roots: roots
        status: status 0: no root,
                       1: root1, root2
                       2: root1 (linear equation)
                       3: root1, root2 (complex).
    """

    if abs(a) < tol and abs(b) < tol:
        print('a = 0, b = 0, not a quadratic equation.')
        status = 0
        return [], status

    if abs(a) < tol:
        print('a = 0, single solution is:', -c / b)
        status = 2
        return [-c / float(b)], status

    D = b ** 2 - 4.0 * a * c

    if D >= 0.0:
        root1 = (-b + np.sqrt(D)) / (2.0 * a)
        root2 = (-b - np.sqrt(D)) / (2.0 * a)
        status = 1
        return [root1, root2], status
    else:
        root1 = (-b + cmath.sqrt(D)) / (2.0 * a)
        root2 = (-b - cmath.sqrt(D)) / (2.0 * a)
        status = 3
        return [root1, root2], status

##################################################################


def quad_fit(mu, dnelecs, tol=1e-12):
    # quadratic fit of mu and nelec.
    """
    Quadratic fit of mu and nelec.
    Args:
        mu: (3,)
        dnelecs: (3,) nelecs - target.
        tol: tolerance.
    Returns:
        mu_new: new mu.
        status: True for sucess.
    """
    # copy = true returns an array copy of the object
    mu_lst = np.array(mu, copy=True)
    dnelecs_lst = np.array(dnelecs, copy=True)
    assert len(mu_lst) == len(dnelecs_lst) and len(mu_lst) == 3

    # make indexed lists of mus and nelecs
    idx1 = np.argsort(mu_lst, kind='mergesort')
    idx2 = np.argsort(dnelecs_lst, kind='mergesort')

    if not (idx1 == idx2).all():
        print("dnelecs is not a monotonic function of mu...")
    mu_lst = mu_lst[idx1]
    dnelecs_lst = dnelecs_lst[idx1]

    # get parabola vertex
    a, b, c, status = get_parabola_vertex(mu_lst, dnelecs_lst, tol=tol)
    if not status:
        print('Duplicated points among three dots', mu_lst, dnelecs_lst)
        return 0, False

    # find the roots of the parabola using quadratic formula
    roots, status = get_roots(a, b, c, tol=tol)

    if status == 0:
        print('root finding error')
        return 0, False
    elif status == 2:
        mu_new = roots[0]
        return mu_new, True
    elif status == 3:
        if abs(roots[0].imag) + abs(roots[1].imag) > 1e-3:
            print('Complex root finding:', roots[0], roots[1])
            return 0, False
        else:
            roots = [roots[0].real, roots[1].real]

    if dnelecs_lst[0] >= 0.0:
        left = -np.inf
        right = mu_lst[0]
    elif dnelecs_lst[1] >= 0.0:
        left = mu_lst[0]
        right = mu_lst[1]
    elif dnelecs_lst[2] >= 0.0:
        left = mu_lst[1]
        right = mu_lst[2]
    else:
        left = mu_lst[2]
        right = np.inf

    if roots[0] < right and roots[0] > left:
        if roots[1] < right and roots[1] > left:
            if abs(roots[0] - mu[0]) < abs(roots[1] - mu[0]):
                return roots[0], True
            else:
                return roots[1], True
        else:
            return roots[0], True
    else:
        if roots[1] < right and roots[1] > left:
            return roots[1], True
        else:
            print("Can not find proper root within the range, ",
                  left, right, 'and roots', roots)
            return 0, False
    # not sure about figuring out which root out of
    # "roots" is chosen and how (lines 124 - 151)
##################################################################


def has_duplicate(dmu, mus, tol=1e-6):
    # return true if any of the arguments are
    # the same within tolerance are the same as dmu
    dmus_abs = np.abs(mus - dmu)
    return (dmus_abs < tol).any()

##################################################################


def violate_previous_mu(dmu, mus, target, nelecs):
    # if new dmu is driving mu in the wrong direction
    x = dmu - mus
    y = target - nelecs
    return ((x * y) < 0.0).any()

##################################################################


def quad_fit_mu(mus, nelecs, filling, step):

    """
    from Zhihao's code

    Use quadratic fit to predict chemical potential.

    Args:
        mus: a list of old mu
        nelecs: a list of old nelectron number
        filling: filling * 2.0 is the target nelec.
        step: max trust step

    Returns:
        dmu: the change in mu.
    """
    mus = np.asarray(mus)
    nelecs = np.asarray(nelecs)
    target = filling * 2.0
    dnelec = nelecs - target
    dnelec_abs = np.abs(dnelec)

    # get three nearest points
    idx_dnelec = np.argsort(dnelec_abs, kind='mergesort')
    mus_sub = mus[idx_dnelec][:3]
    dnelec_sub = dnelec[idx_dnelec][:3]

    # quadratic fit
    dmu, status = quad_fit(mus_sub, dnelec_sub, tol=1e-12)

    # check duplicates
    if has_duplicate(dmu, mus):
        print("duplicate in extrapolation.")
        status = False

    if not status:
        print('quadratic fit failed or duplicated, use linear regression')
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(dnelec_sub, mus_sub)
        dmu = intercept

    # check monotonic for the predict mu:
    if violate_previous_mu(dmu, mus, target, nelecs):
        print("predicted mu violates previous mus. Try linear regression.")
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(dnelec_sub, mus_sub)
        dmu = intercept

        if violate_previous_mu(dmu, mus, target, nelecs):
            print("predicted mu from linear regression also violates",
                  " use finite step.")
            step = min(step, 1e-3)
            dmu = math.copysign(step, (target - nelecs[-1])) + mus[-1]
            # array[-1] - last term in array, the most recent mu, nelec pair

    if abs(dmu - mus[-1]) > step:
        print("extrapolation dMu %20.12f more than trust step %20.12f",
              dmu - mus[-1], step)
        dmu = math.copysign(step, dmu - mus[-1]) + mus[-1]

    if has_duplicate(dmu, mus):
        print("duplicate in extrapolation.")
        dmu = math.copysign(step, dmu - mus[-1]) + mus[-1]

    if ((dmu - mus[-1]) * (target - nelecs[-1]) < 0
            and abs(dmu - mus[-1]) > 2e-3):
        print("extrapolation gives wrong direction, use finite difference")
        dmu = math.copysign(step, (target - nelecs[-1])) + mus[-1]

    return dmu
