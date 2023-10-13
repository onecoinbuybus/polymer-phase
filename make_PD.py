import numpy as np

N_a = 5.0  # Length of polymer A
N_b = 1.0  # Length of polymer B (since 1, we can think of this as the solvent)
volumeFrac = np.arange(0.001, 1, 0.001)
floryParams = np.arange(0.01, 2, 0.005)


def freeEnergy(phi, Na, Nb, xi):
    """
    Calculates the Flory-Huggins mean-field free energy per unit volume in units of kT.

    Return free energy vector of same length as phi.

    Paramters:
    phi: 1D array of volume fractions
    Na: polymer A has length Na
    Nb: polymer B has length Nb
        [note: make Na or Nb 1.0 if considering a single polymer species in some solvent]
    xi: Flory parameter. Encapsulates A-A, A-B, and B-B interactions.

    """
    f = (
        ((phi / Na) * (np.log(phi)))
        + (((1.0 - phi) / Nb) * (np.log(1 - phi)))
        + xi * phi * (1.0 - phi)
    )
    return f


def chemicalPotential(phi, Na, Nb, xi):
    """
    Return the chemical potential vector of same length as phi and where the
    chemical potential is zero.

    This is the first derivative of the free energy of mixing per unit volume with
    respect to composition (or volume fraction). See Eq 4.44 in Polymer Physics by
    Rubinstein and Colby.

    Paramters:
    phi: 1D array of volume fractions
    Na: polymer A has length Na
    Nb: length of species B
    xi: Flory parameter. Encapsulates A-A, A-B, and B-B interactions.
    """
    mu = (
        (((np.log(phi)) / Na) + (1.0 / Na))
        - (((np.log(1 - phi)) / Nb))
        - (1.0 / Nb)
        + xi * (1 - (2 * phi))
    )
    where_chempot_is_zero = whereSignChanges(mu)
    return mu, where_chempot_is_zero


def secondDerivF(phi, Na, Nb, xi):
    """
    Return the second derivative of the free energy as 1D array of same length as phi.
    Parameters:
    phi: 1D array of volume fractions
    Na: polymer A has length Na
    Nb: length of species B
    xi: Flory parameter. Encapsulates A-A, A-B, and B-B interactions.
    """
    fdoubleprime = (1.0 / (Na * phi)) + (1 / (Nb * (1.0 - phi))) - (2 * xi)
    where_dubprime_is_zero = whereSignChanges(fdoubleprime)
    return fdoubleprime, where_dubprime_is_zero


def xispinodal(phi, Na, Nb):
    xispin = (1.0 / 2) * (((1.0 / Na)) + (1.0 / (Nb * (1.0 - phi))))
    return xispin


def slope_f(phi, F, chempot, chempotential_to_look_at):
    """
    Finds the slope between two points on the free energy vs. volume fraction plot.

    Paramters:
    phi: 1D array of volume fraction
    F: 1D array of free energy (for each volume fraction in phi)
    chempot: 1D array of chemical potential
    chempotential_to_look_at:
    """

    # Creates a 1D array of values equal to given chemical potential
    f = (
        np.ones((len(chempot))) * chempotential_to_look_at
    )  # value of free energy we want
    idx = np.argwhere(np.diff(np.sign(f - chempot)) != 0).reshape(-1) + 0
    if len(idx) > 2:
        point1 = idx[
            0
        ]  # First point where the chemical potential equals value of interest
        point2 = idx[
            -1
        ]  # Last point where the chemical potential equals value of interest
    else:
        return 0, 0, 0, 0, 0, 0, 0
    phi1 = phi[point1]  # Volume fraction of first point
    phi2 = phi[point2]  # Volume fraction of last point
    f_point1 = F[point1]  # Free energy of first point
    f_point2 = F[point2]  # Free energy of last point
    slope_between_12 = (f_point2 - f_point1) / (phi2 - phi1)
    return slope_between_12, phi1, phi2, f_point1, f_point2, point1, point2


def whereSignChanges(data):
    """
    Returns the *index* of the vector data where the sign of data changes. That is,
    where data goes from positive to negative or negative to positive.

    The returned variable, idx, may have one or more (or less numbers)
    """
    idx = np.argwhere(np.diff(np.sign(data)) != 0).reshape(-1) + 0
    return idx


def constructPlots(phi, N_a, N_b, chi):
    f_en = freeEnergy(phi, N_a, N_b, chi)
    chem_pot, where_chempot_zero = chemicalPotential(
        phi, N_a, N_b, chi
    )  # Calculate chemical potential
    fdoubleprime, dubprime_zero = secondDerivF(
        phi, N_a, N_b, chi
    )  # Calculate second derivative of free energy
    # Initialize whether there is a min/max in the first derivative of the free energy
    minmax_in_chem_pot = False
    if len(dubprime_zero) > 1:
        minmax_in_chem_pot = True

    # initializing at what volume fractions the second derivative of free energy is zero
    sec_der_zero = [np.nan, np.nan]
    if len(dubprime_zero) > 0:
        if len(dubprime_zero) > 1:
            sec_der_zero = [phi[dubprime_zero[0]], phi[dubprime_zero[1]]]

    if minmax_in_chem_pot:
        # Chemical potential at volume fractions where the second derivative
        # of the free energy is zero
        chem_pot_at_loc_max_1 = chem_pot[dubprime_zero[0]]
        chem_pot_at_loc_max_2 = chem_pot[dubprime_zero[1]]

        # Now let's find at what volume fractions have those chemical potentials
        where_chem_pot_1 = whereSignChanges(chem_pot - chem_pot_at_loc_max_1)
        where_chem_pot_2 = whereSignChanges(chem_pot - chem_pot_at_loc_max_2)
        # print(where_chem_pot_1,where_chem_pot_2,dubprime_zero[0],dubprime_zero[1])
        # testing the slopes between two points on Fmix...
        the_slps = (
            []
        )  # empty list to store the slope between two points on the free energy plot
        the_chempots = []  # empty list to store chemical potential
        the_phis = []  # empty list to store volume fractions
        the_fs = []  # empty list to store free energies
        the_points = []  # empty list to store indices of phi, free energy, etc

        # Loop through a certain range of volume fractions and find the slope between two points
        #  on the free energy vs volume fraction plot for two points that have equal chemical
        #  potentials.
        for i in range(1, dubprime_zero[0]):
            slp, phi1, phi2, f1, f2, p1, p2 = slope_f(phi, f_en, chem_pot, chem_pot[i])
            if p1 > 0 and p2 > 0:
                the_slps.append(slp)
                the_chempots.append(chem_pot[i])
                the_phis.append([phi1, phi2])
                the_fs.append([f1, f2])
                the_points.append([p1, p2])

        # Take the difference of the list of slopes and the list of chemical potentials. Use this to then
        #  find the two points (points on the free energy vs volume frac plot) where the value of their
        #  chemical potential (they'll have the same chem. pot.) equals the slope of the line connecting them.
        slps_minus_chempots = np.array(the_slps) - np.array(the_chempots)
        the_line_idx = whereSignChanges(slps_minus_chempots)
        if len(the_line_idx) > 0:
            phi1, phi2 = the_phis[the_line_idx[0]]
            f1, f2 = the_fs[the_line_idx[0]]
            p1, p2 = the_points[the_line_idx[0]]

            return phi1, phi2, sec_der_zero[0], sec_der_zero[1]



#positions of FHPD
pts = np.zeros((len(floryParams),4))
for i,chi in enumerate(floryParams):
    pts[i] = constructPlots(volumeFrac, N_a, N_b, chi)

import matplotlib.pyplot as plt
for i,pt in enumerate(pts):                       #orignal phase graph
    plt.plot(pt[0:2], [floryParams[i], floryParams[i]],c='0.6',lw=5)
    plt.plot(pt[2:4], [floryParams[i], floryParams[i]],c='0.2',lw=5)


def getNearestValue(floryParams, y):
    idx = np.abs(floryParams - y).argmin()
    return idx

def get_phase(x,y,pts,floryParams):
    idx = getNearestValue(floryParams,y)
    phase_range = pts[idx]
    if pts[idx][2]<x<pts[idx][3]:
        return 'two phase'
    elif pts[idx][3]<x<pts[idx][1] or pts[idx][0]<x<pts[idx][2]:
        return 'meta phase'
    else:
        return 'single phase'

pts_r = np.nan_to_num(pts)

# generation of grid positions and labels
gap = 20
x = np.linspace(0,1,gap) 
y = np.linspace(floryParams[200],floryParams[-1],gap) 
xx, yy = np.meshgrid(x,y)
positions = np.vstack([xx.ravel(), yy.ravel()]).T
labels = [get_phase(i[0],i[1],pts,floryParams) for i in positions]

cor = []
for idx, label in enumerate(labels):
    if label != 'range error':
        cor.append([positions[idx],label])
#pd.DataFrame(cor).to_excel('cor.xlsx')


fig = plt.figure(figsize=(9, 9), dpi=300)
ax1 = fig.subplots()

for i in positions:
    phase = get_phase(i[0],i[1],pts_r,floryParams)
    cc = []
    if phase == 'single phase':
        cc = '#2470a0'
    elif phase == 'two phase':
        cc = '#a696c8'
    elif phase == 'meta phase':
        cc = '#fad3cf'
    plt.scatter(i[0],i[1],c=cc)
ax1.set_ylim(1,2)
ax1.set_xlabel('φ',fontsize=20)
ax1.set_ylabel('Interaction parameter χ',fontsize=20)
plt.show()