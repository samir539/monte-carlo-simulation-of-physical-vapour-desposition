import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.stats import binom

#######################################################################################
# GLOBAL VALUES
#######################################################################################
cellA = 0 # Empty site
cellB = 1 # Added atom

def orderRandom(Z, f):
    """
    ORDERRANDOM produces a distribtion function of the order parameter for
    random surface coverage. The order parameter is just the number of AB bonds
    around a site. The distribution is computed from the binomial distribution.

       Input arguments
           Z  The number of neighbouring sites per site
           f  The surface coverage

       Output arguments
           N       List of possible number of neighbours
           P       The probability distribution of the order parameter
    """
    #
    # Initialise the probability distribution
    N = np.linspace(0,Z,Z+1)
    P = np.linspace(0,0,Z+1)
    #
    # Run over allowed number of AB bonds around each site and compute the
    # probability distribution
    for n in range(Z+1):
        P[n] = (1.0-f)*binom.pmf(n,Z,f) + f*binom.pmf(n,Z,1.0-f)
    return N, P

def order2D(config):
    """
    ORDER2D produces a distribtion function of the order
    parameter. The order parameter is just the number of AB bonds
    around a site.

    Input arguments
    config  The configuration

    Output arguments
    N       List of possible number of neighbours
    P       The probability distribution of the order parameter
    """
    #
    # Get dimensions of configuration
    nx, ny = config.shape
    #
    # Table of neighbour displacements
    dxtab = [0,  0, 1, -1]
    dytab = [1, -1, 0,  0]
    #
    # Initialise the probability distribution
    N = np.linspace(0,4,5)
    P = np.linspace(0,0,5)
    #
    # Run over sites and count the number of AB bonds around each site
    for ix1 in range(nx):
        for iy1 in range(ny):
            t1 = config[ix1, iy1]
            #
            # Scan over the neighbouring sites
            nab = 0
            for d12 in range(4):
                #
                # Find new x coordinate
                ix2 = (ix1 + dxtab[d12])%nx
                #
                # Find new y coordinate
                iy2 = (iy1 + dytab[d12])%ny
                #
                # Count up number of AB bonds
                t2 = config[ix2, iy2]
                if t1 != t2:
                    nab = nab + 1
            #
            # Update the probability distribution
            P[nab] = P[nab] + 1
    #
    # Normalise the distribution
    s = np.sum(P)
    P = P / s
    return N, P

def addInfo(ixa, iya, nBox, config, Ematrix, Eadd ):
    """
    ADDINFO Returns the energy change following the addition or removal of a particle

    Input arguments
       ixa         X coordinate of site
       iya         Y coordinate of site
       config      The configuration of surface atoms
       nBox        System size
       Ematrix     The 2x2 matrix of bond energies
       Eadd        The energy change on adding a particle to a clean surface

    Output arguments
       dE          Energy change following the addition or removal of a particle
    """
    #
    # Count the number of empty and occupied neighbours for site
    nna = 0
    nnb = 0
    for n in range(4):
        ix2, iy2 = getNeighbour(nBox, ixa, iya, n)
        if config[ix2, iy2] == cellA:
            nna = nna+1
        else:
            nnb = nnb+1
    #
    # Compute the energy with and without an atom on site
    dE_empty = Ematrix[cellA, cellA]*nna + Ematrix[cellA, cellB]*nnb
    dE_atom  = Ematrix[cellB, cellA]*nna + Ematrix[cellB, cellB]*nnb + Eadd
    #
    # Compute the energy change
    if config[ixa, iya] == cellA:
        dE = dE_atom - dE_empty
    else:
        dE = dE_empty - dE_atom

    return dE

def swapInfo(ixa, iya, dab, nBox, config, Ematrix):
    """
    SWAPINFO Returns the position of the neighbour and the energy change following a swap

    Input arguments
       ixa         X coordinate of first atom
       iya         Y coordinate of first atom
       dab         Direction of second atom relative to first. There are four
                   possible directions, so this takes values between 1 and
                   4. Together with ixa and ixb, this allows the position
                   of the second atom to be computed. This calculation is
                   done by getNeighbour
       config      The configuration of surface atoms
       nBox        System size
       Ematrix     The 2x2 matrix of bond energies

    Output arguments
       ixb         X coordinate of second atom
       iyb         Y coordinate of second atom
       dE          Energy change following swap
    """
    #
    # Find neighbour atom
    ta = config[ixa, iya]
    ixb, iyb = getNeighbour(nBox, ixa, iya, dab)
    tb = config[ixb, iyb]
    #
    # Find energy change
    if ta == tb:
        dE = 0.0
    else:
        #
        # Count the number of neighbours for site 1
        nna = 0
        for n in range(4):
            ix2, iy2 = getNeighbour(nBox, ixa, iya, n)
            if ((ix2 != ixb) or (iy2 != iyb)) and (config[ix2, iy2] == cellB):
                nna = nna+1
        #
        # Count the number of alloy neighbours for site 1
        nnb = 0
        for n in range(4):
            ix2, iy2 = getNeighbour(nBox, ixb, iyb, n)
            if ((ix2 != ixa) or (iy2 != iya)) and (config[ix2, iy2] == cellB):
                nnb = nnb+1
        #
        # Compute the total energy
        dEa = (Ematrix[tb, cellB]-Ematrix[ta, cellB])*nna + (Ematrix[tb, cellA]-Ematrix[ta, cellA])*(3-nna)
        dEb = (Ematrix[ta, cellB]-Ematrix[tb, cellB])*nnb + (Ematrix[ta, cellA]-Ematrix[tb, cellA])*(3-nnb)
        dE = dEa + dEb

    return ixb, iyb, dE

def getNeighbour (nBox, ix1, iy1, d12):
    """
    GETNEIGHBOUR returns the position of a neighbouring atom

    Input arguments
       nBox    The size of the simulation box
       ix1     X coordinate of first atom
       iy1     Y coordinate of first atom
       d12     Direction of second atom relative to first

    Output arguments
       ix2     X coordinate of second atom
       iy2     Y coordinate of second atom
    """
    #
    # Find new x coordinate
    dxtab = [0, 1, 0, -1]
    dx = dxtab[d12]
    ix2 = (ix1 + dx)%nBox
    #
    # Find new y coordinate
    dytab = [1, 0, -1, 0]
    dy = dytab[d12]
    iy2 = (iy1 + dy)%nBox
    #
    # Return the new coordinates
    return ix2, iy2

def surfaceMC(nBox, nSweeps, nEquil, T, Eaa, Eadd, P, job):
    """
    surfaceMC Performs Metropolis Monte Carlo of a lattice gas model of particles on
    a crystal surface. The added particles are represented as a 2 dimensional
    lattice gas in which particles can exchange position with neighbouring sites
    using the Metropolis alogorithm.

    Input arguments
    nBox    The size of the 2-D grid
    nSweeps The total number of Monte Carlo moves
    nEquil  The number of Monte Carlo moves used to equilibrate the system
    T       The temperature (K)
    Eaa     Particle-particle interaction energy (eV)
    Eadd    The energy change on adding a particle to a clean surface (eV) from vapour at the background pressure.
    P       Pressure in units of the background pressure P0
    job     Name or number given to this simulation. Useful for creating file names

    Output arguments
    fAtoms  The fraction of sites occupied by added particles
    nBar    The average number of unlike neighbours
    Ebar    The average energy
    C       The heat capacity
    """
    #
    # Set some interactions to zero
    # Eam     Particle-empty site interaction energy (eV)
    # Emm     Empty site-empty site interaction energy (eV)
    Eam = 0.0
    Emm = 0.0
    #
    # Compute kT in eV
    kB = 8.617332e-5
    kT = kB*T
    #
    # Compute the energy gained by adding a particle
    Eadd = Eadd - kT*m.log(P)
    #
    # Initialize the configuartion
    #   0   Empty site
    #   1   Particle
    nAtoms = 0
    config = np.zeros((nBox, nBox), dtype=int)
    #
    # Set up energy matrix
    Ematrix = np.zeros((2,2))
    Ematrix[cellA,cellA] = Emm
    Ematrix[cellA,cellB] = Eam
    Ematrix[cellB,cellA] = Eam
    Ematrix[cellB,cellB] = Eaa
    #
    # Initialise energy
    E = nBox*nBox*2*Emm
    #
    # Initialise data gathering
    Etable = np.zeros(nSweeps//1000 + 1)
    fTable = np.zeros(nSweeps//1000 + 1)
    nTable = 0
    Etable[nTable] = E
    nStats = 0
    Ebar = 0.0
    E2bar = 0.0
    fAtoms = 0.0
    #
    # Carry out the random swaps
    for n in range(nSweeps):
        if (n+1)%50000 == 0:
            print('.')
        elif (n+1)%1000 == 0:
            print('.', end='')
            nTable = nTable + 1
            Etable[nTable] = E
            fTable[nTable] = nAtoms/(nBox*nBox)
        ix = np.random.randint(0, nBox-1)
        iy = np.random.randint(0, nBox-1)
        #
        # Select move type
        if nAtoms > 0:
            im = np.random.randint(0, nAtoms)
        else:
            im = 0
        #
        if im == 0:
            #
            # Add or remove particle
            dE = addInfo(ix, iy, nBox, config, Ematrix, Eadd )
            if m.exp(-dE/kT) > np.random.random():
                if config[ix, iy] == cellA:
                    config[ix, iy] = cellB
                    nAtoms = nAtoms + 1
                else:
                    config[ix, iy] = cellA
                    nAtoms = nAtoms - 1
                E = E + dE
        else:
            #
            # Swap pair of particles
            ixn, iyn, dE = swapInfo(ix, iy, np.random.randint(0, 3), nBox, config, Ematrix)
            if m.exp(-dE/kT) > np.random.random():
                temp = config[ix, iy]
                config[ix, iy] = config[ixn, iyn]
                config[ixn, iyn] = temp
                E = E + dE
        #
        # Update statistics
        if n > nEquil:
            nStats = nStats + 1
            Ebar = Ebar + E
            E2bar = E2bar + E*E
            fAtoms = fAtoms + nAtoms/(nBox*nBox)
    #
    # Plot the configuration
    plt.figure(0)
    plt.pcolor(config)
    plt.savefig(job+'-config.png')
    plt.close(0)
    #
    # Plot the energy
    plt.figure(1)
    plt.plot (Etable[0:nTable+1])
    plt.title ("Energy")
    plt.xlabel("Time step / 1000")
    plt.ylabel("Energy")
    plt.savefig(job+'-energy.png')
    plt.close(1)
    #
    # Plot the coverage
    plt.figure(2)
    plt.plot (fTable[0:nTable+1])
    plt.title ("Coverage")
    plt.xlabel("Time step / 1000")
    plt.ylabel("Coverage")
    plt.savefig(job+'-coverage.png')
    plt.close(2)
    #
    # Plot the final neighbour distribution
    N, P = order2D(config)
    fAtoms = fAtoms/nStats
    N0, P0 = orderRandom(4, fAtoms)
    plt.figure(3)
    bar_width = 0.35
    plt.bar(N , P, bar_width, label="Simulation")
    plt.bar(N0+bar_width, P0, bar_width, label="Random")
    plt.title ("Distribution of unlike neighbours")
    plt.xlabel("Number of unlike neighbours")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(job+'-order.png')
    plt.close(3)
    #
    # Print statistics
    nBar = np.dot(N,P)
    Ebar = Ebar/nStats
    E2bar = E2bar/nStats
    C = (E2bar - Ebar*Ebar)/(kT*kT)
    print('')
    print('Coverage                                   = {0:7.3f}'.format(fAtoms))
    print('Heat capacity                              = {0:7.3f}'.format(C),' kB')
    print('The average number of unlike neighbours is = {0:7.3f}'.format(nBar))
    #
    # Return the statistics
    return fAtoms, nBar, Ebar, C

##########################################
# MAIN FUNCTION
##########################################
#
# This invokes the operations in the required order
def main():
    #
    # Define the simulation parameters
    nBox      = 30
    nEquil    = 500000
    nSweeps   = 1000000
    T_list    = [300]
    P_list    = [1, 10, 100, 1000, 10000, 100000, 1000000]
    Eaa_list  = [-0.02, 0.00, 0.02]
    Eadd_list = [-0.10, 0.00, 0.10]
    #
    # Open file to save the statistics
    file = open ("stats.csv", "w")
    file.write('Job number, Coverage, Temperature (K), Bond energy (eV), Adhesion energy (eV), Pressure (P0), Average number of unlike neighbours, Average energy (eV), Heat capacity (kB)\n')
    #
    # Loop over values
    count = 0
    for Eadd in Eadd_list:
        for P in P_list:
            for T in T_list:
                for Eaa in Eaa_list:
                    count = count + 1
                    job = '{:04d}'.format(count)
                    #
                    # Echo the parameters back to the user
                    print ("")
                    print ("Simulation ", job)
                    print ("----------------")
                    print ("Cell size                     = ", nBox)
                    print ("Total number of moves         = ", nSweeps)
                    print ("Number of equilibration moves = ", nEquil)
                    print ("Temperature                   = ", T, "K")
                    print ("Bond energy                   = ", Eaa, "eV")
                    print ("Adhesion energy               = ", Eadd, "eV")
                    print ("Pressure                      = ", P, "P0")
                    #
                    # Run the simulation
                    fAtoms, nBar, Ebar, C = surfaceMC(nBox, nSweeps, nEquil, T, Eaa, Eadd, P, job)
                    #
                    # Write out the statistics
                    file.write('{0:4d}, {1:6.4f}, {2:10.4f}, {3:7.4f}, {4:7.4f}, {5:7.4f}, {6:7.4f}, {7:14.7g}, {8:14.7g}\n'.format(count, fAtoms, T, Eaa, Eadd, P, nBar, Ebar, C))
    #
    # Close the file
    file.close()
    #
    # Sign off
    print('')
    print ("Simulations completed.")
#
# Ensure main is invoked
if __name__== "__main__":
    main()

#code adapted from Prof. Horsfield's Materials Modelling Course