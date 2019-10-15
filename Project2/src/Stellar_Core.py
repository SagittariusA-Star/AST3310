import numpy as np 
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt 
from energy_production import EnergyProduction


class StellarCore(EnergyProduction):
    """Class simulates stellar core and radiation zone"""
    G = 6.67e-11        # [m^3 kg^-1 s^-2]   Gravitational constant
    c = 299792458       # [m/s]              Speed of light
    sigma = 5.67036e-8  # [W m^-2 K^-4]      Stefan-Boltzmann's constant
    kB = 1.38e-23       # [J/K]              Boltzmann's constant
    mu = 1.6605e-27     # [kg]               Aomic mass unit in kg
    L_sun   = 3.846e26                          # [W]                Solar luminocity
    M_sun   = 1.989e30                          # [kg]               Solar mass
    R_sun   = 6.96e8                            # [m]                Solar radius
    rho_sun = M_sun/(4*np.pi/3*(R_sun)**3)      # [kg m^-3]          Mean solar density


    def __init__(self, R0, L0, T0, rho0, M0, X=0.7, Y3=1e-10,
                 Y=0.29, Z=0.01, Z_Li7=1e-7, Z_Be7=1e-7):
        EnergyProduction.__init__(self, X=0.7, Y3=1e-10,
                                  Y=0.29, Z=0.01, Z_Li7=1e-13, Z_Be7=1e-13)
        self.rho0 = rho0                                # [kg m^-3] Initial density
        self.M0 = M0                                    # [kg] Initial mass
        self.X = X                                      # H-1 mass fraction
        self.Y = Y                                      # He-4 mass fraction
        self.Y3 = Y3                                    # H3-3 mass fraction
        self.Z = Z                                      # Metalicity
        self.Z_Li7 = Z_Li7                              # Li-7 mass fraction
        self.Z_Be7 = Z_Be7                              # Be-7 mass fraction
        self.my = (1/(2*self.X + self.Y3 + 3/4*self.Y 
                  + 4/7*self.Z_Li7 + 5/7*self.Z_Be7 + 0.5*self.Z))   # Mean atomic weight
        
        y0 = np.zeros(4)                                # Generating inital value:
        y0[0] = R0                                      # Initial radius
        y0[2] = L0                                      # Initial Luminocity
        y0[3] = T0                                      # Initial temperature
        y0[1] = self.P(rho0, T0)                        # Inital Pressure
        self.y0 = y0
        self.kappa = self.read_kappa()                  # Generating callable log(kappa)
                                                        # function (see read_kappa methode).

    def read_kappa(self):
        """The kappa function reads kappa data from .txt file
           and generates a callable function of log(T) and log(R)
           (T is Temperature and R is rho/(T*1e-6)^3). 
           Returned callable function returns log(kappa) in cgs units."""

        infile = open("opacity.txt", "r")
        lines = infile.readlines()
        N = len(lines)
        z = np.zeros(shape=(N-2, 19))
        x = np.array((lines[0].split()[1:]), dtype=float)
        y = np.zeros(N-2, dtype=float)
        for i, line in enumerate(lines[2:]):
            string = line.split()
            y[i] = float(string[0])
            z[i, :] = string[1:]
        return interp2d(x, y, z, kind="linear")

    def solve(self, m_span, p, m_frac, variable_step=True, debug=False):
        """This function solves the coupled differential equations
           for radius (r), pressure (P), luminocity (L) and temperature (T)
           using a Runge-Kutta 4 algorithm (RK4).
           Inputs:
           m_span       : Tuple or array with initial and final mass (m0, mf).
           p            : Tolerance for variable step lenth.
           frac         : Step length used in static step length. 
                          Also used to generate length of arrays storing PDE solutions.
           variable_step: If true it used adaptive steps, 
                          else it uses static steps.
           debug        : If true function values and dm printed out every 50-th
                          loop iteration"""
        m0, mf = m_span
        N = int(1/abs(m_frac))                      # Max length of arrays and for loops.
        y = np.zeros(shape=(len(self.y0), N + 1))   # r, P, L and T values.
        m = np.zeros(N + 1)                         # Mass values.
        y[:, 0] = self.y0                           #Initial values
        m[0] = m0                                   #Initial mass
        c = 0                                       #Counter for debug mode

        if variable_step:
            for i in range(N):
                """RK4 with variable step length"""
                k1 = self.f(m[i], y[:, i])
                dm = -np.min(p*y[:, i]/np.abs(k1))
                k2 = self.f(m[i] + 0.5*dm, y[:, i] + 0.5*dm*k1)
                k3 = self.f(m[i] + 0.5*dm, y[:, i] + 0.5*dm*k2)
                k4 = self.f(m[i] + dm, y[:, i] + dm*k3)

                y[:, i + 1] = y[:, i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)*dm
                m[i + 1] = m[i] + dm
                c += 1
                if c == 50 and debug:
                    c = 0  #Resetting debug counter
                    print("--------------------------------------------")
                    print("m  : {0} M_sun".format(m[i]/self.M_sun))
                    print("r  : {0} R_sun".format(y[0, i]/self.R_sun))
                    print("rho: {0} g/cm^3"
                          .format(self.rho(y[1, i], y[3, i])*1e-3))
                    print("P  : {0} PPa"
                          .format(self.P(self.rho(y[1, i], y[3, i]), y[3, i])*1e-15))
                    print("L  : {0} L_sun".format(y[2, i]/self.L_sun))
                    print("T  : {0} MK".format(y[3, i]*1e-6))
                    print("dm : {0} M_sun".format(dm/self.M_sun))
                    print("--------------------------------------------")

                if m[i + 1] <= 0 or np.any(y[:, i] < 0):
                    """Breaking loop if mass, r, P, L 
                       or T becomes negative"""
                    print("Mass or functional value is negative! Breaking loop!")
                    break
        else:
            dm = - m_frac*self.M_sun
            for i in range(N):
                """RK4 with static step length"""
                k1 = self.f(m[i], y[:, i])
                k2 = self.f(m[i] + 0.5*dm, y[:, i] + 0.5*dm*k1)
                k3 = self.f(m[i] + 0.5*dm, y[:, i] + 0.5*dm*k2)
                k4 = self.f(m[i] + dm, y[:, i] + dm*k3)

                y[:, i + 1] = y[:, i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)*dm
                m[i + 1] = m[i] + dm
                c += 1
                if c == 50 and debug:
                    c = 0   # Resetting debug counter
                    print("--------------------------------------------")
                    print("m  : {0} M_sun".format(m[i]/self.M_sun))
                    print("r  : {0} R_sun".format(y[0, i]/self.R_sun))
                    print("rho: {0} g/cm^3"
                          .format(self.rho(y[1, i], y[3, i])*1e-3))
                    print("P  : {0} PPa"
                          .format(self.P(self.rho(y[1, i], y[3, i]), y[3, i])*1e-15))
                    print("L  : {0} L_sun".format(y[2, i]/self.L_sun))
                    print("T  : {0} MK".format(y[3, i]*1e-6))
                    print("dm : {0} M_sun".format(dm/self.M_sun))
                    print("--------------------------------------------")

                if abs(m[i + 1]) <= 0 or np.any(y[:, i] < 0):
                    """Breaking loop if mass, r, P, L 
                       or T becomes negative"""
                    print("Mass or functional value is negative! Breaking loop!")
                    break
        self.m = m[:i]
        self.y = y[:, :i]
        return m[:i], y[:, :i]

    def P(self, rho, T):
        """This function calculates the pressure P 
           for a given density rho and temperature T.
           Mostly used for initial pressure."""

        PG = rho*self.kB*T/(self.my*self.mu)    # [Pa] Gass pressure
        
        PR = (4*self.sigma*T**4)/(3*self.c)     # [Pa] Radiation pressure
        return PG + PR

    def rho(self, P, T):
        """This function returns the density rho for given
           pressure P and temperature T."""
        PR = (4*self.sigma*T**4)/(3*self.c)     # [Pa] Radiation pressure
        PG = P - PR                             # [Pa] Gass pressure
        return self.my*self.mu*PG/(self.kB*T)

    def f(self, m, y):
        """Function generates an array of derivatives
           (radius r, pressure P, luminocity L and temperature T wrt. mass)
            for the coupled differential equations. 

           m: mass at i-th loop iteration of solve method.
           y: array containing function values of radius r, pressure P,
              luminocity L and temperature T."""
        r, P, L, T = y
        T9 = T*1e-9
        T6 = T*1e-6
        
        rho = self.rho(P, T)                        # [kg m^-3]
        eps = self.epsilon(T9, rho)                 # Total energy production for given T and rho.
        logR = np.log10(rho) - 3 - 3*np.log10(T6)   # np.log10(rho*1e-3/T6**3)
        logT = np.log10(T)
        kap = 10**(self.kappa(logR, logT))*1e-1     # Calculating kappa.

        drdm = 1/(4*np.pi*r**2*rho)                             # Calculating derivatives:
        dPdm = - self.G*m/(4*np.pi*r**4)                        # ----------||----------
        dLdm = eps                                              # ----------||----------
        dTdm = - 3*kap*L/(256*np.pi**2*self.sigma*r**4*T**3)    # ----------||----------
        return np.array([drdm, dPdm, dLdm, dTdm])

if __name__ == "__main__":
    """Sanity check"""
    R0 = 0.72*6.96e8    # [m] Initial radius
    L0 = 3.846e26       # [W]  Initial luminocity
    T0 = 5.7e6          # [K]  Initial temperature
    rho0 = 5.1*(1.989e30/(4*np.pi/3*(6.96e8)**3))   # [kg m^-3] Initial density
    M0 = 1.989e30       # [kg] Initial mass
    
    Core = StellarCore(R0*0.258, L0, T0, rho0*1.1, M0*0.427)
    m, y = Core.solve((M0*0.427, 0), 0.01, 1e-4, variable_step=False, debug=False)
    R, P, L, T = y
    Rho = Core.rho(P, T)

    T6 = T0*1e-6
    logR = np.log10(rho0) - 3 - 3*np.log10(T6)   # np.log10(rho*1e-3/T6**3)
    logT = np.log10(T0)
    kap = 10**(Core.kappa(logR, logT))*1e-1     # Calculating kappa.

    print(kap)
    print(T0)
    print(rho0)

    plt.subplots(2, 2, figsize = (12, 6))
    plt.tight_layout(pad = 3.5)

    plt.subplot(2, 2, 1)    
    plt.plot(m/1.989e30, R/6.96e8, label="R(m)")
    plt.xlabel(r"$M/M_\odot$", fontsize=12)
    plt.ylabel(r"$R/R_\odot$", fontsize=12)
    plt.legend(loc=0)
    
    plt.subplot(2, 2, 2)
    plt.plot(m/1.989e30, Rho/(1.989e30/(4*np.pi/3*(6.96e8)**3)), label=r"$\rho(m)$")
    plt.xlabel(r"$M/M_\odot$", fontsize=12)
    plt.ylabel(r"$\rho/\rho_\odot$", fontsize=12)
    plt.legend(loc=0)

    plt.subplot(2, 2, 3)
    plt.plot(m/1.989e30, L/3.846e26, label="L(m)")
    plt.xlabel(r"$M/M_\odot$", fontsize=12)
    plt.ylabel(r"$L/L_\odot$", fontsize=12)
    plt.legend(loc=0)

    plt.subplot(2, 2, 4)
    plt.plot(m/1.989e30, T*1e-6, label="T(m)")
    plt.xlabel(r"$M/M_\odot$", fontsize=12)
    plt.ylabel(r"$T$[MK]", fontsize=12)
    plt.legend(loc=0)
    
    plt.show()
