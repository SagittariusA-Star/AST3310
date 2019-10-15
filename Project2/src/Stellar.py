import numpy as np 
import matplotlib.pyplot as plt 
from Stellar_Core import StellarCore

class Star(StellarCore):
    def __init__(self, R0, L0, T0, rho0, M0, alpha = 1, delta = 1, sanity_debug = False,
                  X=0.7, Y3=1e-10, Y=0.29, Z=0.01, Z_Li7=1e-7, Z_Be7=1e-7):
        StellarCore.__init__(self, R0, L0, T0, rho0, M0, X=0.7, Y3=1e-10,
                              Y=0.29, Z=0.01, Z_Li7=1e-7, Z_Be7=1e-7)

        self.cP             = 5/2*self.kB/(self.mu*self.my)     # [J K^-1 m^-1] Spesific hear capacity at constant pressure
        self.alpha          = alpha                             # Dimensionless parameter
        self.delta          = delta                             # Dimensionless parameter
        self.nabla_ad       = 2/5                               # Adiabatic temperature gradient
        self.sanity_debug   = sanity_debug                      # Debug parameters:
        self.debug_T        = 0.9e6                             # [K] Initial Temperaure
        self.debug_R        = 0.84*self.R_sun                   # [m] Initial Radius
        self.debug_M        = 0.99*self.M_sun                   # [kg] Initial Mass
        self.debug_rho      = 55.9                              # [kg m^-3]   Initial deisity
        self.debug_nab_s    = 3.26                              # Initial nabla_stable
        if sanity_debug:
            self.kappa = lambda logR, logT: np.array([np.log10(3.98) + 1]) # [m^2 kg^-1] Initial kappa
            self.my = 0.6                                           # Initial mean molecular weight

    def g(self, r, m):
        """Method returns gravitational acceleration g,
           for given radius r and mass m"""
        return self.G*m/r**2

    def HP(self, T, r, m):
        """Method returns pressure scale height
           for given temperature T, radius r and mass m."""
        return self.kB*T/(self.g(r, m)*self.my*self.mu)
    
    def lm(self, T, r, m):
        """Method returns mixing length for a rising blob,
           for given temperature T, radius r and mass m."""
        return self.alpha*self.HP(T, r, m)

    def U(self, T, rho, m, r):
        """Returns coefficient factor used in third degree
           equation to find Xi, se Xi methode."""
        T6      = T*1e-6                                                # [MK]
        logR    = np.log10(rho) - 3 - 3*np.log10(T6)   
        logT    = np.log10(T)
        kap     = 10**(self.kappa(logR, logT))*1e-1                     # Calculating kappa.
        u       = 64*self.sigma*T**3/(3*kap*rho**2*self.cP)             # [m^2]
        u      *= np.sqrt(self.HP(T, r, m)/(self.g(r, m)*self.delta))   # [m^2]
        return u[0]
    
    def Xi(self, nabla_stable, T, rho, m, r):
        """Method solves third degree equation for 
           Xi = sqrt(nabla_* - nabla_p). A, B, C and D
           are the coefficients of the equation"""
        A       = 1
        B       = self.U(T, rho, m, r)/self.lm(T, r, m)**2
        C       = 4*self.U(T, rho, m, r)**2/self.lm(T, r, m)**4 
        D       = self.U(T, rho, m, r)/(self.lm(T, r, m)**2)*(self.nabla_ad - nabla_stable)
        roots   = np.roots(np.array([A, B, C, D]))

        if np.sum(np.iscomplex(roots)) < 2:
            print("Warning! Less then two complex roots.\
                 All roots are real. Not able to deside which root to use")
            print("Roots: ", roots)

        real_roots = np.where(np.iscomplex(roots) == False) # Extracting real roots
        real_roots = np.real(roots[real_roots])             # Canceling 0j factor
        return real_roots[0]

    def v(self, T, r, m, nabla_stable, rho):
        """Method returns velocity of a gas blob for given temperature
           T, radius r, mass m, radiation temperature gradient nabla_stable
           as well as the density rho."""
        return (np.sqrt(self.g(r, m)*self.lm(T, r, m)**2/
               (4*self.HP(T, r, m)))*self.Xi(nabla_stable, T, rho, m, r))

    def nabla_stable(self, rho, T, r, m, L):
        """Method returns radiation temperature gradient for given
           density rho, temperature T, mass m and luminosity L."""
        T6              = T*1e-6                               #[MK]
        logR            = np.log10(rho) - 3 - 3*np.log10(T6)   # np.log10(rho*1e-3/T6**3)
        logT            = np.log10(T)
        kap             = 10**(self.kappa(logR, logT))*1e-1    # Calculating kappa.
        nabla_stable    = (3 * kap * rho * self.HP(T, r, m) * L 
                          / (64 * np.pi * r**2 * self.sigma * T**4))
        return nabla_stable
    
    def nabla_star(self, rho, T, r, m, L):
        """method returns temperature gradient of the star for given 
           density rho, temperature T, radius r, 
           mass m and luminosity L."""
        T6              = T*1e-6                               # [MK]
        logR            = np.log10(rho) - 3 - 3*np.log10(T6)   # np.log10(rho*1e-3/T6**3)
        logT            = np.log10(T)
        kap             = 10**(self.kappa(logR, logT))*1e-1    # Calculating kappa.
        nabla_stable    = self.nabla_stable(rho, T, r, m, L)   # Radiaution temperature gradiant 
        Xi              = self.Xi(nabla_stable, T, rho, m, r)  # Sqrt of difference between stars and gas blobs temp gradient
        stable          = nabla_stable < self.nabla_ad         # Radiative temperature gradient
        FC              = (rho*self.cP*T*np.sqrt(self.g(r, m)*self.delta)   
                            * self.HP(T, r, m)**(-3/2)*(self.lm(T, r, m)/2)**2*Xi**3)   # [W/m^2] Convective energy flux
        if stable:
            """If inside a radiation zone"""
            FC = 0  # [W/m^2] Convective energy flux
        nabla_star      = 3*kap*rho*self.HP(T, r, m)*(L/(4*np.pi*r**2) - FC)/(16*self.sigma*T**4)   # Actual temperature gradient
        return nabla_star

    def flux(self, rho, T, r, m, L):
        """Method returns convective and radiative flux, FC and FR,
           for given density rho, temperature T, radius r, mass m
           and luminosity L."""
        nabla_stable = self.nabla_stable(rho, T, r, m, L)   # Radiatitive temperature gradient
        stable = nabla_stable < self.nabla_ad             # Instability criterion
        T6 = T*1e-6                                         # [MK]
        logR = np.log10(rho) - 3 - 3*np.log10(T6)   
        logT = np.log10(T)
        kap = 10**(self.kappa(logR, logT))*1e-1             # Calculating kappa.
        Xi = self.Xi(nabla_stable, T, rho, m, r)            # Sqrt of difference between stars and gas blobs temp gradient
        FC = (rho*self.cP*T*np.sqrt(self.g(r, m)*self.delta)
                * self.HP(T, r, m)**(-3/2)*(self.lm(T, r, m)/2)**2*Xi**3)   # [W/m^2] Convective energy flux
        if stable:
            """If convectively unstable FC = 0"""
            FC = 0  # [W/m^2] Convective energy flux
        FR = L/(4*np.pi*r**2) - FC # [W/m^2] Radiative energy flux
        return FC, FR

    def f(self, m, y):
        """method returns defivatives used to 
           solve PDEs in solver method"""
        drdm, dPdm, dLdm, dTdm = StellarCore.f(self, m, y)  # Derivatives
        r, P, L, T = y                                      # Current function values
        rho = self.rho(P, T)                                # [kg m^-3] density
        T6 = T*1e-6                                         # [MK]
        logR = np.log10(rho) - 3 - 3*np.log10(T6)   
        logT = np.log10(T)
        kap = 10**(self.kappa(logR, logT))*1e-1             # Calculating kappa.
        nabla_stable = self.nabla_stable(rho, T, r, m, L)   # Radiative temperature gradient
        unstable = nabla_stable > self.nabla_ad             # Instability criterion
        FC, FR = self.flux(rho, T, r, m, L)                 # [W/m^2] Convective and radiative energy flux 
        if unstable:
            """If unstable temperature gradient is updated to
               include convective energy transport"""
            nabla_star = self.nabla_star(rho, T, r, m, L)               # Actual temperature gradient
            dTdm = -T/self.HP(T, r, m)*nabla_star/(4*np.pi*r**2*rho)    # Actual temperature gradient in terms of derivatives

        if self.sanity_debug:
            """If test used in sanity checks"""
            nabla_star = self.nabla_star(rho, T, r, m, L)
            print(f"Nabla* = {nabla_star[0]:.2f}")
            print(f"Nabla_stable = {nabla_stable[0]:.2f}")
            print(f"FC/(FR+FC) = {FC/(FR+FC):.3f} ")
            print(f"FR/(FR+FC) = {FR/(FR+FC):.3f} ")

        return np.array([drdm, dPdm, dLdm, dTdm])

    def PPchain(self, T, rho):
        """Energy production of each PP-branch"""
        r           = self.reaction_rate(T, rho)          # [kg^-1 s^-1] Reaction rates
        E_PPI       = (2*self.Q[0] + self.Q[1]) * r[1]    # [W/kg] Energy produced by PPI  
       
        E_PPII      =  (self.Q[0] + self.Q[2]) * r[2]     # [W/kg] Energy produced by PPII
        E_PPII     +=  self.Q[3]*r[3] + self.Q[4]*r[4]
       
        E_PPIII     = (self.Q[0] + self.Q[2]) * r[2]      # [W/kg] Energy produced by PPIII
        E_PPIII    += self.Q[5]*r[5]
        return E_PPI, E_PPII, E_PPIII

    @property 
    def FC_list(self):
        """Property generates plotable arrays of relative
           convective and radiative flux, FC and FR."""
        r, P, L, T  = self.y        # Current function values
        rho         = self.rho(P,T) # [kg m^-3] density
        n           = len(r)        # Array length
        F_C_list    = np.zeros(n) 
        F_R_list    = np.zeros(n)
        for i in range(n):
            fc, fr = self.flux(rho[i], T[i], r[i], self.m[i], L[i])
            F_C_list[i] = fc/abs(fc + fr)   # Relative convective flux
            F_R_list[i] = fr/abs(fc + fr)   # Relative radiative flux
        return F_C_list, F_R_list   

if __name__ == "__main__":
    """First Sanity check"""
    R0 = 6.96e8         # [m] Initial radius
    L0 = 3.846e26       # [W]  Initial luminocity
    M0 = 1.989e30       # [kg] Initial mass
    rho0 = 1.42e-7*1.408e3  # [kg m^-3] Initial density
    T0 = 5770               # [K]  Initial temperature
    
    """Debug parameters"""
    debug_T  = 0.9e6        # [K] debug temperature
    debug_R  = 0.84*R0      # [m] debug radius
    debug_M  = 0.99*M0      # [kg] debug mass
    debug_rho = 55.9        # [kgm^-3] debug density
    debug_nab_s = 3.26
    """Generating plottable objects and printing sanity check"""

    Star1 = Star(R0, L0, T0, rho0, M0, sanity_debug = True)
    debug_P = Star1.P(debug_rho, debug_T)

    debug_HP = Star1.HP(debug_T, debug_R, debug_M)
    debug_U = Star1.U(debug_T, debug_rho, debug_M, debug_R)
    debug_Xi = Star1.Xi(debug_nab_s, debug_T, debug_rho, debug_M, debug_R)
    debug_v = Star1.v(debug_T, debug_R, debug_M, debug_nab_s, debug_rho)

    print("First sanity check: ")
    print(f"HP = {debug_HP*1e-6:.2f} Mm")
    print(f"U = {debug_U:.3e}")
    print(f"Xi = {debug_Xi:.3e}")
    print(f"v = {debug_v:.4f} m/s")
    y = np.array([debug_R, debug_P, L0, debug_T])
    Star1.f(debug_M, y)
    
    """Plotting second and third sanity check"""
    SanityStar = Star(R0, L0, T0, rho0, M0, sanity_debug = False)
    m, y = SanityStar.solve((M0, 0), 0.01, 1e-4, variable_step=True, debug=False)
    R, P, L, T = y
    rho = SanityStar.rho(P, T)

    nabla_stable = np.zeros_like(R)
    nabla_star = np.zeros_like(R)
    for i in range(len(R)):
        nabla_stable[i] = SanityStar.nabla_stable(rho[i], T[i], R[i], m[i], L[i])
        nabla_star[i] = SanityStar.nabla_star(rho[i], T[i], R[i], m[i], L[i])

    plt.semilogy(R/R0, nabla_stable, "r", label=r"$\nabla_{stable}$")
    plt.semilogy(R/R0, nabla_star, "b", label=r"$\nabla_{*}$")
    plt.semilogy(R/R0, SanityStar.nabla_ad*np.ones_like(m), "g", label=r"$\nabla_{ad}$")
    plt.legend(loc = 0)
    plt.xlabel(r"$R/R_\odot$")
    plt.ylabel(r"$\nabla$")

    """Cross section sanity check"""

    R0 = 6.96e8    # [m] Initial radius
    L0 = 3.846e26       # [W]  Initial luminocity
    M0 = 1.989e30       # [kg] Initial mass
    rho0 = 1.42e-7*1.408e3   # [kg m^-3] Initial density
    T0 = 5770         # [K]  Initial temperature

    SanityStar = Star(R0, L0, T0, rho0, M0, sanity_debug=False)
    m, y = SanityStar.solve((M0, 0), 0.01, 1e-4, variable_step=True, debug=False)
    R_values, P, L_values, T = y                # Function values
    rho = SanityStar.rho(P, T)                  # [kgm^-3]
    n = len(m)                                  # Array length

    F_C_list, F_R_list = SanityStar.FC_list     # Relative convective and radiative energy fluxes

    R_values /= 6.96e8                          # Normalizing radial distance array
    L_values /= L0                              # Normalizing luminocity array
    R0 /= 6.96e8                                # Normalizing initial radius

    show_every = 5
    core_limit = 0.995

    plt.figure(figsize=(8, 5))
    plt.grid()
    fig = plt.gcf() # get current figure
    ax = plt.gca()  # get current axis
    rmax = 1.2*R0
    ax.set_xlim(-rmax,rmax)
    ax.set_ylim(-rmax,rmax)
    ax.set_aspect('equal')	# make the plot circular
    j = show_every
    for k in range(0, n-1):
        j += 1
        if j >= show_every:	# don't show every step - it slows things down
            if(L_values[k] > core_limit):	# outside core
                if(F_C_list[k] > 0.0):		# convection
                    circR = plt.Circle((0,0),R_values[k],color='red',fill=False)
                    ax.add_artist(circR)
                else:				# radiation
                    circY = plt.Circle((0,0),R_values[k],color='yellow',fill=False)
                    ax.add_artist(circY)
            else:				# inside core
                if(F_C_list[k] > 0.0):		# convection
                    circB = plt.Circle((0,0),R_values[k],color='blue',fill = False)
                    ax.add_artist(circB)
                else:				# radiation
                    circC = plt.Circle((0,0),R_values[k],color='cyan',fill = False)
                    ax.add_artist(circC)
            j = 0
    circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)		# These are for the legend (drawn outside the main plot)
    circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
    circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
    circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
    ax.legend([circR, circY, circC, circB], ['Convection outside core', 'Radiation outside core', 'Radiation inside core', 'Convection inside core'], loc = 0) # only add one (the last) circle of each colour to legend
    plt.xlabel(r"$R/R_\odot$")
    plt.ylabel(r"$R/R_\odot$")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 2)
    plt.title('Cross-section of star')

    # Show all plots
    plt.show()
        
