import numpy as np 


class EnergyProduction:
    """Class calculates the energy production by PP-chains of the
       stellar radiation zone as a function of temperature T
       and density rho"""
    mu = 1.6605e-27     # [kg] atomic mass unit in kg
    NA = 6.022e23       # Avogadros constant
    def __init__(self, X = 0.7, Y3 = 1e-10, 
                 Y = 0.29, Z = 0.01, Z_Li7 = 1e-7, Z_Be7 = 1e-7):
        self.X      = X         # H-1 mass fraction
        self.Y3     = Y3        # He-3 mass fraction
        self.Y      = Y         # H3-4 mass fraction
        self.Z      = Z         # Metalicity
        self.Z_Li7  = Z_Li7     # Li-7 mass fraction
        self.Z_Be7  = Z_Be7     # Be-7 mass fraction
        self.Q      = np.array([0.15 + 1.02 + 5.49, 12.86, 1.59,
                                0.05, 17.35, 0.14 + 6.88 + 3.00 + 1.02])*1.6e-13 # PP-chain reaction energies

    def number_density(self, rho):
        """Returns number density of each isotope, needed
           in the PP-chains, as a function of mass density rho.""" 
        n = np.array([self.X, self.Y3/3, self.Y/4, 1, self.Z_Li7/7, self.Z_Be7/7])
        n *= rho/self.mu
        n[3] = n[0] + 2*n[1] + 2*n[2] + 1*n[4] + 2*n[5]
        return n 
    
    def lamb(self, T, rho):
        """Returns reaction proportionality functions
           for each of the PP-chain reaction using 
           temperature T and mass density rho.           
           Suffix description:
            pp: 
                1)     H1 + H1 --> D2 + e^+ + nu_e
                2)     D2 + H1 --> 2*H1
            33 (PP-I):
                a)     He3 + He3 --> He4 + 2*H1 
            34 (PP-II):
                b)     He3 + He4 --> Be7
            e7 (PP-II):
                c)     Be7 + e^- --> Li7 + nu_e
            17_ (PP-II):
                d)     Li7 + H1 --> 2*He4
            17 (PP-III):
                e)     Be7 + H1 --> B8
                f)     B8 --> Be8 + e^+ + nu_e
                g)     Be8 --> 2*He4 
           """
        n           = self.number_density(rho) # Number densities

        Tstar       = T/(1 + 4.95e-2*T)
        Tstar2      = T/(1 + 0.759*T)
        """Proportionality functions"""
        lambpp      = (4.01e-15*T**(-2/3)*np.exp(-3.380*T**(-1/3))
                        *(1 + 0.123*T**(1/3) + 1.09*T**(2/3) + 0.938*T))
        
        lamb33      = (6.04e10*T**(-2/3)*np.exp(-12.276*T**(-1/3))
                        *(1 + 0.034*T**(1/3) - 0.522*T**(2/3) - 0.124*T 
                        + 0.353*T**(4/3) + 0.213*T**(-5/3)))
        
        lamb34      = (5.61e6*Tstar**(5/6)*T**(-3/2)*np.exp(-12.826*Tstar**(-1/3)))

        lambe7      = (1.34e-10*T**(-1/2)*(1 - 0.537*T**(1/3) + 3.86*T**(2/3)
                        + 0.0027/T*np.exp(2.515e-3/T)))

        lamb17_  = (1.096e9*T**(-2/3)*np.exp(-8.472*T**(-1/3)) 
                        - 4.830e8*Tstar2**(5/6)*T**(-3/2)*np.exp(-8.472*Tstar2**(-1/3))
                        + 1.06e10*T**(-3/2)*np.exp(-30.442/T))
        
        lamb17      = (3.11e5*T**(-2/3)*np.exp(-10.262*T**(-1/3)) 
                        + 2.53e3*T**(-3/2)*np.exp(-7.306/T))

        if T <= 1e-3 and lambe7 > 1.57e-7/n[3]:
            """Correcting for temperature dependent
               efficiency of reaction c)""" 
            lambe7 = 1.57e-7/n[3]
            
        return np.array([lambpp, lamb33, lamb34, lambe7, lamb17_, lamb17])*1e-6/self.NA

    def reaction_rate(self, T9, rho):
        """Returns reaction rate for input temperature
           T9 [GK] and density rho"""
        l     = self.lamb(T9, rho)       #Proportionality functions
        n     = self.number_density(rho)#Number densities
        r     = l/rho                   #Reaction rates
        """Reaction rates corresponding to 
           reactions in lamb function:"""
        r[0] *= 0.5*n[0]**2     # 1) and 2)
        r[1] *= 0.5*n[1]**2     # a)
        r[2] *= n[1]*n[2]       # b)
        r[3] *= n[3]*n[5]       # c)
        r[4] *= n[0]*n[4]       # d)
        r[5] *= n[0]*n[5]       # e), f) and g)
        
        if r[0] < 2*r[1] + r[2]:
            """Making sure no more He3 is used
               than is produced"""
            K = r[0]/(2*r[1] + r[2])
            r[1] = K*r[1]
            r[2] = K*r[2]
        
        if r[2] < r[3] + r[5]:
            """Making sure no more Be7 is used
            than is produced"""
            K = r[2]/(r[3] + r[5])
            r[3] = K*r[3]
            r[5] = K*r[5]

        if r[3] < r[4]:
            """Making sure no more Li7 is used
               than is produced"""
            r[4] = r[3] 
        return r
    
    def epsilon(self, T, rho):
        """Returns total energy produced per unit mass 
           by the PP-chains as a function
           of temperature T and density rho"""
        r = self.reaction_rate(T, rho)
        return np.inner(r, self.Q)

if __name__ == "__main__":
    """Sanity chenck for Appendix C"""
    #T = 1.57e7*1e-9
    T = 0.1
    rho = 1.62e5
    E = EnergyProduction()
    r = E.reaction_rate(T, rho)

    Sanity = r*E.Q*rho
    print(Sanity)
 
