# visulaliser
import FVis3 as FVis
import numpy as np
from astropy import constants as const
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

class Convection2D:
    """Defining physical constants"""
    G       = const.G.value
    M_sun   = const.M_sun.value
    R_sun   = const.R_sun.value
    g       = G * M_sun / R_sun**2
    kB      = const.k_B.value
    mu      = const.u.value
    my      = 0.61

    def __init__(self, init_cond, p, sig, mean, \
                 nx = 300, ny = 100, X = 12e6, Y = 4e6, pert = 0, pert_factor = 0.5):
        """Defining varables"""
        self.init_cond  = init_cond # (T0, P0) Initial temperature T0 and pressure P0 in tuple
        self.p          = p         # Accuracy parameter used in variable step length
        self.nx         = nx        # Grid elements in horizontal (x) axis
        self.ny         = ny        # Grid elements in vertical (y) axis
        self.X          = X         # [m] Length along horizontal (x) axis
        self.Y          = Y         # [m] Length along vertical (y) axis
        self.xarr       = np.linspace(0, self.X, self.nx)   # [m] x-axis
        self.yarr       = np.linspace(0, self.Y, self.ny)   # [m] y-axis
        self.dx         = np.abs(self.xarr[0] - self.xarr[1])   # [m] x-axis differential
        self.dy         = np.abs(self.yarr[0] - self.yarr[1])   # [m] y-axis differetial
        self.gamma      = 5 / 3 # Adiabatic constant for point particle gas
        self.nabla      = 2 / 5 + 1e-2  # Double logarithmic temperature gradient
        self.T          = np.zeros(shape = (self.ny, self.nx))  # [K] Defining empty arrays of temperature  
        self.P          = np.zeros(shape = (self.ny, self.nx))  # [Pa] Defining empty arrays of pressure
        self.u          = np.zeros(shape = (self.ny, self.nx))  # [m/s] Defining empty arrays of horizontal velocity
        self.w          = np.zeros(shape = (self.ny, self.nx))  # [m/s] Defining empty arrays of vertical velocity
        self.pert       = pert  # Gaussian temperature pertubation on (vaule 1) of off (vaule 0)
        self.e_flux     = []
        #self.e_flux     = np.zeros_like(self.yarr)
        self.t_array    = []
        self.FC_tot     = []
        self.t_iterate  = 0
        self.fig        = plt.figure()
        self.ax         = self.fig.add_subplot(111, autoscale_on = False, \
                                               ylim = [0, self.Y/1e6], xlim = [-2e15, 2e15])
        #self.ax.set_xlim([0, self.X / 1e6])
        self.ax.set_ylim([0, self.Y / 1e6])
        self.line,      = self.ax.plot([])
        self.time_text  = self.ax.text(x = 0.03, y = 0.92, s = "", transform = self.ax.transAxes, fontsize = 12)
        self.totalflux_text = self.ax.text(x = 0.63, y = 0.92, s = "", transform = self.ax.transAxes, fontsize = 12)
        self.pert_factor        = pert_factor # Factor of initial temperature T0, amplitude of temperature pertubation.
        self.sigx, self.sigy    = sig   # [m] Standard deviation of temperature pertubation, roughly blob size.
        self.meanx, self.meany  = mean  # [m] Centre position of temperature pertubation.
        self.Xgrid, self.Ygrid  = np.meshgrid(self.xarr, self.yarr) # [m] Grid matrises


    def T_initialize(self, y):
        """Method returns initial temperature profile
           at hydrostatic equilibrium (without pertubation)."""
        T0, P0  = self.init_cond    # Photosphere temperature and pressure
        T       = T0 - self.mu * self.my * self.nabla * self.g * (y - self.Y) / self.kB # [k]
        return T

    def P_initialize(self, y):
        """Method returns initial pressure profile
           at hydrostatic equilibrium."""
        T0, P0  = self.init_cond    # Photosphere temperature and pressure
        P       = P0 * (1 - (self.mu * self.my * self.g * (y - self.Y) * self.nabla) \
                            / (self.kB * T0))**(1 / self.nabla) # [Pa]
        return P

    def rho_initialize(self):
        """Method returns initial density prophile."""
        T0, P0  = self.init_cond    # Photosphere temperature and pressure
        rho     = (self.gamma - 1) * self.e[:, :] * self.mu * self.my / (self.kB * self.T[:, :]) # [kg m^-3]
        return rho

    def e_initialize(self):
        """Method returns initial internal specific energy prophile."""
        T0, P0  = self.init_cond    # Photosphere temperature and pressure
        e       = self.P[:, :] / (self.gamma - 1) # [J m^-3]
        return e

    def gaussian_pert(self):
        """Gaussian pertubation function. Centred at (self.meanx, self.meany),
           horizontal width self.sigx and vertical width self.sigy."""
        pertubation = np.exp(-0.5 * (self.sigx**-2 * (self.Xgrid - self.meanx)**2 \
                             + self.sigy**-2 * (self.Ygrid - self.meany)**2))
        #pertubation  = np.exp(-0.5 * (self.Ygrid - self.meany)**2 *self.sigy **-2)
        #pertubation *= np.sin(3 * 2 * np.pi * self.Xgrid/ self.X) 
        return pertubation

    def initialize(self):
        """initialize temperature T, pressure P, density rho 
           and internal specific energy e. A temperature
           pertubation is added if self.pert = 1 and 
           not added when self.pert = 0."""
        T0, P0  = self.init_cond    # Temperature and pressure at top of grid
        T_init  = self.T_initialize(self.yarr)  # [K] Generating initial temperature profile array
        P_init  = self.P_initialize(self.yarr)  # [Pa] Generating initial pressure profile array
        for i in range(self.nx):
            self.T[:, i] = T_init   # [K] Filling in temperature grid
            self.P[:, i] = P_init   # [Pa] Filling in pressure grid

        self.e          = self.e_initialize()   # [J m^-3] # Defining specific internal energy grid
        self.T[:, :]   += self.pert_factor * T0 * self.pert * self.gaussian_pert() # Adding a temperature pertubation if self.pert = 1
        self.rho        = self.rho_initialize() # [kg m^-3] Defining density grid
        
    def get_timestep(self):
        """Method calculates and returns time step dt.
           If calculated dt is within 1e-4s <= dt <= 1e-2s
           variable step length is used, else is static dt = 1e-2s
           if dt >= 1e-2s or dt = 1e-7s if dt <= 1e-7s."""
        rel_rho = np.max(np.abs(self.drhodt[:, :] / self.rho[:, :]))    # Calculating ratios
        rel_e   = np.max(np.abs(self.dedt[:, :]   / self.e[:, :]))      # between derivatives and variables.
        rel_x   = np.max(np.abs(self.u[:, :]      / self.dx))           # -------------||--------------
        rel_y   = np.max(np.abs(self.w[:, :]      / self.dy))           # -------------||--------------
        rels    = np.array([rel_rho, rel_e, rel_x, rel_y])              # Array of ratios
        rels    = rels[np.where(np.isinf(rels) == False)]               # Filtering out inf
        rels    = rels[np.where(np.isnan(rels) == False)]               # Filtering out NaN
        delta   = np.nanmax(rels)                                       # Maximum of ratios
        dt      = self.p / delta                                        # [s] Calculating time differential dt

        if dt >= 1e-1:
            """Setting dt static if dt too big"""
            dt = 1e-1   # [s]

        if dt <= 1e-7:
            """Setting dt static if dt too small"""
            dt = 1e-7   # [s]
        return dt

    def set_boundary_conditions(self):
        """Boundary conditions for energy, density and velocity
           at vertical upper and lower boundaries."""
        self.e[0, :]    = (4 * self.e[1, :] - self.e[2, :]) \
                          / (3 - 2 * self.dy * self.mu * self.my *self.g / (self.kB * self.T[0, :]))

        self.e[-1, :]   = (4 * self.e[-2, :] - self.e[-3, :]) \
                          / (3 + 2 * self.dy * self.mu * self.my * self.g / (self.kB * self.T[-1, :]))
        
        self.rho[0, :]  = (self.gamma - 1) * self.e[0, :] * self.mu * self.my / (self.kB * self.T[0, :])
        self.rho[-1, :] = (self.gamma - 1) * self.e[-1, :] * self.mu * self.my / (self.kB * self.T[-1, :])

        self.w[0, :]    = np.zeros(self.nx)
        self.w[-1, :]   = np.zeros(self.nx)

        self.u[0, :]    = (4 * self.u[1, :] - self.u[2, :]) / 3
        self.u[-1, :]   = (4 * self.u[-2, :] - self.u[-3, :]) / 3

    def central_x(self, func):
        """Central difference scheme in x-direction.
           Automatically takes care of periodic 
           horizontal boundries using np.roll()."""
        diff = (np.roll(func, -1, axis = 1) - np.roll(func, 1, axis = 1))/(2 * self.dx)
        return diff

    def central_y(self, func):
        """Central difference scheme in y-direction.
           Periodic boundary due to np.roll() is later
           fixed through set_boundary_conditions() method."""
        diff = (np.roll(func, -1, axis = 0) - np.roll(func, 1, axis = 0))/(2 * self.dy)
        return diff

    def upwind_x(self, func, u):
        """Upwind difference scheme in x-direction
           Automatically takes care of periodic 
           horizontal boundries using np.roll()."""
        diff                = np.zeros_like(u)  # Empty array of derivatives
        mask                = np.where(u >= 0)  # Boolian array, True where velovity u >= 0
        diff_pos            = (func - np.roll(func, 1, axis = 1)) / (self.dx)   # Upwind difference scheme when u >= 0. 
        diff_neg            = (np.roll(func, -1, axis = 1) - func) / (self.dx)  # Upwind difference scheme when u < 0.
        diff[mask]          = diff_pos[mask]            # Applying mask and filling array with
        diff[mask == False] = diff_neg[mask == False]   # corresponding derivaives.
        return diff

    def upwind_y(self, func, u):
        """Upwind difference scheme in y-direction.
           Periodic boundary due to np.roll() is later
           fixed through set_boundary_conditions() method."""
        diff                = np.zeros_like(u)  # Empty array of derivatives
        mask                = np.where(u >= 0)  # Boolian array, True where velovity u >= 0
        diff_pos            = (func - np.roll(func, 1, axis = 0)) / (self.dy)   # Upwind difference scheme when u >= 0.
        diff_neg            = (np.roll(func, -1, axis = 0) - func) / (self.dy)  # Upwind difference scheme when u < 0.
        diff[mask]          = diff_pos[mask]            # Applying mask and filling array with
        diff[mask == False] = diff_neg[mask == False]   # corresponding derivaives.
        return diff

    def hydro_solver(self):
        """Method solves the four hydrodynamic equations for
           next time step. """

        """Calling upon derivatives for continouity equation."""
        dudx_cent       = self.central_x(self.u[:,:])
        dwdy_cent       = self.central_y(self.w[:,:])
        drhodx_upwind_u   = self.upwind_x(self.rho[:, :], self.u[:, :])
        drhody_upwind_w   = self.upwind_y(self.rho[:, :], self.w[:, :])
        self.drhodt     = - self.rho[:,:]*(dudx_cent + dwdy_cent) \
                          - self.u[:,:] * drhodx_upwind_u - self.w[:, :] * drhody_upwind_w

        """Calling upon derivatives for x-component of momentum conservation equation."""
        dudx_upwind_u     = self.upwind_x(self.u[:, :], self.u[:, :])
        dwdy_upwind_u     = self.upwind_y(self.w[:, :], self.u[:, :])
        drho_udx_upwind_u = self.upwind_x(self.rho[:, :] * self.u[:, :], self.u[:, :])
        drho_udy_upwind_w = self.upwind_y(self.rho[:, :] * self.u[:, :], self.w[:, :])
        dPdx              = self.central_x(self.P[:, :])
        self.drho_udt     = - self.rho[:, :] * self.u[:, :] * ( dudx_upwind_u + dwdy_upwind_u) \
                          - self.u[:, :] * drho_udx_upwind_u - self.w[:, :] * drho_udy_upwind_w \
                          - dPdx
        """Calling upon derivatives for y-component of momentum conservation equation."""
        drho_wdx_upwind_u = self.upwind_x(self.rho[:, :] * self.w[:, :], self.u[:, :])
        drho_wdy_upwind_w = self.upwind_y(self.rho[:, :] * self.w[:, :], self.w[:, :])
        dwdy_upwind_w     = self.upwind_y(self.w[:, :], self.w[:, :])
        dudx_upwind_w     = self.upwind_x(self.u[:, :], self.w[:, :])
        dPdy              = self.central_y(self.P[:, :])
        self.drho_wdt     = - self.rho[:, :] * self.w[:, :] * (dwdy_upwind_w + dudx_upwind_w) \
                            - self.w[:, :] * drho_wdy_upwind_w - self.u[:, :] * drho_wdx_upwind_u \
                            - dPdy - self.rho[:, :] * self.g

        """Calling upon derivatives for of energy conservation equation."""        
        dedx_upwind_u     = self.upwind_x(self.e[:, :], self.u[:, :])
        dedy_upwind_w     = self.upwind_y(self.e[:, :], self.w[:, :])
        self.dedt         = - self.e[:, :] * dudx_cent - self.u[:, :] * dedx_upwind_u \
                            - self.e[:, :] * dwdy_cent - self.w[:, :] * dedy_upwind_w \
                            - self.P[:, :] * (dudx_cent + dwdy_cent)

        dt = self.get_timestep()    # [s] Calculating time step dt
        rho_old = self.rho[:, :]    # [kg m^-3] Current mass density

        """Updating primary variables"""
        self.rho[:, :]  = self.rho[:, :]           + self.drhodt[:, :]   * dt   # Mass density next time step
        self.u[:, :]    = (rho_old * self.u[:, :]  + self.drho_udt[:, :] * dt) / self.rho[:, :] # Horizontal velocity next time step
        self.w[:, :]    = (rho_old * self.w[:, :]  + self.drho_wdt[:, :] * dt) / self.rho[:, :] # Vertical velocity next time step
        self.e[:, :]    = self.e[:, :]             + self.dedt[:, :]     * dt   # Specific internal energy next time step

        self.set_boundary_conditions() # Imposing boundry conditions

        """Updating secondary variables"""
        self.P[:, :]    = (self.gamma - 1) * self.e[:, :]   # Pressure next time step
        self.T[:, :]    = self.e[:, :] * self.mu * self.my * (self.gamma - 1)   
        self.T[:, :]   /= (self.rho[:, :] * self.kB)        # Temperature next time step
        self.t_iterate += dt
        self.e_flux.append(np.sum(self.e[:,:] * self.w[:, :], axis = 1))
        self.t_array.append(self.t_iterate)
        self.FC_tot.append(np.sum(self.e[:, :] * self.w[:, :]))
        # self.e_flux[:] = np.sum(self.e[:, :] * np.sqrt(self.w[:, :]**2 + self.u[:, :]**2), axis = 1)
        return dt

    def FC_init(self):
        self.ax.set_xlabel(r"PW/m$^2$", fontsize = 12)
        self.ax.set_ylabel(r"$y$ [Mm]", fontsize = 12)
        self.ax.set_xticklabels(np.arange(-2, 3, 1), fontsize = 12)
        self.ax.set_yticklabels([0, 1, 2, 3, 4], fontsize = 12)
        self.line.set_data(self.e_flux, self.yarr/1e6)
        self.time_text.set_text("")
        self.totalflux_text.set_text("")
        return self.line, self.time_text, self.totalflux_text

    def FC_animate(self, i):
        self.line.set_data(self.e_flux[i], self.yarr/1e6)
        self.time_text.set_text(f"t = {self.t_array[i]:.2f} s")
        self.totalflux_text.set_text(f"$F_C^{{total}} = ${self.FC_tot[i]:.2g} W/m$^2$")
        return self.line, self.time_text, self.totalflux_text

if __name__ == '__main__':
    T0 = 5778    # [K]
    P0 = 1.8e8   # [Pa]
    y0 = 0       # [m]
    init_cond = np.array([T0, P0]) # Photoshpere temperature and pressure
    p = 1e-2    # Accuracy parameter for variable step length
    fps = 1.0         # Frames per second in animation

    vis = FVis.FluidVisualiser(fontsize = 16)

    if "sanity" in sys.argv:
        """Sanity check is run if correct command line argument is given"""
        solver = Convection2D(init_cond, p, (0.8e6, 0.8e6), \
                              (6e6, 2e6), pert = 0, pert_factor = 0)
        solver.initialize()
        vis.save_data(60, solver.hydro_solver, rho = solver.rho, u = solver.u, \
                    w = solver.w, e = solver.e, P = solver.P, T = solver.T, \
                        sim_fps = fps, folder = "Sanity check")
        
        vis.animate_2D("T", cmap = "plasma", extent = [0, 12, 0, 4], \
                       units = {"Lx": "Mm", "Lz": "Mm"}, quiverscale = 2, \
                       save = True, video_name = "Sanity", folder = "Sanity check")
        
        vis.delete_current_data()

    else:
        """Sanity check is not run if 'sanity_check' is not a command line argument"""
        times = [0, 50, 100, 150, 200, 250, 300]
        solver = Convection2D(init_cond, p, (0.8e6, 0.8e6), (6e6, 2e6), pert = 1, pert_factor = 0.5)
        solver.initialize()
        vis.save_data(300, solver.hydro_solver, rho = solver.rho, u = solver.u, \
                      w = solver.w, e = solver.e, P = solver.P, T = solver.T, \
                      sim_fps = fps, folder = "Animations")

        # vis.animate_2D("T", cmap = "plasma", extent = [0, 12, 0, 4], \
        #                units = {"Lx": "Mm", "Lz": "Mm"}, quiverscale = 3, \
        #                save = False, video_name="for_fun", folder = "for_fun_new", snapshots = times)
        all_args = [(frame_no, time) for frame_no, time in enumerate(solver.t_array)]
        ani = animation.FuncAnimation(solver.fig, solver.FC_animate, len(solver.t_array),
                                      init_func = solver.FC_init, interval = 10, blit = True)

        # vis.animate_energyflux(extent = [0, 1, 0, 4], units ={"Lz": "Mm"}, save = False, \
        #                        video_name="convective_flux", folder = "Animations")#, snapshots = times)
        plt.show()
        vis.delete_current_data()
