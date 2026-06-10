from dolfin import *
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .postprocessing import cmap 
set_log_level(LogLevel.CRITICAL)

class DoubleGyre():

    def __init__(self, Lx, Ly, nx, ny, mu, gamma, dt, T):
        '''
        Initialize double gyre flow problem

        Inputs
            domain length               (`float`)
            domain height               (`float`)
            horizontal refinement       (`int`)
            vertical refinement         (`int`)
            viscosity                   (`float`)
            control effort              (`float`)
            time step                   (`float`)
            final time                  (`float`)
        '''

        # define mesh
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.mesh = RectangleMesh(Point(0, 0), Point(self.Lx, self.Ly), self.nx, self.ny)

        # define function spaces
        self.Vh = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Ph = FunctionSpace(self.mesh, "CG", 1)
        self.nvelocity = self.Vh.dim()
        self.npressure = self.Ph.dim()

        # extract coordinates for velocity degrees of freedom
        Vh_coords = self.Vh.tabulate_dof_coordinates().reshape((-1, self.mesh.geometry().dim()))
        self.x1 = Vh_coords[::2, 0]
        self.x2 = Vh_coords[::2, 1]

        # define grid for streamlines plotting
        self.x1_grid, self.x2_grid = np.meshgrid(np.linspace(0, self.Lx, self.nx), np.linspace(0, self.Ly, self.ny))

        # set parameters
        self.mu = mu
        self.gamma = gamma
        self.dt = dt
        self.T = T
        self.ntimesteps = round(T / dt) + 1
        self.times = np.linspace(0.0, self.T, self.ntimesteps)

    def vortex(self, vortex_coords = None, delta = 0.25, intensity = 0.1):
        '''
        Define a vortex

        Inputs
            coordinates of the vortex center        (`tuple[float]`, len: 2, optional)

        Output
            vortex velocity                         (`dolfin.cpp.function.Function`)
        '''
        
        # set vortex center
        if vortex_coords is None:
            vortex_coords = (self.Lx / 2, self.Ly / 2)
        elif vortex_coords[0] < 0.0 or vortex_coords[0] > self.Lx or vortex_coords[1] < 0.0 or vortex_coords[1] > self.Ly:
            raise ValueError("Vortex must be placed within the domain")
        else:
            pass

        # define stream function
        psi = Function(self.Ph)
        psi = project(Expression('intensity * exp(- ((x[0] - x1_vortex)*(x[0] - x1_vortex) + (x[1] - x2_vortex)*(x[1] - x2_vortex))/(delta*delta))', degree = 1, x1_vortex = vortex_coords[0], x2_vortex = vortex_coords[1], delta = delta, intensity = intensity), self.Ph)
        
        # define velocity
        v0 = Function(self.Vh)
        v01 = project(psi.dx(1), self.Vh.sub(0).collapse())
        v02 = project(-psi.dx(0), self.Vh.sub(1).collapse())
        v0.vector()[:] = np.ravel(np.column_stack((v01.vector()[:].reshape(-1,1), v02.vector()[:].reshape(-1,1))))

        return v0

    def double_gyre_flow(self, amplitude, frequency, intensity = 0.1):
        '''
        Compute double gyre flow

        Inputs
            time vector                 (`np.array[float]`, shape: (ntimesteps,))
            amplitude                   (`float`)
            frequency                   (`float`)
            intensity                   (`float`, optional)            
            
        Output
            velocity values             (`np.array[float]`, shape: (ntimesteps, nvelocity)
        '''

        x1 = self.x1[:, None]
        x2 = self.x2[:, None]
        times = np.asarray(self.times)[None, :]
        vt = np.empty((times.shape[1], 2*x1.shape[0]))

        f = lambda x,t: amplitude * np.sin(frequency * t) * x**2 + x - 2 * amplitude * np.sin(frequency * t) * x
        vt[:,::2] = (-np.pi * intensity * np.sin(np.pi * f(x1, times)) * np.cos(np.pi * x2)).T
        vt[:,1::2] = (np.pi * intensity * np.cos(np.pi * f(x1, times)) * np.sin(np.pi * x2) * (2 * amplitude * np.sin(frequency * times) * x1 + 1.0 - 2 * amplitude * np.sin(frequency * times))).T

        return vt
    
    def assembly_CT(self):
        '''
        Assemble the constant matrices for the Chorin-Temam projection method

        Output
            list of constant matrices           (`list[dolfin.cpp.la.Matrix]`, len: 3)
        '''
            
        w = TestFunction(self.Vh)
        q = TestFunction(self.Ph)
        v = TrialFunction(self.Vh)
        p = TrialFunction(self.Ph)

        # Second step
        a2 = inner(grad(p), grad(q)) * dx
        A2 = assemble(a2)
        
        # Third step
        a3 = inner(v, w) * dx
        A3 = assemble(a3)
        
        # Fourth step
        a4 = p * q * dx
        A4 = assemble(a4)
        
        return A2, A3, A4 
    
    def solve_NS(self, vortex_coords, v0_val = None, p0_val = None, control = False, ut_val = None, amplitude_ref = 0.25, frequency_ref = 5.0, ntimesteps = None):
        '''
        Solve Navier-Stokes equations via incremental Chorin-Temam projection method

        Inputs
            coordinates of the initial vortex center    (`tuple[float]`, len: 2)
            initial velocity values                     (`np.array[float]`, shape: (nvelocity), optional)
            initial pressure values                     (`np.array[float]`, shape: (npressure), optional)
            control on/off                              (`bool`, optional)
            control values                              (`np.array[float]`, shape: (ntimesteps-1, nvelocity), optional)
            double gyro flow amplitude                  (`float`, optional)
            double gyro flow frequency                  (`float`, optional)
        
        Output
            list of velocity, pressure and (if employed) control and reference values (`list[np.array]`, shapes: (ntimesteps, nvelocity), (ntimesteps, npressure), (ntimesteps, nvelocity), (ntimesteps, nvelocity))
        '''

        # set initial conditions
        if v0_val is None:
            v0 = self.vortex(vortex_coords = vortex_coords)
        else:
            v0 = Function(self.Vh)
            v0.vector()[:] = v0_val
        if p0_val is None:
            p0 = Function(self.Ph)
        else:
            p0 = Function(self.Ph)
            p0.vector()[:] = p0_val
               
        # set number of time steps to simulate
        if ntimesteps is None:
            ntimesteps = self.ntimesteps

        # set reference flow if control is on
        if control and ut_val is None:
            v_ref0 = Function(self.Vh)
            v_ref1 = Function(self.Vh)
            vt_ref = self.double_gyre_flow(amplitude_ref, frequency_ref)
        
        # define control 
        u = Function(self.Vh)   

        # define test and trial functions
        w = TestFunction(self.Vh)
        q = TestFunction(self.Ph)
        v = TrialFunction(self.Vh)
        v1 = Function(self.Vh)
        p1 = Function(self.Ph)

        # assemble constant matrices
        A2, A3, A4 = self.assembly_CT()
        
        # define boundary conditions
        bc = DirichletBC(self.Ph, 0.0, "on_boundary")

        # initialize tensors to store solution
        vt = np.empty((ntimesteps, self.nvelocity))
        vt[0] = v0.vector()[:]
        pt = np.empty((ntimesteps, self.npressure))
        pt[0] = p0.vector()[:]
        if control and ut_val is None:
            ut = np.empty((ntimesteps-1, self.nvelocity))

        for i in trange(1, ntimesteps):
            
            # update reference flow and compute control
            if control:
                if ut_val is None:
                    v_ref0.vector()[:] = vt_ref[i-1]
                    v_ref1.vector()[:] = vt_ref[i]
                    u = project((1 / self.dt) * (v_ref1 - v_ref0) + self.mu * div(grad(v_ref0)) - self.gamma * (v0 - v_ref0) + grad(v_ref0) * v0, self.Vh) 
                else:
                    u = Function(self.Vh)
                    u.vector()[:] = ut_val[i-1]

            # first Chorin-Temam step
            a1 = (1 / self.dt) * inner(v, w) * dx + self.mu * inner(grad(v), grad(w)) * dx + inner(grad(v) * v0, w) * dx
            A1 = assemble(a1)
            L1 = (1 / self.dt) * inner(v0, w) * dx - inner(grad(p0), w) * dx + inner(u, w) * dx
            b1 = assemble(L1)
            solve(A1, v1.vector(), b1)
            
            # second Chorin-Temam step
            L2 = - (1 / self.dt) * div(v1) * q * dx
            b2 = assemble(L2)
            bc.apply(A2, b2)
            solve(A2, p1.vector(), b2)
            
            # third Chorin-Temam step
            L3 = inner(v1, w) * dx - self.dt * inner(grad(p1), w) * dx
            b3 = assemble(L3)
            solve(A3, v1.vector(), b3)
            
            # fourth Chorin-Temam step
            L4 = p0 * q * dx + p1 * q * dx
            b4 = assemble(L4)
            solve(A4, p1.vector(), b4)

            # update state
            v0.assign(v1)
            p0.assign(p1)

            # store results
            vt[i] = v1.vector()[:]
            pt[i] = p1.vector()[:]
            if control and ut_val is None:
                ut[i-1] = u.vector()[:]
        
        if control and ut_val is None:
            out = [vt, pt, ut, vt_ref]
        else:
            out = [vt, pt]

        return out

    def plot_velocity(self, v):
        '''
        Plot velocity

        Inputs
            velocity values             (`np.array[float]`, shape: (nvelocity,))
        '''

        v_plot = Function(self.Vh)
        v_plot.vector()[:] = v
        v_norm = project(sqrt(inner(v_plot, v_plot)), self.Vh.sub(0).collapse())  
        plot(v_norm, cmap = cmap)
        plt.xlim(0, self.Lx)
        plt.ylim(0, self.Ly)
        
    def plot_vorticity(self, v, cmap = "seismic", vmin = None, vmax = None):
        '''
        Plot vorticity

        Inputs
            velocity values             (`np.array[float]`, shape: (nvelocity,))
        '''
        
        # evaluate velocity on a grid
        v_plot = Function(self.Vh)
        v_plot.vector()[:] = v
        V1 = np.zeros_like(self.x1_grid)
        V2 = np.zeros_like(self.x2_grid)
        for i in range(self.nx):
            for j in range(self.ny):
                try:
                    v_plot_val = v_plot(Point(self.x1_grid[j,i], self.x2_grid[j,i]))
                    V1[j,i] = v_plot_val[0]
                    V2[j,i] = v_plot_val[1]
                except:
                    V1[j,i] = np.nan
                    V2[j,i] = np.nan

        # plot streamlines and vorticity
        plot(project(v_plot.sub(1).dx(0) - v_plot.sub(0).dx(1), self.Ph), cmap = cmap, vmin = vmin, vmax = vmax)
        plt.streamplot(self.x1_grid, self.x2_grid, V1, V2, color = 'black', linewidth = 1, density = 1)
        plt.gca().add_patch(patches.Rectangle((0, 0), 2, 1, linewidth = 5, edgecolor = 'black', facecolor = 'none'))
        plt.xlim(0, self.Lx)
        plt.ylim(0, self.Ly)

    def plot_pressure(self, p):
        '''
        Plot pressure

        Inputs
            pressure values             (`np.array[float]`, shape: (npressure,))
        '''

        p_plot = Function(self.Ph)
        p_plot.vector()[:] = p
        plot(p_plot, cmap = "jet")
        plt.xlim(0, self.Lx)
        plt.ylim(0, self.Ly)