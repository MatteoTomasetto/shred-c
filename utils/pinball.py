from dolfin import *
from dolfin_adjoint import *
import numpy as np
from tqdm import trange
from .postprocessing import cmap, white
set_log_level(LogLevel.CRITICAL)

class Pinball():

    def __init__(self, diffusion, mu, beta, beta_g, dt, T):
        '''
        Initialize fluidic pinball problem

        Inputs
            density diffusion           (`float`)
            fluid viscosity             (`float`)
            control effort              (`float`)
            control smoothness          (`float`)
            time step                   (`float`)
            final time                  (`float`)
        '''

        # define mesh
        self.mesh = Mesh("data/Pinball/mesh.xml")
        self.h = 0.05       # mesh step size
        self.eps = 1e-8     # Nitsche's method parameter

        # define coarse mesh for plotting
        self.mesh_coarse = Mesh("data/Pinball/mesh_coarse.xml")

        # define boundaries
        class GammaOBS1(SubDomain): # Obstacle boundary
            def inside(self, x, on_boundary):
                d = sqrt((x[0] + 0.5)*(x[0] + 0.5) + (x[1] + 0.5)*(x[1] + 0.5))
                return on_boundary and d <= 0.15 + DOLFIN_EPS
        class GammaOBS2(SubDomain): # Obstacle boundary
            def inside(self, x, on_boundary):
                d = sqrt((x[0] - 0.5)*(x[0] - 0.5) + (x[1] + 0.5)*(x[1] + 0.5))
                return on_boundary and d <= 0.15 + DOLFIN_EPS
        class GammaOBS3(SubDomain): # Obstacle boundary
            def inside(self, x, on_boundary):
                d = sqrt(x[0]*x[0] + (x[1] - 0.5)*(x[1] - 0.5))
                return on_boundary and d <= 0.15 + DOLFIN_EPS
        boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        GammaOBS1().mark(boundaries, 1)
        GammaOBS2().mark(boundaries, 2)
        GammaOBS3().mark(boundaries, 3)
        self.ds = Measure("ds", domain = self.mesh, subdomain_data = boundaries)

        # define function spaces
        self.Yh = FunctionSpace(self.mesh, "CG", 1)
        self.VPh = FunctionSpace(self.mesh, VectorElement('CG', self.mesh.ufl_cell(), 2) * FiniteElement('CG', self.mesh.ufl_cell(), 1))
        self.Vh = self.VPh.sub(0).collapse()
        self.Ph = self.VPh.sub(1).collapse()
        self.Vh_coarse = FunctionSpace(self.mesh_coarse, VectorElement('CG', self.mesh_coarse.ufl_cell(), 1))
        self.nstate = self.Yh.dim()
        self.nvelocity = self.Vh.dim()
        self.npressure = self.Ph.dim()
        
        # set parameters
        self.diffusion = diffusion
        self.mu = mu
        self.beta = beta
        self.beta_g = beta_g
        self.dt = dt
        self.T = T
        self.ntimesteps = round(T / dt) + 1
        self.times = np.linspace(0.0, self.T, self.ntimesteps)

    def gaussian(self, mean_coords):
        '''
        Generate a 2D Gaussian starting from the mean position

        Inputs
            mean coordinate                         (`tuple[float]`, len: 2)


        Output
            two-dimensional Gaussian function       (`dolfin.cpp.function.Function`)
        '''

        y = Expression('10 / pi * exp(- 10*(x[0] - x1)*(x[0] - x1) - 10*(x[1] - x2)*(x[1] - x2))', degree = 1, x1 = mean_coords[0], x2 = mean_coords[1])
        
        return interpolate(y, self.Yh)

    def solve_NS(self, pinball_velocities):
        '''
        Solve steady Navier-Stokes via Newton method
        
        Inputs
            cylinder velocities             (`tuple[float]`, len: 3)
        
        Output
            steady Navier-Stokes velocity   (`dolfin.cpp.function.Function`)
        '''
        
        n = FacetNormal(self.mesh)
        t = as_vector([n[1], -n[0]])

        # define free-slip BC on the wall
        wall = DirichletBC(self.VPh.sub(0), (0.0, 0.0), "on_boundary && (x[0] >= 0.9 || x[0] < -0.9 || x[1] < -0.9 || x[1] > 0.9)")

        # solve state equation
        w, q = TestFunctions(self.VPh)
        vp = Function(self.VPh)
        v, p = split(vp)

        F = self.mu * inner(grad(v), grad(w)) * dx + dot(dot(grad(v), v), w) * dx - p * div(w) * dx - q * div(v) * dx + 1 / self.eps * inner(v, n) * inner(w, n) * (self.ds(1) + self.ds(2) + self.ds(3)) - 1 / self.eps * inner(v, t) * inner(w, t) * (self.ds(1) + self.ds(2) + self.ds(3)) + 1 / self.eps * pinball_velocities[0] * inner(w, t) * self.ds(1) + 1 / self.eps * pinball_velocities[1] * inner(w, t) * self.ds(2) + 1 / self.eps * pinball_velocities[2] * inner(w, t) * self.ds(3)
        solve(F == 0, vp, wall)
        v, p = vp.split()
        v = project(v, self.Vh)
        
        return v

    def solve_FP(self, pinball_velocities, ut = None):
        '''
        Solve Fokker-Planck equation

        Inputs
            cylinder velocities      (`tuple[float]`, len: 3)
            control values           (`np.array[float]`, shape: (ntimesteps-1, nvelocity), optional)
        
        Outputs
            density values           (`np.array[float]`, shape: (ntimesteps, nstate))
            velocity values          (`np.array[float]`, shape: (nvelocity))
            loss function            (`float`)
        '''
    
        # set control
        if ut is None:
            ut = []
            for i in range(self.ntimesteps - 1):
                ut.append(Function(self.Vh))
        u = Function(self.Vh)

        # compute velocity
        v = self.solve_NS(pinball_velocities)
        v_in_mod = project(v[1], self.Yh)(0,-1)

        # define initial condition
        mean_coords = (0.0, 0.0)
        y0 = self.gaussian(mean_coords)

        # initialize tensors to store solution
        yt = np.empty((self.ntimesteps, self.nstate))
        yt[0] = y0.vector()[:]

        # initialize loss
        J = 0.0

        for i in trange(1, self.ntimesteps):

            # update control
            u.assign(ut[i-1])

            # set test and trial functions
            w = TestFunction(self.Yh)
            y = TrialFunction(self.Yh)

            # solve Fokker-Planck
            a = inner(y, w) * dx + 0.5 * self.dt * (self.diffusion + v_in_mod * self.h / 2) * inner(grad(y), grad(w)) * dx - 0.5 * self.dt * y * project(v[0], self.Yh) * w.dx(0) * dx - 0.5 * self.dt * y * project(v[1], self.Yh) * w.dx(1) * dx - 0.5 * self.dt * y * project(u[0], self.Yh) * w.dx(0) * dx - 0.5 * self.dt * y * project(u[1], self.Yh) * w.dx(1) * dx
            L = inner(y0, w) * dx - 0.5 * self.dt * (self.diffusion + v_in_mod * self.h / 2) * inner(grad(y0), grad(w)) * dx + 0.5 * self.dt * y0 * project(v[0], self.Yh) * w.dx(0) * dx + 0.5 * self.dt * y0 * project(v[1], self.Yh) * w.dx(1) * dx + 0.5 * self.dt * y0 * project(u[0], self.Yh) * w.dx(0) * dx + 0.5 * self.dt * y0 * project(u[1], self.Yh) * w.dx(1) * dx 
            y = Function(self.Yh)        
            solve(a == L, y, DirichletBC(self.Yh, 0.0, "on_boundary"))
            
            # update state
            y0.assign(y)

            # store results
            yt[i] = y.vector()[:]
            
            # update mean
            mean_coords = mean_coords + self.dt * v(mean_coords)
            
            # update loss
            yd = self.gaussian(mean_coords)
            J += 0.5 * assemble(inner(y - yd, y - yd)* dx + 10 * inner(y, y) * self.ds + self.beta * inner(u, u) * dx + self.beta_g * inner(grad(u), grad(u)) * dx)

        return yt, v.vector()[:], J
    
    def solve_OCP(self, pinball_velocities):
        '''
        Solve optimal control problem

        Input
            cylinder velocities      (`tuple[float]`, len: 3)
        
        Outputs
            optimal density values          (`np.array[float]`, shape: (ntimesteps, nstate))
            optimal control values          (`np.array[float]`, shape: (ntimesteps-1, nvelocity))
            velocity values                 (`np.array[float]`, shape: (nvelocity))
            loss function                   (`float`)
        '''
        
        set_working_tape(Tape())

        # solve forward problem
        ut = []
        for i in range(self.ntimesteps - 1):
            ut.append(Function(self.Vh))
        yt, v, J = self.solve_FP(pinball_velocities, ut)

        # solve optimal control problem
        control = [Control(u) for u in ut]
        Jhat = ReducedFunctional(J, control)
        ut_opt = minimize(Jhat, method = 'L-BFGS-B', tol = 1e-3, options = {'disp': True, 'maxiter': 50})

        # compute optimal state
        yt_opt, v, J_opt = self.solve_FP(pinball_velocities, ut_opt)

        # store optimal control values
        ut_opt_vals = np.empty((self.ntimesteps-1, self.nvelocity))
        for i in range(self.ntimesteps-1):
            ut_opt_vals[i] = ut_opt[i].vector()[:] 

        return yt_opt, ut_opt_vals, v, J_opt
    
    def plot_state(self, y, vmin = None, vmax = None):
        '''
        Plot state

        Inputs
            state values             (`np.array[float]`, shape: (nstate,))
        '''

        y_plot = Function(self.Yh)
        y_plot.vector()[:] = y
        plot(y_plot, cmap = "jet", vmin = vmin, vmax = vmax)

    def plot_control(self, u, vmin = None, vmax = None):
        '''
        Plot control

        Inputs
            control values             (`np.array[float]`, shape: (nvelocity,))
        '''

        u_plot = Function(self.Vh)
        u_plot.vector()[:] = u
        u_plot.set_allow_extrapolation(True)
        plot(sqrt(u_plot**2), cmap = cmap, vmin = vmin, vmax = vmax)
        plot(project(u_plot, self.Vh_coarse), cmap = white, alpha = 0.9, scale = 0.5, minlength = 0, width = 0.004)
