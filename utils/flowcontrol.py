from dolfin import *
from dolfin_adjoint import *
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .postprocessing import cmap, black
set_log_level(LogLevel.CRITICAL)

class FlowControl():

    def __init__(self, mu, v_in_mod, beta, beta_g, burnin, dt, T):
        '''
        Initialize flow control problem

        Inputs
            fluid viscosity             (`float`)
            inflow intensity            (`float`)
            control effort              (`float`)
            control smoothness          (`float`)
            burn-in time steps          (`float`)
            time step                   (`float`)
            final time                  (`float`)
        '''

        # define mesh
        self.mesh = Mesh("data/FlowControl/mesh.xml")

        # define boundaries
        class GammaC(SubDomain):   # Control boundary
            def inside(self, x, on_boundary):
                return on_boundary and between(x[0], (0.79, 1.0)) and between(x[1], (0.79, 1.21))
        class GammaOBS(SubDomain):   # Obstacle boundary
            def inside(self, x, on_boundary):
                return on_boundary and between(x[0], (1.0, 1.21)) and between(x[1], (0.79, 1.21))
        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        GammaC().mark(self.boundaries, 1)
        GammaOBS().mark(self.boundaries, 2)
        self.ds = Measure("ds", domain = self.mesh, subdomain_data = self.boundaries)

        # define function spaces
        self.Vh = FunctionSpace(self.mesh, VectorElement(NodalEnrichedElement(FiniteElement("CG", self.mesh.ufl_cell(), 1), FiniteElement("Bubble", self.mesh.ufl_cell(), 3))))
        self.Ph = FunctionSpace(self.mesh, "CG", 1)
        self.nvelocity = self.Vh.dim()
        self.npressure = self.Ph.dim()

        # extract control indices
        bc_dummy = DirichletBC(self.Vh, (1.0, 2.0), self.boundaries, 1)
        fun_dummy = Function(self.Vh)
        bc_dummy.apply(fun_dummy.vector())
        self.idx_control_x1 = fun_dummy.vector() == 1
        self.idx_control_x2 = fun_dummy.vector() == 2
        self.idx_control = self.idx_control_x1 | self.idx_control_x2
        self.ncontrol = sum(self.idx_control == True)

        # set parameters
        self.mu = mu
        self.v_in_mod = v_in_mod       
        self.beta = beta
        self.beta_g = beta_g
        self.burnin = burnin
        self.dt = dt
        self.T = T
        self.ntimesteps = round(T / dt) + 1 - self.burnin
        self.times = np.linspace(0.0, self.T, self.burnin + self.ntimesteps)

    def BC(self, alpha_in):
        '''
        Define Dirichlet boundary conditions

        Input
            angle of attack                     (`float`)
        
        Output
            velocity boundary conditions        (`list[fenics_adjoint.types.dirichletbc.DirichletBC]`, len: 3)
            pressure boundary conditions        (`fenics_adjoint.types.dirichletbc.DirichletBC`)
        '''
                
        # parabolic BC on the inflow
        v_in = Expression(('v_in_mod * cos(alpha_in)','x[1] * (2.0 - x[1]) * v_in_mod * sin(alpha_in)'), degree = 1, v_in_mod = self.v_in_mod, alpha_in = alpha_in)  
        inflow = DirichletBC(self.Vh, v_in, "on_boundary && x[0] <= 0.01")
    
        # free-slip BC on the wall
        wall = DirichletBC(self.Vh.sub(1), 0.0, "on_boundary && (x[1] >= 1.99 || x[1] < 0.01)")

        # free-slip BC on the wall
        obstacle = DirichletBC(self.Vh, (0.0, 0.0), self.boundaries, 2)
        
        # velocity BC
        v_bc = [inflow, wall, obstacle]
        
        # pressure BC
        p_bc = DirichletBC(self.Ph, 0.0, "on_boundary && x[0] >= 7.99")
        
        return v_bc, p_bc
    
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
    
    def solve_NS(self, alpha_in, v0_val = None, p0_val = None, ut = None, ntimesteps = None):
        '''
        Solve Navier-Stokes equations via incremental Chorin-Temam projection method

        Inputs
            angle of attack             (`float`)
            initial velocity values     (`np.array[float]`, shape: (nvelocity), optional)
            initial pressure values     (`np.array[float]`, shape: (npressure), optional)
            control functions           (`list[fenics.Function]`, len: (ntimesteps-1), optional)
            number of time steps        (`int`, optional)

        
        Outputs
            velocity values             (`np.array[float]`, shape: (ntimesteps, nvelocity))
            pressure values             (`np.array[float]`, shape: (ntimesteps, npressure))
            loss function               (`float`)
        '''

        # set initial conditions
        v0 = Function(self.Vh)
        if v0_val is not None:
            v0.vector()[:] = v0_val
        p0 = Function(self.Ph)
        if p0_val is not None:
            p0.vector()[:] = p0_val
               
        # set number of time steps to simulate
        if ntimesteps is None:
            ntimesteps = self.ntimesteps

        # set control
        if ut is None:
            ut = []
            for i in range(ntimesteps - 1):
                ut.append(Function(self.Vh)) 
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
        v_bc, p_bc = self.BC(alpha_in)

        # initialize tensors to store solution
        vt = np.empty((ntimesteps, self.nvelocity))
        vt[0] = v0.vector()[:]
        pt = np.empty((ntimesteps, self.npressure))
        pt[0] = p0.vector()[:]

        # initialize loss
        J = 0.0

        for i in trange(1, ntimesteps):
            
            # update control
            u.assign(ut[i-1])

            # control BC
            u_bc = DirichletBC(self.Vh, u, self.boundaries, 1)
            if i == 1:
                v_bc.append(u_bc)
            else:
                v_bc[-1] = u_bc

            # first Chorin-Temam step
            a1 = (1 / self.dt) * inner(v, w) * dx + self.mu * inner(grad(v), grad(w)) * dx + inner(grad(v) * v0, w) * dx
            A1 = assemble(a1)
            L1 = (1 / self.dt) * inner(v0, w) * dx - inner(grad(p0), w) * dx
            b1 = assemble(L1)
            [bc.apply(A1, b1) for bc in v_bc]
            solve(A1, v1.vector(), b1)
        
            # second Chorin-Temam step
            L2 = - (1 / self.dt) * div(v1) * q * dx
            b2 = assemble(L2)
            p_bc.apply(A2, b2)
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

            # update loss
            grad_u = grad(u)
            norm_grad_u = sum(abs(grad_u[i, j]) for i in range(self.mesh.geometry().dim()) for j in range(self.mesh.geometry().dim()))
            J += assemble(0.5 * inner(grad(v1), grad(v1)) * dx + 0.5 * self.beta * inner(u, u) * self.ds(1) + 0.5 * self.beta_g * norm_grad_u * self.ds(1))

        return vt, pt, J

    def solve_OCP(self, alpha_in):
        '''
        Solve optimal control problem

        Input
            angle of attack                 (`float`)
        
        Outputs
            optimal velocity values         (`np.array[float]`, shape: (burnin + ntimesteps - 1, nvelocity))
            optimal pressure values         (`np.array[float]`, shape: (burnin + ntimesteps - 1, npressure))
            optimal control values          (`np.array[float]`, shape: (ntimesteps-1, ncontrol))
            loss function                   (`float`)
        '''

        # burn-in
        if self.burnin > 0:
            vt_burnin, pt_burnin, J_burnin = self.solve_NS(alpha_in, ntimesteps = self.burnin)
        else:
            vt_burnin = np.zeros((1, self.nvelocity))
            pt_burnin = np.zeros((1, self.npressure))
        set_working_tape(Tape());

        # solve forward problem
        ut = []
        for i in range(self.ntimesteps - 1):
            ut.append(Function(self.Vh))
        vt, pt, J = self.solve_NS(alpha_in, v0_val = vt_burnin[-1], p0_val = pt_burnin[-1], ut = ut)

        # solve optimal control problem
        control = [Control(u) for u in ut]
        Jhat = ReducedFunctional(J, control)
        ut_opt = minimize(Jhat, method = 'L-BFGS-B', options = {'disp': True, 'maxiter': 100})

        # compute optimal state
        vt_opt, pt_opt, J_opt = self.solve_NS(alpha_in, vt_burnin[-1], pt_burnin[-1], ut_opt)

        # store optimal control values
        ut_opt_vals = np.empty((self.ntimesteps-1, self.ncontrol))
        for i in range(self.ntimesteps-1):
            ut_opt_vals[i] = ut_opt[i].vector()[self.idx_control] 

        return np.concatenate((vt_burnin, vt_opt[1:]), axis = 0), np.concatenate((pt_burnin, pt_opt[1:]), axis = 0), ut_opt_vals, J_opt
    
    def plot_velocity(self, v):
        '''
        Plot velocity

        Inputs
            velocity values            (`np.array[float]`, shape: (nvelocity,))
        '''

        v_plot = Function(self.Vh)
        v_plot.vector()[:] = v
        v_norm = project(sqrt(inner(v_plot, v_plot)), self.Vh.sub(0).collapse())  
        plot(v_norm, cmap = cmap)
        
    def plot_pressure(self, p):
        '''
        Plot pressure

        Inputs
            pressure values            (`np.array[float]`, shape: (npressure,))
        '''

        p_plot = Function(self.Ph)
        p_plot.vector()[:] = p
        plot(p_plot, cmap = "jet")

    def plot_control(self, u):
        '''
        Plot control

        Inputs
            control values             (`np.array[float]`, shape: (ncontrol,))
        '''
        
        u_plot_vec = np.zeros(self.nvelocity)
        u_plot_vec[self.idx_control] = u
        u_plot = Function(self.Vh)
        u_plot.vector()[:] = u_plot_vec
        
        rectangle = patches.Rectangle((0.0, 0.0), 8.0, 2.0, edgecolor = 'silver', facecolor = 'silver', alpha = 0.5, zorder = 0)
        circle = patches.Circle((1, 1), 0.2, edgecolor = 'black', facecolor = 'white', linewidth = 3, zorder = 0)
        plt.gca().add_patch(rectangle)
        plt.gca().add_patch(circle)
        plot(u_plot, scale = 60, cmap = cmap, minlength = 0, width = 0.008)
        plt.axis('equal')
        plt.xlim((0.75, 1.25))
        plt.ylim((0.75, 1.25))

    def plot_velocity_with_control(self, v, u):
        '''
        Plot control with velocity

        Inputs
            velocity values            (`np.array[float]`, shape: (nvelocity,))
            control values             (`np.array[float]`, shape: (ncontrol,))
        '''

        v_plot = Function(self.Vh)
        v_plot.vector()[:] = v
        v_norm = project(sqrt(inner(v_plot, v_plot)), self.Vh.sub(1).collapse())  

        u_plot_vec = np.zeros(self.nvelocity)
        u_plot_vec[self.idx_control] = u
        u_plot = Function(self.Vh)
        u_plot.vector()[:] = u_plot_vec

        vmin = v_norm.vector()[:].min()
        vmax = v_norm.vector()[:].max()
        
        plot(v_norm, cmap = cmap, vmin = vmin, vmax = vmax)
        plot(u_plot, scale = 60, cmap = cmap, minlength = 0, width = 0.008, clim=(vmin,vmax))          
        plt.axis('equal')
        plt.xlim((0.75, 1.25))
        plt.ylim((0.75, 1.25))
