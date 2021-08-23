# FENICS/DOLFIN validation of free-free beam vibration

import dolfinx
#from dolfinx.io import XDMFFile
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.linalg
from petsc4py import PETSc
from slepc4py import SLEPc

# beam properties
n_ele = 50
start_x = 0.0
end_x = 1.0
E = 1e6
nu = 0.3
G = E / (2*(1 + nu))
I = 2e-9
kappa = 1
A = 0.0002
rho = 1.0

# 1D mesh and its function space
def create_mesh():
    mesh = dolfinx.IntervalMesh(MPI.COMM_WORLD, n_ele, [start_x, end_x])
    P2 = ufl.FiniteElement("CG", "interval", 2)
    P1 = ufl.FiniteElement("CG", "interval", 1)
    ME = ufl.MixedElement([P2, P1])
    V = dolfinx.FunctionSpace(mesh, ME)
    return mesh, V

def create_matrices(V, lumped = False):

    def chi(Phi):
        return Phi.dx(0)
    def gamma(w, Phi):
        return (w.dx(0) - Phi)
    def M(Phi):
        return E * I * chi(Phi)
    def Q(w, Phi):
        return kappa * G * A * gamma(w,Phi)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    (w, phi) = ufl.split(u)
    (w_ , phi_) = ufl.split(v)

    # Timoshenko stiffness
    stiffness_form = (ufl.inner(M(phi), chi(phi_)) + ufl.inner(Q(w, phi), gamma(w_, phi_))) * ufl.dx 
    KM = dolfinx.fem.assemble_matrix(stiffness_form)
    KM.assemble()

    # consistent mass matrix
    mass_form = rho * (A * ufl.dot(w, w_) + I * ufl.dot(phi, phi_)) * ufl.dx
    MM = dolfinx.fem.assemble_matrix(mass_form)
    MM.assemble()

    # make lumped mass matrix
    if lumped:
        UU = dolfinx.Function(V)
        unity = UU.vector.copy()
        unity.set(1.0)
        lm = MM*unity 
        MM.zeroEntries()
        MM.setDiagonal(lm)
    
    return KM, MM

def solve_eigenmodes(KM, MM, dims = 5, target_freq = 0.0, print_status = True):

    pc = PETSc.PC().create(MPI.COMM_WORLD)
    pc.setType(pc.Type.CHOLESKY)

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setType(ksp.Type.PREONLY)
    ksp.setPC(pc)
    #ksp.setTolerances(rtol=1E-9)
    
    st = SLEPc.ST().create(MPI.COMM_WORLD)
    st.setType(st.Type.SINVERT)
    st.setKSP(ksp)

    solver = SLEPc.EPS().create(MPI.COMM_WORLD)
    solver.setST(st)
    solver.setOperators(KM, MM)
    solver.setType(solver.Type.KRYLOVSCHUR)
    solver.setDimensions(dims)
    solver.setTolerances(1e-5, 1000)
    solver.setWhichEigenpairs(solver.Which.TARGET_REAL)
    solver.setTarget(target_freq)
    solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    solver.setFromOptions()
    solver.solve()

    if print_status:
        its = solver.getIterationNumber()
        sol_type = solver.getType()
        nev, ncv, mpd = solver.getDimensions()
        tol, maxit = solver.getTolerances()
        nconv = solver.getConverged()
        print("")
        print("Number of iterations of the method: %i" % its)
        print("Solution method: %s" % sol_type)
        print("Number of requested eigenvalues: %i" % nev)
        print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
        print("Number of converged eigenpairs: %d" % nconv)

    return solver

def extract_modes(results, N = 3, min_freq = 0.01, max_error = 1e-4):
    vr, vi = KM.getVecs()
    n_okay = 0
    freqs = []
    modes = []
    
    for i in range(results.getConverged()):
        k = results.getEigenpair(i, vr, vi)
        error = results.computeError(i)
        if np.isnan(k.real) or k.real <= min_freq or error > max_error:
            continue
        n_okay += 1
        if n_okay <= N:
            freqs.append(np.sqrt(k.real))
            modes.append(vr.copy())
    
    return freqs, modes

def invert_stiffness(K):
    ai, aj, av = K.getValuesCSR()
    K_np = scipy.sparse.csr_matrix((av, aj, ai)).todense()
    return scipy.linalg.inv(K_np, overwrite_a=False, check_finite=True)

def calc_modal_thing(modes, what, V):
    out = []
    tmp = dolfinx.Function(V).vector
    for m in modes:
        what.mult(m, tmp)
        out.append(m.dot(tmp))
    return out

def postprocess(mesh, V, KM, results, min_freq = 0.01, max_error = 1e-4):

    eigenmode = dolfinx.Function(V)
    nodes = mesh.geometry.x[:, 0]
    order = np.argsort(nodes)
    vr, vi = KM.getVecs()
    n_okay = 0

    print("")
    print("        k          ||Ax-kx||/||kx|| ")
    print("----------------- ------------------")
    
    for i in range(results.getConverged()):
        k = results.getEigenpair(i, vr, vi)
        error = results.computeError(i)
        if not np.isnan(k.real) and k.real > min_freq and error < max_error:
            n_okay += 1
            print(" %12f       %12g" % (np.sqrt(k.real) / (2*np.pi), error)) # rad/s to Hz
            if n_okay <= 4:
                eigenmode.vector.array[:] = PETSc.Vec.getArray(vr)
                eigenmode.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                result = eigenmode.sub(0).compute_point_values()
                plt.plot(nodes[order], result[order] / np.max(np.abs(result[order])))

    print("")
    plt.legend(["2-node", "3-node", "4-node", "5-node"])
    plt.show()

# Newmark-Beta integrator (gamma = 0.5, beta = 1/6 are coefs for linear acceleration)
def update_eta_newmark(dt, dis, vel, acc, wn, zeta, F, gamma = 0.5, beta = 1.0/6.0):
    
    a1 = 1 / (beta * np.pow(dt, 2)) + (gamma * 2 * zeta * wn) / (beta * dt)
    a2 = 1 / (beta * dt) + (gamma / beta - 1) * (2 * zeta * wn)
    a3 = (1 / (2 * beta) - 1) + dt * 2 * zeta * wn * (gamma / (2 * beta) - 1)

    new_disp = (F + a1 * dis + a2 * vel + a3 * acc) / (np.pow(wn, 2) + a1)
    new_veloc = (gamma / (beta * dt)) * (new_disp - dis) + (1 - gamma / beta) * vel + dt * (1 - gamma / (2 * beta)) * acc
    new_accel = 1 / (beta * np.pow(dt, 2)) * (new_disp - dis) - (1 / (beta * dt)) * vel - (1 / (2 * beta) - 1) * acc

    return new_disp, new_veloc, new_accel

# Wilson-Theta integrator
def update_eta_wilson(dt, dis, vel, acc, wn, zeta, F, theta = 1.4):
    # equation of motion coefs
    mg = 1.0
    cg = 2.0 * zeta * wn
    kg = wn**2
    # effective stiffness
    k = 6.0/(theta**2 * dt**2)*mg + 3.0/(theta * dt)*cg + kg
    # predicted displacement
    dis_theta = F / k
    # generalised coords new state
    new_acc = 6.0/(theta**3 * dt**2) * (dis_theta - dis) - 6.0/(theta**2 * dt) * vel + (1.0 - 3.0/theta) * acc
    new_vel = vel + dt*0.5 * (acc + new_acc)
    new_dis = dis + dt * vel + dt**2/6.0 * (new_acc + 2*acc)
    return new_dis, new_vel, new_acc

def effective_force(dt, mode, dis, vel, acc, wn, zeta, F):
    # prediction coef
    theta = 1.4
    # equation of motion coefs
    mg = 1.0
    cg = 2.0 * zeta * wn
    return mode.dot(F) + (6.0/(theta**2 * dt**2)*mg + 3.0/(theta*dt)*cg)*dis + (6.0/(theta*dt)*mg +2*cg)*vel + (2*mg + theta*dt*0.5)*acc

mesh, V = create_mesh()
KM, MM = create_matrices(V)
results = solve_eigenmodes(KM, MM)
freqs, modes = extract_modes(results)
K_inv = invert_stiffness(KM)
modal_masses = calc_modal_thing(modes, MM, V)
modal_stiffness = calc_modal_thing(modes, KM, V)
# TODO allow damping as CM = a*MM + b*KM

postprocess(mesh, V, KM, results)
