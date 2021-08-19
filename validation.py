# FENICS/DOLFIN validation of free-free beam vibration

import dolfinx
#from dolfinx.io import XDMFFile
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

# beam properties
n_ele = 100
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
    ksp.setType(ksp.Type.LGMRES)
    ksp.setTolerances(rtol=1E-5)
    ksp.setPC(pc)

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

def postprocess(mesh, KM, results, min_freq = 0.01, max_error = 1e-4):

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
        if np.isnan(k.real) or k.real <= min_freq or error > max_error:
            continue

        n_okay += 1
        print(" %12f       %12g" % (np.sqrt(k.real) / (2*np.pi), error)) # rad/s to Hz
        if n_okay <= 3:
            er = PETSc.Vec.getArray(vr)
            eigenmode.vector.array[:] = (er / np.max(np.abs(er)))
            eigenmode.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            result = eigenmode.sub(0).compute_point_values()
            plt.plot(nodes[order], result[order])

    print("")
    plt.show()


mesh, V = create_mesh()
KM, MM = create_matrices(V)
results = solve_eigenmodes(KM, MM)
postprocess(mesh, KM, results)
