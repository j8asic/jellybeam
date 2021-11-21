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
from celluloid import Camera

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
    nodes = mesh.geometry.x[:, 0]
    order = np.argsort(nodes)
    return mesh, V, order

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

def extract_modes(results, V, N = 3, min_freq = 0.01, max_error = 1e-4):
    vr, vi = KM.getVecs()
    freqs = []
    modes = []
    
    for i in range(results.getConverged()):
        k = results.getEigenpair(i, vr, vi)
        error = results.computeError(i)
        if len(freqs) < N and not np.isnan(k.real) and k.real > min_freq and error < max_error:
            m = dolfinx.Function(V)
            m.vector.array[:] = PETSc.Vec.getArray(vr)
            modes.append(m)
            freqs.append(np.sqrt(k.real))
    
    return freqs, modes

def invert_stiffness(K):
    ai, aj, av = K.getValuesCSR()
    K_np = scipy.sparse.csr_matrix((av, aj, ai)).todense()
    return scipy.linalg.inv(K_np, overwrite_a=False, check_finite=True)

def calc_modal_thing(modes, what, V):
    out = []
    tmp = dolfinx.Function(V)
    for m in modes:
        what.mult(m.vector, tmp.vector)
        #result = np.dot(m.sub(0).vector.array[order], )
        out.append(m.vector.dot(tmp.vector))
    return out

def postprocess(mesh, V, order, KM, results, min_freq = 0.01, max_error = 1e-4):

    eigenmode = dolfinx.Function(V)
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
                plt.plot(mesh.geometry.x[order, 0], result[order] / np.max(np.abs(result[order])))

    print("")
    plt.legend(["2-node", "3-node", "4-node", "5-node"])
    plt.show()

# Newmark-Beta integrator (gamma = 0.5, beta = 1/6 are coefs for linear acceleration)
def update_eta_newmark(dt, dis, vel, acc, wn, zeta, F, Fprev, gamma = 0.5, beta = 1.0/6.0):
    
    a1 = 1 / (beta * np.power(dt, 2)) + (gamma * 2 * zeta * wn) / (beta * dt)
    a2 = 1 / (beta * dt) + (gamma / beta - 1) * (2 * zeta * wn)
    a3 = (1 / (2 * beta) - 1) + dt * 2 * zeta * wn * (gamma / (2 * beta) - 1)

    new_disp = (F + a1 * dis + a2 * vel + a3 * acc) / (np.power(wn, 2) + a1)
    new_veloc = (gamma / (beta * dt)) * (new_disp - dis) + (1 - gamma / beta) * vel + dt * (1 - gamma / (2 * beta)) * acc
    new_accel = 1 / (beta * np.power(dt, 2)) * (new_disp - dis) - (1 / (beta * dt)) * vel - (1 / (2 * beta) - 1) * acc

    return new_disp, new_veloc, new_accel

# Complementary Function and Particular Integral (CF&PI) method
# "Fluid–Structure Interaction Using a Modal Approach", doi:10.1115/1.4004859
def update_eta_cfpi(dt, dis, vel, acc, wn, zeta, F, Fprev):
    #Fprev = F
    s = np.sqrt(1 - zeta**2)
    wts = wn * dt * s
    e = np.exp(-zeta * wn * dt)
    S, C = np.sin(wts), np.cos(wts)

    # complementary solution
    new_dis = dis * e * (C + zeta/s*S) + vel * e * 1/(wn*s)*S
    new_vel = vel * e * (C - zeta/s*S) - dis * e * wn/s*S

    # particular solution
    new_dis += \
        -e * Fprev * ((zeta*wn*dt + 2*zeta**2 - 1)/(wn**2*wts) * S + (wn*dt+2*zeta)/(wn**3*dt)*C) \
        + Fprev * 2*zeta/(wn**3 * dt) \
        + e * F * ((2*zeta**2 - 1)/(wn**2*wts) * S + (2*zeta)/(wn**3*dt)*C) \
        + F * (wn*dt - 2*zeta)/(wn**3*dt)

    new_vel += \
          e * Fprev * ((zeta + wn*dt)/(wn*wts) * S + 1/(wn**2*dt)*C) \
        - Fprev * 1/(wn**2 * dt) \
        - e * F * (zeta/(wn*wts) * S + 1/(wn**2*dt)*C) \
        + F * 1/(wn**2*dt)

    return new_dis, new_vel, acc

mesh, V, order = create_mesh()
KM, MM = create_matrices(V)
results = solve_eigenmodes(KM, MM)
freqs, modes = extract_modes(results, V)
n_modes = len(freqs)
K_inv = invert_stiffness(KM)
modal_masses = calc_modal_thing(modes, MM, V)
modal_stiffness = calc_modal_thing(modes, KM, V)
# TODO allow damping as CM = a*MM + b*KM

#postprocess(mesh, V, order, KM, results)

def calc_force(t, mesh, order):
    F = dolfinx.Function(V)
    u,v = F.split()
    if t < 0.03:
        u.interpolate(lambda x: ((x[0] > 0.5) * (1 - (4*(x[0] - 0.8))**2)) * t/0.03 * 5e-2)
    return F

dt, t, i = 1e-2, 0.0, 0
dis, vel, acc, Fmodal = np.zeros(n_modes), np.zeros(n_modes), np.zeros(n_modes), np.zeros(n_modes)
ds, ts = [[] for i in range(n_modes)], []
fig = plt.figure()
camera = Camera(fig)

while t < 0.2:

    F = calc_force(t, mesh, order)

    yeah = dolfinx.Function(V)
    yeah.vector.array[:] = np.dot(K_inv, F.vector.array)
        
    for j in range(n_modes):
        wn = freqs[j]
        mode = modes[j]
        Fprev = Fmodal[j]
        Fmodal[j] = mode.vector.dot(F.vector)
        dis[j], vel[j], acc[j] = update_eta_cfpi(dt, dis[j], vel[j], acc[j], wn, 0.05, Fmodal[j], Fprev)
        ds[j].append(dis[j])
    if i % 20:
        result = np.zeros_like(F.sub(0).compute_point_values())
        for j in range(n_modes):
            result += modes[j].sub(0).compute_point_values() * dis[j]
        #plt.clf()
        #plt.ylim([-2, 2])
        #plt.plot(mesh.geometry.x[order, 0], result[order], c='black', lw=4)
        #plt.scatter(mesh.geometry.x[order, 0], result[order], s=1e4*np.absolute(F.sub(0).compute_point_values()[order]), color='red')
        #plt.draw()
        #plt.pause(0.001)
        #camera.snap()

    ts.append(t)
    i += 1
    t += dt
#plt.show()

#animation = camera.animate(interval = 30)
#animation.save('/home/josip/Temp/beam_01.mp4')

for j in range(n_modes):
    plt.plot(ts, ds[j])
plt.xlim([0, 0.2])
plt.ylim([-0.004, 0.004])
plt.legend(["1", "2", "3"])
plt.show()