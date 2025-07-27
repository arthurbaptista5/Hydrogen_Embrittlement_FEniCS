#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:45:16 2024

@author: arthur
"""

from fenics import * 
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd

# Initialize MPI
# comm = MPI.comm_world
# rank = comm.Get_rank()
# size = comm.Get_size()

# print(f"Process {rank} out of {size} processes")

# Variaveis
E_ = Constant(210e9)                                       # Young's Modulus (N/m²)
poisson = Constant(0.3)                                    # Poisson Modulus
Gc_ = Constant(2700.0)                                     # Critical energy release rate (J/m2)
l = Constant(7.50e-6)                                      # Phase field length scale (m)
lambda_ = 121.15e9
mu = 80.76e9
rho = Constant(0.0)                                        # density (kg/m²)
g = Constant(0.0)                                          # gravity (m/s²)
wt_ppm = Constant(0.1)
C_nvalue = Constant(wt_ppm)                                # Inside body (mol/m3)
C_nboundary = Constant(wt_ppm)                             # (mol/m3)
Rb = Constant(8.31450)                                     # Constante de Boltzmann (N*m/mol*K)
T_O = Constant(300.0)                                      # Temperature Kelvin
nu = Constant(2.0e-6)                                      # Coeficiente de dilatacao (m3/mol)
D = Constant(1.27e-8)                                      # Coeficiente de difusao (m2/s)
M = D/(Rb*T_O)                                             # Coeficiente de mobilidade
del_Gibbs = Constant(30.0e3)                               # Gibbs free energy difference between the decohering interface and the surrounding material (N*m/mol)
qui = Constant(0.890)
k = Constant(1.0e-7)

# Dominio
mesh = Mesh('mesh.xml')                                    # mesh (unit: m)

# Define a funcao de espaco escalar (Phase-field)
V = FunctionSpace(mesh, 'CG', 1)

# Define a funcao de espaco escalar (Difusao)
E = FunctionSpace(mesh, 'P', 2)

# Define a funcao de espaco vetorial
W = VectorFunctionSpace(mesh,'CG', 1)

# Define a funcao de espaco escalar Discontinuous Lagrange para H+
WW = FunctionSpace(mesh, 'DG', 0)

# Contorno
tol_bd = 1.0e-6
def Crack(x):
    return abs(x[1]) < 1.0e-6 and x[0] <= 0.0

def left_bd(x, on_boundary):
    return on_boundary and near(x[0], -0.00050, tol_bd)
    
def bottom_bd(x, on_boundary):
    return on_boundary and near(x[1], -0.00050, tol_bd)

def right_bd(x, on_boundary):
    return on_boundary and near(x[0], 0.00050, tol_bd)

def top_bd(x, on_boundary):
    return on_boundary and near(x[1], 0.00050, tol_bd)

# CC e CI (Mecanico)
load = Expression("t", t = 0.0, degree = 1)                         # expressao dependente do tempo

bc_bottom_m = DirichletBC(W, Constant((0.0, 0.0)), bottom_bd)                
bc_top_m = DirichletBC(W.sub(1), load, top_bd)                      # deslocamento preescrito
bc_m = [bc_bottom_m, bc_top_m]

# CC e CI (Dano)
bc_phi = DirichletBC(V, Constant(1.0), Crack)                       # crack

# CC e CI (Difusao)  
C_n = project(C_nvalue, E)                                          # IC body
bcd_left = DirichletBC(E, C_nboundary, left_bd)
# bcd_top = DirichletBC(E, C_nboundary, top_bd)
bcd_right = DirichletBC(E, C_nboundary, right_bd)
bcd_bottom = DirichletBC(E, C_nboundary, bottom_bd)
bcd_crack = DirichletBC(E, C_nboundary, Crack)                      # crack

bc_d = [bcd_crack, bcd_left, bcd_bottom, bcd_right]

# Define Energia de degradacao dependente do hidrogenio
def teta(C_n):
    return ((C_n*5.50e-5)/((C_n*5.50e-5) + exp(-del_Gibbs/(Rb*T_O))))
def Gc(C_n):
    return Gc_*(1.0 - qui*teta(C_n))

# Define strain
def epsilon(u, C_n):
    E_elastic = sym(grad(u))   # Elastic strain tensor
    E_chemical = (nu/3)*C_n*Identity(d) 
    return E_elastic #+ E_chemical

# Define stress
def sigma(u, C_n):
    return ((1.0-pold)**2 + k)*(lambda_*tr(epsilon(u, C_n))*Identity(d) + 2*mu*epsilon(u, C_n))

def sigma_m(u, C_n):
    return (1/3) * tr(sigma(u, C_n))

def psi(u):
    return 0.5 * (lambda_+mu) * (0.5 * (tr(epsilon(u, C_n)) + abs(tr(epsilon(u, C_n)))))**2 + mu * inner(dev(epsilon(u, C_n)), dev(epsilon(u, C_n)))

def H(uold, unew, Hold):
    return conditional(lt(psi(uold), psi(unew)), psi(unew), Hold)

# Parametros da solucao
t = 0.0                   # T inicial
u_r = 1.0e-5              # Descolamento resultante (m)
dt = 0.10               # Time-step
tol = 1e-6
T_f = 1                 # T final (s)
u_step = u_r/T_f          # Passo do deslocamento

# Problema Dano
p = TrialFunction(V)
q = TestFunction(V)
pnew, pold, Hold = Function(V), Function(V), Function(WW)
pnew.assign(Constant(0.0))

# Problema Mecanico
u_trial = TrialFunction(W)
v = TestFunction(W)
unew, uold = Function(W), Function(W)

d = unew.geometric_dimension()
f = Constant((0, -rho*g))
T = Constant((0,0))

# Define Diffusion
C = Function(E)
v_d = TestFunction(E)

# Forma variacional problema elasticidade
F_m = inner(sigma(u_trial, C_n), epsilon(v, C_n))* dx - dot(f, v) * dx - dot(T, v) * ds
a, L = lhs(F_m), rhs(F_m)  

# Forma variacional problema difusao
F_d = (C - C_n) / dt * v_d * dx + D * dot(grad(C), grad(v_d)) * dx - M * nu * C_n * dot(grad(sigma_m(unew, C_n)), grad(v_d)) * dx

# Forma variacional problema dano 
F_phi = (Gc(C_n) * l * inner(grad(p), grad(q)) + ((Gc(C_n)/l) + 2.0 * H(uold, unew, Hold)) * inner(p, q) - 2.0 * H(uold, unew, Hold) * q) * dx
a_phi, L_phi = lhs(F_phi), rhs(F_phi)

#Contorno para output forca
top = CompiledSubDomain("near(x[1], 0.0005) && on_boundary")
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top.mark(boundaries, 1)
ds = Measure("ds")(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Output files
file_U = File("./DisplacementDir/Displacement.pvd")
file_von_Mises = File('./VonMisesDir/von_Mises.pvd')
file_Phi = File('./Phidir/phi.pvd')
file_C = File("./Concentration_Dir/Concentration.pvd")
fname = open('ForcevsDisp.txt', 'w')

# Solver parameters
p_disp = LinearVariationalProblem(a, L, unew, bc_m)
p_phi = LinearVariationalProblem(a_phi, L_phi, pnew, bc_phi)
J = derivative(F_d, C)
p_d = NonlinearVariationalProblem(F_d, C, bc_d, J=J)
solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)
solver_d = NonlinearVariationalSolver(p_d)

# Solve 
while t < T_f:
    t += dt
    if t >=(0.4*T_f):
        dt = 0.01

    load.t = t * u_step
    
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        solver_disp.solve()
        solver_phi.solve()
        solver_d.solve()
        
        # Error               
        err_u = errornorm(unew, uold, norm_type = 'l2', mesh = None)
        err_phi = errornorm(pnew, pold, norm_type = 'l2', mesh = None)
        err_d = errornorm(C, C_n, norm_type = 'l2', mesh = None)  
        err = max(err_u, err_phi, err_d)

        uold.assign(unew)
        pold.assign(pnew)
        Hold.assign(project(psi(unew), WW))    
        C_n.assign(C)

    if err < tol:
        print('Iterations:', iter, ', Total time', t)
  
        if round(t * 1e4) % 10 == 0: 

            # # Plot u no tempo
            file_U << (unew, t) 
                    
            # Plot von_mises stress no tempo
            s = sigma(unew, C_n) - (1/3)*tr(sigma(unew, C_n))*Identity(d) # deviatoric stress
            von_Mises_v = sqrt(3./2*inner(s, s)) 
            von_Mises = project(von_Mises_v, V)
            file_von_Mises << (von_Mises, t)

            # Plot phi no tempo
            file_Phi << (pnew, t)

            # Plot C
            file_C << (C, t)

            # Carga x deslocamento 
            Traction = dot(sigma(unew, C_n), n)
            fy = Traction[1]*ds(1)
            fname.write(str(t*u_step) + "\t")
            fname.write(str(assemble(fy)/1000) + "\n")
            print('fy =', assemble(fy)/1000)
            if (assemble(fy)/1000) < 10:
                t = T_f

print('Simulation completed')