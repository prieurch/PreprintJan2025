# To illustrate Theorem 1 of the preprint
# Stability of the wave equation with a saturated dynamic boundary control
# contact: christophe.prieur@gipsa-lab.fr
# January 2025

import numpy as np
import control
import scipy.linalg as la
import scipy.integrate as integrate
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

plt.close('all')

# saturation level
u0=1.5

# time horizon
T=6

# length of the domain
L=1

# matrices of the finite dimensional system
A=np.zeros((2,2))
A[0,0]=-1; A[0,1]=-1; A[1,1]=-2
B=np.zeros(2)
B[1]=1
B=B.reshape(2,1)
n=len(B)
    
CtrbMatrix= control.ctrb(A,B) 

if np.linalg.matrix_rank(CtrbMatrix)==n:
    print('Controllable system')
else:
    print('Uncontrollable system')

C= np.zeros(2)
C[0]=0.6; C[1]=2.3
#C[0]=0.5; C[1]=2 # is also possible
C=C.reshape(1,2)
  
d=0.5*np.eye(1)

# check first assumptions of Theorem
e, v= np.linalg.eig(A) 
if max(e.real)>=0 or d<0:
    print('assumption of Theorem 1 is not satisfied')
    
# choice of the initial condition, z_0, z_1 and w_0
def z0(x):              # Initial condition for z
    return 5*x*(L-x)**2

def z1(x):              # Initial condition for zt
    return 0

w0=np.zeros(2)          # initial condition for w
w0[0]=0; w0[1]=0

def f(x): # z_x ** 2 for the previous initial condition
    return (5-20*x+15*x**2)**2

y,err= integrate.quad(f, 0, 1)

lyap0=y # this if the value of V(t=0) (it does not depend on P for this initial condition)

def f(x): # z_xx ** 2 for the previous initial condition
    return (30*x-20)**2

y,err= integrate.quad(f, 0, 1)
m=np.sqrt(y) # the condition | A (initial) | <= m is satisfied
print('First condition of the basin of attraction satisfied')
 
def checkMI(tau1,tau2,eta,P,g,r,G,verboseValue=True):
    # to check the matrix inequalities in Theorem 1
    # we will build the three matrices
    Test=True
    M11=np.dot(A.transpose(),P)+np.dot(P,A)
    M12=np.dot(P,B)-C.transpose()
    M31=-eta*(C+G)
    M1=np.hstack((M11,M12,M31.transpose()))
    M23=(-1-eta*(d+g))*np.eye(1)
    M2=np.hstack((M12.transpose(),-2*d*np.eye(1),M23))
    M3=np.hstack((M31,M23.transpose(),-2*eta*np.eye(1)))
    firstM= np.vstack((M1,M2,M3))
    e,v=la.eig(firstM) # eigenvalues should be negative
    if max(e)>0:
        if verboseValue:
            print('First Matrix Inequality is false')
        Test=False
    
    M11=tau1*np.dot(A.transpose(),A)+tau2*P
    M12=tau1*np.dot(A.transpose(),B)
    M13=G.transpose()
    M1=np.hstack((M11,M12,M13))
    M22=tau1*np.dot(B.transpose(),B)
    M23=g*np.eye(1)
    M2=np.hstack((M12.transpose(),M22,M23))
    M33=u0**2*np.eye(1)
    M3=np.hstack((M13.transpose(),M23.transpose(),M33))
    secondM= np.vstack((M1,M2,M3))
    e,v=la.eig(secondM) # eigenvalues should be positive
    if min(e)<0:
        if verboseValue:
            print('Second Matrix Inequality is false')
        Test=False
        
    thirdM=m*tau1+r*tau2-1   # should be negative
    if thirdM>=0:
        if verboseValue:
            print('Third Inequality is false')  
        Test=False
    if Test and verboseValue:
        print('All inequalities satisfied')
    return Test
    
def ComputeVariables(tau2,eta,verboseValue=True):
    # to compute the variables in Theorem 1 from tau2 and eta
    # we will use a LMI solver for that
    tau1 = cp.Variable((1,1))
    P = cp.Variable((2,2), PSD=True)
    g = cp.Variable((1,1))
    r = cp.Variable((1,1))
    G=cp.Variable((1,2))

    # build the constraints
    constr = []

    M11=A.transpose()@P+P@A
    M12=P@B-C.transpose()
    M31=-eta*(C+G)
    M1=cp.hstack((M11,M12,cp.transpose(M31)))
    M23=(-1-eta*(d+g))
    M2=cp.hstack((cp.transpose(M12),-2*d,M23))
    M3=cp.hstack((M31,cp.transpose(M23),-2*eta*np.eye(1)))
    firstM= cp.vstack((M1,M2,M3))

    constr += [firstM<<0]

    M11=tau1*np.dot(A.transpose(),A)+tau2*P
    M12=tau1*np.dot(A.transpose(),B)
    M13=cp.transpose(G)
    M1=cp.hstack((M11,M12,M13))
    M22=tau1@np.dot(B.transpose(),B)
    M23=g
    M2=cp.hstack((cp.transpose(M12),M22,M23))
    M33=u0**2*np.eye(1)
    M3=cp.hstack((cp.transpose(M13),cp.transpose(M23),M33))
    secondM=cp.vstack((M1,M2,M3))

    constr += [secondM>>0]

    thirdM=m*tau1+r*tau2-0.9999 # I added a small margin on the third inequality
    constr += [thirdM<=0]

    fourthM=tau1-0.01 # I added a small margin to be sure that tau1 is positive
    constr += [fourthM>=0]

    prob = cp.Problem(cp.Maximize(r),constr)
    try:
        result = prob.solve(solver='CVXOPT',verbose=verboseValue)
    except:
        pass
    if prob.status=='infeasible' or prob.status==None: 
        # print('no solution found')
        return 0,0,0,0,0
    else:     
        tau1=tau1.value; P=P.value; g=g.value; r=r.value; G=G.value
        ResultTest=checkMI(tau1,tau2,eta,P,g,r,G,verboseValue)
        if ResultTest:
            return tau1,P,g,r,G
        else:
            return 0,0,0,0,0
rmax=0

# tau2=0.1
# eta=1
# tau1,P,g,r,G= ComputeVariables(tau2,eta)
# ResultTest=checkMI(tau1,tau2,eta,P,g,r,G)

# tau2=0.001
# eta=0.005
# tau1,P,g,r,G= ComputeVariables(tau2,eta)
# ResultTest=checkMI(tau1,tau2,eta,P,g,r,G)

verboseValue=False # select False to show only essentials

# tau2 could be between 0.1 and 10
for tau2 in tqdm(np.linspace(0.0005, 10, 50)):
    #tau2=1 # just for the time being.

    # eta could be between 0.001 and 1 
    for eta in np.linspace(0.005, 10, 50):
        #eta= 1
        tau1,P,g,r,G= ComputeVariables(tau2, eta,verboseValue)

        if r>rmax:
            print('We computed r='+str(r[0][0]))
            Pmax=P; gmax=g; rmax=r; Gmax=G
            tau1max=tau1; tau2max=tau2; etamax=eta

        if r > lyap0:
            print('Second condition of the basin of attraction satisfied, still trying to optimize')

# computed values after the previous optimization
# tau1max = 0.00999999
# tau2max = 0.20457142857142857
# etamax = 1.4328571428571426
# rmax = 4.3989531
# gmax = -0.07766258
# Gmax = [ 0.01555488, -0.50830176]
    
# to get the best values:
if 0:
    tau1=tau1max;tau2=tau2max;eta=etamax
    P=Pmax;g=gmax;r=rmax;G=Gmax

# to round the best values (as done in the preprint):
if 1:
    def round_it(x,sig):
        return float('{:g}'.format(float('{:.{p}g}'.format(x, p=sig))))
    sig=6 # significant figures
    tau1=round_it(tau1max[0][0],sig); tau2=round_it(tau2max,sig)
    eta=round_it(etamax,sig); P=np.zeros((2,2)); P[0,0]=round_it(Pmax[0,0],sig)
    P[0,1]=round_it(Pmax[0,1],sig); P[1,0]=P[0,1]
    P[1,1]=round_it(Pmax[1,1],sig); g=round_it(gmax[0][0],sig)
    r=round_it(rmax[0][0],sig); G=np.zeros((1,2))
    G[0,0]=round_it(Gmax[0,0],sig); G[0,1]=round_it(Gmax[0,1],sig)

# checking the matrix inequalities with these values
ResultTest=checkMI(tau1,tau2,eta,P,g,r,G)
e,v=la.eig(P)

if r > lyap0:
    print('Both conditions of the basin of attraction satisfied')
    
if not(ResultTest) or min(e)<0:
    print('Matrix inequalities are not satisfied')      

# CFL condition (needs to be <1)
CFL=0.9

dt=0.001
Nt = int(round(T/dt))
t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time

dx = dt/CFL # to ensure the CFL condition
Nx = int(round(L/dx))
x = np.linspace(0, L, Nx+1)

C2 = (dt/dx)**2               # Help variable in the scheme

def sat(u):
    if np. absolute (u)<u0:
        sigma=u
    else:
        sigma=u0*np.sign(u)
    return sigma

def lyap(z_1,z,w):
    # compute the Lyapunov function candidate 
    # we neglect the exp(mu)
    zx= (z[1:]-z[:-1])/dx
    zt= (z[1:]-z_1[1:])/dt
    tmp = 0.5*np.sum( ( zt +zx )**2*dx )   # int of square of zt+zx
    tmp+= 0.5*np.sum( ( zt -zx )**2*dx ) # int of square of zt-zx
    tmp+= np.dot(np.dot(w.transpose(),P),w)
    return tmp

# view the plots after the simulations
def view(z,w,control,lyap,names):
    # inputs
    # z: solution to the PDE -> 3D plot
    # w: solution to the ODE -> 2D plot
    # control: boundary control -> 2D plot
    # lyap: time evolution of the lyapunov values -> 2D plot
    # names: for the figure savings
        
    if factor==1:
        add='1'
    else:
        add='2'
    
    POV=[12,-60] # point of view, elevation and azimuth
    
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ST, SX = np.meshgrid(t, x)
    ax.plot_surface(ST, SX, z.T, cmap='jet')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('z(t,x)')
    ax.view_init(elev=POV[0], azim=POV[1]) # adjust view
    plt.savefig(names[0]+add+'.eps',format='eps')

    fig , ax= plt.subplots()
    for i in range(2):
        if i==0:
            line='r:'
        else:
            line='b-'
        ax.plot(t,w[:,i], line, label='w_'+str(i+1))
    ax.set_xlabel('t')
    ax.set_ylabel('w(t)')
    ax.legend()
    plt.savefig(names[1]+add+'.eps',format='eps')
    
    fig , ax= plt.subplots()
    ax.plot(t,control)
    ax.set_xlabel('t')
    ax.set_ylabel('sat(t)')
    plt.savefig(names[2]+add+'.eps',format='eps')

    fig , ax= plt.subplots()
    ax.plot(t,lyap)
    ax.set_xlabel('t')
    ax.set_ylabel('V(t)')
    plt.savefig(names[3]+add+'.eps',format='eps',bbox_inches='tight')
    
    color = 'red'
    fig , ax= plt.subplots()
    ax.set_xlabel('t')
    ax.set_ylabel('sat(t)', color=color)
    line='r:'
    line1, = ax.plot(t,control, line,label='sat(t)')
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'blue'
    ax2.set_ylabel('V(t)', color=color)  # we already handled the x-label with ax1
    line='b-'
    line2, = ax2.plot(t, lyap,  line,label='V(t)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax.legend(handles=[line1, line2],loc=1)
    
    plt.savefig(names[2]+names[3]+add+'.eps',format='eps',bbox_inches='tight')

# multitplicative factor for the initial condition
# factor =1 for the converging solution
# factor =100 for the diverging one

for factor in [1,100]:
    
    ww = np.zeros((len(t),2))       # to save the w solution
    zz = np.zeros((len(t),len(x)))  # to save the solution
    lyap_zz = np.zeros(len(t))      # to save the Lyapunov values
    control_zz = np.zeros(len(t))   # to save the boundary control

    w  = np.zeros(2)        # Solution w at new time level
    w_1 = np.zeros(2)       # Solution w at 1 time level back
    z   = np.zeros(Nx+1)   # Solution array at new time level
    z_1 = np.zeros(Nx+1)   # Solution at 1 time level back
    z_2 = np.zeros(Nx+1)   # Solution at 2 time levels back
    

    # Set initial condition z(0,x) = z0(x)
    zz[0,:]= factor*z0(x)
    ww[0]  = factor*w0      

    # apply formula for n=0
    z_2[:] = factor*z0(x)
    w_1[:] = factor*w0
                      
    # apply formula for n=1
    z_1[1:-1] = z_2[1:-1] + dt*factor*z1(x[1:-1])  +C2/2*(z_2[2:] - 2*z_2[1:-1] + z_2[:-2])

    z_1[0] = 0 # Enforce Dirichlet boundary conditions

    # initial value of the control
    control_zz[0]= sat(d*factor*z1(x[-1])+np.dot(C,w_1))  
    z_1[-1] =  z_2[-1] -dx*control_zz[0]
 
    lyap_zz[0]=lyap(z_2,z_1,w_1)

    for n in range(1, Nt+1):
        # Update all inner mesh points at time t[n+1]  
        z[1:-1] = 2*z_1[1:-1] - z_2[1:-1] +C2*(z_1[2:] - 2*z_1[1:-1] + z_1[:-2])
          
        # compute control    
        control_zz[n]=sat(d*(z_1[-1]-z_2[-1])/dt +np.dot(C,w_1))
    
        # Insert boundary conditions
        # Dirichlet at x=0
        z[0] = 0;
        # controlled Neumann condition at x=1
        # taking into account the previous equation
        z[-1] = z[-2] -dx*control_zz[n]
    
        tmp=np.dot(B,(z_1[-1]-z_2[-1])/dt)
        tmp=tmp.reshape(2)
        w= w_1+dt*(np.dot(A,w_1)+tmp)
    
        # compute the Lyapunov function value
        lyap_zz[n]=lyap(z_1,z,w)
    
        # Switch variables before next step
        z_2[:], z_1[:] = z_1, z
        w_1[:] = w
    
        # save result
        zz[n,:]=z 
        ww[n,:]=w 
    
    fig , ax= plt.subplots()
    ax.set_title('Initial condition')
    ax.plot(x, z0(x),'g-', label='$z_0$')
    ax.legend()
    ax.set_rasterized(True)
    plt.savefig('to_check.eps',format='eps',bbox_inches='tight')

    fig , ax= plt.subplots()
    n=30
    ax.set_title('Solution at t='+str(t[n]))
    ax.plot(x, zz[n,:],'g-', label='z(t,.)')
    ax.legend()
  
    names=['pde','ode','control','lyapunov']
    view(zz,ww,control_zz,lyap_zz,names)

print('DONE')

