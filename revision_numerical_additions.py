import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import root
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"figures"; DATA=ROOT/"data"
OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

def q_nu(beta,gamma):
    gamma=np.asarray(gamma,float); q=np.empty_like(gamma); nu=np.empty_like(gamma)
    small=np.abs(gamma)<1e-12; q[small]=beta; nu[small]=-beta**3/3
    g=gamma[~small]
    if g.size:
        z=beta*g
        q[~small]=np.tanh(z)/g
        nu[~small]=(beta*g/np.cosh(z)**2-np.tanh(z))/(2*g**3)
    return q,nu

def response(x,beta,gamma):
    x=np.asarray(x,float); gamma=np.asarray(gamma,float)
    R=np.sqrt(x*x+gamma*gamma); y=np.zeros_like(x); nz=R>1e-14
    y[nz]=x[nz]/R[nz]*np.tanh(beta*R[nz]); return y

def response_prime(x,beta,gamma):
    x=np.asarray(x,float); gamma=np.asarray(gamma,float)
    R=np.sqrt(x*x+gamma*gamma); gp=np.empty_like(x); nz=R>1e-12
    if np.any(nz):
        rn=R[nz]; xn=x[nz]; gn=gamma[nz]; z=beta*rn
        gp[nz]=(gn**2/rn**3)*np.tanh(z)+beta*(xn**2/rn**2)/np.cosh(z)**2
    gp[~nz]=beta; return gp

def linear_data(beta,alpha,gamma,J):
    q,nu=q_nu(beta,gamma); M=np.diag(q)@J@np.diag(alpha)
    vals,vr=eig(M); i=int(np.argmax(vals.real)); lam=float(vals[i].real)
    v=vr[:,i].real
    if np.sum(v)<0: v=-v
    v/=np.linalg.norm(v)
    valsl,vl=eig(M.T); j=int(np.argmin(np.abs(valsl.real-lam))); w=vl[:,j].real
    if w@v<0: w=-w
    w/=w@v; tc=1/lam; K=J@np.diag(alpha)
    kappa=-(tc**3)*np.sum(w*nu*(K@v)**3)
    return q,tc,v,float(kappa)

def solve_eq(tau,beta,alpha,gamma,J,init):
    F=lambda m:m-response(tau*(J@(alpha*m)),beta,gamma)
    sol=root(F,init,method="hybr",tol=1e-12)
    if np.linalg.norm(sol.fun)>1e-9: raise RuntimeError("root solve failed")
    return sol.x

def stability(m,tau,beta,alpha,gamma,J):
    x=tau*(J@(alpha*m)); gp=response_prime(x,beta,gamma)
    A=-np.eye(len(m))+tau*np.diag(gp)@J@np.diag(alpha)
    return float(np.max(np.linalg.eigvals(A).real))

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# Figures 8-9: broad cooperative branch and stability
beta=1.30; alpha=np.array([.55,.45]); gamma=np.array([.55,1.00])
J=np.array([[1.40,.75],[.75,1.10]])
_,tc,v,_=linear_data(beta,alpha,gamma,J)
td=tc*np.linspace(2.20,1.001,140); branch=[]; init=np.array([.95,.85])
for t in td:
    m=solve_eq(t,beta,alpha,gamma,J,init)
    if m@v<0:m=-m
    branch.append(m); init=m
branch=np.asarray(branch); stab=np.array([stability(m,t,beta,alpha,gamma,J) for m,t in zip(branch,td)])
o=np.argsort(td); t=td[o]; branch=branch[o]; stab=stab[o]
fig,ax=plt.subplots(); ax.plot(t/tc,branch[:,0],label="$m_1$"); ax.plot(t/tc,branch[:,1],label="$m_2$")
ax.axvline(1,ls=":"); ax.set(xlabel="$\\tau/\\tau_c$",ylabel="Equilibrium community orientation",title="Community-resolved cooperative equilibrium branch"); ax.legend()
save(fig,"fig8_community_branches")
fig,ax=plt.subplots(); ax.plot(t/tc,stab); ax.axhline(0,ls=":"); ax.axvline(1,ls=":")
ax.set(xlabel="$\\tau/\\tau_c$",ylabel="Largest real Jacobian eigenvalue",title="Stability along the cooperative branch")
save(fig,"fig9_branch_stability")
np.savetxt(DATA/"broad_cooperative_branch.csv",np.column_stack([t/tc,t,branch,stab]),delimiter=",",header="tau_over_tauc,tau,m1,m2,max_real_jacobian",comments="")

# Figure 10: robustness under small asymmetry
beta_h=1.40; Jh=np.array([[1.,-1.2],[-1.2,1.]])
a0=np.array([.5,.5]); g0=np.array([.55,.55]); _,tch,_,_=linear_data(beta_h,a0,g0,Jh)
th=1.4*tch; m0=solve_eq(th,beta_h,a0,g0,Jh,np.array([.75,-.75]))
if m0[0]<0:m0=-m0
da=np.linspace(-.05,.05,41); dg=np.linspace(-.10,.10,41)
C=np.empty((41,41)); S=np.empty_like(C); Opp=np.empty_like(C,dtype=int)
for i,xg in enumerate(dg):
    for j,xa in enumerate(da):
        a=np.array([.5+xa,.5-xa]); g=np.array([.55+xg,.55-xg])
        m=solve_eq(th,beta_h,a,g,Jh,m0)
        if m[0]<0:m=-m
        p=np.sqrt(np.sum(a*m*m)); C[i,j]=abs(a@m)/p
        S[i,j]=stability(m,th,beta_h,a,g,Jh); Opp[i,j]=int(m[0]*m[1]<0)
fig,ax=plt.subplots(); im=ax.imshow(C,origin="lower",aspect="auto",extent=[da[0],da[-1],dg[0],dg[-1]])
ax.set(xlabel=r"Weight asymmetry $\delta_\alpha$",ylabel=r"Response asymmetry $\delta_\Gamma$",title=r"Cancellation ratio $|M_{\rm glob}|/P_{\rm com}$ at $\tau=1.4\tau_c$")
fig.colorbar(im,ax=ax); save(fig,"fig10_hidden_asymmetry_robustness")
with open(DATA/"hidden_asymmetry_grid.csv","w") as f:
    f.write("delta_alpha,delta_gamma,cancellation_ratio,max_real_jacobian,oppositional\n")
    for i,xg in enumerate(dg):
        for j,xa in enumerate(da): f.write(f"{xa},{xg},{C[i,j]},{S[i,j]},{Opp[i,j]}\n")

# Figure 11: K=4 branch
b4=1.30; a4=np.array([.34,.27,.22,.17]); g4=np.array([.35,.65,.95,1.25])
J4=np.array([[1.35,.80,.25,.10],[.80,1.20,.50,.20],[.25,.50,1.05,.60],[.10,.20,.60,.95]])
q4,tc4,v4,k4=linear_data(b4,a4,g4,J4); td=tc4*np.linspace(2,1.001,120)
br=[]; init=np.ones(4)*.75
for x in td:
    m=solve_eq(x,b4,a4,g4,J4,init)
    if m@v4<0:m=-m
    br.append(m); init=m
br=np.asarray(br); st4=np.array([stability(m,x,b4,a4,g4,J4) for m,x in zip(br,td)])
o=np.argsort(td); t4=td[o]; br=br[o]; st4=st4[o]
fig,ax=plt.subplots()
for i in range(4): ax.plot(t4/tc4,br[:,i],label=fr"$m_{i+1}$")
ax.axvline(1,ls=":"); ax.set(xlabel="$\\tau/\\tau_c$",ylabel="Equilibrium community orientation",title="Four-community branch with heterogeneous responses"); ax.legend(ncol=2)
save(fig,"fig11_four_community_branch")
np.savetxt(DATA/"four_community_branch.csv",np.column_stack([t4/tc4,t4,br,st4]),delimiter=",",header="tau_over_tauc,tau,m1,m2,m3,m4,max_real_jacobian",comments="")

summary=(
f"two-community broad branch: tau_c={tc:.10f}; stability max={stab.max():.10f}; at 2.2 tau_c m={branch[-1].tolist()}\n"
f"hidden asymmetry: tau={th:.10f}=1.4 tau_c(sym); max cancellation ratio={np.max(C):.10f}; all oppositional={bool(np.all(Opp))}; stability max={np.max(S):.10f}\n"
f"K4: q={q4.tolist()}; tau_c={tc4:.10f}; v_c={v4.tolist()}; kappa={k4:.10f}; stability max={st4.max():.10f}\n")
(DATA/"revision_numerical_summary.txt").write_text(summary)
print(summary)
