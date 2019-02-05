### PLASSIER : Minimisation par precessus gaussien
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from numpy.random import randn # randn(k)=k var gausiennes centrées réduites

plt.ion()
plt.show()

##Q1
Y_reel=lambda x: -np.sin(np.pi*x/2)
p=4
sigma_mes=0.1
lc=1
n=150 #Nombre observations
X = np.linspace(-1,2.5,num=n)
espsilon=sigma_mes*randn(n)
Y_obs=Y_reel(X)+espsilon
C_mod=lambda x,sigma_mod: sigma_mod**2*np.exp(-abs(x[0]-x[1])**2/lc**2)

plt.figure(1)
plt.clf()
plt.axis([-1,2.5,-1,2])
XX=np.linspace(-1,2.5,10**3)
plt.plot(XX,Y_reel(XX),label='Y_reel')
plt.plot(X,Y_obs,'*',label='Y_obs')
plt.title('Observations de Y_reel(x) obtenues pour n=%s'%(n))
plt.legend()

##Q2:
H=np.zeros((p,n))
for i in range(p):
    H[i]=X**i
H=H.T

[xx,yy]=np.meshgrid(X,X)
def Fonction(sigma_mod):
    R=C_mod([xx,yy],sigma_mod)+sigma_mes*np.eye((n))
    R_inv=np.linalg.inv(R)
    matrice1=np.dot(H.T,R_inv)
    Q_post=np.linalg.inv(np.dot(matrice1,H))
    Beta_post=np.dot(np.dot(Q_post,matrice1),Y_obs)
    Y_bis=Y_obs-np.dot(H,Beta_post)
    return np.log(np.linalg.det(R))+np.dot(Y_bis.T,np.dot(R_inv,Y_bis))+1000*(sigma_mod<0)

from scipy.optimize import fmin#sinon mon python bug
sigma_mod=sc.optimize.fmin(Fonction,1)[0] #Armijo ou Wolfe possibles
XX=np.linspace(0,1,10**2)

plt.figure(2)
plt.clf()
plt.plot(XX,[Fonction(x) for x in XX],color='darkslateblue',label='Fonction(sigma)')
plt.title('Recherche de sigma_mod par maximum de vraissemblance pour n=%s'%(n))
plt.legend()

##Q3:
h=lambda x: x**np.arange(0,p,1)
r=lambda x: C_mod([x,X],sigma_mod)

R=C_mod([xx,yy],sigma_mod)+sigma_mes*np.eye((n))
R_inv=np.linalg.inv(R)
matrice1=np.dot(H.T,R_inv)
Q_post=np.linalg.inv(np.dot(matrice1,H))
Beta_post=np.dot(np.dot(Q_post,matrice1),Y_obs)
Y_bis=Y_obs-np.dot(H,Beta_post)
Y_post=lambda x: np.dot((h(x)).T,Beta_post)+np.dot(r(x).T,np.dot(R_inv,Y_bis))
A=np.zeros((n+p,n+p))
A[p:,:p]=H
A[:p,p:]=H.T
A[p:,p:]=R
A=np.linalg.inv(A)
vecteur=lambda x: np.array(list(h(x))+list(r(x)))
Var_post=lambda x,sigma_mod: sigma_mod**2-np.dot(vecteur(x),np.dot(A,vecteur(x).T))

XX=np.linspace(-1,2.5,10**3)
Z1=np.array([Y_post(x) for x in XX])
Z2=2*np.array([np.sqrt(Var_post(x,sigma_mod)) for x in XX])
Y_bas=Z1-Z2
Y_haut=Z1+Z2

plt.figure(3)
plt.clf()
plt.axis([-1,2.5,-1,1])
plt.plot(XX,Y_bas,label='Y_bas')
plt.plot(XX,Y_haut,label='Y_haut')
plt.plot(XX,Y_reel(XX),label='Y_reel')
plt.title('Tube de confiance pour n=%s'%(n))
plt.legend()

##Q4:
TT=X #réalisations en ces points
l=len(TT)
M=np.zeros((l,n+p))

for i in range(l):
        M[i]=list(np.array(h(TT[i])))+list(r(TT[i]))
[xx,yy]=np.meshgrid(TT,TT)
covariance=C_mod([xx,yy],sigma_mod)-np.dot(M,np.dot(A,M.T))

m=np.min(np.linalg.eigvals(covariance))
S_cov=np.linalg.cholesky(covariance+10*abs(m)*np.eye(l))
C=np.dot(S_cov,randn(l))
Y_realisation_post=[Y_post(TT[i]) for i in range(l)]+C

plt.figure(4)
plt.clf()
plt.axis([-1,2.5,-1,2])
plt.plot(TT,Y_realisation_post,'ro',label='Y_realisation_post')
plt.plot(XX,Y_reel(XX),label='Y_reel')
plt.plot(X,Y_obs,'*',label='Y_obs')
plt.title('Réalisation de Y_reel(x) selon la loi à postériori n=%s'%(n))
plt.legend()

##Q5
M=1000 # Nombre itérations par boucle
Y_derivee_reel=lambda x: -np.pi/2*np.cos(np.pi/2*x)

def diff_fini(Z1):
    Y_der_mc=(Z1[1:]-Z1[:-1])
    return Y_der_mc*len(Z1)/3.5

Yder_mc=diff_fini(Y_obs)
Var_p=[Var_post(X[j],sigma_mod) for j in range(n)]
Y_p=[Y_post(X[j]) for j in range(n)]

L=np.zeros((M,n-1))

for i in range(M):
    C=np.dot(S_cov,randn(n))
    Y_realisation=Y_p+C
    Y_deri=diff_fini(Y_realisation)
    L[i]=Y_deri
    Yder_mc=Yder_mc+Y_deri
Yder_mc=Yder_mc/M # la dérivée par Monte Carlo

J=L-np.dot(np.ones((M,n-1)),np.diag(Yder_mc))
J=J**2
Var_der_mc=np.mean(J,axis=0)+sigma_mod**2 # variance empirique 
lowtube=Yder_mc-1.96*np.sqrt(Var_der_mc)
uppertube=Yder_mc+1.96*np.sqrt(Var_der_mc)

plt.figure(5)
plt.clf()
plt.axis([-1,2.5,-2.7,3.5])
plt.plot(X,Y_derivee_reel(X),label='Y_dérivée réelle')
plt.plot(X[:-1],lowtube, label='lowtube')
plt.plot(X[:-1],uppertube, label='uppertube')
plt.plot(X[:-1],Yder_mc,label='Y_mc')
plt.title('Monte Carlo pour n=%s observations'%(n))
plt.legend()

##Q6 Pour un seul jeu d'observations
nb_it=5000
liste=[]
for i in range(nb_it):
    C=np.dot(S_cov,randn(n))
    Y_realisation=Y_p+C
    Y_derive=diff_fini(Y_realisation)
    a=np.min(Y_derive)
    k=list(Y_derive).index(a)
    liste.append([X[k],a])

liste=np.array(liste)

plt.figure(6)
plt.clf()
plt.subplot(2,1,1)
plt.hist(liste[:,0],bins=1000,range=[-1,2.5],normed=1)
plt.title('Position du minimum pour %s itérations' %(nb_it))
plt.subplot(2,1,2)
plt.hist(liste[:,1],bins=1000,range=[-1-np.pi/2,1-np.pi/2],normed=1)
plt.title('Valeur de ce minimum')

##Q6_Bis plusieurs jeux d'observations
M=500
nb_it=0
liste_finale=[]
for k in range (nb_it):
    espsilon=sigma_mes*randn(n)
    Y_obs=Y_reel(X)+espsilon # nouvelles observations
    def Fonction(sigma_mod):
        R=C_mod([xx,yy],sigma_mod)+sigma_mes*np.eye((n))
        R_inv=np.linalg.inv(R)
        matrice1=np.dot(H.T,R_inv)
        Q_post=np.linalg.inv(np.dot(matrice1,H))
        Beta_post=np.dot(np.dot(Q_post,matrice1),Y_obs)
        Y_bis=Y_obs-np.dot(H,Beta_post)
        return np.log(np.linalg.det(R))+np.dot(Y_bis.T,np.dot(R_inv,Y_bis))+1000*(sigma_mod<0)
    # sigma_mod=sc.optimize.fmin(Fonction,0.2)[0]
    sigma_mod=sc.optimize.fmin(Fonction,0.23)[0]
    R=C_mod([xx,yy],sigma_mod)+sigma_mes*np.eye((n))
    R_inv=np.linalg.inv(R)
    matrice1=np.dot(H.T,R_inv)
    Q_post=np.linalg.inv(np.dot(matrice1,H))
    Beta_post=np.dot(np.dot(Q_post,matrice1),Y_obs)
    Y_bis=Y_obs-np.dot(H,Beta_post)
    Y_post=lambda x: np.dot((h(x)).T,Beta_post)+np.dot(r(x).T,np.dot(R_inv,Y_bis))
    Yder_mc=diff_fini(Y_obs)
    Var_p=[Var_post(X[j],sigma_mod) for j in range(n)]
    Y_p=[Y_post(X[j]) for j in range(n)]
    for i in range(M):
        C=np.dot(S_cov,randn(n))
        Y_realisation=Y_p+C
        Y_deri=diff_fini(Y_realisation)
        Yder_mc=Yder_mc+Y_deri
    Yder_mc=Yder_mc/M
    a=np.min(Y_deri)
    k=list(Y_deri).index(a)
    liste_finale.append([X[k],a])
liste_finale=np.array(liste_finale)

# plt.figure(61)
# plt.clf()
# plt.subplot(2,1,1)
# plt.hist(lis[:,0],bins=100,range=[-1,2.5],normed=1)
# plt.title('Position du minimum pour %s itérations' %(nb_it))
# plt.subplot(2,1,2)
# plt.hist(lis[:,1],bins=100,range=[-1-np.pi/2,1-np.pi/2],normed=1)
# plt.title('Valeur de ce minimum')