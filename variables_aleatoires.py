#exercice 1.1 

import matplotlib.pyplot as plt #pour les graphes 
import numpy as np #bibliothèque pour le calcul scientifique
from scipy.stats import binom 

p=0.5 #probabilité d'obtenir pile 
n=20 #nombre de lancer 

esperance_X=p
variance_X=p*(1-p)

print("loi de X_k : Bernoulli(",p,")")
print("E(X_k=)",esperance_X )
print("V(X_k)=",variance_X)

esperance_S_n=n*p
variance_S_n=n*p*(1-p)

print("loi de S_n : Binomiale(",n,",",p,")")
print("E(S_n =)",esperance_S_n )
print("V(S_n)=",variance_S_n)


#calcul de P(S_n=k) 

for k in range(n+1):
    probabilite = binom.pmf(k,n,p)
    print(f"P(Sn={k}) = {probabilite:6f}")

#.pmf signifie Probability Mass Function (fonction de masse de probabilité), ce qui correspond à la loi de probabilité d'une variable discrète 
#le premier f signifie formatted (chaîne formatée). Placé tout au début, il active la fonctionnalité d'évaluation des variables entre {}
#le deuxième f signifie float (nombre à virgule). C'est un code de formatage mathématique pour préciser qu'on applique la règle des 6 décimales à un nombre décimal.


#calcule de F_S_n(k) fonction de répartition

for k in range(n+1):
    F_k= binom.cdf(k,n,p)
    print(f"F({k})=P(Sn<={k})={F_k:.6f}")

#.cdf cumulative distribution function pour calculer la fonction de répartition 


#représentation graphique

x=np.arange(0,n+1) #axe des abscisse
F=binom.cdf(x,n,p)


plt.figure(figsize=(9,5))
plt.stem(x, F)
plt.scatter(x,F)

plt.xlabel("k")
plt.ylabel("F(k)=P(Sn<= k)")
plt.title("fonction de répartition de la loi binomiale")
plt.grid(True)

plt.ylim(0, 1.05)

plt.show()


#pour le calcule de médiane soit .median ou .ppf signifie Percent Point Function, c'est le nom donné en statistiques à la fonction quantile, c'est-à-dire l'inverse de la fonction de répartition
#on verifie que graphiquement on tombe sur 10

mediane= binom.median(n,p)
medianebis=binom.ppf(0.5,n,p) #quantile pour le premier quartile c'est 0,25 par ex

print("mediane de Sn=", mediane)




#exercice 1.2

#nb de bonne reponse loi bin(12,1/5) binom.pmf pour calculer la réponse

n=12
p=1/5

probabilite=binom.cdf(4,n,p) #cdf pour calculer inégalité on trouve 0.92 donc a 92%de chances d'avoir 
print(probabilite)



#exercice 1.3

#moralité la loi poisson = loi limite de la binomiale que si np<=10
 
#axe des abscisse de 0 à 50 et ordonnée on fais binom.pmf pour chaque points des abscisse
#graphe steps pour les barres

from scipy.stats import poisson

n=50
p=0.08
lambda_poisson=n*p

#valeur en abscisse 
x=np.arange(0,51)

#np.arange(début, fin) : C'est une fonction de NumPy qui génère une liste ordonnée de nombres (un tableau).


#calcule de P(X=x) valeurs ordonnée 
proba_binomiale=binom.pmf(x,n,p) #ici x varie 

plt.figure(figsize=(10,5))
plt.stem(x,proba_binomiale)


#plt.figure(figsize=(10, 5))
#Rôle : Crée une nouvelle figure graphique (une nouvelle fenêtre de dessin) et définit ses dimensions.
#figsize=(10, 5) : Spécifie la largeur et la hauteur de l'image en pouces (ici 10 de large sur 5 de haut). 
# Cela permet d'avoir un graphique bien lisible et étiré en largeur.

#plt.stem(x, proba_binomiale)Rôle : trace un graphe en bâtons (ou diagramme en tiges/marqueurs)
# .x (axe horizontal) : les valeurs possibles de la variable aléatoire (de 0 à 50 dans ton code)
# .proba_binomiale (axe vertical) : la probabilité associée à chaque valeur $P(X = x)$.



#on calcule proba de poisson P(X=x)
proba_poisson=poisson.pmf(x,lambda_poisson)

plt.figure(figsize=(10,5))
plt.stem(x,proba_poisson)


#exercice 1.4
n=50
p=0.4
from scipy.stats import norm

#objectif montrer l'approximation de la binomiale par la loi normale N(np,np(1-p))=N(esperance,variance) 
#quand n >= 30 et np>= 5 et n(1-p)>=5

#valeur abscisse 
x=np.arange(0,51)

#calcule de P(X=x) valeurs ordonnée 
proba_binomiale=binom.pmf(x,n,p) #ici x varie 

proba_normal=norm.pdf(x,n*p,np.sqrt(n * p * (1 - p))) #pour calculer loi pour va continue .pdf(x,mu,ecart type)

plt.figure(figsize=(10,5))
plt.stem(x, proba_binomiale, linefmt="blue", markerfmt="bo", label="Binomiale")
plt.stem(x, proba_normal, linefmt="red", markerfmt="ro", label="Normale")

#linefmt="blue" (Line Format) : Définit le style et la couleur de la tige verticale. Ici, on lui demande de tracer des lignes en bleu.
#markerfmt="bo" (Marker Format) : Définit l'apparence du point au sommet de chaque tige.

#label="Binomiale" : Attribue un nom à ce tracé. C'est ce texte qui apparaîtra dans la légende quand tu ajoutes plt.legend().


#exercice 1.5

from scipy.stats import randint

#X suit une loi uniforme sur [1,6] P(X=k)=1/6
#fonction de répartition

for k in range(7):
    F_k= randint.cdf(k, 1, 6) 
    print(f"F({k})=P(X<={k})={F_k:.6f}")

#La fonction randint du module scipy.stats représente la loi uniforme discrète (discrete uniform distribution).
#numpy.random.randint : Génère juste des nombres entiers aléatoires.

#calcul de l'espérance 

#toujours initialiser la somme à 0 avant la boucle


esperance_uniforme=0
for k in range(7):
    probabilite_uniforme=randint.pmf(k,1,7)
    produit=k*probabilite_uniforme
    esperance_uniforme+=produit
print(esperance_uniforme)




esperance_carree=0
for k in range(7):
    probabilite_uniforme=randint.pmf(k,1,7)
    produit=k**2*probabilite_uniforme
    esperance_carree+=produit
print(esperance_carree)

    
variance_uniforme= esperance_carree - esperance_uniforme**2
print(variance_uniforme)


#exercice 6

#borne sup exclue
lancers = np.random.randint(1, 4, size=10)
print(lancers)


#exercice 7
#on va étudier loi uniforme 
#la densité de la va X qui suit une loi uniforme sur [0,25] est f(x)=1/25 pour x entre [0,25]

#unif.pdf pour la densité 
#unif.cdf pour la fonction de répartition


#on veux calculer la fonction de répartition



#esperance a+b/2
#variance (a-b)**2/12



#exercice 8

#on cherche L tq calcul P(T>=L)=0.6 L 
#la fonction de rép en L P(T<L)=1-0.6=0.4
#L quantile de la loi exp d'ordre 0.4
#d'ou F(L)=0.4 donc L=F^-1(0.4)
#.ppf(0.4,1)

#0.4=F(L) = 1-e^-L => L=-ln(0.6)

#on veux dessiner la fonction de répartition F de T
#Ft(x)=1-e^-x pour x>=0

from scipy.stats import expon

#calcul de L de deux manière 
#1 calcul numérique
L=-np.log(0.6)
#2 avec ppf
L=expon.ppf(0.4)
#L quantile d'ordre 0.4

#fonction de rép
#plt.plot pour une courbe continue

#approche 1 on va définir la fonction f(x)

x=np.linspace(-1,5,1000) axe abscisse pas de 1000 
#plus on est fin et mieux c'est 

F= np.where(t<0,0,1-np.exp(-t))


plt.figure(figsize=(8,5))
plt.plot(t,F)

plt.xlabel("t(années)")
plt.ylabel("F(t)")
plt.title("Fonction de repartitionde T")

#approche 2 on va faire comme précedemment
lambda=1
F_scipy=expon.cdf(x, scale=1/lambda)

plt.figure(figsize=(8,5))
plt.plot(t,F)

plt.xlabel("t(années)")
plt.ylabel("F(t)")
plt.title("Fonction de repartitionde T")


#calcul esperance expo.mean
#calcul variance expo.var



#on modélise la durée de vie du premier composant T1,..,T5
#on veut qu'au moins l'un deux ait une durée de vie inf à L
#càd il existe 1<i<5 tq Ti<L càd min(Ti)<L
#P(min Ti < L)=1-P(min Ti>L)=1-P(T1,..,T5>L)=1-P(T1>L)^5= 1-0.6**5

n=1-0.6**5
print(n)

#exercice 9


#par calule
#avec python P(T>1000)=1-expon.cdf(1000,scale=1/lambda)



#exercice 10 on le fais pas en tp 
#on retient le résultat X une va continue F fonction de répartition  alors F(X) suit une loi uniforme sur 0,1
#moralité à partir de n'importe quelle va on peux se ramener à une va qui suit une loi uniforme

#la réciproque F^-1(U) suit la loi de X U uniforme 
#moralité on simule des lois a partir de la loi uniforme 



#exercice 11
#il y a des lois construites à partir de la loi normale

from scipy.stats import norm
from scipy.integrate import quad


#densité c'est .pdf 
def dnorm(x):
    return(1/np.sqrt(2**np.pi))*np.exp(-x**2/2)

print("dnorm(0)=",dnorm(0))


integrale, erreur= quad(dnorm,-np.inf, np.inf)

print("integral", integrale)
print(erreur, erreur)

#on va représenter la densité 
x=np.linspace(-4,4,1000) 
plt.figure(figsize=(8,5))
plt.plot(x,dnorm(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("densité de la loi normale")
plt.show()

#on va représenter la fonction de rép

x=np.linspace(-4,4,1000) 
rep=norm.cdf(x,0,1) #=norm.cdf(x)

plt.figure(figsize=(8,5))
plt.plot(x,rep)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("fonction de repartition de la loi normale")
plt.show()



#calculons des probabilité

proba1= norm.cdf(2.2)
print(proba1)

proba2=1-norm.cdf(1.7)



#exercice 12 

#plus sigma grand plus cloche étalée
#centrée la variable on retranche l'esperance 
#reduire la va càd divisé par l'écrat type 


def densite(x,mu,sigma):
    return(1/(sigma * np.sqrt(2*pi)*exp()))
