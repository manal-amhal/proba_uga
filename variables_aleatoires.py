"""
==============================================================================
UE : Probabilités Appliquées — M1 SSD (UGA)
Script : variables_aleatoires.py
Description : Implémentation, simulation et illustration graphique des lois 
              de probabilités usuelles (discrètes et continues) et théorèmes 
              limites selon le polycopié de cours.
==============================================================================
"""
import matplotlib.pyplot as plt #pour les graphes 
import numpy as np #bibliothèque pour le calcul scientifique
from scipy.integrate import quad
from scipy.stats import binom, expon, norm, poisson, randint


# ==============================================================================
# EXERCICE 1.1 : Loi de Bernoulli et Loi Binomiale
# ------------------------------------------------------------------------------
# Théorie :
# - Variable de Bernoulli $X_k \sim \mathcal{B}(p)$ sur $\{0, 1\}$ :
#     $P(X_k = 1) = p$,  $P(X_k = 0) = 1 - p$
#     $\mathbb{E}[X_k] = p$,  $\text{Var}(X_k) = p(1 - p)$
# - $S_n$ est définie comme la somme de $n$ variables de Bernoulli i.i.d. :
#     $S_n = \sum_{k=1}^n X_k \sim \mathcal{B}(n, p)$ sur $\{0, \dots, n\}$
#     Par linéarité : $\mathbb{E}[S_n] = n \cdot p$
#     Par indépendance : $\text{Var}(S_n) = n \cdot p(1 - p)$
#     Formule générale de la masse de probabilité : $P(S_n = k) = \binom{n}{k} p^k (1 - p)^{n - k}$
# ==============================================================================


# Paramètres du problème
p = 0.5  # Probabilité de succès (lancer de pièce)
n = 20   # Nombre de lancers indépendants

# 1. Moments théoriques de la loi de Bernoulli
esperance_X_k = p
variance_X_k = p * (1 - p)

print("--- Exercice 1.1 ---")
print("Loi de X_k : Bernoulli(", p, ")")
print("E(X_k) =", esperance_X_k)
print("V(X_k) =", variance_X_k)
print()

# 2. Moments théoriques de la loi Binomiale

esperance_S_n = n * p
variance_S_n = n * p * (1 - p)

print("Loi de S_n : Binomiale(n,p)")
print("E(S_n) = esperance_S_n")
print("V(S_n) = variance_S_n")
print()

# 3. Calcul des probabilités ponctuelles P(S_n = k) via .pmf

for k in range(n+1):
    probabilite = binom.pmf(k,n,p)
    print(f"P(Sn={k}) = {probabilite:6f}")

# .pmf signifie probability mass function (fonction de masse de probabilité), ce qui correspond à la loi de probabilité d'une variable discrète 
# le premier f signifie formatted (chaîne formatée); placé tout au début, il active la fonctionnalité d'évaluation des variables entre {}
# le deuxième f signifie float (nombre à virgule); c'est un code de formatage mathématique pour préciser qu'on applique la règle des 6 décimales à un nombre décimal


# 4. Fonction de répartition $F_S_n(k) = P(S_n <= k)$ via .cdf

for k in range(n + 1):
    F_k = binom.cdf(k, n, p)
    print(f"F({k}) = P(S_n <= {k}) = {F_k:.6f}")

# .cdf signifie cumulative distribution function (fonction de répartition)


# 5. Représentation graphique de la fonction de répartition (étape discrète)

x = np.arange(0, n + 1) # axe des abscisses 
F = binom.cdf(x, n, p) # axe des ordonnées

plt.figure(figsize=(9, 5))
plt.stem(x, F)
plt.scatter(x, F)
plt.xlabel("k")
plt.ylabel("F(k) = P(S_n <= k)")
plt.title("Fonction de répartition de la loi Binomiale B(20, 0.5)")
plt.grid(True)
plt.ylim(0, 1.05)
plt.show()

# 6. Médiane de S_n 

# $on cherche m tel que P(S_n <= m) >= 0.5 et P(S_n >= m) >= 0.5$
# pour le calcule de médiane soit on utilise .median ou .ppf qui signifie percent point function (fonction quantile)

mediane = binom.median(n, p)
medianebis = binom.ppf(0.5, n, p)

print(f"\nMédiane de S_n (méthode direct) : {mediane}")
print(f"Médiane de S_n (via fonction quantile .ppf) : {medianebis}\n")


# ==============================================================================
# EXERCICE 1.2 : Modélisation d'un QCM
# ------------------------------------------------------------------------------
# Théorie :
# Soit $X$ le nombre de réponses correctes parmi $n = 12$ questions.
# Chaque question comporte 5 choix ($p = 1/5 = 0.2$ de succès au hasard).
# $X \sim \mathcal{B}(12, 0.2)$. On cherche $P(X \le 4) = F_X(4)$.
# ==============================================================================

# Paramètres du problème
n = 12
p = 1 / 5

probabilite = binom.cdf(4, n, p)

print("--- Exercice 1.2 ---")
print("P(X <= 4) =", probabilite)
print()


# ==============================================================================
# EXERCICE 1.3 : Approximation de la Binomiale par la Loi de Poisson
# ------------------------------------------------------------------------------
# Théorie (Loi des événements rares) :
# Si $X \sim \mathcal{B}(n, p)$ avec $n \ge 30$ et $n \cdot p \le 10$,
# alors la loi de $X$ est approchée par une loi de Poisson $\mathcal{P}(\lambda)$
# avec $\lambda = n \cdot p$. Ici : $n = 50$, $p = 0.08 \implies \lambda = 4$.
# ==============================================================================

# Paramètres du problème
n = 50
p = 0.08
lambda_poisson = n * p  

x = np.arange(0, 51) # valeur en abscisse 
proba_binomiale = binom.pmf(x, n, p) # valeurs ordonnée P(X=x) (ici x varie)

# np.arange(début, fin) : c'est une fonction de NumPy qui génère une liste ordonnée de nombres (un tableau)

plt.figure(figsize=(10, 5))
plt.stem(x, proba_binomiale)
plt.xlabel("x")
plt.ylabel("P(X = x)")
plt.title("Loi Binomiale B(50, 0.08)")
plt.grid(True)
plt.show()


# plt.figure(figsize=(10, 5))
# rôle : crée une nouvelle figure graphique et définit ses dimensions.
# figsize=(10, 5) : Spécifie la largeur et la hauteur de l'image en pouces (ici 10 de large sur 5 de haut). 
# cela permet d'avoir un graphique bien lisible et étiré en largeur

# .x (axe horizontal) : les valeurs possibles de la variable aléatoire (de 0 à 50 dans ton code)
# .proba_binomiale (axe vertical) : la probabilité associée à chaque valeur $P(X = x)$.

proba_poisson = poisson.pmf(x, lambda_poisson)

plt.figure(figsize=(10, 5))
plt.stem(x, proba_poisson)     # diagramme en bâtons pour l'approximation
plt.xlabel("x")
plt.ylabel("P(X = x)")
plt.title("Loi de Poisson P(4) - Approximation")
plt.grid(True)
plt.show()


# ==============================================================================
# EXERCICE 1.4 : Théorème de Moivre-Laplace (Binomiale -> Normale)
# ------------------------------------------------------------------------------
# Théorie :
# Si $X \sim \mathcal{B}(n, p)$ sous les conditions $n \ge 30$, $np \ge 5$ et $n(1-p) \ge 5$,
# $X$ est approchée par $Y \sim \mathcal{N}(\mu = np, \sigma^2 = np(1-p))$
# Ici : $n = 50$, $p = 0.4 \implies \mu = 20, \sigma = \sqrt{12} \approx 3.46$
# ==============================================================================

# Paramètres du problème
n = 50
p = 0.4

x = np.arange(0, 51) # valeur en abscisse 
proba_binomiale = binom.pmf(x, n, p) # calcule de P(X=x) (x varie) - valeurs ordonnée 
proba_normal = norm.pdf(x, n * p, np.sqrt(n * p * (1 - p))) # pour calculer loi pour va continue .pdf(x,mu,ecart type)

plt.figure(figsize=(10,5))
plt.stem(x, proba_binomiale, linefmt="blue", markerfmt="bo", label="Binomiale")
plt.stem(x, proba_normal, linefmt="red", markerfmt="ro", label="Normale")

# linefmt="blue" : définit le style et la couleur de la tige verticale
# markerfmt="bo" : définit l'apparence du point au sommet de chaque tige
# label="Binomiale" : attribue un nom au tracé


# ==============================================================================
# EXERCICE 1.5 : Loi Uniforme Discrète
# ------------------------------------------------------------------------------
# Théorie :
# $X \sim \mathcal{U}(\{1, \dots, N\})$ avec $N = 6$ (Lancer de dé).
# $P(X = k) = \frac{1}{N}$ pour $k \in \{1, \dots, N\}$
# $\mathbb{E}[X] = \frac{N + 1}{2} = 3.5$
# $\text{Var}(X) = \frac{N^2 - 1}{12} = \frac{35}{12} \approx 2.9167$
# ==============================================================================


# Calcul de la fonction de répartition P(X <= k)
for k in range(7):
    F_k = randint.cdf(k, 1, 7)
    print(f"F({k}) = P(X <= {k}) = {F_k:.6f}")
print()

# la fonction randint du module scipy.stats représente la loi uniforme discrète (discrete uniform distribution)
# numpy.random.randint : génère juste des nombres entiers aléatoires


# Calcul de l'espérance : $E[X] = sum(k * P(X = k))$
# toujours initialiser la somme à 0 avant la boucle

esperance_uniforme = 0
for k in range(1, 7):
    probabilite_uniforme = randint.pmf(k, 1, 7)
    produit = k * probabilite_uniforme
    esperance_uniforme += produit

print("Espérance calculée :", esperance_uniforme)

# Calcul du moment d'ordre 2 : $E[X^2] = sum(k^2 * P(X = k))$

esperance_carree = 0
for k in range(1, 7):
    probabilite_uniforme = randint.pmf(k, 1, 7)
    produit = (k**2) * probabilite_uniforme
    esperance_carree += produit

print("E[X^2] calculée :", esperance_carree)

# Variance : $Var(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2$
variance_uniforme = esperance_carree - (esperance_uniforme**2)
print("Variance calculée :", variance_uniforme)
print()
variance_uniforme= esperance_carree - esperance_uniforme**2
print(variance_uniforme)


# ==============================================================================
# EXERCICE 1.6 : Simulation de tirages aléatoires
# ==============================================================================

lancers = np.random.randint(1, 4, size=10) # borne sup exclue
print(lancers)

