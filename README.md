# REXIA – Projet 2026

Projet du cours *Responsible and Explainable AI* (CentraleSupélec / Université Paris-Saclay).

Le projet est en trois parties, une par type de données. Chaque partie a son propre notebook dans `projet/`.

## Partie 1 – Données tabulaires

Notebook : `projet/REXIA_Partie1_Donnees_Tabulaires.ipynb`

Jeu de données RH anonymisé (`RH_dataset.csv`, ~24k lignes). On cherche à prédire si un employé va démissionner dans les 6 prochains mois à partir de variables comme l'ancienneté, le salaire, la dernière promotion, etc.

Le notebook contient :

- une exploration du dataset (distributions, valeurs manquantes, corrélations, comparaison d'un profil qui démissionne vs. un qui reste),
- une réflexion sur les variables sensibles et les proxies possibles (âge, statut marital, parent, salaire, famille d'emploi…),
- l'entraînement de trois modèles transparents : un arbre de décision, une régression logistique, et un modèle logit avec splines (type GAM),
- un rééquilibrage par SMOTE pour traiter le fort déséquilibre des classes,
- une évaluation de l'équité par sous-groupes (FPR/FNR, disparate impact), avec des courbes de précision-rappel et une discussion du trade-off performance / équité,
- une explication post-hoc via permutation importance, puis un réentraînement sur un jeu réduit aux variables les plus utiles.

## Partie 2 – Données images

Notebook : `projet/REXIA_Partie2_Donnees_Visuelles.ipynb`

Dataset **CelebA** (visages + 40 attributs booléens). L'analyse porte sur :

- la description du dataset et les corrélations entre attributs (dont quelques corrélations artificielles exhibées),
- l'identification des variables sensibles (`Male`, `Pale_Skin`…) et le calcul de demographic parity et disparate impact pour chaque attribut cible,
- l'entraînement d'un modèle de classification sur l'attribut `Smiling` à partir des images,
- l'évaluation de l'équité par sous-groupes (`Male`, `Pale_Skin`) : accuracy, FPR, FNR, écarts entre groupes,
- des explications post-hoc : saliency maps par gradients et LIME, appliquées sur des cas correctement classifiés, mal classifiés et minoritaires.

## Partie 3 – Données textuelles

Notebook : `projet/REXIA_Partie3_Donnees_textuelles.ipynb`

Dataset **Civil Comments** (30 000 commentaires en ligne annotés pour la toxicité et avec des attributs démographiques, via Hugging Face).

Le notebook couvre :

- une analyse descriptive (longueurs, équilibre du dataset, distribution de `toxicity`),
- le nettoyage du texte et la tokenisation avec spaCy,
- des nuages de mots et une analyse TF-IDF,
- une analyse des groupes démographiques présents dans les commentaires,
- un modèle de classification (régression logistique) avec interprétation via ses coefficients et explications locales LIME,
- une évaluation de la fairness, à la fois du dataset lui-même et du modèle.

## Lancer les notebooks

```bash
jupyter notebook projet/
```

Dépendances principales : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `spacy`, `lime`, `datasets` (Hugging Face), plus les librairies habituelles de deep learning pour la partie images.

`pip install -r requirements.txt`