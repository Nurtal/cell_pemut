# CELLNOTEBOOK

## OVerview
L'idée est de générer un notebook jupyter pour Patrice, un truc qui prends en entrées une série de fichier genéré par OMIC avec des méta données et qui  réalise des analyses de voisinage / "permutation", etc ...

## TODO

- [x] check moyenne des count computation (normalement tout est ok), l'idée c'est qu'on compte pour chaque pop ses voisins et qu'on divise par le nombre d'individu dans la pop
- [x] Générer matrice de proximité en moyenne de count et en pourcentage
- [x] sauvegarde automatique des figures générées (avec et sans chiffres)
- [x] Add column dans les matrice file (pardois l'index est pas là, ie same que nm des cols, ie noms ds pops)
- [ ] Rajouter barre de std aux histogrammes
- [ ] separate generation du multimatrix et filtre des pops
- [ ] Ajouter filtrage des groupes avant histo
- [ ] Check M6 result
