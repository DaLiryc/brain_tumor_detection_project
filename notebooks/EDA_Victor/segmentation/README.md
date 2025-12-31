# Modèle de segmentation en 2 dimensions U-Net

Ce module implémente une architecture **U-Net** pour la segmentation sémantique d'images IRM en 2D. L'objectif est de générer des masques binaires localisant les tumeurs à partir des coupes IRM.

## PATH du code pour le modèle de segmentation

Le code principal et l'entraînement du modèle se trouvent dans le notebook :
`notebooks/EDA_Victor/segmentation/segmentation_u-net.ipynb`

## Données
Pour cette application nous utilisons les données [Brain MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Ce jeu de données est composé de fichier `.tif` avec les images des scanners cérébraux et puis les scanners dit `mask` qui sont les contours de la tumeur.


## Architecture du Modèle
Nous utilisons une architecture **U-Net** standard (Encoder-Decoder) implémentée avec **TensorFlow/Keras**.

* **Entrée (Input) :** Images IRM (taille : 256x256x3).
* **Encoder (Contraction) :** Série de couches de convolution `Conv2D` suivies de `MaxPooling2D` pour extraire les caractéristiques profondes.
* **Bottleneck :** La couche la plus profonde capturant le contexte abstrait.
* **Decoder (Expansion) :** Couches de `Conv2DTranspose` concaténées avec les sorties correspondantes de l'encodeur (Skip Connections) pour récupérer la résolution spatiale.
* **Sortie (Output) :** Une couche `Conv2D` avec activation **Sigmoid** pour générer un masque de probabilité binaire (0 = fond, 1 = tumeur).

## Pré-requis et Installation

Le projet nécessite un environnement Python avec les bibliothèques suivantes (voir `requirements.txt` à la racine) :
* `tensorflow` / `keras`
* `opencv-python` (cv2)
* `numpy`, `pandas`
* `matplotlib` (pour la visualisation)
* `scikit-learn`

Pour installer les dépendances :
```bash
pip install -r requirements.txt
