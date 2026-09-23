# 📸 Photothèque Ultime

Script Python en ligne de commande qui **scanne une photothèque en vrac, l'analyse (EXIF, doublons, OCR, visages, objets, lieux), la catalogue dans une base SQLite** et la **range chronologiquement** dans une photothèque cible, avec renommage des fichiers et moteur de recherche intégré.

> ✅ **Sécurité par défaut** : rien n'est déplacé ni modifié tant que `--apply` n'est pas passé explicitement.

---

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Modes d'exécution](#modes-dexécution)
- [Options](#options)
- [Structure de rangement](#structure-de-rangement)
- [Nommage des fichiers](#nommage-des-fichiers)
- [Recherche](#recherche)
- [Reconnaissance des visages](#reconnaissance-des-visages)
- [Base de données](#base-de-données)
- [Bon à savoir / limites](#bon-à-savoir--limites)
- [Dépannage](#dépannage)

---

## Fonctionnalités

| Fonction | Détail |
|---|---|
| **Inventaire** | Photos, RAW et vidéos (voir formats ci-dessous) |
| **Métadonnées EXIF** | Date de prise de vue, dimensions, appareil, GPS (via ExifTool) |
| **Géolocalisation inverse** | Ville / pays à partir du GPS, **100 % hors ligne** |
| **Doublons exacts** | Hash MD5 |
| **Doublons visuels** | pHash + BK-Tree (distance de Hamming ≤ 5) |
| **OCR** | Texte des images (Tesseract, français + anglais) |
| **Détection d'objets** | YOLOv8, 80 catégories COCO traduites en français (chien, voiture, vélo…) avec mots-clés de regroupement (`animal`, `véhicule`, `nourriture`) |
| **Détection de visages** | OpenCV (comptage, `portrait`, `groupe`) |
| **Reconnaissance nominative** | Optionnelle, via `face_recognition` |
| **Rangement chronologique** | `Année / Mois`, vidéos et doublons séparés |
| **Renommage** | Lieu, date, appareil, numéro chronologique |
| **Tags dans les photos** | Mots-clés, lieu, texte OCR écrits en IPTC/XMP |
| **Sidecars `.xmp`** | Créés pour les fichiers RAW (le RAW n'est jamais modifié) |
| **Recherche** | Plein texte (SQLite FTS5) + filtres + similarité visuelle + export CSV/JSON |

**Formats reconnus**

- Photos : `jpg jpeg png heic heif webp tiff tif bmp gif`
- RAW : `cr2 cr3 nef arw raf dng orf rw2 pef srw`
- Vidéos : `mov mp4 m4v avi mkv webm mts m2ts 3gp`

---

## Installation

### Prérequis

- Python 3.8+
- macOS (testé) ; Linux devrait fonctionner de la même façon

### Outils système (macOS / Homebrew)

```bash
brew install exiftool tesseract tesseract-lang
```

### Dépendances Python

```bash
pip install pyexiftool pillow imagehash reverse_geocoder pytesseract numpy opencv-python ultralytics
```

### Optionnel : reconnaissance faciale nominative

```bash
brew install cmake
pip install face_recognition
```

> 💡 **Toutes les dépendances sont optionnelles.** Si l'une manque, le script continue en mode dégradé et affiche un `⚠` au démarrage indiquant la fonction désactivée.

> 🌐 **Premier lancement** : le modèle YOLO (`yolov8m.pt`) est téléchargé automatiquement une seule fois. Ensuite, tout fonctionne hors ligne.

---

## Démarrage rapide

```bash
# 1. Simulation complète : analyse + rapport, rien n'est déplacé
python3 phototheque_V5.py /chemin/SOURCE --target /chemin/CIBLE

# 2. Relire le rapport généré (rapport_phototheque.txt / .json)

# 3. Application réelle
python3 phototheque_V5.py /chemin/SOURCE --target /chemin/CIBLE --apply
```

**Conseil** : faites d'abord un essai sur un petit échantillon.

```bash
python3 phototheque_V5.py /chemin/SOURCE --target /chemin/CIBLE --limit 100
```

---

## Modes d'exécution

| Commande | Effet |
|---|---|
| `--scan` | Inventaire seul (comptage par type), aucune analyse |
| `--analyse` | Hash, EXIF, OCR, détection → catalogue SQLite, **sans plan de rangement** |
| `--dry-run` *(défaut)* | Analyse complète + rapport, **aucun fichier déplacé** |
| `--apply` | Déplace et renomme réellement les fichiers |

Le pipeline complet se déroule en 5 étapes : inventaire → EXIF → hash/OCR/détection (multi-thread) → quasi-doublons visuels → plan de rangement.

> ⚠️ Même en `--dry-run`, le **catalogue SQLite est créé/mis à jour**. Seuls vos fichiers photo ne sont pas touchés.

---

## Options

| Option | Description | Défaut |
|---|---|---|
| `source` | Dossier source à analyser | *(requis)* |
| `--target` | Dossier cible (photothèque triée) | requis sauf pour `--scan` |
| `--db` | Chemin de la base SQLite | `<target>/CATALOGUE/catalogue.sqlite` |
| `--workers` | Nombre de threads d'analyse | `8` |
| `--limit N` | Ne traite que les N premiers fichiers | illimité |
| `--report` | Préfixe des fichiers de rapport | `rapport_phototheque` |
| `--name-sep` | Séparateur dans les noms de fichiers | `-` |
| `--no-detect` | Désactive visages / objets / animaux (bien plus rapide) | détection active |
| `--detect-conf` | Seuil de confiance YOLO (0 à 1) | `0.35` |
| `--no-write-tags` | N'écrit **rien** dans les photos (catalogue seul enrichi) | écriture active |
| `--known-faces` | Dossier de visages de référence | — |
| `--face-tolerance` | Tolérance de reconnaissance faciale (0 à 1, plus bas = plus strict) | `0.5` |

### Exemples

```bash
# Rangement rapide par date uniquement (sans IA)
python3 phototheque_V5.py SOURCE --target CIBLE --no-detect --apply

# Sans modifier les fichiers photo (catalogue uniquement)
python3 phototheque_V5.py SOURCE --target CIBLE --no-write-tags --apply

# Détection d'objets plus sélective
python3 phototheque_V5.py SOURCE --target CIBLE --detect-conf 0.5

# Avec reconnaissance de personnes
python3 phototheque_V5.py SOURCE --target CIBLE --known-faces ./visages --apply
```

---

## Structure de rangement

```
CIBLE/
├── 2019/
│   ├── 07 Juillet/
│   └── 08 Aout/
├── 2024/
│   └── 01 Janvier/
├── 0000_SansDate/          ← fichiers sans date exploitable
│   └── 00/
├── 00_VIDEOS/
│   └── 2024_07/
├── 00_DOUBLONS_EXACTS/
├── 00_DOUBLONS_VISUELS/
├── 00_ERREURS/             ← fichiers illisibles ou de type inconnu
└── CATALOGUE/
    ├── catalogue.sqlite
    ├── catalogue_backup/   ← 5 dernières sauvegardes de la base
    └── phototheque.log
```

- La date utilisée est la **date de prise de vue** (EXIF), avec repli sur la **date de modification** du fichier.
- Les fichiers sont **déplacés** (pas copiés) : source et cible doivent être sur le **même disque** pour un rangement instantané.
- Les dossiers de la source commençant par `00_` ou nommés `CATALOGUE` sont ignorés lors du scan.

---

## Nommage des fichiers

Format généré (photos, RAW et vidéos rangés normalement) :

```
LIEU-MM-AAAA-APPAREIL-NNN.ext
```

Exemple : `COLLIOURE-07-2019-SM-G930F-001.jpeg`

- **LIEU** : ville (ou pays à défaut), en majuscules — omis si pas de GPS
- **MM-AAAA** : mois et année de la prise de vue
- **APPAREIL** : marque + modèle (vidéos : omis)
- **NNN** : numéro chronologique **au sein de chaque dossier** (le tri alphabétique redonne l'ordre réel des photos)
- Séparateur modifiable avec `--name-sep`
- Les doublons et fichiers en erreur **gardent leur nom d'origine**

---

## Recherche

Une fois le catalogue construit :

```bash
python3 phototheque_V5.py search "chien + plage" --db CIBLE/CATALOGUE/catalogue.sqlite
```

### Syntaxe

| Requête | Signification |
|---|---|
| `chien plage` ou `chien + plage` | chien **ET** plage |
| `chien OR chat` | chien **OU** chat |
| `chien -plage` | chien **SANS** plage |
| `"Plage de Palavas"` | phrase exacte |
| `voiture` | trouve aussi `voitures` (préfixe automatique) |
| `type:photo` | filtre par type (`photo`, `raw`, `video`) |
| `format:jpg` | filtre par extension |
| `date:2024-01-01..2024-08-31` | plage de dates de prise de vue |
| `similaire:IMG_1234.JPG` | photos visuellement proches (pHash ≤ 10) |

Les filtres se combinent :

```bash
python3 phototheque_V5.py search "animal + date:2024-01-01..2024-08-31 type:photo" --db catalogue.sqlite
```

Champs indexés en plein texte : nom de fichier, texte OCR, personnes détectées, objets détectés, ville, pays.

### Options de recherche

| Option | Description | Défaut |
|---|---|---|
| `--db` | Chemin vers `catalogue.sqlite` | *(requis)* |
| `--limit` | Nombre max de résultats | `50` |
| `--export csv\|json` | Exporte les résultats | — |
| `--output` | Fichier d'export | `resultats.<format>` |

---

## Reconnaissance des visages

Trois niveaux, selon ce qui est installé :

1. **OpenCV seul** : nombre de visages, tags `portrait` / `groupe`.
2. **`face_recognition` sans `--known-faces`** : chaque personne différente reçoit un tag stable `visage_01`, `visage_02`… sur tout le catalogue.
3. **`face_recognition` + `--known-faces`** : les personnes connues sont identifiées par leur nom.

### Organisation du dossier de référence

Un sous-dossier par personne, contenant une ou plusieurs photos (plusieurs angles = meilleure fiabilité) :

```
visages/
├── Alice/
│   ├── alice1.jpg
│   └── alice2.jpg
└── Bob/
    └── bob1.jpg
```

---

## Base de données

Le catalogue `catalogue.sqlite` contient :

| Table | Contenu |
|---|---|
| `media` | Chemin, taille, type, MD5, pHash, dates, dimensions, appareil, statut |
| `metadata` | Ville, pays, GPS, personnes, objets, texte OCR |
| `duplicates` | Lien doublon → original (`exact` / `visual`, distance) |
| `fts_search` | Index plein texte FTS5 |

**Statuts** : `indexed`, `moved`, `duplicate_exact`, `duplicate_visual`, `error`.

Le script est **incrémental** : relancé sur une source déjà cataloguée, il ne réinsère pas les fichiers connus. La base est sauvegardée automatiquement avant chaque `--apply`.

---

## Bon à savoir / limites

- **Tags écrits dans les photos = modification des fichiers.** Par défaut, `--apply` écrit OCR, lieu et mots-clés (IPTC/XMP) dans les photos, avec `-overwrite_original` : **aucune copie `.ext_original` n'est conservée**. Utilisez `--no-write-tags` si vous voulez laisser vos fichiers strictement intacts. Les mots-clés existants sont conservés (ajout uniquement).
- **Vérification d'intégrité** : chaque fichier est vérifié par MD5 avant/après déplacement ; toute anomalie est signalée dans le log.
- **Fichiers RAW** : non analysés visuellement (pas de pHash/OCR/détection) ; un sidecar `.xmp` est créé à côté.
- **HEIC/HEIF** : Pillow ne lit pas ces formats sans plugin supplémentaire (`pillow-heif`, non géré par le script). Sans lui, ces fichiers peuvent être classés dans `00_ERREURS` lors de l'analyse d'image. Vérifiez le rapport en `--dry-run` avant `--apply`.
- **Doublons visuels** : ils sont *proposés* dans `00_DOUBLONS_VISUELS` ; vérifiez-les avant de supprimer quoi que ce soit (le script ne supprime jamais rien).
- **Détection d'objets** : dépend de 80 catégories COCO ; un seuil bas (`--detect-conf`) augmente les faux positifs.
- **Performances** : la détection d'objets et l'OCR ralentissent nettement l'analyse. `--no-detect` accélère fortement un simple tri par date.

---

## Dépannage

| Problème | Solution |
|---|---|
| `⚠ pyexiftool absent` | `brew install exiftool` puis `pip install pyexiftool` |
| Pas de lieu dans les noms | Installer `reverse_geocoder` ; les photos doivent contenir un GPS |
| Pas d'OCR | `brew install tesseract tesseract-lang` + `pip install pytesseract` |
| Détection d'objets désactivée | `pip install ultralytics` ; connexion internet requise au 1er lancement |
| `database is locked` | Fermez tout autre programme utilisant la base ; le script gère déjà WAL + délai d'attente |
| Trop lent | `--no-detect`, ou augmenter `--workers` |
| Beaucoup de fichiers en `00_ERREURS` | Consulter `CIBLE/CATALOGUE/phototheque.log` et le rapport `.txt` |

---

## Licence

À compléter selon votre choix (MIT, GPL, etc.).
