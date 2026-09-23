#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phototheque Ultime - v3.0
==========================
Scanne une photothèque source, hash/analyse chaque fichier, catalogue le
tout dans une base SQLite (catalogue.sqlite) et propose (puis, sur demande,
applique) un rangement chronologique dans une photothèque cible, sur le
même disque.

v1.1 : renommage des fichiers (date, appareil, lieu, numéro).
v2.0 : OCR (Tesseract) sur les photos, sidecars .xmp automatiques pour les
fichiers RAW, et un moteur de recherche en ligne de commande.
v2.1 : correction des plantages sur doublons + verrous SQLite.
v3.0 : - reconnaissance du contenu des images : visages (OpenCV) et
         objets / animaux / véhicules... (YOLOv8, 80 catégories COCO
         traduites en français) ;
       - nouveau format de nom : NNNN_Appareil_Lieu_JJ-MM-AAAA.ext ;
       - écriture des tags dans les photos ACTIVÉE PAR DÉFAUT
         (--no-write-tags pour la désactiver).

Exemple de recherche :

    python3 phototheque_V5.py search "chien + plage" --db CIBLE/CATALOGUE/catalogue.sqlite

Aucune écriture ni suppression n'a lieu tant que --apply n'est pas passé
explicitement. --dry-run (mode par défaut si aucun mode n'est donné) ne
fait QUE produire un rapport.

Voir GUIDE_UTILISATION.md pour l'installation et des exemples complets.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------
# Dépendances optionnelles : le script tourne en mode dégradé si elles
# sont absentes (utile pour un premier test rapide), mais avertit
# clairement l'utilisateur. Sur le Mac cible, installez-les (voir le
# guide) pour bénéficier de l'extraction EXIF et de la détection des
# quasi-doublons visuels.
# ----------------------------------------------------------------------
try:
    import exiftool  # pip install pyexiftool
    HAVE_EXIFTOOL = True
except ImportError:
    HAVE_EXIFTOOL = False

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

try:
    import imagehash  # pip install imagehash
    HAVE_IMAGEHASH = True
except ImportError:
    HAVE_IMAGEHASH = False

try:
    import reverse_geocoder as rgeo  # pip install reverse_geocoder
    HAVE_GEOCODER = True
except ImportError:
    HAVE_GEOCODER = False

try:
    import pytesseract  # pip install pytesseract (+ brew install tesseract tesseract-lang)
    HAVE_OCR = True
except ImportError:
    HAVE_OCR = False

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    import cv2  # pip install opencv-python
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from ultralytics import YOLO  # pip install ultralytics
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False

try:
    import face_recognition  # pip install face_recognition (nécessite cmake + dlib)
    HAVE_FACE_RECOGNITION = True
except ImportError:
    HAVE_FACE_RECOGNITION = False

# Détection de visages : possible dès qu'OpenCV est là (cascade fournie
# avec le paquet, aucun téléchargement).
HAVE_FACES = HAVE_CV2 and HAVE_NUMPY
# Détection d'objets/animaux : nécessite ultralytics + numpy. Le modèle
# yolov8m.pt (~6 Mo) est téléchargé UNE seule fois au premier lancement,
# puis tout fonctionne hors ligne.
HAVE_OBJECTS = HAVE_YOLO and HAVE_NUMPY

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp", ".gif"}
RAW_EXTS   = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng", ".orf", ".rw2", ".pef", ".srw"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".3gp"}

PHASH_EXACT_DUP = 0       # distance de Hamming pour un doublon exact de pHash
PHASH_VISUAL_DUP = 5      # distance <= 5 -> proposé comme doublon visuel
PHASH_SIMILARITY_SEARCH = 10  # distance <= 10 -> "similaire:" en recherche (moins strict)

OCR_TEXT_MAX_LEN = 4000   # tronque le texte OCR stocké (les images très riches en texte n'ont pas besoin d'être stockées intégralement pour rester utiles à la recherche)

CHUNK_SIZE_EXIFTOOL = 150  # nb de fichiers par appel batch à exiftool
DB_WRITE_BATCH = 50        # nb de résultats accumulés avant un COMMIT

# ---- Reconnaissance du contenu (v3.0) --------------------------------
YOLO_MODEL_NAME = "yolov8m.pt"   # le plus petit/rapide ; "yolov8s.pt" = plus précis, ~3x plus lent
DETECT_CONF = 0.35               # seuil de confiance par défaut (0.0 - 1.0)
DETECT_MAX_SIDE = 640            # les images sont réduites à cette taille avant détection (vitesse)
DETECT_MAX_LABELS = 12           # nb max d'étiquettes différentes conservées par image
FACE_MIN_SIZE = 40               # un visage plus petit que 40x40 px est ignoré (bruit)
FACE_TOLERANCE = 0.5             # seuil de reconnaissance faciale (plus bas = plus strict)

# Noms de mois pour les dossiers de rangement ("08" -> "08 Aout")
MONTHS_FR = {
    "01": "Janvier", "02": "Fevrier", "03": "Mars", "04": "Avril",
    "05": "Mai", "06": "Juin", "07": "Juillet", "08": "Aout",
    "09": "Septembre", "10": "Octobre", "11": "Novembre", "12": "Decembre",
}

# Séparateur entre les blocs du nom de fichier (numéro / appareil / lieu
# / date). Modifiable en ligne de commande avec --name-sep "-" si vous
# préférez des tirets partout.
NAME_SEPARATOR = "_"

# Les 80 catégories COCO reconnues par YOLO, traduites pour que la
# recherche fonctionne en français ("chien", "voiture", "vélo"...).
COCO_FR = {
    "person": "personne", "bicycle": "vélo", "car": "voiture", "motorcycle": "moto",
    "airplane": "avion", "bus": "bus", "train": "train", "truck": "camion",
    "boat": "bateau", "traffic light": "feu de circulation", "fire hydrant": "bouche d'incendie",
    "stop sign": "panneau stop", "parking meter": "parcmètre", "bench": "banc",
    "bird": "oiseau", "cat": "chat", "dog": "chien", "horse": "cheval",
    "sheep": "mouton", "cow": "vache", "elephant": "éléphant", "bear": "ours",
    "zebra": "zèbre", "giraffe": "girafe", "backpack": "sac à dos", "umbrella": "parapluie",
    "handbag": "sac à main", "tie": "cravate", "suitcase": "valise", "frisbee": "frisbee",
    "skis": "skis", "snowboard": "snowboard", "sports ball": "ballon", "kite": "cerf-volant",
    "baseball bat": "batte de baseball", "baseball glove": "gant de baseball",
    "skateboard": "skateboard", "surfboard": "planche de surf", "tennis racket": "raquette de tennis",
    "bottle": "bouteille", "wine glass": "verre à vin", "cup": "tasse", "fork": "fourchette",
    "knife": "couteau", "spoon": "cuillère", "bowl": "bol", "banana": "banane",
    "apple": "pomme", "sandwich": "sandwich", "orange": "orange", "broccoli": "brocoli",
    "carrot": "carotte", "hot dog": "hot-dog", "pizza": "pizza", "donut": "donut",
    "cake": "gâteau", "chair": "chaise", "couch": "canapé", "potted plant": "plante",
    "bed": "lit", "dining table": "table", "toilet": "toilettes", "tv": "télévision",
    "laptop": "ordinateur portable", "mouse": "souris", "remote": "télécommande",
    "keyboard": "clavier", "cell phone": "téléphone", "microwave": "micro-ondes",
    "oven": "four", "toaster": "grille-pain", "sink": "évier", "refrigerator": "réfrigérateur",
    "book": "livre", "clock": "horloge", "vase": "vase", "scissors": "ciseaux",
    "teddy bear": "ours en peluche", "hair drier": "sèche-cheveux", "toothbrush": "brosse à dents",
}

# Sous-ensembles utiles pour les mots-clés de regroupement ajoutés
# automatiquement (permettent de chercher "animal" ou "véhicule" sans
# connaître l'espèce exacte).
COCO_ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "teddy bear"}
COCO_VEHICLES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train",
                 "truck", "boat"}
COCO_FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "bottle", "wine glass", "cup"}

# Le modèle YOLO est chargé une seule fois, paresseusement, et partagé
# par tous les threads d'analyse. L'inférence est sérialisée par un
# verrou : le modèle n'est pas garanti thread-safe, et le parallélisme
# utile est de toute façon déjà interne à PyTorch.
_yolo_model = None
_yolo_lock = threading.Lock()
_yolo_failed = False

# Reconnaissance faciale nominative (v3.1) : `_known_faces` est chargée
# UNE seule fois au démarrage depuis --known-faces (un sous-dossier par
# personne). `_unknown_faces` se remplit au fil de l'analyse : chaque
# visage qui ne correspond à personne de connu reçoit un identifiant
# "visage_NN" stable, réutilisé pour cette même personne sur toutes ses
# autres photos du catalogue (le rangement par lots multi-thread rend ce
# verrou nécessaire, comme pour le modèle YOLO ci-dessus).
_known_faces = []              # liste de (nom, encodage_128d)
_unknown_faces = []            # liste de (label "visage_NN", encodage_128d)
_unknown_faces_lock = threading.Lock()
_unknown_face_counter = 0

# La cascade de visages, elle, n'est pas thread-safe non plus mais est
# légère : on en crée une par thread.
_thread_local = threading.local()
DETECT_ENABLED = True            # basculé à False par --no-detect

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_extension TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    media_type TEXT CHECK(media_type IN ('photo','raw','video','unknown')),
    hash_md5 TEXT UNIQUE,
    hash_phash TEXT,
    date_shot DATETIME,
    date_modified DATETIME,
    width INTEGER,
    height INTEGER,
    camera_make TEXT,
    camera_model TEXT,
    status TEXT DEFAULT 'indexed',
    dest_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    location_city TEXT,
    location_country TEXT,
    gps_lat REAL,
    gps_lon REAL,
    persons_detected TEXT,
    objects_detected TEXT,
    ocr_text TEXT,
    rating INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS duplicates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    duplicate_of_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    kind TEXT CHECK(kind IN ('exact','visual')),
    distance INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_search USING fts5(
    file_name,
    ocr_text,
    persons_detected,
    objects_detected,
    location_city,
    location_country,
    media_id UNINDEXED
);

CREATE TRIGGER IF NOT EXISTS trg_media_ai AFTER INSERT ON media BEGIN
    INSERT INTO fts_search(rowid, file_name, media_id) VALUES (new.id, new.file_name, new.id);
END;

CREATE TRIGGER IF NOT EXISTS trg_metadata_ai AFTER INSERT ON metadata BEGIN
    UPDATE fts_search SET ocr_text=new.ocr_text, persons_detected=new.persons_detected,
        objects_detected=new.objects_detected, location_city=new.location_city,
        location_country=new.location_country WHERE media_id = new.media_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_media_ad AFTER DELETE ON media BEGIN
    DELETE FROM fts_search WHERE media_id = old.id;
END;
"""

log = logging.getLogger("phototheque")


def open_db(db_path):
    """Ouvre une connexion SQLite robuste : mode WAL (meilleure tolérance
    aux accès concurrents) + busy_timeout (attend/retente au lieu de lever
    immédiatement 'database is locked')."""
    conn = sqlite3.connect(db_path, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=8000")
    return conn


def close_db(conn):
    """Force la fusion du journal WAL dans le fichier principal avant de
    fermer, pour que la prochaine connexion (étape suivante du pipeline,
    process séparé, etc.) ne trouve jamais un journal WAL en attente qui
    pourrait déclencher un 'database is locked' évitable."""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    conn.close()


# ----------------------------------------------------------------------
# BK-Tree pour la recherche de similarité visuelle (pHash), avec
# insertion incrémentale (voir correction du cahier des charges §2.4).
# ----------------------------------------------------------------------
class BKTree:
    def __init__(self):
        self.tree = None  # (item, children_dict{distance: node})

    @staticmethod
    def _distance(a, b):
        return bin(a ^ b).count("1")

    def add(self, item, value):
        node = (item, value, {})
        if self.tree is None:
            self.tree = node
            return
        cur = self.tree
        while True:
            cur_item, cur_value, children = cur
            d = self._distance(value, cur_value)
            if d == 0:
                d = 1  # évite d'écraser un noeud pour deux hash identiques
            if d in children:
                cur = children[d]
            else:
                children[d] = (item, value, {})
                return

    def find_within(self, value, radius):
        if self.tree is None:
            return []
        results = []
        candidates = [self.tree]
        while candidates:
            node = candidates.pop()
            item, node_value, children = node
            d = self._distance(value, node_value)
            if d <= radius:
                results.append((item, d))
            for dist, child in children.items():
                if d - radius <= dist <= d + radius:
                    candidates.append(child)
        return results


# ----------------------------------------------------------------------
# Étape 1 : Inventaire
# ----------------------------------------------------------------------
def classify_extension(ext):
    ext = ext.lower()
    if ext in PHOTO_EXTS:
        return "photo"
    if ext in RAW_EXTS:
        return "raw"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def inventory(source_dir, limit=None):
    records = []
    source_dir = Path(source_dir)
    for root, dirs, files in os.walk(source_dir):
        # on ignore nos propres dossiers de sortie s'ils sont sous la source
        dirs[:] = [d for d in dirs if not d.startswith("00_") and d != "CATALOGUE"]
        for fname in files:
            if fname.startswith("."):
                continue  # fichiers cachés macOS (.DS_Store, etc.)
            fpath = Path(root) / fname
            try:
                stat = fpath.stat()
            except OSError as e:
                log.warning("Impossible de lire %s (%s)", fpath, e)
                continue
            ext = fpath.suffix.lower()
            records.append({
                "path": str(fpath),
                "name": fname,
                "ext": ext,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime),
                "media_type": classify_extension(ext),
            })
            if limit and len(records) >= limit:
                return records
    return records


# ----------------------------------------------------------------------
# Étape 2 : Métadonnées EXIF (batch, process persistant exiftool)
# ----------------------------------------------------------------------
def extract_metadata_batch(records):
    """Remplit date_shot / width / height / camera_make / camera_model /
    gps_lat / gps_lon / location_city / location_country directement dans
    chaque dict de `records` (mutation en place)."""
    if not HAVE_EXIFTOOL:
        log.warning("pyexiftool non installé : les métadonnées EXIF seront absentes "
                     "(dates de prise de vue -> repli sur la date de modification du fichier).")
        return
    targets = [r for r in records if r["media_type"] in ("photo", "raw", "video")]
    if not targets:
        return
    path_to_record = {r["path"]: r for r in targets}
    with exiftool.ExifToolHelper() as et:
        for i in range(0, len(targets), CHUNK_SIZE_EXIFTOOL):
            chunk = targets[i:i + CHUNK_SIZE_EXIFTOOL]
            paths = [r["path"] for r in chunk]
            try:
                # -n : sort les valeurs numériques (GPS en degrés décimaux
                # signés) plutôt que des chaînes formatées type "48 deg 51'...".
                metas = et.get_metadata(paths, params=["-n"])
            except Exception as e:
                # Un seul fichier à problème (corrompu, verrouillé...) peut
                # faire échouer l'appel groupé. On isole le lot en repassant
                # fichier par fichier plutôt que de perdre les métadonnées
                # des 149 autres photos du lot.
                log.warning("Erreur ExifTool sur un lot de %d fichiers (%s) -> nouvel essai fichier par fichier.",
                            len(paths), e)
                metas = []
                for p in paths:
                    try:
                        metas.extend(et.get_metadata([p], params=["-n"]))
                    except Exception as e2:
                        log.error("Métadonnées illisibles, fichier ignoré (le reste du traitement continue) : %s (%s)", p, e2)
            for meta in metas:
                src = meta.get("SourceFile")
                rec = path_to_record.get(src)
                if not rec:
                    continue
                date_str = (meta.get("EXIF:DateTimeOriginal") or meta.get("QuickTime:CreateDate")
                            or meta.get("EXIF:CreateDate") or meta.get("XMP:DateCreated"))
                rec["date_shot"] = parse_exif_date(date_str)
                rec["width"] = meta.get("EXIF:ImageWidth") or meta.get("File:ImageWidth") or meta.get("QuickTime:ImageWidth")
                rec["height"] = meta.get("EXIF:ImageHeight") or meta.get("File:ImageHeight") or meta.get("QuickTime:ImageHeight")
                rec["camera_make"] = meta.get("EXIF:Make")
                rec["camera_model"] = meta.get("EXIF:Model")
                rec["gps_lat"] = meta.get("EXIF:GPSLatitude") or meta.get("Composite:GPSLatitude")
                rec["gps_lon"] = meta.get("EXIF:GPSLongitude") or meta.get("Composite:GPSLongitude")

    reverse_geocode_batch(targets)


def reverse_geocode_batch(records):
    """Résout la ville/pays le plus proche pour chaque enregistrement
    disposant de coordonnées GPS valides, en une seule passe, entièrement
    en local (base embarquée de ~33 000 villes, aucun accès réseau).
    Une coordonnée manquante ou invalide (chaîne vide, etc.) sur UNE
    photo ne doit jamais empêcher la géolocalisation des autres."""
    if not HAVE_GEOCODER:
        return
    geo_records = []
    coords = []
    for r in records:
        lat, lon = r.get("gps_lat"), r.get("gps_lon")
        if lat in (None, "", 0) or lon in (None, "", 0):
            continue
        try:
            coords.append((float(lat), float(lon)))
        except (TypeError, ValueError):
            continue  # coordonnée illisible sur cette photo -> on l'ignore, sans bloquer les autres
        geo_records.append(r)
    if not geo_records:
        return
    try:
        results = rgeo.search(coords, mode=1)
    except Exception as e:
        log.error("Géolocalisation inverse impossible : %s", e)
        return
    for rec, res in zip(geo_records, results):
        rec["location_city"] = res.get("name")
        rec["location_country"] = res.get("cc")


def parse_exif_date(date_str):
    if not date_str:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str.split("+")[0].split(".")[0], fmt.split("%z")[0].strip())
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------------
# Étape 3 : Hash MD5 + pHash (parallélisé, threads)
# ----------------------------------------------------------------------
def compute_md5(path, block_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Reconnaissance du contenu des images (v3.0)
#   - visages      : OpenCV Haar cascade (fournie avec opencv-python)
#   - objets/animaux : YOLOv8 nano (80 catégories COCO)
# Les deux sont facultatifs : si la dépendance manque, le reste du
# pipeline tourne exactement comme avant.
# ----------------------------------------------------------------------
def get_yolo_model():
    """Charge le modèle YOLO une seule fois pour tout le processus. En cas
    d'échec (pas de réseau au premier lancement, fichier corrompu...), on
    mémorise l'échec pour ne pas réessayer sur chaque photo."""
    global _yolo_model, _yolo_failed
    if _yolo_model is not None or _yolo_failed:
        return _yolo_model
    with _yolo_lock:
        if _yolo_model is None and not _yolo_failed:
            try:
                _yolo_model = YOLO(YOLO_MODEL_NAME)
                log.info("Modèle de reconnaissance chargé : %s", YOLO_MODEL_NAME)
            except Exception as e:
                _yolo_failed = True
                log.warning("Modèle YOLO indisponible (%s) -> détection d'objets désactivée. "
                            "Le premier lancement nécessite une connexion internet pour "
                            "télécharger %s (~6 Mo), ensuite tout est hors ligne.",
                            e, YOLO_MODEL_NAME)
    return _yolo_model


def get_face_cascade():
    """Une cascade par thread : l'objet OpenCV n'est pas thread-safe."""
    cascade = getattr(_thread_local, "face_cascade", None)
    if cascade is None:
        xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(xml))
        if cascade.empty():
            cascade = False  # marqueur d'échec, évite de retenter
        _thread_local.face_cascade = cascade
    return cascade or None


def _prepare_array(img):
    """Convertit une image PIL en tableau BGR réduit (côté le plus long
    ramené à DETECT_MAX_SIDE) : la détection n'a pas besoin de la pleine
    résolution et devient ainsi beaucoup plus rapide."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = DETECT_MAX_SIDE / max(w, h)
    if scale < 1:
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    arr = np.asarray(rgb)
    return arr[:, :, ::-1].copy()  # RGB -> BGR (convention OpenCV/YOLO)


def detect_faces(bgr):
    """Retourne le nombre de visages détectés (0 si aucun / indisponible)."""
    cascade = get_face_cascade()
    if cascade is None:
        return 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
    )
    return len(faces)


def detect_objects(bgr, conf=DETECT_CONF):
    """Retourne un dictionnaire {étiquette_anglaise: nombre d'occurrences}."""
    model = get_yolo_model()
    if model is None:
        return {}
    with _yolo_lock:
        results = model.predict(bgr, conf=conf, verbose=False)
    counts = defaultdict(int)
    for res in results:
        names = res.names
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        for cls_id in boxes.cls.tolist():
            label = names.get(int(cls_id)) if isinstance(names, dict) else names[int(cls_id)]
            if label:
                counts[label] += 1
    return dict(counts)


def load_known_faces(folder):
    """Charge les visages de référence pour la reconnaissance nominative :
    un sous-dossier de `folder` par personne (le nom du dossier = le nom
    de la personne), contenant une ou plusieurs photos d'elle. Plusieurs
    photos de référence par personne améliorent la fiabilité (angles,
    luminosité différents). Renvoie une liste de (nom, encodage_128d)."""
    known = []
    folder = Path(folder)
    if not folder.is_dir():
        log.warning("Dossier de visages connus introuvable : %s", folder)
        return known
    for person_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        name = person_dir.name
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in PHOTO_EXTS:
                continue
            try:
                img = face_recognition.load_image_file(str(img_path))
                for enc in face_recognition.face_encodings(img):
                    known.append((name, enc))
            except Exception as e:
                log.warning("Visage de référence illisible (%s) : %s", img_path, e)
    log.info("Visages connus chargés : %d référence(s) pour %d personne(s).",
              len(known), len({n for n, _ in known}))
    return known


def identify_faces(rgb, tolerance=FACE_TOLERANCE):
    """Détecte les visages d'une image (tableau RGB) et tente de les
    identifier : nom de la personne si elle correspond à un visage connu
    (--known-faces), sinon un identifiant "visage_NN" stable qui reste le
    même pour cette personne sur toutes les photos du catalogue (elle
    pourra être renommée après coup, par ex. en cherchant/remplaçant ce
    tag une fois qu'on sait qui c'est). Renvoie une liste de noms, une
    entrée par visage détecté sur la photo."""
    global _unknown_face_counter
    locations = face_recognition.face_locations(rgb)
    if not locations:
        return []
    labels = []
    for enc in face_recognition.face_encodings(rgb, locations):
        best_name, best_dist = None, tolerance
        for name, ref_enc in _known_faces:
            dist = np.linalg.norm(ref_enc - enc)
            if dist < best_dist:
                best_name, best_dist = name, dist
        if best_name:
            labels.append(best_name)
            continue
        with _unknown_faces_lock:
            match_label = None
            for label, ref_enc in _unknown_faces:
                if np.linalg.norm(ref_enc - enc) < tolerance:
                    match_label = label
                    break
            if match_label is None:
                _unknown_face_counter += 1
                match_label = f"visage_{_unknown_face_counter:02d}"
                _unknown_faces.append((match_label, enc))
        labels.append(match_label)
    return labels


def pluralize_fr(label):
    """Met au pluriel l'étiquette française d'un objet ("2 chien" est
    disgracieux dans un tag lu par un humain). Seul le premier mot est
    accordé : "gant de baseball" -> "gants de baseball"."""
    words = label.split(" ")
    w = words[0]
    if w.endswith(("s", "x", "z")):
        pass
    elif w.endswith(("eau", "eu")):
        w += "x"
    elif w.endswith("al"):
        w = w[:-2] + "aux"
    else:
        w += "s"
    words[0] = w
    return " ".join(words)


def format_objects_fr(counts):
    """Transforme {dog: 2, car: 1} en une chaîne cherchable en français :
    "2 chiens, 1 voiture, animal". Les mots-clés de regroupement
    (animal, véhicule, nourriture) sont ajoutés pour pouvoir chercher
    large sans connaître l'espèce exacte."""
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:DETECT_MAX_LABELS]
    parts = []
    for label, n in ordered:
        fr = COCO_FR.get(label, label)
        parts.append(f"{n} {pluralize_fr(fr)}" if n > 1 else fr)
    groups = []
    labels = set(counts)
    if labels & COCO_ANIMALS:
        groups.append("animal")
    if labels & COCO_VEHICLES:
        groups.append("véhicule")
    if labels & COCO_FOOD:
        groups.append("nourriture")
    return ", ".join(parts + groups)


def format_persons_fr(n_faces, n_persons, face_names=None):
    """Chaîne lisible et cherchable pour la colonne persons_detected.
    Si face_names est fourni (reconnaissance nominative active), les noms
    reconnus (ou "visage_NN" pour une personne non identifiée) passent en
    tête, pour être cherchables et pour finir en tag sur la photo."""
    parts = []
    if face_names:
        parts.extend(dict.fromkeys(face_names))  # dédoublonne, garde l'ordre
    if n_persons:
        parts.append(f"{n_persons} personnes" if n_persons > 1 else "1 personne")
    if n_faces:
        parts.append(f"{n_faces} visages" if n_faces > 1 else "1 visage")
    if n_faces >= 3 or n_persons >= 3:
        parts.append("groupe")
    if n_faces == 1 and n_persons <= 1:
        parts.append("portrait")
    return ", ".join(parts) or None


def analyse_photo_content(path, detect_conf=DETECT_CONF, face_tolerance=FACE_TOLERANCE):
    """Ouvre l'image UNE seule fois pour calculer le pHash, le texte OCR
    et la reconnaissance du contenu (visages + objets/animaux) : évite
    autant de relectures disque du même fichier."""
    ph, text, persons, objects = None, None, None, None
    with Image.open(path) as img:
        img.load()
        if HAVE_IMAGEHASH:
            ph = imagehash.phash(img)
        if HAVE_OCR:
            raw_text = pytesseract.image_to_string(img, lang="fra+eng")
            text = raw_text.strip()[:OCR_TEXT_MAX_LEN] or None
        if DETECT_ENABLED and (HAVE_FACES or HAVE_OBJECTS or HAVE_FACE_RECOGNITION):
            bgr = _prepare_array(img)
            # Chaque détecteur est protégé séparément : si la détection
            # d'objets échoue sur une image, on garde quand même les
            # visages (et réciproquement), et jamais l'image n'est
            # classée "en erreur" pour ça.
            counts = {}
            if HAVE_OBJECTS:
                try:
                    counts = detect_objects(bgr, conf=detect_conf)
                except Exception as e:
                    log.debug("Détection d'objets impossible sur %s : %s", path, e)
            n_faces = 0
            face_names = []
            if HAVE_FACE_RECOGNITION:
                # Reconnaissance nominative (--known-faces) : remplace le
                # simple comptage par une identification visage par visage.
                try:
                    face_names = identify_faces(bgr[:, :, ::-1], tolerance=face_tolerance)
                    n_faces = len(face_names)
                except Exception as e:
                    log.debug("Reconnaissance faciale impossible sur %s : %s", path, e)
            elif HAVE_FACES:
                try:
                    n_faces = detect_faces(bgr)
                except Exception as e:
                    log.debug("Détection de visages impossible sur %s : %s", path, e)
            n_persons = counts.get("person", 0)
            persons = format_persons_fr(n_faces, n_persons, face_names)
            objects = format_objects_fr({k: v for k, v in counts.items() if k != "person"})
    return ph, text, persons, objects


def analyse_worker(record, detect_conf=DETECT_CONF, face_tolerance=FACE_TOLERANCE):
    """Exécuté dans un thread du pool : calcule hash + phash (+ OCR et
    reconnaissance du contenu pour les photos). Ne touche jamais au
    fichier original (lecture seule)."""
    try:
        record["hash_md5"] = compute_md5(record["path"])
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"Hash MD5 impossible : {e}"
        return record

    needs_open = HAVE_IMAGEHASH or HAVE_OCR or (DETECT_ENABLED and (HAVE_FACES or HAVE_OBJECTS or HAVE_FACE_RECOGNITION))
    if record["media_type"] == "photo" and HAVE_PIL and needs_open:
        try:
            ph, ocr_text, persons, objects = analyse_photo_content(
                record["path"], detect_conf=detect_conf, face_tolerance=face_tolerance)
            record["hash_phash"] = str(ph) if ph is not None else None
            record["ocr_text"] = ocr_text
            record["persons_detected"] = persons
            record["objects_detected"] = objects
        except Exception as e:
            record["status"] = "error"
            record["error"] = f"Image illisible : {e}"
            return record

    record["status"] = "indexed"
    return record


# ----------------------------------------------------------------------
# Étape 4 : Producer-Consumer -> écriture SQLite (thread unique)
# ----------------------------------------------------------------------
def writer_thread_func(db_path, result_queue, done_event, stats):
    conn = open_db(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        buffer = []

        def flush():
            if not buffer:
                return
            for rec in buffer:
                try:
                    # Vérifie AVANT d'insérer plutôt que d'insérer et de
                    # rattraper l'erreur : plus robuste, et évite qu'une
                    # gestion de doublon puisse elle-même déclencher une
                    # 2e violation de contrainte non rattrapée.

                    # Cas 1 : ce chemin est déjà catalogué (relance du script
                    # sur une source déjà scannée, reprise après interruption...).
                    # On ne réinsère pas, on retrouve juste sa ligne existante.
                    existing = conn.execute("SELECT id FROM media WHERE file_path=?",
                                             (rec["path"],)).fetchone()
                    if existing:
                        rec["media_id"] = existing[0]
                        stats["indexed"] += 1
                        continue

                    # Cas 2 : même contenu (MD5) qu'un fichier déjà catalogué
                    # ailleurs -> doublon exact.
                    dup = None
                    if rec.get("hash_md5"):
                        dup = conn.execute("SELECT id FROM media WHERE hash_md5=?",
                                            (rec["hash_md5"],)).fetchone()

                    if dup:
                        cur = conn.execute(
                            "INSERT INTO media (file_path,file_name,file_extension,file_size,"
                            "media_type,hash_md5,hash_phash,date_shot,date_modified,status) "
                            "VALUES (?,?,?,?,?,NULL,?,?,?, 'duplicate_exact')",
                            (rec["path"], rec["name"], rec["ext"], rec["size"], rec["media_type"],
                             rec.get("hash_phash"),
                             rec.get("date_shot").isoformat() if rec.get("date_shot") else None,
                             rec["mtime"].isoformat()),
                        )
                        rec["media_id"] = cur.lastrowid
                        rec["status"] = "duplicate_exact"
                        conn.execute("INSERT INTO duplicates (media_id,duplicate_of_id,kind) VALUES (?,?,'exact')",
                                     (cur.lastrowid, dup[0]))
                        stats["duplicates_exact"] += 1
                    else:
                        cur = conn.execute(
                            "INSERT INTO media (file_path,file_name,file_extension,file_size,media_type,"
                            "hash_md5,hash_phash,date_shot,date_modified,width,height,camera_make,"
                            "camera_model,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (rec["path"], rec["name"], rec["ext"], rec["size"], rec["media_type"],
                             rec.get("hash_md5"), rec.get("hash_phash"),
                             rec.get("date_shot").isoformat() if rec.get("date_shot") else None,
                             rec["mtime"].isoformat(), rec.get("width"), rec.get("height"),
                             rec.get("camera_make"), rec.get("camera_model"), rec.get("status", "indexed")),
                        )
                        rec["media_id"] = cur.lastrowid
                        stats["indexed"] += 1
                        if (rec.get("gps_lat") is not None or rec.get("location_city")
                                or rec.get("ocr_text") or rec.get("persons_detected")
                                or rec.get("objects_detected")):
                            conn.execute(
                                "INSERT INTO metadata (media_id,location_city,location_country,"
                                "gps_lat,gps_lon,ocr_text,persons_detected,objects_detected) "
                                "VALUES (?,?,?,?,?,?,?,?)",
                                (rec["media_id"], rec.get("location_city"), rec.get("location_country"),
                                 rec.get("gps_lat"), rec.get("gps_lon"), rec.get("ocr_text"),
                                 rec.get("persons_detected"), rec.get("objects_detected")),
                            )
                    conn.commit()
                except Exception as e:
                    # Quoi qu'il arrive pour CE fichier, on continue avec les
                    # suivants : une seule ligne à problème ne doit jamais
                    # interrompre tout le traitement.
                    log.error("Erreur écriture DB pour %s : %s", rec["path"], e)
                    stats["errors"] += 1
                    try:
                        conn.rollback()
                    except Exception:
                        pass
            buffer.clear()

        while True:
            try:
                rec = result_queue.get(timeout=0.5)
            except queue.Empty:
                if done_event.is_set() and result_queue.empty():
                    break
                continue
            if rec is None:
                break
            buffer.append(rec)
            if len(buffer) >= DB_WRITE_BATCH:
                flush()
            result_queue.task_done()
        flush()
    finally:
        # Garantit que la connexion est toujours proprement fermée (et le
        # WAL fusionné), même si une erreur inattendue survient ci-dessus -
        # sinon les étapes suivantes du pipeline se retrouvent bloquées par
        # un verrou SQLite fantôme (voir correction §"database is locked").
        close_db(conn)


def run_pipeline(records, db_path, workers=8, detect_conf=DETECT_CONF, face_tolerance=FACE_TOLERANCE):
    """Lance le pool de threads d'analyse (producers) + le thread unique
    d'écriture (consumer), conformément au motif Producer-Consumer requis
    par le cahier des charges."""
    import concurrent.futures

    result_queue = queue.Queue()
    done_event = threading.Event()
    stats = {"indexed": 0, "duplicates_exact": 0, "errors": 0}

    writer = threading.Thread(target=writer_thread_func, args=(db_path, result_queue, done_event, stats))
    writer.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(analyse_worker, r, detect_conf, face_tolerance) for r in records]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            rec = fut.result()
            result_queue.put(rec)
            if i % 50 == 0 or i == len(records):
                print(f"\r  Analyse : {i}/{len(records)} fichiers", end="", flush=True)
    print()

    done_event.set()
    writer.join()
    return stats


# ----------------------------------------------------------------------
# Étape 5 : Détection des quasi-doublons visuels (BK-Tree)
# ----------------------------------------------------------------------
def detect_visual_duplicates(db_path):
    if not HAVE_IMAGEHASH:
        return 0
    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT id, hash_phash FROM media WHERE media_type='photo' AND hash_phash IS NOT NULL "
        "AND status NOT IN ('duplicate_exact','error') ORDER BY id"
    ).fetchall()
    tree = BKTree()
    count = 0
    for media_id, phash_str in rows:
        try:
            value = int(phash_str, 16)
        except (TypeError, ValueError):
            continue
        matches = tree.find_within(value, PHASH_VISUAL_DUP)
        if matches:
            orig_id, dist = min(matches, key=lambda m: m[1])
            conn.execute("UPDATE media SET status='duplicate_visual' WHERE id=?", (media_id,))
            conn.execute("INSERT INTO duplicates (media_id,duplicate_of_id,kind,distance) VALUES (?,?,'visual',?)",
                         (media_id, orig_id, dist))
            count += 1
        else:
            tree.add(media_id, value)
    conn.commit()
    close_db(conn)
    return count


# ----------------------------------------------------------------------
# Étape 6 : Proposition de classement (dossiers + noms de fichiers)
# ----------------------------------------------------------------------
def sanitize_folder_component(name):
    return "".join(c for c in name if c.isalnum() or c in " _-").strip() or "Inconnu"


def sanitize_component(value):
    """Nettoie un fragment de nom de fichier : pas de séparateurs de
    chemin ni de caractères spéciaux, espaces -> tirets."""
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|]', "", value)
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    return value


def parse_iso(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def compute_folder(row, target_dir):
    """Détermine uniquement le DOSSIER cible (pas encore le nom de fichier)."""
    target_dir = Path(target_dir)
    status = row["status"]
    media_type = row["media_type"]

    if status == "duplicate_exact":
        return target_dir / "00_DOUBLONS_EXACTS"
    if status == "duplicate_visual":
        return target_dir / "00_DOUBLONS_VISUELS"
    if status == "error":
        return target_dir / "00_ERREURS"
    if media_type == "unknown":
        return target_dir / "00_ERREURS"

    dt = parse_iso(row["date_shot"]) or parse_iso(row["date_modified"])
    yyyy = dt.strftime("%Y") if dt else "0000_SansDate"
    mm = dt.strftime("%m") if dt else "00"
    mm_label = f"{mm} {MONTHS_FR[mm]}" if mm in MONTHS_FR else mm

    if media_type == "video":
        return target_dir / "00_VIDEOS" / f"{yyyy}_{mm}"

    return target_dir / yyyy / mm_label


def build_camera_label(make, model):
    make = (make or "").strip()
    model = (model or "").strip()
    if not make and not model:
        return ""
    if make and model and make.lower() in model.lower():
        label = model
    elif make and model:
        label = f"{make}-{model}"
    else:
        label = model or make
    return sanitize_component(label)


def build_location_label(city, country):
    if city:
        return sanitize_component(city)
    if country:
        return sanitize_component(country)
    return ""


def generate_filename(row, seq, is_video=False, sep="-"):
    """Construit un nom de fichier au format :
    LIEU-MM-AAAA-APPAREIL-NNN.ext
    Exemple : COLLIOURE-07-2019-SM-G930F-001.jpeg
    """
    ext = Path(row["file_name"]).suffix.lower()
    dt = parse_iso(row["date_shot"]) or parse_iso(row["date_modified"])
    
    date_label = dt.strftime("%m-%Y") if dt else ""
    cam_label = "" if is_video else build_camera_label(row["camera_make"], row["camera_model"])
    loc_label = build_location_label(row["location_city"], row["location_country"]).upper()

    parts = [p for p in (loc_label, date_label, cam_label) if p]
    if not parts:
        parts = [sanitize_component(Path(row["file_name"]).stem) or "Media"]

    return sep.join(parts + [f"{seq:03d}"]) + ext


def build_report(db_path, target_dir, name_sep="-"):
    conn = open_db(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT m.id AS id, m.file_path AS file_path, m.file_name AS file_name, "
        "m.media_type AS media_type, m.status AS status, m.date_shot AS date_shot, "
        "m.date_modified AS date_modified, m.camera_make AS camera_make, "
        "m.camera_model AS camera_model, md.location_city AS location_city, "
        "md.location_country AS location_country "
        "FROM media m LEFT JOIN metadata md ON md.media_id = m.id "
        "ORDER BY m.id"
    ).fetchall()
    close_db(conn)

    # Les fichiers "de rangement normal" (photo/raw/video correctement
    # classés) reçoivent un nom parlant et numéroté. Les doublons/erreurs
    # gardent leur nom d'origine (plus simple à comparer/retrouver), avec
    # une numérotation de secours uniquement en cas de collision.
    RENAME_STATUSES = {"indexed", "moved"}

    # Le numéro étant désormais en tête du nom de fichier, il doit suivre
    # l'ordre chronologique et non l'ordre de scan : on calcule d'abord le
    # dossier de chaque fichier, puis on numérote dossier par dossier en
    # triant par date de prise de vue. Un tri alphabétique dans le Finder
    # redonne ainsi l'ordre réel des photos.
    folders = {row["id"]: compute_folder(row, target_dir) for row in rows}

    def sort_key(row):
        dt = parse_iso(row["date_shot"]) or parse_iso(row["date_modified"])
        return (dt is None, dt or datetime.min, row["file_name"])

    seq_by_id = {}
    by_folder = defaultdict(list)
    for row in rows:
        if row["status"] in RENAME_STATUSES and row["media_type"] != "unknown":
            by_folder[str(folders[row["id"]])].append(row)
    for folder_key, folder_rows in by_folder.items():
        for i, row in enumerate(sorted(folder_rows, key=sort_key), 1):
            seq_by_id[row["id"]] = i

    plan = []
    seen_paths = set()
    for row in rows:
        folder = folders[row["id"]]

        if row["id"] in seq_by_id:
            new_name = generate_filename(row, seq_by_id[row["id"]],
                                         is_video=(row["media_type"] == "video"),
                                         sep=name_sep)
            dest = folder / new_name
        else:
            dest = folder / row["file_name"]

        dest = str(dest)
        if dest in seen_paths:  # garde-fou en cas de collision improbable
            p = Path(dest)
            n = 2
            while f"{p.stem}_{n}{p.suffix}" in seen_paths:
                n += 1
            dest = str(p.with_name(f"{p.stem}_{n}{p.suffix}"))
        seen_paths.add(dest)

        plan.append({
            "media_id": row["id"], "source": row["file_path"], "file_name": row["file_name"],
            "media_type": row["media_type"], "status": row["status"], "dest": dest,
        })
    return plan


def print_and_save_report(plan, out_prefix):
    by_status = {}
    for item in plan:
        by_status.setdefault(item["status"], 0)
        by_status[item["status"]] += 1

    print("\n=== RAPPORT ===")
    print(f"Total de fichiers analysés : {len(plan)}")
    for status, count in sorted(by_status.items()):
        print(f"  - {status:20s} : {count}")

    json_path = f"{out_prefix}.json"
    txt_path = f"{out_prefix}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Rapport Phototheque Ultime\n")
        f.write(f"Généré le : {datetime.now().isoformat()}\n\n")
        for item in plan:
            f.write(f"[{item['status']}] {item['source']}  ->  {item['dest']}\n")
    print(f"\nRapport détaillé : {txt_path}")
    print(f"Rapport JSON      : {json_path}")


# ----------------------------------------------------------------------
# Étape 7 : Application (--apply) : déplacement réel + vérification
# ----------------------------------------------------------------------
def backup_database(db_path):
    db_path = Path(db_path)
    backup_dir = db_path.parent / "catalogue_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"catalogue_{stamp}.sqlite"
    shutil.copy2(db_path, backup_path)
    # ne garde que les 5 dernières sauvegardes
    backups = sorted(backup_dir.glob("catalogue_*.sqlite"))
    for old in backups[:-5]:
        old.unlink()
    return backup_path


def write_xmp_sidecar(et, raw_path):
    """Crée (sans jamais écraser) un sidecar .xmp à côté du fichier RAW
    déplacé, contenant les métadonnées déjà lues par ExifTool. Le fichier
    RAW original n'est jamais modifié (portabilité totale, §1 du cahier
    des charges)."""
    raw_path = Path(raw_path)
    xmp_path = raw_path.with_suffix(raw_path.suffix + ".xmp")
    if xmp_path.exists():
        return xmp_path, "déjà présent"
    try:
        et.execute("-o", str(xmp_path), str(raw_path))
        return xmp_path, "créé"
    except Exception as e:
        return None, f"échec ({e})"


def build_keywords(persons_detected, objects_detected, location_city, location_country):
    """Transforme les colonnes de détection en une liste de mots-clés
    simples et propres pour les champs IPTC:Keywords / XMP:Subject, que
    Photos, Lightroom, digiKam... savent tous lire et filtrer."""
    keywords = []
    for raw in (persons_detected, objects_detected):
        if not raw:
            continue
        for chunk in raw.split(","):
            # "2 chiens" -> "chiens" : on garde le mot, pas le compte
            word = re.sub(r"^\s*\d+\s+", "", chunk).strip()
            if word and word not in keywords:
                keywords.append(word)
    for loc in (location_city, location_country):
        if loc and loc not in keywords:
            keywords.append(loc)
    return keywords


def embed_metadata_in_photo(et, photo_path, ocr_text=None, location_city=None, location_country=None,
                            persons_detected=None, objects_detected=None):
    """Écrit directement dans le fichier photo le texte OCR et le lieu.
    Le commutateur -overwrite_original est utilisé pour ne pas créer de
    fichiers de sauvegarde .ext_original."""
    args = ["-overwrite_original"]
    if ocr_text:
        clean = " ".join(ocr_text.split())[:1000]
        args += [f"-XMP-dc:Description={clean}", f"-IPTC:Caption-Abstract={clean}"]
    if location_city:
        args += [f"-IPTC:City={location_city}", f"-XMP-photoshop:City={location_city}"]
    if location_country:
        args += [f"-IPTC:Country-PrimaryLocationName={location_country}"]
    for kw in build_keywords(persons_detected, objects_detected, location_city, location_country):
        # '+=' ajoute le mot-clé sans effacer ceux déjà présents dans la
        # photo : on n'écrase jamais un classement fait à la main.
        args += [f"-IPTC:Keywords+={kw}", f"-XMP-dc:Subject+={kw}"]
    if len(args) == 1:
        return "rien à écrire"
    try:
        et.execute(*args, str(photo_path))
        return "écrit"
    except Exception as e:
        return f"échec ({e})"


def apply_plan(plan, db_path, logfile, write_tags=True):
    backup_path = backup_database(db_path)
    log.info("Sauvegarde de la base : %s", backup_path)

    conn = open_db(db_path)
    moved, failed = 0, 0

    et_ctx = exiftool.ExifToolHelper() if HAVE_EXIFTOOL else None
    et = et_ctx.__enter__() if et_ctx else None
    try:
        with open(logfile, "a", encoding="utf-8") as lf:
            lf.write(f"\n--- Application du {datetime.now().isoformat()} ---\n")
            for item in plan:
                src = Path(item["source"])
                dest = Path(item["dest"])
                if not src.exists():
                    lf.write(f"MANQUANT: {src}\n")
                    failed += 1
                    continue
                try:
                    src_md5 = compute_md5(src)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest))
                    dest_md5 = compute_md5(dest)
                except Exception as e:
                    lf.write(f"ERREUR (déplacement) : {src} -> {dest} ({e})\n")
                    failed += 1
                    continue

                if src_md5 != dest_md5:
                    # Le fichier a été déplacé mais son contenu diffère à
                    # l'arrivée : on le signale comme échec d'intégrité, sans
                    # toucher au fichier (il est déjà à `dest`, à vérifier
                    # manuellement).
                    lf.write(f"ALERTE INTEGRITE (MD5 different) : {src} -> {dest}\n")
                    failed += 1
                    continue

                # Le déplacement physique a réussi : on compte le fichier comme
                # déplacé même si la mise à jour du catalogue échoue ensuite
                # (ex : verrou SQLite passager) - ce qui compte pour l'utilisateur
                # c'est l'état réel de ses fichiers sur le disque.
                moved += 1
                try:
                    conn.execute("UPDATE media SET file_path=?, status='moved' WHERE id=?",
                                 (str(dest), item["media_id"]))
                    conn.commit()
                    lf.write(f"OK: {src} -> {dest}\n")
                except Exception as e:
                    lf.write(f"OK (déplacé) mais catalogue non mis à jour : {src} -> {dest} ({e})\n")

                if item["media_type"] == "raw" and et is not None:
                    xmp_path, xmp_status = write_xmp_sidecar(et, dest)
                    lf.write(f"XMP ({xmp_status}): {xmp_path or dest}\n")

                if item["media_type"] == "photo" and write_tags and et is not None:
                    meta_row = conn.execute(
                        "SELECT ocr_text, location_city, location_country, "
                        "persons_detected, objects_detected FROM metadata WHERE media_id=?",
                        (item["media_id"],),
                    ).fetchone()
                    if meta_row:
                        ocr_text, loc_city, loc_country, persons, objects = meta_row
                        tag_status = embed_metadata_in_photo(
                            et, dest, ocr_text, loc_city, loc_country, persons, objects)
                        lf.write(f"TAGS ({tag_status}): {dest}\n")
    finally:
        if et_ctx:
            et_ctx.__exit__(None, None, None)
        close_db(conn)
    return moved, failed


# ----------------------------------------------------------------------
# Moteur de recherche CLI (v2) : FTS5 + filtres + similarité visuelle
# ----------------------------------------------------------------------
_RE_SIMILAR = re.compile(r"similaire:(\S+)", re.IGNORECASE)
_RE_DATE_RANGE = re.compile(r"date:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})")
_RE_TYPE = re.compile(r"type:(\w+)", re.IGNORECASE)
_RE_FORMAT = re.compile(r"format:(\w+)", re.IGNORECASE)
_RE_NEGATION = re.compile(r'(?<!\S)-(\w+)')


_FTS_KEYWORDS = {"AND", "OR", "NOT", "NEAR"}


def add_prefix_wildcards(text):
    """Ajoute un '*' à chaque mot simple de la requête, pour que "chien"
    trouve aussi "chiens" et "voiture" trouve "voitures".

    FTS5 fait sinon une correspondance mot-à-mot stricte, ce qui est
    pénible en français où le script stocke le pluriel dès qu'il détecte
    plusieurs objets ("2 chiens"). Les phrases entre guillemets, les
    opérateurs FTS5 et les mots déjà suivis d'une étoile sont laissés
    intacts."""
    if not text:
        return text
    out = []
    for token in re.findall(r'"[^"]*"|\S+', text):
        if (token.startswith('"') or token.endswith("*")
                or token.upper() in _FTS_KEYWORDS
                or not re.fullmatch(r"\w+", token, flags=re.UNICODE)):
            out.append(token)
        else:
            out.append(token + "*")
    return " ".join(out)


def parse_search_query(raw_query):
    """Extrait les filtres spéciaux (date:, type:, format:, similaire:) du
    texte de requête et traduit le reste en syntaxe FTS5 valide :
    - "a + b"      -> "a b"      (ET implicite en FTS5, pas besoin de '+')
    - "a OR b"     -> inchangé   (mot-clé FTS5 natif)
    - "a -b"       -> "a NOT b"  (FTS5 n'a pas de raccourci '-')
    - '"expr"'     -> inchangé   (phrase exacte, syntaxe FTS5 native)
    """
    text = raw_query
    filters = {}

    m = _RE_SIMILAR.search(text)
    if m:
        filters["similar_file"] = m.group(1)
        text = _RE_SIMILAR.sub("", text)

    m = _RE_DATE_RANGE.search(text)
    if m:
        filters["date_start"], filters["date_end"] = m.group(1), m.group(2)
        text = _RE_DATE_RANGE.sub("", text)

    m = _RE_TYPE.search(text)
    if m:
        filters["media_type"] = m.group(1).lower()
        text = _RE_TYPE.sub("", text)

    m = _RE_FORMAT.search(text)
    if m:
        filters["file_extension"] = "." + m.group(1).lower().lstrip(".")
        text = _RE_FORMAT.sub("", text)

    text = re.sub(r"\s*\+\s*", " ", text)                 # "+" -> ET implicite
    text = _RE_NEGATION.sub(r"NOT \1", text)               # "-mot" -> "NOT mot"
    text = re.sub(r"\s+", " ", text).strip()
    text = add_prefix_wildcards(text)

    filters["fts_query"] = text or None
    return filters


def search_similar(conn, filename, limit):
    ref = conn.execute(
        "SELECT id, hash_phash FROM media WHERE file_name LIKE ? AND hash_phash IS NOT NULL LIMIT 1",
        (f"%{filename}%",),
    ).fetchone()
    if not ref or not ref["hash_phash"]:
        return []
    try:
        ref_value = int(ref["hash_phash"], 16)
    except ValueError:
        return []

    rows = conn.execute(
        "SELECT id, file_path, file_name, media_type, date_shot, hash_phash FROM media "
        "WHERE media_type='photo' AND hash_phash IS NOT NULL AND id != ?",
        (ref["id"],),
    ).fetchall()

    tree = BKTree()
    for row in rows:
        try:
            tree.add(row["id"], int(row["hash_phash"], 16))
        except ValueError:
            continue
    matches = sorted(tree.find_within(ref_value, PHASH_SIMILARITY_SEARCH), key=lambda m: m[1])[:limit]
    by_id = {row["id"]: row for row in rows}
    return [dict(by_id[mid], distance=dist) for mid, dist in matches]


def run_search(db_path, raw_query, limit=50):
    conn = open_db(db_path)
    conn.row_factory = sqlite3.Row
    try:
        filters = parse_search_query(raw_query)

        if filters.get("similar_file"):
            return search_similar(conn, filters["similar_file"], limit)

        sql = "SELECT m.id,m.file_path,m.file_name,m.media_type,m.date_shot,m.camera_make,m.camera_model FROM media m"
        where, params = [], []
        if filters.get("fts_query"):
            sql += " JOIN fts_search fs ON fs.media_id = m.id"
            where.append("fts_search MATCH ?")
            params.append(filters["fts_query"])
        if filters.get("media_type"):
            where.append("m.media_type = ?")
            params.append(filters["media_type"])
        if filters.get("file_extension"):
            where.append("LOWER(m.file_extension) = ?")
            params.append(filters["file_extension"])
        if filters.get("date_start"):
            where.append("date(m.date_shot) BETWEEN date(?) AND date(?)")
            params.extend([filters["date_start"], filters["date_end"]])
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY m.date_shot DESC LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            log.error("Requête invalide (%s) : %s", filters.get("fts_query"), e)
            return []
        return [dict(r) for r in rows]
    finally:
        close_db(conn)


def print_search_results(results, raw_query):
    if not results:
        print(f"Aucun résultat pour : {raw_query}")
        return
    print(f"{len(results)} résultat(s) pour : {raw_query}\n")
    for r in results:
        date = (r.get("date_shot") or "")[:19]
        extra = f"   (distance visuelle = {r['distance']})" if "distance" in r else ""
        print(f"[{r.get('media_type', '?'):6s}] {date:19s}  {r['file_path']}{extra}")


def export_search_results(results, fmt, output_path):
    if not results:
        return
    if fmt == "csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    elif fmt == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main_search(argv):
    parser = argparse.ArgumentParser(prog="phototheque.py search",
                                      description="Recherche dans une photothèque déjà cataloguée.")
    parser.add_argument("query", help='Ex: "chien + plage", \'"Plage de Palavas"\', '
                                       '"date:2024-01-01..2024-08-31 type:photo", "similaire:IMG_1234.JPG"')
    parser.add_argument("--db", required=True, help="Chemin vers catalogue.sqlite")
    parser.add_argument("--export", choices=["csv", "json"], default=None)
    parser.add_argument("--output", default=None, help="Fichier de sortie (défaut : resultats.<format>)")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not Path(args.db).exists():
        print(f"Base introuvable : {args.db}")
        return

    results = run_search(args.db, args.query, limit=args.limit)
    print_search_results(results, args.query)

    if args.export and results:
        output_path = args.output or f"resultats.{args.export}"
        export_search_results(results, args.export, output_path)
        print(f"\nExporté : {output_path}")


def main_run(argv):
    parser = argparse.ArgumentParser(description="Phototheque Ultime - rangement local de médias")
    parser.add_argument("source", help="Dossier source à analyser")
    parser.add_argument("--target", help="Dossier cible (photothèque triée). Requis sauf en --scan seul.")
    parser.add_argument("--db", default=None, help="Chemin de la base catalogue.sqlite "
                         "(par défaut : <target>/CATALOGUE/catalogue.sqlite)")
    parser.add_argument("--scan", action="store_true", help="Inventaire seul, aucune analyse.")
    parser.add_argument("--analyse", action="store_true", help="Hash + métadonnées, pas de plan de rangement.")
    parser.add_argument("--dry-run", action="store_true", help="Analyse complète + rapport, RIEN n'est déplacé (mode par défaut).")
    parser.add_argument("--apply", action="store_true", help="Applique réellement les déplacements.")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de threads d'analyse (défaut : 8).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Ne traite que les N premiers fichiers trouvés (pratique pour un premier test).")
    parser.add_argument("--report", default="rapport_phototheque", help="Préfixe des fichiers de rapport.")
    parser.add_argument("--write-tags", dest="write_tags", action="store_true", default=True,
                         help="(activé par défaut) En plus du catalogue, écrit le texte OCR, le lieu et "
                              "les mots-clés du contenu détecté directement dans les photos (IPTC/XMP). "
                              "Ne touche jamais aux fichiers RAW. ExifTool conserve automatiquement une "
                              "copie de sauvegarde (.ext_original) à chaque écriture.")
    parser.add_argument("--no-write-tags", dest="write_tags", action="store_false",
                         help="Désactive l'écriture des tags dans les photos : le catalogue SQLite est "
                              "alors seul enrichi, et aucun fichier photo n'est modifié.")
    parser.add_argument("--no-detect", dest="detect", action="store_false", default=True,
                         help="Désactive la reconnaissance du contenu (visages, objets, animaux). "
                              "Nettement plus rapide si vous ne voulez qu'un rangement par date.")
    parser.add_argument("--detect-conf", type=float, default=DETECT_CONF,
                         help=f"Seuil de confiance de la détection d'objets, entre 0 et 1 "
                              f"(défaut : {DETECT_CONF}). Plus bas = plus de détections, mais plus "
                              f"d'erreurs ; plus haut = seulement les objets évidents.")
    parser.add_argument("--name-sep", default="-",
                         help="Séparateur entre les blocs du nom de fichier (défaut : \"-\").")
    parser.add_argument("--known-faces", default=None,
                         help="Dossier contenant un sous-dossier par personne (avec une ou plusieurs "
                              "photos d'elle) pour reconnaître qui est sur chaque photo. Les visages non "
                              "reconnus reçoivent un tag \"visage_01\", \"visage_02\"... stable sur tout le "
                              "catalogue, à renommer plus tard. Nécessite 'pip install face_recognition'.")
    parser.add_argument("--face-tolerance", type=float, default=FACE_TOLERANCE,
                         help=f"Tolérance de la reconnaissance faciale, entre 0 et 1 (défaut : "
                              f"{FACE_TOLERANCE}). Plus bas = plus strict (moins de faux positifs, "
                              f"risque de ne pas reconnaître) ; plus haut = plus permissif.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    global DETECT_ENABLED
    DETECT_ENABLED = args.detect

    if not any([args.scan, args.analyse, args.dry_run, args.apply]):
        args.dry_run = True  # mode par défaut : sûr

    if (args.dry_run or args.apply or args.analyse) and not args.target and not args.scan:
        parser.error("--target est requis pour --analyse / --dry-run / --apply")

    target_dir = Path(args.target) if args.target else Path(args.source)
    db_path = args.db or str(target_dir / "CATALOGUE" / "catalogue.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logfile = target_dir / "CATALOGUE" / "phototheque.log" if args.target else Path("phototheque.log")

    print(f"Source : {args.source}")
    if args.target:
        print(f"Cible  : {args.target}")
    print(f"Base   : {db_path}")
    if not HAVE_EXIFTOOL:
        print("⚠ pyexiftool absent -> pas d'extraction EXIF (voir GUIDE_UTILISATION.md).")
    if not HAVE_PIL:
        print("⚠ Pillow absent -> pas de pHash ni d'OCR possibles sur les photos.")
    if not HAVE_IMAGEHASH:
        print("⚠ imagehash absent -> pas de détection des doublons visuels.")
    if not HAVE_GEOCODER:
        print("⚠ reverse_geocoder absent -> les noms de fichiers n'incluront pas le lieu.")
    if not HAVE_OCR:
        print("⚠ pytesseract absent -> pas d'OCR (voir GUIDE_UTILISATION.md).")
    if not args.detect:
        print("ℹ Reconnaissance du contenu désactivée (--no-detect).")
    else:
        if not HAVE_FACES:
            print("⚠ opencv-python absent -> pas de détection de visages "
                  "(pip install opencv-python).")
        if not HAVE_OBJECTS:
            print("⚠ ultralytics absent -> pas de détection d'objets/animaux "
                  "(pip install ultralytics).")
        if HAVE_FACES or HAVE_OBJECTS:
            actifs = " + ".join(n for n, ok in (("visages", HAVE_FACES),
                                                 ("objets/animaux", HAVE_OBJECTS)) if ok)
            print(f"✓ Reconnaissance du contenu active : {actifs} "
                  f"(seuil {args.detect_conf}).")
        if args.known_faces:
            if not HAVE_FACE_RECOGNITION:
                print("⚠ face_recognition absent -> pas de reconnaissance nominative "
                      "(pip install face_recognition).")
            else:
                global _known_faces
                _known_faces = load_known_faces(args.known_faces)
                n_people = len({n for n, _ in _known_faces})
                print(f"✓ Reconnaissance nominative active : {n_people} personne(s) connue(s) "
                      f"depuis {args.known_faces} (tolérance {args.face_tolerance}). "
                      f"Les visages non reconnus seront tagués visage_01, visage_02...")
        elif HAVE_FACE_RECOGNITION:
            print("ℹ face_recognition est installé mais --known-faces n'est pas fourni : "
                  "les visages non reconnus seront quand même tagués visage_01, visage_02... "
                  "(sans --known-faces, aucun nom ne pourra être reconnu).")
    if args.write_tags:
        print("✓ Écriture des tags dans les photos active "
              "(--no-write-tags pour la désactiver).")

    t0 = time.time()
    print("\n[1/5] Inventaire...")
    records = inventory(args.source, limit=args.limit)
    print(f"  {len(records)} fichiers trouvés.")
    if not records:
        print("Aucun fichier à traiter.")
        return

    if args.scan:
        by_type = {}
        for r in records:
            by_type[r["media_type"]] = by_type.get(r["media_type"], 0) + 1
        print("\nRépartition par type :")
        for t, c in by_type.items():
            print(f"  - {t:10s} : {c}")
        return

    print("\n[2/5] Extraction des métadonnées EXIF...")
    extract_metadata_batch(records)

    print("\n[3/5] Hash MD5 + pHash + OCR + reconnaissance du contenu (multi-thread)...")
    # base fraîche pour ce run (le fichier persists entre les lancements ;
    # on ne le recrée pas s'il existe déjà, pour permettre des scans incrémentaux)
    stats = run_pipeline(records, db_path, workers=args.workers,
                          detect_conf=args.detect_conf, face_tolerance=args.face_tolerance)
    print(f"  Indexés : {stats['indexed']} | Doublons exacts : {stats['duplicates_exact']} | Erreurs : {stats['errors']}")

    if args.analyse:
        print(f"\nTerminé en {time.time()-t0:.1f}s. Base : {db_path}")
        return

    print("\n[4/5] Détection des quasi-doublons visuels (BK-Tree)...")
    n_visual = detect_visual_duplicates(db_path)
    print(f"  {n_visual} quasi-doublons visuels détectés (distance pHash <= {PHASH_VISUAL_DUP}).")

    print("\n[5/5] Calcul du plan de rangement...")
    plan = build_report(db_path, target_dir, name_sep=args.name_sep)
    print_and_save_report(plan, args.report)

    if args.apply:
        moved, failed = apply_plan(plan, db_path, logfile, write_tags=args.write_tags)
        print(f"\nAppliqué : {moved} fichiers déplacés, {failed} échecs. Log : {logfile}")
        if any(item["media_type"] == "raw" for item in plan) and HAVE_EXIFTOOL:
            print("Sidecars .xmp créés pour les fichiers RAW déplacés (voir le log pour le détail).")
        if args.write_tags and HAVE_EXIFTOOL:
            print("Texte OCR / lieu écrits directement dans les photos (voir le log pour le détail).")
    else:
        print("\nMode simulation (--dry-run) : aucun fichier n'a été déplacé.")
        print("Relancez avec --apply une fois le rapport validé.")

    print(f"\nTerminé en {time.time()-t0:.1f}s.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        main_search(sys.argv[2:])
    else:
        main_run(sys.argv[1:])


if __name__ == "__main__":
    main()