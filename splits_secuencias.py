# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import shutil
import random

#######################################
# CONFIGURACIÓN GENERAL
#######################################

# Ruta base
base = Path(r"C:\Users\jmoli\Desktop\TFG mates\Datos\Archivos procesados")

# Pool unificado
slices_dir = base / "Unified_sec" / "Slices"
masks_dir  = base / "Unified_sec" / "Masks"

# Modalidades disponibles
MODS = ["FLAIR", "T1", "T2"]

# Definimos rangos de pacientes por split y dataset
splits_pacientes = {
    "train": {"nature": range(1, 54)},   # 1–53
    "val":   {"nature": range(54, 65)},  # 54–64
    "test":  {"nature": range(65, 76)},  # 65–75
}

# Carpeta destino (multimodal)
dest_base = base / "borrar_carpeta"
for split in ["train", "val", "test"]:
    (dest_base / split / "Slices").mkdir(parents=True, exist_ok=True)
    (dest_base / split / "Masks").mkdir(parents=True, exist_ok=True)

#######################################
# 1) CONSTRUCCIÓN DE LOS SPLITS EN DISCO
#######################################

# Parámetros de filtrado
BLACK_THR = 0.05        # por debajo de esto consideramos "muy oscuro"
MIN_BRAIN_FRAC = 0.10   # si menos del 10% de píxeles superan BLACK_THR => demasiado oscuro
RATIO_NEG_TRAIN = 0.3   # en train: nos quedamos con ~30% de los negativos

# (De tu script original; no se usan actualmente)
VAR_FRAC_THR = 0.00
PIX_DIFF_THR = 0.00

random.seed(0)  # opcional: reproducibilidad del muestreo de negativos

for split_name, cfg in splits_pacientes.items():
    print(f"\n=== Construyendo split: {split_name} ===")

    num_kept = 0
    num_skipped_empty = 0
    num_skipped_neg = 0
    num_skipped_missing_mod = 0

    for dataset_name, pac_range in cfg.items():
        for p in pac_range:
            pid = f"p{p:02d}"  # p01, p02, ...

            # Usamos SOLO máscaras FLAIR como "lista maestra" de cortes
            pattern_mask_flair = f"{dataset_name}_FLAIR_mask_{pid}_*.npy"

            for mask_flair_path in sorted(masks_dir.glob(pattern_mask_flair)):
                # Ej: nature_FLAIR_mask_p01_z039.npy
                z_part = mask_flair_path.name.split("_")[-1]  # "z039.npy"

                # Construir paths de las 3 imágenes para el MISMO (pid,z)
                img_paths = {}
                missing = False
                for m in MODS:
                    img_name = f"{dataset_name}_{m}_slice_{pid}_{z_part}"
                    img_path = slices_dir / img_name
                    if not img_path.exists():
                        print(f"⚠ Falta imagen {m} para {dataset_name} {pid} {z_part}")
                        missing = True
                        break
                    img_paths[m] = img_path

                if missing:
                    num_skipped_missing_mod += 1
                    continue

                # Cargamos FLAIR + máscara (única) para filtros
                img_flair = np.load(str(img_paths["FLAIR"]))
                mask = np.load(str(mask_flair_path))

                # 1) Filtrar cortes con muy poco cerebro (solo TRAIN) usando FLAIR
                if split_name == "train":
                    brain_frac = (img_flair > BLACK_THR).mean()
                    if brain_frac < MIN_BRAIN_FRAC:
                        num_skipped_empty += 1
                        continue

                # 2) Muestreo de negativos (solo TRAIN)
                has_lesion = (mask.sum() > 0)
                if split_name == "train" and (not has_lesion):
                    if random.random() > RATIO_NEG_TRAIN:
                        num_skipped_neg += 1
                        continue

                # 3) Copiar las 3 imágenes al split
                for m in MODS:
                    dest_slice = dest_base / split_name / "Slices" / img_paths[m].name
                    shutil.copy2(img_paths[m], dest_slice)

                # 4) Copiar SOLO 1 máscara (la FLAIR) renombrada SIN modalidad
                new_mask_name = f"{dataset_name}_mask_{pid}_{z_part}"  # nature_mask_p01_z039.npy
                dest_mask = dest_base / split_name / "Masks" / new_mask_name
                shutil.copy2(mask_flair_path, dest_mask)

                num_kept += 1

    print(f"{split_name}: {num_kept} cortes (pid,z) copiados.")
    if split_name == "train":
        print(f"  - descartados por casi vacíos: {num_skipped_empty}")
        print(f"  - negativos descartados por muestreo: {num_skipped_neg}")
    print(f"  - descartados por falta de modalidad: {num_skipped_missing_mod}")

#######################################
# 2) (OPCIONAL) CARGA EN MEMORIA DE LOS SPLITS
#######################################

X_splits = {"train": [], "val": [], "test": []}  # (3,H,W) apilado [FLAIR,T1,T2]
Y_splits = {"train": [], "val": [], "test": []}  # (H,W)

for split_name in ["train", "val", "test"]:
    print(f"\n=== Generando listas multimodales para split: {split_name} ===")

    split_slices_dir = dest_base / split_name / "Slices"
    split_masks_dir  = dest_base / split_name / "Masks"

    X_list, Y_list = [], []

    # Iteramos SOLO sobre FLAIR para no duplicar (una muestra por corte)
    for flair_path in sorted(split_slices_dir.glob("*_FLAIR_slice_*.npy")):
        # Ej: nature_FLAIR_slice_p01_z039.npy
        parts = flair_path.stem.split("_")
        # parts: ['nature','FLAIR','slice','p01','z039']
        dataset_name = parts[0]
        pid = parts[-2]            # 'p01'
        z_part = parts[-1] + ".npy"  # 'z039.npy'

        # Construir paths T1/T2
        t1_path = split_slices_dir / f"{dataset_name}_T1_slice_{pid}_{z_part}"
        t2_path = split_slices_dir / f"{dataset_name}_T2_slice_{pid}_{z_part}"

        mask_path = split_masks_dir / f"{dataset_name}_mask_{pid}_{z_part}"

        if (not t1_path.exists()) or (not t2_path.exists()) or (not mask_path.exists()):
            print(f"⚠ Muestra incompleta en {split_name}: {dataset_name} {pid} {z_part}")
            continue

        flair = np.load(str(flair_path))
        t1    = np.load(str(t1_path))
        t2    = np.load(str(t2_path))
        mask  = np.load(str(mask_path))

        x = np.stack([flair, t1, t2], axis=0)  # (3,H,W)
        X_list.append(x)
        Y_list.append(mask)

    X_splits[split_name] = X_list
    Y_splits[split_name] = Y_list

    print(f"{split_name}: {len(X_list)} muestras multimodales (3ch) y {len(Y_list)} máscaras")

#######################################
# RESUMEN FINAL EN DISCO
#######################################

print("\n=== Resumen final en disco (multimodal) ===")
for split_name in splits_pacientes.keys():
    n_slices = len(list((dest_base / split_name / "Slices").glob("*.npy")))
    n_masks  = len(list((dest_base / split_name / "Masks").glob("*.npy")))
    # OJO: n_slices debería ser ~3 * n_masks (si todo está completo)
    print(f"{split_name.upper():5s} -> Slices: {n_slices:5d} | Masks: {n_masks:5d}")


