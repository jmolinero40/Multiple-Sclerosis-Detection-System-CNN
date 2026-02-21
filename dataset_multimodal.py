# -*- coding: utf-8 -*-
"""
dataset_multimodal.py

Dataset + augments para experimento MULTIMODAL (3 canales):
    canal 0 = FLAIR
    canal 1 = T1
    canal 2 = T2
y una única máscara GT por corte.

"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# transforms para data augmentation
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class MSLesionDatasetMultimodal(Dataset):
    """
    Dataset multimodal: apila [FLAIR, T1, T2] del mismo (paciente,z).

    Devuelve:
      img : torch.Tensor (3,H,W) float32
      mask: torch.Tensor (1,H,W) float32 {0,1}
    """

    def __init__(self, base_dir, split="train", transform=None,
                 dataset_name="nature", check_files=True):
        """
        base_dir: ruta a tu carpeta de splits multimodales (p.ej. Unified_splits_res_multimodal)
        split: 'train', 'val', 'test'
        transform: función opcional f(img, mask) -> (img, mask)
        dataset_name: 'nature' (si solo tienes este, déjalo fijo)
        check_files: si True, valida que existen T1/T2/mask para cada FLAIR
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.dataset_name = dataset_name

        self.slices_dir = self.base_dir / split / "Slices"
        self.masks_dir  = self.base_dir / split / "Masks"

        if not self.slices_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de slices: {self.slices_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de masks: {self.masks_dir}")

        # Tomamos SOLO FLAIR como "lista maestra" de muestras (una por corte).
        pattern = f"{dataset_name}_FLAIR_slice_*.npy"
        self.flair_paths = sorted(self.slices_dir.glob(pattern))
        if len(self.flair_paths) == 0:
            raise RuntimeError(f"No se han encontrado FLAIR con patrón '{pattern}' en {self.slices_dir}")

        # Precomputamos samples: (flair, t1, t2, mask)
        self.samples = []
        missing = 0
        for fpath in self.flair_paths:
            name = fpath.name  # nature_FLAIR_slice_p01_z039.npy

            t1_path = fpath.parent / name.replace("_FLAIR_slice_", "_T1_slice_")
            t2_path = fpath.parent / name.replace("_FLAIR_slice_", "_T2_slice_")

            # máscara única SIN modalidad: nature_mask_p01_z039.npy
            mask_name = name.replace("_FLAIR_slice_", "_mask_")
            mask_path = self.masks_dir / mask_name

            if check_files:
                if not t1_path.exists() or not t2_path.exists() or not mask_path.exists():
                    missing += 1
                    continue

            self.samples.append((fpath, t1_path, t2_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No se han podido construir samples multimodales. "
                f"¿Están T1/T2 y la máscara en su sitio? Missing={missing}"
            )

        if missing > 0:
            print(f"[MSLesionDatasetMultimodal] Aviso: {missing} FLAIR sin pareja completa (T1/T2/mask) -> omitidos.")

        print(f"[MSLesionDatasetMultimodal] Split='{split}' → {len(self.samples)} muestras encontradas.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, t1_path, t2_path, mask_path = self.samples[idx]

        flair = np.load(str(fpath)).astype(np.float32)
        t1    = np.load(str(t1_path)).astype(np.float32)
        t2    = np.load(str(t2_path)).astype(np.float32)

        if flair.shape != t1.shape or flair.shape != t2.shape:
            raise ValueError(
                f"Shapes distintas para {fpath.name}: "
                f"FLAIR{flair.shape} T1{t1.shape} T2{t2.shape}"
            )

        img = np.stack([flair, t1, t2], axis=0).astype(np.float32)  # (3,H,W)

        mask = np.load(str(mask_path)).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # (1,H,W)

        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(mask)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)

        return img_t, mask_t


def train_augment(
    img, mask,
    p_geom=0.8,
    p_int=0.7,
    brightness_delta=0.06,          # ±6% (asumiendo min-max en [0,1])
    contrast_range=(0.90, 1.10),
    gamma_range=(0.90, 1.10),
    noise_std=0.01,
    brain_thr=0.05                  # umbral para detectar cerebro (sobre max entre canales)
):
    """
    Augment compatible con 3 canales multimodales (FLAIR/T1/T2).
    Mantiene tu estilo actual (pensado para intensidades en [0,1]).

    img : torch.Tensor (C,H,W) float
    mask: torch.Tensor (1,H,W) float {0,1}
    """
    img = img.float()
    mask = mask.float()

    # ---- 0) Brain mask (para NO tocar fondo) ----
    # En multimodal, NO usamos el "canal central", usamos el máximo entre canales.
    # brain: (1,H,W)
    max_hw = img.max(dim=0).values
    brain = (max_hw > brain_thr).float().unsqueeze(0)

    # ---------- 1) AUGMENT GEOMÉTRICO (img, mask y brain) ----------
    if random.random() < p_geom:
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            brain = TF.hflip(brain)

        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
            brain = TF.vflip(brain)

        angle = random.uniform(-10, 10)
        img  = TF.rotate(img,  angle, interpolation=InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        brain = TF.rotate(brain, angle, interpolation=InterpolationMode.NEAREST)

    # ---------- 2) AUGMENT DE INTENSIDAD (solo img, SOLO dentro de brain) ----------
    if random.random() < p_int:
        # brillo (mismo delta para los 3 canales)
        delta = random.uniform(-brightness_delta, brightness_delta)
        img = img + delta

        # contraste: escalamos respecto a la media dentro del cerebro
        cfac = random.uniform(*contrast_range)
        mean = (img * brain).sum() / (brain.sum() * img.size(0) + 1e-8)
        img = (img - mean) * cfac + mean

        # gamma (aplicado a los 3 canales)
        g = random.uniform(*gamma_range)
        img = torch.clamp(img, 0.0, 1.0) ** g

        # ruido (solo en cerebro)
        if noise_std and noise_std > 0:
            img = img + torch.randn_like(img) * noise_std * brain

    # ---- 3) FORZAR fondo a 0 SIEMPRE ----
    img = img * brain
    img = torch.clamp(img, 0.0, 1.0)

    return img, mask
