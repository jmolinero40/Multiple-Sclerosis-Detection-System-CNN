# -*- coding: utf-8 -*-
"""
TEST: Reporte global y por paciente (F1 y Dice medio por imagen)
- Con post-procesado: eliminar componentes pequeñas (remove_small_components_2d)
- Modelo 3 canales (in_channels=3)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Spyder/Windows OMP

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.ndimage import label

#from dataset import MSLesionDataset
from dataset_multimodal import MSLesionDatasetMultimodal


# ---------- 0. Modelo (misma UNet que en train) ----------

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class DoubleConv(nn.Module):
    """(Conv2d -> GroupNorm -> ReLU) x2  (+ SE opcional)"""
    def __init__(self, in_channels, out_channels, use_se: bool = True, se_reduction: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, use_se: bool = True, se_reduction: int = 16):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels, use_se=use_se, se_reduction=se_reduction)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, use_se=use_se, se_reduction=se_reduction)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, use_se=use_se, se_reduction=se_reduction)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, use_se=use_se, se_reduction=se_reduction)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, use_se=use_se, se_reduction=se_reduction)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, use_se=use_se, se_reduction=se_reduction)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, use_se=use_se, se_reduction=se_reduction)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, use_se=use_se, se_reduction=se_reduction)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, use_se=use_se, se_reduction=se_reduction)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        xb = self.bottleneck(self.pool(x4))

        x = self.up4(xb)
        x = torch.cat([x4, x], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)  # logits


# ---------- Dataset wrapper: devuelve también el nombre del slice ----------
class MSLesionDatasetWithName(MSLesionDatasetMultimodal):
    def __getitem__(self, idx):
        img_t, mask_t = super().__getitem__(idx)
        flair_path = self.samples[idx][0]   # <- primer elemento es el FLAIR path
        return img_t, mask_t, flair_path.name


def parse_patient_key(name: str) -> str:
    """
    name ejemplo: nature_slice_p01_z039.npy
    key devuelto: "nature_p01"
    """
    dataset = name.split("_slice_")[0]
    pid = name.split("_slice_")[1].split("_")[0]  # p01
    return f"{dataset}_{pid}"


def remove_small_components_2d(bin_mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    """
    Elimina componentes conexas pequeñas en una máscara binaria 2D.
    bin_mask: (H,W) {0,1}
    """
    bin_mask = bin_mask.astype(bool)
    structure = np.ones((3, 3), dtype=np.int32)  # 8-conectividad
    labeled, n = label(bin_mask, structure=structure)
    if n == 0:
        return bin_mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())  # counts[0]=fondo
    remove = counts < min_size
    remove[0] = False
    cleaned = bin_mask.copy()
    cleaned[remove[labeled]] = False
    return cleaned.astype(np.uint8)


def parse_patient_key(name: str) -> str:
    # name: nature_FLAIR_slice_p01_z039.npy
    parts = name.replace(".npy","").split("_")
    dataset = parts[0]          # "nature"
    pid = parts[-2]             # "p01"
    return f"{dataset}_{pid}"   # "nature_p01"


# ---------- 1. Rutas, dispositivo y test loader ----------
base_dir = Path(r"C:\Users\jmoli\Desktop\TFG mates\Datos\Archivos procesados\Unified_splits_multimodal")
modelo_path = Path(r"C:\Users\jmoli\Desktop\TFG mates\Modelos\unet2d_nature_multimodal_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

test_ds = MSLesionDatasetWithName(base_dir, split="test", transform=None)
print(f"[MSLesionDataset] Split='test' → {len(test_ds)} muestras encontradas.")


test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)


# ---------- 2. Cargar modelo ----------
model = UNet2D(in_channels=3, out_channels=1, base_channels=32).to(device)
state_dict = torch.load(modelo_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Modelo cargado desde:", modelo_path)


# ---------- 3. Evaluación en test (global + por paciente) ----------
def eval_en_test(umbral=0.75, min_size=20):
    eps = 1e-8

    # Global pixel stats
    tp = fp = fn = tn = 0

    # Global dice medio por imagen
    dice_sum = 0.0
    num_imgs = 0

    # Por paciente
    per_pat = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "dice_sum": 0.0, "num_imgs": 0
    })

    with torch.no_grad():
        for imgs, masks, names in test_loader:
            imgs = imgs.to(device).float()    # (B,3,H,W)
            masks = masks.to(device).float()  # (B,1,H,W)

            logits = model(imgs)
            probs = torch.sigmoid(logits)

            # predicción binaria + POSPROCESADO (quitar puntitos)
            preds_np = (probs > umbral).detach().cpu().numpy().astype(np.uint8)  # (B,1,H,W)

            for i in range(preds_np.shape[0]):
                preds_np[i, 0] = remove_small_components_2d(preds_np[i, 0], min_size=min_size)

            preds = torch.from_numpy(preds_np).to(device=device, dtype=torch.float32)  # (B,1,H,W)

            # --- métricas pixel a pixel (GLOBAL) ---
            preds_f = preds.view(-1)
            masks_f = masks.view(-1)

            tp_b = torch.sum((preds_f == 1) & (masks_f == 1)).item()
            fp_b = torch.sum((preds_f == 1) & (masks_f == 0)).item()
            fn_b = torch.sum((preds_f == 0) & (masks_f == 1)).item()
            tn_b = torch.sum((preds_f == 0) & (masks_f == 0)).item()

            tp += tp_b
            fp += fp_b
            fn += fn_b
            tn += tn_b

            # --- Dice medio por imagen (GLOBAL) ---
            B = masks.shape[0]
            preds_b = preds.view(B, -1)
            masks_b = masks.view(B, -1)

            inter = (preds_b * masks_b).sum(dim=1)
            union = preds_b.sum(dim=1) + masks_b.sum(dim=1)
            dice_batch = (2 * inter + eps) / (union + eps)  # (B,)

            dice_sum += dice_batch.sum().item()
            num_imgs += B

            # --- DESGLOSE POR PACIENTE ---
            for i in range(B):
                key = parse_patient_key(names[i])

                pi = preds[i].view(-1)
                mi = masks[i].view(-1)

                tp_i = torch.sum((pi == 1) & (mi == 1)).item()
                fp_i = torch.sum((pi == 1) & (mi == 0)).item()
                fn_i = torch.sum((pi == 0) & (mi == 1)).item()
                tn_i = torch.sum((pi == 0) & (mi == 0)).item()

                per_pat[key]["tp"] += tp_i
                per_pat[key]["fp"] += fp_i
                per_pat[key]["fn"] += fn_i
                per_pat[key]["tn"] += tn_i

                per_pat[key]["dice_sum"] += dice_batch[i].item()
                per_pat[key]["num_imgs"] += 1

    # ---------- Métricas globales ----------
    total = tp + tn + fp + fn + eps
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    dice_pixel = (2 * tp) / (2 * tp + fp + fn + eps)
    dice_por_imagen = dice_sum / (num_imgs + eps)

    confusion = np.array([[tn, fp],
                          [fn, tp]], dtype=np.int64)

    # ---------- Métricas por paciente ----------
    per_patient_results = []
    for key, d in per_pat.items():
        tp_p, fp_p, fn_p, tn_p = d["tp"], d["fp"], d["fn"], d["tn"]

        precision_p = tp_p / (tp_p + fp_p + eps)
        recall_p = tp_p / (tp_p + fn_p + eps)
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p + eps)

        dice_img_p = d["dice_sum"] / (d["num_imgs"] + eps)

        per_patient_results.append({
            "patient": key,
            "num_slices": d["num_imgs"],
            "f1": f1_p,
            "dice_img": dice_img_p,
            "tp": tp_p, "fp": fp_p, "fn": fn_p, "tn": tn_p
        })

    per_patient_results.sort(key=lambda x: x["patient"])

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "dice_pixel": dice_pixel,
        "dice_por_imagen": dice_por_imagen,
        "confusion": confusion,
        "per_patient": per_patient_results
    }


if __name__ == "__main__":
    umbral = 0.7
    min_size = 10

    res = eval_en_test(umbral=umbral, min_size=min_size)

    print(f"\n=== Resultados en TEST (umbral {umbral:.2f} | min_size {min_size}) ===")
    print(f"TP: {res['tp']:.0f}  FP: {res['fp']:.0f}  FN: {res['fn']:.0f}  TN: {res['tn']:.0f}")
    print(f"Accuracy      : {res['accuracy']:.4f}")
    print(f"Precision     : {res['precision']:.4f}")
    print(f"Sensibilidad  : {res['recall']:.4f}")
    print(f"Especificidad : {res['specificity']:.4f}")
    print(f"F1-score      : {res['f1']:.4f}")
    print(f"Dice (pixel)  : {res['dice_pixel']:.4f}")
    print(f"Dice (medio por imagen): {res['dice_por_imagen']:.4f}")

    print("\n--- Por paciente (F1 y Dice medio por imagen) ---")
    for r in res["per_patient"]:
        print(f"{r['patient']} | slices={r['num_slices']:4d} | F1={r['f1']:.4f} | Dice_img={r['dice_img']:.4f}"
              f" | TP={r['tp']:.0f} FP={r['fp']:.0f} FN={r['fn']:.0f} TN={r['tn']:.0f}")

    print("\nMatriz de confusión (valores absolutos, pixel a pixel):")
    print(res["confusion"])

