# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:58:26 2025

@author: jmoli
"""
import time

from pathlib import Path

import torch
import torch.nn as nn #capas, funciones de pérdida, etc.
import torch.optim as optim #optimizadores como Adam o SGD
from torch.utils.data import DataLoader #A partir de nuestra clase de datos hace los batches
import torch.nn.functional as F

#from dataset_z import MSLesionDataset, train_augment
from dataset_multimodal import MSLesionDatasetMultimodal, train_augment

# ---------- MODELO: U-Net 2D sencilla ----------

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2  (+ SE opcional)"""

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
        self.use_se = use_se
        self.se = SEBlock(out_channels, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)
        return x



class UNet2D(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, use_se: bool = True, se_reduction: int = 16):
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
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        xb = self.bottleneck(self.pool(x4))

        # Decoder con skip connections
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

        logits = self.out_conv(x) 
        return logits


# ---------- MÉTRICA Y FUNCIÓN DE PÉRDIDA ----------

def dice_coefficient(pred, target, eps=1e-6):
    """
    Dice sobre tensores (B,1,H,W) en [0,1].
    pred: probabilidades después de sigmoid
    target: máscara 0/1
    """
    #contiguous se hace solamente para asgeurar formato
    #view comprime cada foto de los n batches en n filas de pixeles 
    pred = pred.contiguous().view(pred.shape[0], -1) 
    target = target.contiguous().view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()



class BCEDiceLoss(nn.Module):
    """
    Combinación sencilla:
    - BCEWithLogitsLoss (estabilidad numérica)
    - Dice loss
    """
    def __init__(self, bce_weight=0.7, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice_loss = 1.0 - dice_coefficient(probs, target)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss



class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (atención por canales).
    x: (B, C, H, W)  -> repondera canales con pesos en (B, C, 1, 1)
    """
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


# ---------- ENTRENAMIENTO BÁSICO ----------

if __name__ == "__main__":
    inicio = time.perf_counter()

    # 1) Datasets y DataLoaders
    base_dir = Path(r"C:\Users\jmoli\Desktop\TFG mates\Datos\Archivos procesados\Unified_splits_multimodal")

    batch_size = 8
    num_epochs = 15

    #train_ds = MSLesionDataset(base_dir, split="train", transform=train_augment)
    #val_ds   = MSLesionDataset(base_dir, split="val")
    train_ds = MSLesionDatasetMultimodal(base_dir, split="train", transform=train_augment, dataset_name="nature")
    val_ds   = MSLesionDatasetMultimodal(base_dir, split="val",   transform=None,        dataset_name="nature")


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # 2) Modelo, dispositivo, pérdida, optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    model = UNet2D(in_channels=3, out_channels=1, base_channels=32).to(device)

    pos_weight = torch.tensor([3.0], device=device)
    criterion  = BCEDiceLoss(bce_weight=0.7, pos_weight=pos_weight)
    
    """
    criterion = AdaptiveBCEFocalDiceLoss(
    bce_weight=0.7,
    pos_weight=pos_weight,     # mantiene tu idea de penalizar positivos
    focal_alpha=0.75,          # peso clase lesión
    gamma_start=0.0,           # empieza como BCE (focal casi apagada)
    gamma_end=2.0,             # focal “real” al final
    ramp_epochs=6              # en 6 épocas migras BCE -> focal
    )
    """

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Carpeta de modelos
    carpeta_modelos = Path(r"C:\Users\jmoli\Desktop\TFG mates\Modelos")
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    # Best + Early stopping (misma métrica: val_dice)
    best_val_dice = 0.0
    patience = 6
    epochs_no_improve = 0

    # 3) Bucle de entrenamiento
    for epoch in range(1, num_epochs + 1):
        # --- Entrenamiento ---
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs  = imgs.to(device)   # (B,3,H,W)
            masks = masks.to(device)  # (B,1,H,W)

            optimizer.zero_grad()
            logits = model(imgs)      # (B,1,H,W)
            #loss = criterion(logits, masks, epoch) 
            loss = criterion(logits, masks)
            loss.backward()
            
            #evitar explosion del gradiente
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        # --- Validación ---
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)   # (B,3,H,W)
                masks = masks.to(device)  # (B,1,H,W)

                logits = model(imgs)
                #loss = criterion(logits, masks, epoch)
                loss = criterion(logits, masks)

                val_loss_sum += loss.item() * imgs.size(0)

                probs = torch.sigmoid(logits)
                val_dice_sum += dice_coefficient(probs, masks).item() * imgs.size(0)

        val_loss = val_loss_sum / len(val_ds)
        val_dice = val_dice_sum / len(val_ds)

        # Scheduler: UNA vez por epoch, con la métrica final
        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}"
        )

        # Guardar mejor modelo + early stopping (alineados)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = carpeta_modelos / "recoge_datos"
            torch.save(model.state_dict(), best_path)
            print(f"✅ Nuevo mejor modelo (val_dice={best_val_dice:.4f}) guardado en: {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping: no mejora en {patience} epochs.")
                break

    fin = time.perf_counter()
    duracion = fin - inicio
    print(f"\nTiempo total de ejecución: {duracion/60:.2f} minutos ({duracion:.1f} segundos)")
