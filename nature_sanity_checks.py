# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 12:18:21 2025

@author: jmoli
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Ruta base al dataset Mendeley_data
ruta = Path(r"C:\Users\jmoli\Desktop\TFG mates\Datos")

ruta_nature = ruta / "Archivos en bruto" / "nature(grande)" / "MSLesSeg Dataset" /\
    "MSLesSeg Dataset" / "unified"
    

# Vía de carpetas y archivos
def rutas_paciente(p):
    carpeta_p = ruta_nature / f"P{p}" / "T1"
    flair = carpeta_p / f"P{p}_T1_FLAIR.nii.gz"    
    mask  = carpeta_p / f"P{p}_T1_MASK.nii.gz" 
    return flair, mask

def comprueba_paciente(p):
    flair_path, mask_path = rutas_paciente(p)
    print(f"\n=== Paciente {p} ===")
    print("FLAIR:", flair_path)
    print("Mask :", mask_path)

    # 1) ¿Existen los archivos?
    if not flair_path.exists():
        print("  ⚠ No se encuentra la FLAIR")
        return None
    if not mask_path.exists():
        print("  ⚠ No se encuentra la máscara")
        return None

    # 2) Cargar imágenes
    flair_img = nib.load(str(flair_path))
    mask_img  = nib.load(str(mask_path))

    flair_data = flair_img.get_fdata()
    mask_data  = mask_img.get_fdata()

    # 3) Comprobar shapes
    same_shape = (flair_data.shape == mask_data.shape)
    print("  Shape FLAIR :", flair_data.shape)
    print("  Shape MASK  :", mask_data.shape)
    print("  ¿Misma shape?:", same_shape)

    # 4) ¿Máscara binaria?
    valores = np.unique(mask_data)
    print("  Valores únicos en máscara:", valores)

    # Permitimos ligeros flotantes tipo 0.0, 1.0
    binaria = set(np.round(valores).tolist()).issubset({0, 1})
    print("  ¿Máscara binaria (0/1)?", binaria)

    # 5) ¿Hay alguna lesión marcada?
    voxeles_lesion = (mask_data > 0.5).sum()
    print("  Nº de voxeles con lesión:", int(voxeles_lesion))

    return {
        "paciente": p,
        "shape": flair_data.shape,
        "same_shape": same_shape,
        "binaria": binaria,
        "voxeles_lesion": int(voxeles_lesion),
    }


"""
resumen = []
for p in range(1, 76):
    info = comprueba_paciente(p)
    if info is not None:
        resumen.append(info)

print("\nResumen rápido:")
print(f"Pacientes comprobados: {len(resumen)}")
print("Pacientes sin lesión:", [r["paciente"] for r in resumen if r["voxeles_lesion"] == 0])
print("Pacientes con shapes distintas:", [r["paciente"] for r in resumen if not r["same_shape"]])
print("Pacientes con máscara no binaria:", [r["paciente"] for r in resumen if not r["binaria"]])

shapes = []
spacings = []

for p in range(1, 61):
    
    carpeta_p = ruta_nature / f"P{p}" / "T1"
    ruta_flair = carpeta_p / f"P{p}_T1_FLAIR.nii.gz"  
    
    img = nib.load(str(ruta_flair))
    shapes.append(img.shape)
    spacings.append(img.header.get_zooms())

print("Shapes distintos:", set(shapes))
print("Espaciados distintos:", set(spacings))
"""

#########################################################
# AQUÍ HACEMOS EL ESCALADO Y SECCIONAMIENTO DE LOS DATOS
#########################################################

# Ruta donde guardarás los slices procesados
ruta_out = ruta / "Archivos procesados" / "Nature_slices_res"
ruta_out.mkdir(parents=True, exist_ok=True)

target_size = (192, 224)  # (width, height)

def procesa_paciente_nature(p):
    flair_path, mask_path = rutas_paciente(p)
    print(f"\n=== Procesando Patient-{p} ===")

    if not flair_path.exists() or not mask_path.exists():
        print("  ⚠ Falta FLAIR o máscara, se salta el paciente")
        return

    # 1) Cargar volúmenes
    img_flair = nib.load(str(flair_path))
    img_mask  = nib.load(str(mask_path))

    data_flair = img_flair.get_fdata()
    data_mask  = img_mask.get_fdata()

    # 2) Normalizar FLAIR a [0,1] por volumen
    min_val = data_flair.min()
    max_val = data_flair.max()
    data_flair = (data_flair - min_val) / (max_val - min_val + 1e-8)

    # 3) Crear carpeta de salida del paciente
    out_p = ruta_out / f"Patient-{p}" / "slices_Z"
    out_p.mkdir(parents=True, exist_ok=True)

    # 4) Recorrer cortes axiales (eje Z)
    Z = data_flair.shape[2]
    print(f"  Volumen shape: {data_flair.shape}, nº cortes Z = {Z}")

    for k in range(Z):
        slice_img = data_flair[:, :, k]
        slice_mask = data_mask[:, :, k]

        # 5) Reescalar al tamaño objetivo
        # imagen: bilinear, máscara: nearest
        slice_img_resized = cv2.resize(slice_img.astype(np.float32), target_size,
                                       interpolation=cv2.INTER_LINEAR)
        slice_mask_resized = cv2.resize(slice_mask.astype(np.uint8), target_size,
                                        interpolation=cv2.INTER_NEAREST)

        # 6) Guardar como .npy (más cómodo para la red)
        nombre_img  = out_p / f"img_p{p:02d}_z{k:03d}.npy"
        nombre_mask = out_p / f"mask_p{p:02d}_z{k:03d}.npy"

        np.save(nombre_img,  slice_img_resized)
        np.save(nombre_mask, slice_mask_resized)

    print(f"  Guardados cortes en: {out_p}")


for p in range(1, 76):
    
    procesa_paciente_nature(p)
    
    
shapes = []
spacings = []

for p in range(1, 76):
    
    carpeta_p = ruta_out / f"Patient-{p}" / "slices_Z"
    ruta_flair = carpeta_p / f"img_p{p:02d}_z000.npy"
    
    img = np.load(str(ruta_flair))
    shapes.append(img.shape)

print("Shapes distintos:", set(shapes))

if len(set(shapes))==1:
    print("El redimensionado ha sido efectivo")

