# Multiple-Sclerosis-Detection-System-CNN
CNN system for segmentation of multiple sclerosis lesions

Segmentación Multimodal de Lesiones de Esclerosis Múltiple (U-Net 2D)
Este repositorio contiene el código fuente para el preprocesamiento, entrenamiento y evaluación de un modelo de red neuronal convolucional (U-Net 2D con bloques Squeeze-and-Excitation) diseñado para segmentar lesiones de Esclerosis Múltiple en imágenes de resonancia magnética (MRI). El modelo utiliza un enfoque multimodal de 3 canales, combinando secuencias FLAIR, T1 y T2.

**Requisitos Previos**
Asegúrate de tener un entorno de Python configurado (se recomienda Python 3.8 o superior). Las principales librerías necesarias para ejecutar este proyecto son:

torch y torchvision

numpy

nibabel (para la lectura de volúmenes médicos .nii.gz)

opencv-python (cv2 para el reescalado de imágenes)

scipy (para el posprocesado de componentes conexas)

matplotlib

**Configuración de Rutas (¡FUNDAMENTAL!)**
Antes de ejecutar cualquier script, debes modificar las rutas base definidas en el código para que apunten a los directorios de tu máquina local. Busca variables como ruta, base, base_dir o carpeta_modelos en las primeras líneas de los scripts (actualmente apuntan a C:\Users\jmoli\Desktop\TFG mates\...).

**Flujo de Ejecución (Pipeline)**
El proyecto está diseñado para ejecutarse secuencialmente siguiendo estos pasos:

**1. Preprocesamiento de Volúmenes (Sanity Checks y Slicing)**
Script: nature_sanity_checks.py

Este script carga los volúmenes 3D originales en formato NIfTI, realiza comprobaciones de integridad (sanity checks), normaliza las intensidades entre 0 y 1, redimensiona los cortes axiales a 192x224 y los guarda como matrices 2D en formato .npy.

Ejecución: python nature_sanity_checks.py

Salida: Carpetas por paciente con slices individuales en .npy.

**2. Creación de Particiones (Train, Val, Test)**
Script: splits_secuencias.py

Toma los cortes generados en el paso anterior y los divide en conjuntos de entrenamiento, validación y prueba según los rangos de pacientes definidos. Además, filtra los cortes que no contienen suficiente información del cerebro (imágenes muy oscuras) y realiza un submuestreo de los cortes negativos (sin lesión) para balancear los datos de entrenamiento.

Ejecución: python splits_secuencias.py

Salida: Directorio multimodal con subcarpetas train, val y test, cada una con sus respectivas carpetas de Slices y Masks.

**3. Entrenamiento del Modelo**
Script: train_unet_simple.py
(Dependencia: dataset_multimodal.py)

Inicia el proceso de entrenamiento de la U-Net 2D. Utiliza el MSLesionDatasetMultimodal para cargar los 3 canales (FLAIR, T1, T2) y aplica técnicas de data augmentation en tiempo real. Entrena utilizando una función de pérdida combinada (BCE + Dice Loss).

Ejecución: python train_unet_simple.py

Salida: El modelo entrenado se evalúa en el conjunto de validación al final de cada época. Los pesos del mejor modelo se guardan automáticamente en la carpeta de modelos especificada.

**4. Evaluación y Pruebas**
Script: test_unet_simple.py

Carga los pesos del mejor modelo entrenado y lo evalúa sobre el conjunto de test. Aplica un posprocesado dinámico para eliminar componentes conexas excesivamente pequeñas (ruido).

Ejecución: python test_unet_simple.py

Salida: Imprime en consola un reporte detallado con métricas globales (Accuracy, Precision, Sensibilidad, Especificidad, F1-score, Dice) y un desglose del rendimiento (F1 y Dice) por cada paciente individual, además de la matriz de confusión.
