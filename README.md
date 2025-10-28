# Deteccion-de-fracturas-Rayos-X-
Este es un visualizador de escritorio simple construido con Python y Tkinter, diseñado para comparar la precisión de dos modelos de detección de objetos (un "Baseline" y un "Pipeline Mejorado") en imágenes de rayos X, centrándose en la detección de fracturas.

🌟 Características

Visualización lado a lado de las predicciones del modelo.

Comparación de las cajas delimitadoras (Bounding Boxes) con la verdad fundamental (Ground Truth, GT).

Cálculo y visualización de métricas clave (IoU promedio, TP, FP, FN, Precision, Recall y F1-Score) por imagen.

Capacidad para ajustar el umbral de exigencia de acierto (IoU Threshold) en tiempo real.

Simulación de datos para demostración.

🛠️ Requisitos

Asegúrate de tener Python 3.x instalado.

Dependencias de Python

Necesitarás las siguientes librerías:

Pillow (PIL) para el procesamiento de imágenes.

numpy para los cálculos matemáticos (promedios, min/max).

tkinter (generalmente viene incluido con Python, pero si no, instálalo).

Para instalar las dependencias, utiliza el archivo requirements.txt:

pip install -r requirements.txt


🚀 Uso

1. Configuración de Rutas

Antes de ejecutar, debes modificar las rutas de datos en la parte superior del archivo mejoras_de_gemini.py:
