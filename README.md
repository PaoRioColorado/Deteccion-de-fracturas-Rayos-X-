# 🦴 Detección de fracturas óseas en radiografías

**ISFT 190 – Procesamiento de Imágenes en Python**

**Autora:** Paola Fernanda Dueña

---

## 📘 Descripción
Este proyecto desarrolla un **pipeline reproducible** para detectar automáticamente la ubicación de fracturas óseas en radiografías usando bounding boxes.  
El objetivo es construir un sistema educativo y de apoyo visual para instructores y estudiantes, mostrando un flujo completo: **entrada → preprocesado → detección → evaluación → visualización**.

El proyecto compara un **baseline sencillo** con un **pipeline mejorado**, evaluando la calidad de las detecciones mediante métricas objetivas.

---

## 🧠 Tecnologías y librerías principales
- **Python 3.10**
- **NumPy:** operaciones numéricas y manejo de arrays
- **Pandas:** manipulación de datos tabulares
- **Matplotlib:** visualización de gráficos y resultados
- **scikit-image:** procesamiento de imágenes (filtros, transformaciones, contraste)
- **OpenCV:** lectura, preprocesado y mejora de imágenes
- **scikit-learn:** extracción de características y clasificación (baseline)
- **Pillow:** carga y manipulación básica de imágenes
- **Tkinter:** interfaz para demo interactiva
- Todas las dependencias están detalladas en `requirements.txt`

---

## 📊 Dataset
**FracAtlas Original Dataset**  
- Fuente: [Kaggle](https://www.kaggle.com/datasets/mahmudulhasantasin/fracatlas-original-dataset)  
- Contiene radiografías con **fracturas etiquetadas** mediante bounding boxes.  
- Licencia: uso académico permitido (CC BY).  
  
---

## Limitaciones del dataset
El dataset presenta algunas limitaciones comunes en radiografías reales:
- Variabilidad en la calidad de imagen y resolución.
- Diferencias de iluminación y contraste.
- Fracturas sutiles que no siempre son fácilmente visibles.
- Posible desbalance entre imágenes con y sin fractura.

Estas limitaciones pueden afectar la precisión de la detección y explican algunos errores observados.

---

## Baseline y pipeline mejorado
Se implementa un baseline sencillo que simula detecciones con mayor ruido, desplazamientos y falsos positivos.
El pipeline mejorado reduce el ruido, ajusta mejor las bounding boxes y disminuye la cantidad de detecciones erróneas.

Ambos enfoques se comparan utilizando métricas objetivas como IoU promedio y F1-score.

---

## Análisis crítico y trabajo futuro
Si bien el pipeline mejorado muestra mejores métricas que el baseline, el sistema presenta limitaciones importantes:
- La detección es simulada y no corresponde a un modelo entrenado real.
- El desempeño depende fuertemente de la calidad de las anotaciones.
- No se evalúa en un entorno clínico real.

Como trabajo futuro se propone:
- Integrar un detector real basado en deep learning (por ejemplo YOLO o RetinaNet).
- Ampliar el dataset y balancear clases.
- Incorporar validación cruzada y métricas adicionales.

---

## Consideraciones éticas
Este proyecto tiene fines educativos y no debe utilizarse como herramienta diagnóstica clínica.
Las imágenes utilizadas son de acceso académico y no contienen información personal identificable.

---

Pipeline: Imagen → lectura y escalado → simulación de detección → cálculo de métricas → visualización comparativa.



