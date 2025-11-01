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
- Contiene radiografías con **fraturas etiquetadas** mediante bounding boxes.  
- Licencia: uso académico permitido (CC BY).  
- Estructura esperada en el repositorio:  

