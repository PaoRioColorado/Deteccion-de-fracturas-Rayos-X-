# 🦴 Detección de fracturas óseas en radiografías

**ISFT 190 – Procesamiento de Imágenes en Python**
**Autora:** Paola Fernanda Dueña

## 📘 Descripción

Este proyecto busca desarrollar un pipeline reproducible para detectar automáticamente si una radiografía presenta fractura o no, utilizando técnicas clásicas de procesamiento de imágenes. El objetivo es construir un sistema educativo que sirva como apoyo al diagnóstico médico, demostrando un flujo completo: entrada → preprocesado → método → evaluación → visualización.

## 🧠 Tecnologías y librerías principales

* Python 3.10
* NumPy: operaciones numéricas y manejo de arrays
* Pandas: manipulación de datos tabulares
* Matplotlib: visualización de gráficos y resultados
* scikit-image: procesamiento de imágenes (filtros, transformaciones, contraste)
* OpenCV: lectura, preprocesado y mejora de imágenes
* scikit-learn: extracción de características y clasificación (SVM, métricas)
* Pillow: carga y manipulación básica de imágenes

> 🔧 Todas las dependencias están detalladas en el archivo `requirements.txt`.

## 📊 Dataset

**FracAtlas Original Dataset**

* Fuente: [Kaggle](https://www.kaggle.com/datasets/mahmudulhasantasin/fracatlas-original-dataset)
* Contiene radiografías clasificadas con y sin fracturas
* Uso con fines académicos únicamente

## 📈 Métricas de evaluación

* Accuracy
* Precision / Recall / F1-Score
* Matriz de confusión

## 🧩 Reproducibilidad

* División de datos train / val / test con semillas fijas
* Resultados guardados en `/results`
* Notebook ejecutable de punta a punta sin errores
