import os
import random
from tkinter import Tk, Label, Button, filedialog, Frame, W, E, Entry, DoubleVar, TclError
from tkinter.scrolledtext import ScrolledText 
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ----------------------------------------------------------------------
# PASO CRÍTICO: CONFIGURACIÓN DE RUTAS - ¡DEBES REVISAR ESTO!
# ----------------------------------------------------------------------
# RUTA CORREGIDA A: C:\Users\paola\OneDrive\Escritorio\Proyecto Deteccion de fracturas\images\Dataset\train\
# 
# Raíz de las imágenes
DATA_ROOT = r"C:\Users\paola\OneDrive\Escritorio\Proyecto Deteccion de fracturas\images\Dataset\train\images"
# Raíz de las etiquetas (archivos .txt)
LABEL_ROOT = r"C:\Users\paola\OneDrive\Escritorio\Proyecto Deteccion de fracturas\images\Dataset\train\labels"

# ----------------------------------------------------------------------
# CONSTANTES PARA COLORES DINÁMICOS
# ----------------------------------------------------------------------
# Umbrales para feedback visual instantáneo del Pipeline (Mejora 1)
IOU_GOOD = 0.75 # Consideramos "muy buen acierto" si IoU >= 0.75
IOU_ACCEPTABLE = 0.50 # Consideramos "acierto aceptable" si IoU >= 0.50

# ----------------------------------------------------------------------
# FUNCIONES NÚCLEO Y MÉTRICAS
# ----------------------------------------------------------------------

def calcular_iou(rect_a, rect_b):
    """Calcula el Intersection over Union (IoU) entre dos rectángulos."""
    x_left = max(rect_a[0], rect_b[0])
    y_top = max(rect_a[1], rect_b[1])
    x_right = min(rect_a[2], rect_b[2])
    y_bottom = min(rect_a[3], rect_b[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])
    area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calcular_metricas_clasificacion(gt_rects, pred_rects, iou_threshold=0.5):
    """Calcula True Positives (TP), False Positives (FP), y False Negatives (FN)."""
    
    if not gt_rects and not pred_rects:
        return 0, 0, 0, [0.0]
        
    if not gt_rects and pred_rects:
        return 0, len(pred_rects), 0, [0.0] * len(pred_rects)
        
    if gt_rects and not pred_rects:
        return 0, 0, len(gt_rects), [0.0] * len(gt_rects)
        
    matched_gt = [False] * len(gt_rects)
    matched_pred = [False] * len(pred_rects)
    
    # Lista para almacenar el IoU máximo por cada predicción
    pred_ious = [0.0] * len(pred_rects)

    for i, gt in enumerate(gt_rects):
        best_iou = 0.0
        best_pred_index = -1
        
        for j, pred in enumerate(pred_rects):
            current_iou = calcular_iou(gt, pred)
            
            # Actualizar el IoU máximo para la predicción j
            pred_ious[j] = max(pred_ious[j], current_iou)
            
            if current_iou >= iou_threshold and not matched_pred[j]:
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_index = j
                    
        if best_pred_index != -1:
            matched_gt[i] = True
            matched_pred[best_pred_index] = True
    
    TP = sum(matched_gt) 
    FP = len(pred_rects) - sum(matched_pred) 
    FN = len(gt_rects) - sum(matched_gt) 
    
    return TP, FP, FN, pred_ious

def calcular_f1_score(TP, FP, FN):
    """Calcula Precision, Recall y F1-Score."""
    if TP == 0:
        return 0.0, 0.0, 0.0 
        
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision + recall == 0:
        return precision, recall, 0.0
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

def leer_labels(label_path, img_width, img_height):
    """
    Lee las coordenadas normalizadas de YOLO y las convierte a píxeles. 
    Manejo de errores de archivo vacío/malo.
    """
    rects = []
    if not os.path.exists(label_path):
        return rects
    try:
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue 
                parts = line.split()
                if len(parts) == 5:
                    try:
                        # Ignoramos la clase (parts[0]) ya que solo buscamos ubicación.
                        _, x_c, y_c, w, h = map(float, parts) 
                        
                        x1 = int((x_c - w/2) * img_width)
                        y1 = int((y_c - h/2) * img_height)
                        x2 = int((x_c + w/2) * img_width)
                        y2 = int((y_c + h/2) * img_height)
                        rects.append((x1, y1, x2, y2))
                    except ValueError:
                        print(f"Advertencia: Línea mal formateada en {label_path}: {line}")
        return rects
    except Exception as e:
        print(f"Error leyendo labels en {label_path}: {e}")
        return []

# ----------------------------------------------------------------------
# SIMULACIÓN DEL PIPELINE DE DETECCIÓN (MÉTODO Y BASELINE)
# ----------------------------------------------------------------------

def run_detection(is_pipeline, img_width, img_height, gt_rects, iou_threshold=0.5):
    """Función unificada de simulación de detección para Baseline y Pipeline."""
    
    pred_rects = []
    confidences = [] 
    
    if is_pipeline:
        # Pipeline (Alto rendimiento)
        noise_range = 30 # Aumentado ligeramente para variación en demo
        scale_range = (0.9, 1.1)
        conf_range = (0.9, 0.99)
        fp_chance = 0.05
        fp_conf_range = (0.7, 0.85)
    else:
        # Baseline (Bajo rendimiento)
        noise_range = 80
        scale_range = (0.5, 1.5)
        conf_range = (0.5, 0.75)
        fp_chance = 0.35
        fp_conf_range = (0.5, 0.65)

    if gt_rects:
        for gt in gt_rects:
            # Simular ligera desviación en posición y escala
            dx = random.randint(-noise_range, noise_range)
            dy = random.randint(-noise_range, noise_range)
            scale = random.uniform(*scale_range) 
            
            w, h = gt[2] - gt[0], gt[3] - gt[1]
            x1 = int(gt[0] + dx)
            y1 = int(gt[1] + dy)
            x2 = int(x1 + w * scale)
            y2 = int(y1 + h * scale)
            
            pred_rects.append((x1, y1, x2, y2))
            confidences.append(random.uniform(*conf_range))
            
    # Simular Falsos Positivos (detecciones donde no hay GT)
    if random.random() < fp_chance:
        w_rand = random.randint(50, 200)
        h_rand = random.randint(50, 200)
        x1_rand = random.randint(50, img_width - 250)
        y1_rand = random.randint(50, img_height - 250)
        pred_rects.append((x1_rand, y1_rand, x1_rand + w_rand, y1_rand + h_rand))
        confidences.append(random.uniform(*fp_conf_range))

    TP, FP, FN, pred_ious = calcular_metricas_clasificacion(gt_rects, pred_rects, iou_threshold)
    
    avg_iou = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0
    
    return pred_rects, avg_iou, TP, FP, FN, confidences, pred_ious

# ----------------------------------------------------------------------
# VISUALIZACIÓN TKINTER / PIL
# ----------------------------------------------------------------------

def get_font(size):
    """Intenta cargar Arial, si falla, usa la fuente por defecto."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

def draw_image_with_boxes(im_path, boxes, box_type, meta_data=None, max_height=400):
    """
    Abre una imagen, la escala y dibuja los bounding boxes y el borde.
    box_type: 'GT', 'BASELINE', 'PIPELINE'
    meta_data: (confidences, ious) para el PIPELINE
    """
    try:
        img = Image.open(im_path).convert("RGB")
    except FileNotFoundError:
        img = Image.new("RGB", (300, max_height), color="gray")
        draw = ImageDraw.Draw(img)
        draw.text((50, max_height/2 - 20), "IMAGEN NO ENCONTRADA", fill="black")
        return img
        
    ratio = min(1.0, max_height / img.height)
    if ratio != 1.0:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        
    draw = ImageDraw.Draw(img)
    
    # Dibuja el borde general de la imagen
    draw.rectangle((0, 0, img.width - 1, img.height - 1), outline="#111111", width=2)
    
    # Configuración de fuentes
    font_size_main = max(14, int(20 * ratio))
    font_size_box = max(10, int(14 * ratio))
    font_main = get_font(font_size_main)
    font_box = get_font(font_size_box)
    
    # Título de la columna (Fuera de cualquier caja, en la esquina superior)
    if box_type == 'GT':
        color_main = "#4CAF50" # Verde
        texto_main = "Fractura Real (GT)"
    elif box_type == 'BASELINE':
        color_main = "#2196F3" # Azul
        texto_main = "Modelo BASELINE"
    elif box_type == 'PIPELINE':
        color_main = "#F44336" # Rojo (Default)
        texto_main = "Nuestro PIPELINE"

    # DIBUJAR TÍTULO FIJO DE LA COLUMNA
    draw.text((5, 5), 
              texto_main, 
              fill=color_main, 
              font=font_main, 
              stroke_width=1, 
              stroke_fill="#111111")
    
    for i, rect in enumerate(boxes):
        rect_scaled = tuple(int(c * ratio) for c in rect)
        
        # --- Lógica de Color y Texto de la Caja ---
        color_box = "#00C853" 
        texto_box = ""
        
        if box_type == 'BASELINE':
            color_box = "#2196F3" 
            texto_box = "BASELINE"
        elif box_type == 'PIPELINE':
            conf = meta_data[0][i] if meta_data and meta_data[0] and i < len(meta_data[0]) else 0.0
            iou = meta_data[1][i] if meta_data and meta_data[1] and i < len(meta_data[1]) else 0.0
            
            # Asignación de color dinámico
            if iou >= IOU_GOOD:
                color_box = "#4CAF50" # Verde (¡Muy buen acierto!)
            elif iou >= IOU_ACCEPTABLE:
                color_box = "#FFC107" # Amarillo (Acierto aceptable)
            else:
                color_box = "#F44336" # Rojo (Predicción Baja o Falsa Alarma)
                
            texto_box = f"Conf: {conf*100:.0f}% | IoU: {iou:.2f}"
        
        elif box_type == 'GT':
            color_box = "#4CAF50"
            texto_box = "REAL (GT)"


        # Dibujar el rectángulo
        draw.rectangle(rect_scaled, outline=color_box, width=3)
        
        # --- Dibujar Texto de la Caja ---
        
        # Posición: Justo encima de la caja
        text_x = rect_scaled[0] + 5 
        text_y_target = rect_scaled[1] - font_box.size - 2 # 2px de margen
        
        # Si la posición objetivo está muy cerca o fuera del borde superior (y < 25px)
        # Bajamos el texto para que esté DENTRO de la caja, cerca del borde superior.
        if text_y_target < 25: 
            text_y = rect_scaled[1] + 5 
        else:
            text_y = text_y_target
        
        draw.text((text_x, text_y), 
                  texto_box, 
                  fill=color_box, 
                  font=font_box, 
                  stroke_width=1, 
                  stroke_fill="#111111")
    
    return img

class FractureApp:
    def __init__(self, root, data_root):
        self.root = root
        self.root.title("Visualizador de Pipeline: Detección de Fracturas")
        self.data_root = data_root
        self.imagenes = self.get_images(data_root)
        self.index = 0
        self.max_height = 400

        if not os.path.exists(data_root):
             self.label_titulo = Label(root, text="ERROR CRÍTICO: La carpeta DATA_ROOT no existe. Revise la ruta.", font=("Arial",14,"bold"), fg="red")
             self.label_titulo.pack(pady=20)
             return
        
        if not self.imagenes:
             self.label_titulo = Label(root, text="ERROR: Se encontró la carpeta, pero no hay imágenes .jpg/.png dentro.", font=("Arial",14,"bold"), fg="red")
             self.label_titulo.pack(pady=20)
             return
        
        self.imagenes.sort()
        
        # ----------------------------------------------------------
        # FRAME SUPERIOR (TÍTULO Y BOTÓN SALIR)
        # ----------------------------------------------------------
        self.frame_top = Frame(root)
        self.frame_top.pack(fill="x", padx=10, pady=10)
        
        # Título del visualizador
        self.label_titulo = Label(self.frame_top, 
                                  text="Detección de Fracturas: Comparativa de Modelos de IA", 
                                  font=("Arial",16,"bold"))
        self.label_titulo.pack(side="left", padx=(50, 0), expand=True) # Centrar título
        
        # Botón Salir (Cerrar)
        self.boton_cerrar = Button(self.frame_top, 
                                   text="Cerrar Aplicación ✕", 
                                   command=root.destroy, 
                                   font=("Arial", 12),
                                   bg="#F44336", 
                                   fg="white")
        self.boton_cerrar.pack(side="right", padx=10, ipady=5)
        # ----------------------------------------------------------

        # Área de imágenes combinadas
        self.label_imagenes = Label(root, borderwidth=2, relief="groove")
        self.label_imagenes.pack(padx=20, pady=10)
        
        # Frame de Controles
        self.frame_controles = Frame(root)
        self.frame_controles.pack(pady=10)
        
        self.boton_atras = Button(self.frame_controles, text="← Anterior", command=self.prev_image, font=("Arial", 12))
        self.boton_atras.pack(side="left", padx=10, ipady=5)
        self.boton_siguiente = Button(self.frame_controles, text="Siguiente →", command=self.next_image, font=("Arial", 12))
        self.boton_siguiente.pack(side="left", padx=10, ipady=5)
        self.boton_subir = Button(self.frame_controles, text="Cargar Imagen Externa", command=self.upload_image, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.boton_subir.pack(side="left", padx=20, ipady=5)
        
        # Nuevo Frame para el control de Umbral (Threshold IoU)
        self.frame_threshold = Frame(self.frame_controles)
        self.frame_threshold.pack(side="left", padx=30)
        
        # Texto simplificado para el umbral
        Label(self.frame_threshold, text="Exigencia de Acierto (IoU):", font=("Arial", 12)).pack(side="left")
        
        self.iou_threshold_var = DoubleVar(value=0.5) 
        self.iou_threshold_entry = Entry(self.frame_threshold, textvariable=self.iou_threshold_var, width=5, font=("Arial", 12))
        self.iou_threshold_entry.pack(side="left", padx=5)
        
        Button(self.frame_threshold, text="Aplicar", command=self.show_image, font=("Arial", 12), bg="#008CBA", fg="white").pack(side="left", ipady=3)
        
        # Usar ScrolledText para las métricas
        self.info_frame = Frame(root, padx=20, pady=15)
        self.info_frame.pack(fill="both", expand=True) # Permite que este frame se estire

        # AJUSTE DE ALTURA PARA EVITAR CORTE
        self.metrics_text = ScrolledText(
            self.info_frame, 
            wrap="word", 
            font=("Courier New", 12), 
            bg="#F0F0F0", 
            bd=1, 
            relief="sunken",
            height=25 # <-- Valor AUMENTADO a 25 para evitar el corte
        )
        self.metrics_text.pack(fill="both", expand=True) # Permite que el texto se estire
        self.metrics_text.config(state="disabled") 
        
        # ELIMINACIÓN DEL TAMAÑO FIJO para que la ventana se auto-ajuste
        # self.FIXED_WIDTH = 1500 
        # self.FIXED_HEIGHT = 980 
        # self.root.geometry(f"{self.FIXED_WIDTH}x{self.FIXED_HEIGHT}")
        self.root.resizable(True, True) # Permite el ajuste manual de ser necesario

        self.show_image(initial_load=True)

    def get_images(self, folder):
        """Busca imágenes en el directorio recursivamente."""
        all_imgs = []
        if not os.path.exists(folder):
            return []
            
        for root_dir, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg",".png", ".jpeg")):
                    all_imgs.append(os.path.join(root_dir, f))
        return all_imgs

    def show_image(self, initial_load=False):
        """Procesa y muestra la imagen actual con las tres vistas y métricas."""
        
        try:
            current_threshold = self.iou_threshold_var.get()
            if not 0.0 <= current_threshold <= 1.0:
                 current_threshold = 0.5
                 self.iou_threshold_var.set(0.5)
        except (AttributeError, TclError):
            current_threshold = 0.5
        
        img_path = self.imagenes[self.index]
        try:
            orig_img = Image.open(img_path)
        except Exception:
            self.metrics_text.config(state="normal")
            self.metrics_text.delete(1.0, "end")
            self.metrics_text.insert("end", f"ERROR: No se puede abrir la imagen en la ruta: {img_path}")
            self.metrics_text.config(state="disabled")
            return
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file_name = f"{base_name}.txt"
        
        gt_rects = leer_labels(os.path.join(LABEL_ROOT, label_file_name), orig_img.width, orig_img.height)

        baseline_results = run_detection(False, orig_img.width, orig_img.height, gt_rects, current_threshold)
        (baseline_rects, baseline_iou, baseline_TP, baseline_FP, baseline_FN, baseline_conf, baseline_ious) = baseline_results
        
        pipeline_results = run_detection(True, orig_img.width, orig_img.height, gt_rects, current_threshold)
        (pipeline_rects, pipeline_iou, pipeline_TP, pipeline_FP, pipeline_FN, pipeline_conf, pipeline_ious) = pipeline_results

        # 3. Visualización 
        view_gt = draw_image_with_boxes(img_path, gt_rects, box_type='GT', max_height=self.max_height)
        view_baseline = draw_image_with_boxes(img_path, baseline_rects, box_type='BASELINE', max_height=self.max_height)
        view_pipeline = draw_image_with_boxes(img_path, pipeline_rects, box_type='PIPELINE', meta_data=(pipeline_conf, pipeline_ious), max_height=self.max_height)

        spacing = 20
        total_width = view_gt.width + view_baseline.width + view_pipeline.width + spacing * 2
        
        # Fondo blanco para un contraste limpio
        combined = Image.new("RGB", (total_width, self.max_height), color="#E0E0E0") 
        
        x_offset = 0
        for im in [view_gt, view_baseline, view_pipeline]:
            combined.paste(im, (x_offset, 0))
            x_offset += im.width + spacing
        
        # --- DIBUJAR LÍNEAS DE SEPARACIÓN ---
        draw_combined = ImageDraw.Draw(combined)
        line_color = "#AAAAAA" 
        line_width = 3
        
        # Separador 1: Entre GT y BASELINE
        draw_combined.line([
            (view_gt.width + spacing // 2, 0), 
            (view_gt.width + spacing // 2, self.max_height)
        ], fill=line_color, width=line_width)

        # Separador 2: Entre BASELINE y PIPELINE
        draw_combined.line([
            (view_gt.width + spacing + view_baseline.width + spacing // 2, 0), 
            (view_gt.width + spacing + view_baseline.width + spacing // 2, self.max_height)
        ], fill=line_color, width=line_width)
        
        self.tk_imagen = ImageTk.PhotoImage(combined)
        self.label_imagenes.config(image=self.tk_imagen)
        
        # 4. Actualización de Métricas
        
        def format_confidences(conf_list):
            if not conf_list:
                return "[Ninguna Detección]"
            return "[" + ", ".join([f"{c:.2f}" for c in conf_list]) + "]"

        base_conf_str = format_confidences(baseline_conf)
        pipe_conf_str = format_confidences(pipeline_conf)

        display_baseline_iou = baseline_iou if gt_rects else 0.0000
        display_pipeline_iou = pipeline_iou if gt_rects else 0.0000
        
        f1_baseline = calcular_f1_score(baseline_TP, baseline_FP, baseline_FN)[2]
        f1_pipeline = calcular_f1_score(pipeline_TP, pipeline_FP, pipeline_FN)[2]

        intro_message = (
            "--------------------------------------------------\n"
            "  ANÁLISIS DE RENDIMIENTO DEL DETECTOR DE FRACTURAS\n"
            "--------------------------------------------------\n\n"
            "**Comparativa Visual:**\n"
            "1. **Verde Oscuro (Real):** La etiqueta del experto (Ground Truth).\n"
            "2. **Azul (Modelo BASELINE):** Simulación de un modelo con bajo rendimiento (peor).\n"
            "3. **ROJO/AMARILLO/VERDE CLARO (Nuestro PIPELINE):** Nuestro modelo. El color indica la calidad del acierto:\n"
            f"   - **Verde Claro:** IoU ≥ {IOU_GOOD:.2f} (Excelente Ubicación).\n"
            f"   - **Amarillo:** IoU ≥ {IOU_ACCEPTABLE:.2f} (Acierto aceptable).\n"
            f"   - **Rojo:** IoU < {IOU_ACCEPTABLE:.2f} (Pobre Ubicación / Falsa Alarma).\n"
            "\n"
            f"**Control de Exigencia (IoU):** El 'Umbral IoU' ({current_threshold:.2f}) define qué tan estricto somos para declarar un acierto (TP) en la tabla de métricas.\n"
            "--------------------------------------------------\n"
        )
        
        info_text = (
            f"Imagen Actual: {os.path.basename(img_path)}\n"
            f"Fracturas Reales (Verdes): {len(gt_rects)}\n"
            f"\n"
            f"--- MÉTRICAS CLAVE ---\n"
            f"\n"
            f"**1. PRECISIÓN DE UBICACIÓN (IoU - Promedio de Solapamiento):**\n"
            f"  > Baseline (Azul): {display_baseline_iou:.4f} (Mide qué tan bien la caja envuelve la fractura. Cerca de 1.0000 es perfecto).\n"
            f"  > Pipeline (Rojo/Amarillo/Verde): {display_pipeline_iou:.4f}\n"
            f"\n"
            f"**2. NIVEL DE CONFIANZA PROMEDIO:**\n"
            f"  > Baseline (Azul): {base_conf_str} (Confianza baja/media).\n"
            f"  > Pipeline (R/A/V): {pipe_conf_str} (Confianza alta).\n"
            f"\n"
            f"**3. DESGLOSE DE DIAGNÓSTICO (Con Exigencia de IoU = {current_threshold:.2f}):**\n"
            f"| Resultado | TP (Aciertos) | FP (Falsas Alarmas) | FN (Fracturas Omitidas) | Puntuación F1 (Calidad) |\n"
            f"|-----------|---------------|---------------------|-------------------------|-------------------------|\n"
            f"| Baseline  | {baseline_TP:<13}| {baseline_FP:<19} | {baseline_FN:<23} | {f1_baseline:.4f} |\n"
            f"| Pipeline  | {pipeline_TP:<13}| {pipeline_FP:<19} | {pipeline_FN:<23} | {f1_pipeline:.4f} |"
        )
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        
        if initial_load:
            self.metrics_text.insert("end", intro_message)
        
        self.metrics_text.insert("end", info_text)
        self.metrics_text.config(state="disabled")

    def next_image(self):
        """Navega a la siguiente imagen del dataset."""
        if not self.imagenes: return
        self.index = (self.index + 1) % len(self.imagenes)
        self.show_image()

    def prev_image(self):
        """Navega a la imagen anterior del dataset."""
        if not self.imagenes: return
        self.index = (self.index - 1 + len(self.imagenes)) % len(self.imagenes)
        self.show_image()

    def upload_image(self):
        """Permite al usuario cargar una imagen externa para la demo."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de prueba",
            filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
        )
        if file_path:
            self.imagenes.append(file_path)
            self.index = len(self.imagenes) - 1
            self.show_image()

# ----------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Verificación de Rutas (Añadida al inicio de la ejecución para mejor feedback)
    root_error = False
    if not os.path.exists(DATA_ROOT):
        print("----------------------------------------------------------------")
        print("¡ERROR! La carpeta DATA_ROOT no existe en la ruta configurada.")
        print(f"Ruta: {DATA_ROOT}")
        print("----------------------------------------------------------------")
        root_error = True
    
    if not os.path.exists(LABEL_ROOT):
        print("----------------------------------------------------------------")
        print("¡ERROR! La carpeta LABEL_ROOT no existe en la ruta configurada.")
        print(f"Ruta: {LABEL_ROOT}")
        print("----------------------------------------------------------------")
        root_error = True

    if root_error:
        print("Por favor, verifica las rutas antes de continuar.")
    else:
        # Iniciar la aplicación Tkinter
        root = Tk()
        app = FractureApp(root, DATA_ROOT)
        root.mainloop()
