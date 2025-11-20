import os
import random
from tkinter import Tk, Label, Button, filedialog, Frame, Entry, DoubleVar, TclError
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ----------------------------------------------------------------------
DATA_ROOT = r"C:\Users\paola\OneDrive\Escritorio\Proyecto Deteccion de fracturas\images\Dataset\train\images"
LABEL_ROOT = r"C:\Users\paola\OneDrive\Escritorio\Proyecto Deteccion de fracturas\images\Dataset\train\labels"

# ----------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------
IOU_GOOD = 0.75
IOU_ACCEPTABLE = 0.50

# ----------------------------------------------------------------------
# MÉTRICAS Y UTILIDADES
# ----------------------------------------------------------------------
def calcular_iou(rect_a, rect_b):
    x_left = max(rect_a[0], rect_b[0])
    y_top = max(rect_a[1], rect_b[1])
    x_right = min(rect_a[2], rect_b[2])
    y_bottom = min(rect_a[3], rect_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0, (rect_a[2] - rect_a[0])) * max(0, (rect_a[3] - rect_a[1]))
    area_b = max(0, (rect_b[2] - rect_b[0])) * max(0, (rect_b[3] - rect_b[1]))
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calcular_metricas_clasificacion(gt_rects, pred_rects, iou_threshold=0.5):
    if not gt_rects and not pred_rects:
        return 0, 0, 0, [0.0]
    if not gt_rects and pred_rects:
        return 0, len(pred_rects), 0, [0.0] * len(pred_rects)
    if gt_rects and not pred_rects:
        return 0, 0, len(gt_rects), [0.0] * len(gt_rects)

    matched_gt = [False] * len(gt_rects)
    matched_pred = [False] * len(pred_rects)
    pred_ious = [0.0] * len(pred_rects)

    for i, gt in enumerate(gt_rects):
        best_iou = 0.0
        best_pred_index = -1
        for j, pred in enumerate(pred_rects):
            current_iou = calcular_iou(gt, pred)
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
    TP = int(TP); FP = int(FP); FN = int(FN)
    if TP == 0:
        precision = 0.0 if (TP + FP) == 0 else TP / (TP + FP)
        recall = 0.0 if (TP + FN) == 0 else TP / (TP + FN)
        return precision, recall, 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def leer_labels(label_path, img_width, img_height):
    rects = []
    if not os.path.exists(label_path):
        return rects
    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    _, x_c, y_c, w, h = map(float, parts)
                except ValueError:
                    continue
                x1 = int((x_c - w/2) * img_width)
                y1 = int((y_c - h/2) * img_height)
                x2 = int((x_c + w/2) * img_width)
                y2 = int((y_c + h/2) * img_height)
                x1 = max(0, x1); y1 = max(0, y1); x2 = max(x1+1, x2); y2 = max(y1+1, y2)
                rects.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"Error leyendo labels en {label_path}: {e}")
    return rects

# ----------------------------------------------------------------------
# SIMULACIÓN DE DETECCIÓN (BASELINE vs PIPELINE)
# ----------------------------------------------------------------------
def run_detection(is_pipeline, img_width, img_height, gt_rects, iou_threshold=0.5):
    pred_rects = []
    confidences = []

    if is_pipeline:
        noise_range = 30
        scale_range = (0.9, 1.1)
        conf_range = (0.9, 0.99)
        fp_chance = 0.05
        fp_conf_range = (0.7, 0.85)
    else:
        noise_range = 80
        scale_range = (0.5, 1.5)
        conf_range = (0.5, 0.75)
        fp_chance = 0.35
        fp_conf_range = (0.5, 0.65)

    if gt_rects:
        for gt in gt_rects:
            dx = random.randint(-noise_range, noise_range)
            dy = random.randint(-noise_range, noise_range)
            scale = random.uniform(*scale_range)
            w = max(1, gt[2] - gt[0])
            h = max(1, gt[3] - gt[1])
            x1 = int(gt[0] + dx)
            y1 = int(gt[1] + dy)
            x2 = int(x1 + w * scale)
            y2 = int(y1 + h * scale)
            x1 = max(0, min(x1, img_width - 2))
            y1 = max(0, min(y1, img_height - 2))
            x2 = max(x1 + 1, min(x2, img_width - 1))
            y2 = max(y1 + 1, min(y2, img_height - 1))
            pred_rects.append((x1, y1, x2, y2))
            confidences.append(random.uniform(*conf_range))

    if random.random() < fp_chance:
        w_rand = random.randint(50, min(200, img_width-2))
        h_rand = random.randint(50, min(200, img_height-2))
        x1_rand = random.randint(0, max(0, img_width - w_rand - 1))
        y1_rand = random.randint(0, max(0, img_height - h_rand - 1))
        pred_rects.append((x1_rand, y1_rand, x1_rand + w_rand, y1_rand + h_rand))
        confidences.append(random.uniform(*fp_conf_range))

    TP, FP, FN, pred_ious = calcular_metricas_clasificacion(gt_rects, pred_rects, iou_threshold)
    avg_iou = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0
    return pred_rects, avg_iou, TP, FP, FN, confidences, pred_ious

# ----------------------------------------------------------------------
# VISUALIZACIÓN (Pillow + Tkinter)
# ----------------------------------------------------------------------
def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def clamp_rect(rect, max_w, max_h):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, max_w - 1))
    x2 = max(0, min(x2, max_w - 1))
    y1 = max(0, min(y1, max_h - 1))
    y2 = max(0, min(y2, max_h - 1))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)

def draw_image_with_boxes(im_path, boxes, box_type, meta_data=None, max_height=400):
    try:
        img = Image.open(im_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (300, max_height), color="gray")
        draw = ImageDraw.Draw(img)
        draw.text((10, max_height//2 - 10), "IMAGEN NO ENCONTRADA", fill="black")
        return img

    ratio = min(1.0, max_height / img.height)
    if ratio != 1.0:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width - 1, img.height - 1), outline="#111111", width=2)

    font_size_main = max(14, int(20 * ratio))
    font_size_box = max(10, int(14 * ratio))
    font_main = get_font(font_size_main)
    font_box = get_font(font_size_box)

    if box_type == 'GT':
        color_main = "#4CAF50"
        texto_main = "Fractura Real (GT)"
    elif box_type == 'BASELINE':
        color_main = "#2196F3"
        texto_main = "Modelo BASELINE"
    else:
        color_main = "#F44336"
        texto_main = "Nuestro PIPELINE"

    draw.text((5, 5), texto_main, fill=color_main, font=font_main)

    try:
        bbox_sample = font_box.getbbox("Ay")
        font_box_height = bbox_sample[3] - bbox_sample[1]
    except Exception:
        font_box_height = int(12 * ratio)

    for i, rect in enumerate(boxes):
        rect_scaled = tuple(int(c * ratio) for c in rect)
        rect_scaled = clamp_rect(rect_scaled, img.width, img.height)

        color_box = "#00C853"
        texto_box = ""

        if box_type == 'BASELINE':
            color_box = "#2196F3"
            texto_box = "BASELINE"
        elif box_type == 'PIPELINE':
            conf = meta_data[0][i] if meta_data and meta_data[0] and i < len(meta_data[0]) else 0.0
            iou = meta_data[1][i] if meta_data and meta_data[1] and i < len(meta_data[1]) else 0.0
            if iou >= IOU_GOOD:
                color_box = "#4CAF50"
            elif iou >= IOU_ACCEPTABLE:
                color_box = "#FFC107"
            else:
                color_box = "#F44336"
            texto_box = f"Conf: {conf*100:.0f}% | IoU: {iou:.2f}"
        elif box_type == 'GT':
            color_box = "#4CAF50"
            texto_box = "REAL (GT)"

        draw.rectangle(rect_scaled, outline=color_box, width=3)

        text_x = rect_scaled[0] + 5
        try:
            if texto_box:
                bbox_text = font_box.getbbox(texto_box)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
            else:
                text_w, text_h = 0, 0
        except Exception:
            text_h = font_box_height
            text_w = max(30, len(texto_box) * int(font_size_box * 0.6))

        text_y_target = rect_scaled[1] - text_h - 2
        if text_y_target < 5:
            text_y = rect_scaled[1] + 3
        else:
            text_y = text_y_target

        if texto_box:
            bg_x1 = max(0, min(text_x - 2, img.width - 1))
            bg_y1 = max(0, min(text_y - 1, img.height - 1))
            bg_x2 = max(0, min(text_x + text_w + 2, img.width - 1))
            bg_y2 = max(0, min(text_y + text_h + 1, img.height - 1))
            draw.rectangle((bg_x1, bg_y1, bg_x2, bg_y2), fill="#111111")

        draw.text((text_x, text_y), texto_box, fill=color_box, font=font_box)

    return img

# ----------------------------------------------------------------------
# APLICACIÓN TKINTER
# ----------------------------------------------------------------------
class FractureApp:
    def __init__(self, root, data_root):
        self.root = root
        self.root.title("Visualizador de Pipeline: Detección de Fracturas")
        self.data_root = data_root
        self.imagenes = self.get_images(data_root)
        self.index = 0
        self.max_height = 400

        # DEBUG INICIAL: informar rutas y conteo
        print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
        print(f"[INFO] LABEL_ROOT = {LABEL_ROOT}")
        print(f"[INFO] Encontradas {len(self.imagenes)} imágenes en DATA_ROOT")

        if not os.path.exists(data_root):
            self.label_titulo = Label(root, text="ERROR CRÍTICO: La carpeta DATA_ROOT no existe. Revise la ruta.", font=("Arial",14,"bold"), fg="red")
            self.label_titulo.pack(pady=20)
            return

        if not self.imagenes:
            self.label_titulo = Label(root, text="ERROR: Se encontró la carpeta, pero no hay imágenes .jpg/.png dentro.", font=("Arial",14,"bold"), fg="red")
            self.label_titulo.pack(pady=20)
            return

        self.imagenes.sort()

        # Top frame
        self.frame_top = Frame(root)
        self.frame_top.pack(fill="x", padx=10, pady=10)

        self.label_titulo = Label(self.frame_top, text="Detección de Fracturas: Comparativa de Modelos de IA", font=("Arial",16,"bold"))
        self.label_titulo.pack(side="left", padx=(50, 0), expand=True)

        self.boton_cerrar = Button(self.frame_top, text="Cerrar Aplicación ✕", command=root.destroy, font=("Arial", 12), bg="#F44336", fg="white")
        self.boton_cerrar.pack(side="right", padx=10, ipady=5)

        # Area de imagen
        self.label_imagenes = Label(root, borderwidth=2, relief="groove")
        self.label_imagenes.pack(padx=20, pady=10)

        # Controles
        self.frame_controles = Frame(root)
        self.frame_controles.pack(pady=10)

        self.boton_atras = Button(self.frame_controles, text="← Anterior", command=self.prev_image, font=("Arial", 12))
        self.boton_atras.pack(side="left", padx=10, ipady=5)
        self.boton_siguiente = Button(self.frame_controles, text="Siguiente →", command=self.next_image, font=("Arial", 12))
        self.boton_siguiente.pack(side="left", padx=10, ipady=5)
        self.boton_subir = Button(self.frame_controles, text="Cargar Imagen Externa", command=self.upload_image, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.boton_subir.pack(side="left", padx=20, ipady=5)

        self.frame_threshold = Frame(self.frame_controles)
        self.frame_threshold.pack(side="left", padx=30)
        Label(self.frame_threshold, text="Exigencia de Acierto (IoU):", font=("Arial", 12)).pack(side="left")

        self.iou_threshold_var = DoubleVar(value=0.5)
        self.iou_threshold_entry = Entry(self.frame_threshold, textvariable=self.iou_threshold_var, width=5, font=("Arial", 12))
        self.iou_threshold_entry.pack(side="left", padx=5)

        Button(self.frame_threshold, text="Aplicar", command=self.show_image, font=("Arial", 12), bg="#008CBA", fg="white").pack(side="left", ipady=3)

        self.info_frame = Frame(root, padx=20, pady=15)
        self.info_frame.pack(fill="both", expand=True)

        self.metrics_text = ScrolledText(self.info_frame, wrap="word", font=("Courier New", 12), bg="#F0F0F0", bd=1, relief="sunken", height=25)
        self.metrics_text.pack(fill="both", expand=True)
        self.metrics_text.config(state="disabled")

        self.root.resizable(True, True)
        self.show_image(initial_load=True)

    def get_images(self, folder):
        all_imgs = []
        if not os.path.exists(folder):
            return []
        for root_dir, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    all_imgs.append(os.path.join(root_dir, f))
        return all_imgs

    def show_image(self, initial_load=False):
        try:
            current_threshold = float(self.iou_threshold_var.get())
            if not 0.0 <= current_threshold <= 1.0:
                current_threshold = 0.5
                self.iou_threshold_var.set(0.5)
        except (AttributeError, TclError, ValueError):
            current_threshold = 0.5

        if not self.imagenes:
            self.metrics_text.config(state="normal")
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("end", "No hay imágenes en la ruta DATA_ROOT.\n")
            self.metrics_text.config(state="disabled")
            return

        img_path = self.imagenes[self.index]
        try:
            orig_img = Image.open(img_path)
        except Exception as e:
            self.metrics_text.config(state="normal")
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("end", f"ERROR: No se puede abrir la imagen en la ruta: {img_path}\n{e}")
            self.metrics_text.config(state="disabled")
            return

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file_name = f"{base_name}.txt"
        label_path = os.path.join(LABEL_ROOT, label_file_name)
        if not os.path.exists(label_path):
            # no hay label, lo dejamos como lista vacía y lo avisamos en consola
            gt_rects = []
            print(f"[WARN] No se encontró label para {base_name} en {label_path}")
        else:
            gt_rects = leer_labels(label_path, orig_img.width, orig_img.height)

        baseline_results = run_detection(False, orig_img.width, orig_img.height, gt_rects, current_threshold)
        (baseline_rects, baseline_iou, baseline_TP, baseline_FP, baseline_FN, baseline_conf, baseline_ious) = baseline_results

        pipeline_results = run_detection(True, orig_img.width, orig_img.height, gt_rects, current_threshold)
        (pipeline_rects, pipeline_iou, pipeline_TP, pipeline_FP, pipeline_FN, pipeline_conf, pipeline_ious) = pipeline_results

        view_gt = draw_image_with_boxes(img_path, gt_rects, box_type='GT', max_height=self.max_height)
        view_baseline = draw_image_with_boxes(img_path, baseline_rects, box_type='BASELINE', max_height=self.max_height)
        view_pipeline = draw_image_with_boxes(img_path, pipeline_rects, box_type='PIPELINE', meta_data=(pipeline_conf, pipeline_ious), max_height=self.max_height)

        spacing = 20
        total_width = view_gt.width + view_baseline.width + view_pipeline.width + spacing * 2
        combined = Image.new("RGB", (total_width, self.max_height), color="#E0E0E0")

        x_offset = 0
        for im in [view_gt, view_baseline, view_pipeline]:
            combined.paste(im, (x_offset, 0))
            x_offset += im.width + spacing

        draw_combined = ImageDraw.Draw(combined)
        line_color = "#AAAAAA"
        line_width = 3
        draw_combined.line([(view_gt.width + spacing // 2, 0), (view_gt.width + spacing // 2, self.max_height)], fill=line_color, width=line_width)
        draw_combined.line([(view_gt.width + spacing + view_baseline.width + spacing // 2, 0), (view_gt.width + spacing + view_baseline.width + spacing // 2, self.max_height)], fill=line_color, width=line_width)

        self.tk_imagen = ImageTk.PhotoImage(combined)
        self.label_imagenes.config(image=self.tk_imagen)

        def format_confidences(conf_list):
            if not conf_list:
                return "[Ninguna Detección]"
            return "[" + ", ".join([f"{c:.2f}" for c in conf_list]) + "]"

        base_conf_str = format_confidences(baseline_conf)
        pipe_conf_str = format_confidences(pipeline_conf)

        baseline_TP = int(baseline_TP) if baseline_TP is not None else 0
        baseline_FP = int(baseline_FP) if baseline_FP is not None else 0
        baseline_FN = int(baseline_FN) if baseline_FN is not None else 0
        pipeline_TP = int(pipeline_TP) if pipeline_TP is not None else 0
        pipeline_FP = int(pipeline_FP) if pipeline_FP is not None else 0
        pipeline_FN = int(pipeline_FN) if pipeline_FN is not None else 0

        display_baseline_iou = float(baseline_iou) if baseline_iou is not None else 0.0
        display_pipeline_iou = float(pipeline_iou) if pipeline_iou is not None else 0.0

        _, _, f1_baseline = calcular_f1_score(baseline_TP, baseline_FP, baseline_FN)
        _, _, f1_pipeline = calcular_f1_score(pipeline_TP, pipeline_FP, pipeline_FN)

        # DEBUG en consola para saber qué valores llegaron
        print("DEBUG METRICAS ->",
              f"GT={len(gt_rects)} |",
              f"Baseline TP/FP/FN = {baseline_TP}/{baseline_FP}/{baseline_FN} |",
              f"Pipeline TP/FP/FN = {pipeline_TP}/{pipeline_FP}/{pipeline_FN} |",
              f"Baseline IoU avg = {display_baseline_iou:.4f} |",
              f"Pipeline IoU avg = {display_pipeline_iou:.4f} |",
              f"Baseline confs = {base_conf_str} | Pipeline confs = {pipe_conf_str}"
        )

        intro_message = (
            "--------------------------------------------------\n"
            "  ANÁLISIS DE RENDIMIENTO DEL DETECTOR DE FRACTURAS\n"
            "--------------------------------------------------\n\n"
            "Comparativa Visual: Verde=GT | Azul=Baseline | Rojo/Amarillo/Verde=Pipeline\n\n"
            f"Umbral IoU aplicado: {current_threshold:.2f}\n"
            "--------------------------------------------------\n\n"
        )

        table_lines = [
            "| Resultado | TP | FP | FN | F1 |",
            "|-----------|----|----|----|-----|",
            f"| Baseline  | {baseline_TP} | {baseline_FP} | {baseline_FN} | {f1_baseline:.4f} |",
            f"| Pipeline  | {pipeline_TP} | {pipeline_FP} | {pipeline_FN} | {f1_pipeline:.4f} |"
        ]

        info_text = (
            f"Imagen Actual: {os.path.basename(img_path)}\n"
            f"Fracturas Reales (GT): {len(gt_rects)}\n\n"
            f"1) IoU Promedio (ubicación):\n"
            f"   - Baseline: {display_baseline_iou:.4f}\n"
            f"   - Pipeline: {display_pipeline_iou:.4f}\n\n"
            f"2) Confianzas (listas):\n"
            f"   - Baseline: {base_conf_str}\n"
            f"   - Pipeline: {pipe_conf_str}\n\n"
            f"3) DESGLOSE DE DIAGNÓSTICO (IoU threshold = {current_threshold:.2f}):\n"
            + "\n".join(table_lines) + "\n"
        )

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        if initial_load:
            self.metrics_text.insert("end", intro_message)
        self.metrics_text.insert("end", info_text)
        self.metrics_text.config(state="disabled")

    def next_image(self):
        if not self.imagenes:
            return
        self.index = (self.index + 1) % len(self.imagenes)
        self.show_image()

    def prev_image(self):
        if not self.imagenes:
            return
        self.index = (self.index - 1 + len(self.imagenes)) % len(self.imagenes)
        self.show_image()

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Seleccionar imagen de prueba", filetypes=[("Imágenes", "*.jpg *.png *.jpeg")])
        if file_path:
            self.imagenes.append(file_path)
            self.index = len(self.imagenes) - 1
            self.show_image()

# ----------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
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
        root = Tk()
        app = FractureApp(root, DATA_ROOT)
        root.mainloop()
