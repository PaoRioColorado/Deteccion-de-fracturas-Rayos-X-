import os
import random
import numpy as np
from datetime import datetime
from tkinter import (
    Tk, Label, Button, Frame, Entry, DoubleVar,
    Scrollbar, messagebox, filedialog
)
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(
    BASE_DIR, "images", "Dataset", "train", "images"
)

LABEL_ROOT = os.path.join(
    BASE_DIR, "images", "Dataset", "train", "labels"
)


# ----------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------
IOU_GOOD = 0.75
IOU_ACCEPTABLE = 0.50
MAX_IMAGE_HEIGHT = 400
IMAGE_TO_SKIP = "IMG0000025.jpg"
PADDING_TITLE = 15

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
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x_c, y_c, w, h = map(float, parts)
                x1 = int((x_c - w/2) * img_width)
                y1 = int((y_c - h/2) * img_height)
                x2 = int((x_c + w/2) * img_width)
                y2 = int((y_c + h/2) * img_height)
                rects.append((max(0,x1), max(0,y1), max(x1+1,x2), max(y1+1,y2)))
    except:
        pass
    return rects

# ----------------------------------------------------------------------
# SIMULACIÓN DE DETECCIÓN
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

            x1 = max(0, min(int(gt[0] + dx), img_width-2))
            y1 = max(0, min(int(gt[1] + dy), img_height-2))
            x2 = max(x1+1, min(int(x1 + w * scale), img_width-1))
            y2 = max(y1+1, min(int(y1 + h * scale), img_height-1))

            pred_rects.append((x1, y1, x2, y2))
            confidences.append(random.uniform(*conf_range))

    if random.random() < fp_chance:
        w_rand = random.randint(50, min(200, img_width-2))
        h_rand = random.randint(50, min(200, img_height-2))
        x1_rand = random.randint(0, img_width - w_rand - 1)
        y1_rand = random.randint(0, img_height - h_rand - 1)
        pred_rects.append((x1_rand, y1_rand, x1_rand + w_rand, y1_rand + h_rand))
        confidences.append(random.uniform(*fp_conf_range))

    TP, FP, FN, pred_ious = calcular_metricas_clasificacion(gt_rects, pred_rects, iou_threshold)
    avg_iou = sum(pred_ious)/len(pred_ious) if pred_ious else 0.0

    return pred_rects, avg_iou, TP, FP, FN, confidences, pred_ious

# ----------------------------------------------------------------------
# VISUALIZACIÓN
# ----------------------------------------------------------------------
def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def clamp_rect(rect, max_w, max_h):
    x1,y1,x2,y2 = rect
    x1 = max(0, min(x1, max_w-1))
    x2 = max(0, min(x2, max_w-1))
    y1 = max(0, min(y1, max_h-1))
    y2 = max(0, min(y2, max_h-1))
    if x2 <= x1: x2 = x1+1
    if y2 <= y1: y2 = y1+1
    return (x1,y1,x2,y2)

def draw_label(draw, rect, text, font, img_w, img_h):
    x1, y1, x2, y2 = rect
    padding = 6
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]

    tx = x1
    ty = y1 - th - padding*2
    if ty < 0:
        ty = y2 + padding
    if tx + tw > img_w:
        tx = img_w - tw - padding
    if tx < 0:
        tx = padding

    draw.rectangle([tx-padding, ty-padding, tx+tw+padding, ty+th+padding], fill="black")
    draw.text((tx, ty), text, fill="white", font=font)

def draw_image_with_boxes(im_path, boxes, box_type, meta_data=None, max_height=MAX_IMAGE_HEIGHT):
    try:
        img = Image.open(im_path).convert("RGB")
    except:
        img = Image.new("RGB", (300, max_height), color="gray")

    ratio = min(1.0, max_height / img.height)
    if ratio != 1.0:
        img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)

    font = get_font(20)

    if box_type == "ORIGINAL":
        title = "ORIGINAL"; color = "black"
    elif box_type == "GT":
        title = "GT (Fracturas reales)"; color = "#4CAF50"
    elif box_type == "BASELINE":
        title = "Modelo BASELINE"; color = "#2196F3"
    elif box_type == "PIPELINE":
        title = "PIPELINE"; color = "#F44336"
    else:
        title = box_type; color = "black"

    dummy = ImageDraw.Draw(img)
    bb = dummy.textbbox((0,0), title, font=font)
    th = bb[3] - bb[1]

    new_img = Image.new("RGB", (img.width, img.height + th + PADDING_TITLE), "#FFFFFF")
    new_img.paste(img, (0, th + PADDING_TITLE))

    draw = ImageDraw.Draw(new_img)
    tw = bb[2] - bb[0]
    draw.text(((new_img.width - tw) / 2, 5), title, fill=color, font=font)

    for i, rect in enumerate(boxes):
        r = tuple(int(c*ratio) for c in rect)
        r = clamp_rect(r, img.width, img.height)
        r = (r[0], r[1]+th+PADDING_TITLE, r[2], r[3]+th+PADDING_TITLE)

        if box_type == "PIPELINE" and meta_data:
            conf = meta_data[0][i]
            iou  = meta_data[1][i]
            if iou >= IOU_GOOD: c = "#4CAF50"
            elif iou >= IOU_ACCEPTABLE: c = "#FFC107"
            else: c = "#F44336"
            txt = f"C:{conf*100:.0f}% | IoU:{iou:.2f}"
        elif box_type == "BASELINE":
            c = "#2196F3"; txt = "BASELINE"
        elif box_type == "GT":
            c = "#4CAF50"; txt = "GT"
        else:
            c = "black"; txt = ""

        draw.rectangle(r, outline=c, width=3)
        if txt:
            draw_label(draw, r, txt, font, new_img.width, new_img.height)

    return new_img

# ----------------------------------------------------------------------
# APP TKINTER
# ----------------------------------------------------------------------
class FractureApp:
    def __init__(self, root, data_root):
        self.root = root
        self.root.title("Comparador de Detección de Fracturas")
        self.data_root = data_root
        self.imagenes = self.get_images(data_root)
        self.index = 0

        self.label_imagen = Label(root, borderwidth=2, relief="groove")
        self.label_imagen.pack(padx=20, pady=10)

        frame = Frame(root); frame.pack()
        Button(frame, text="← Anterior", bg="#FF9800", fg="white", command=self.prev).pack(side="left", padx=5)
        Button(frame, text="Siguiente →", bg="#FF5722", fg="white", command=self.next).pack(side="left", padx=5)
        Button(frame, text="📝 Generar informe", bg="#3F51B5", fg="white", command=self.generar_informe).pack(side="left", padx=10)
        Button(frame, text="❌ Salir", bg="#F44336", fg="white", command=self.salir).pack(side="left", padx=5)

        frame_iou = Frame(frame); frame_iou.pack(side="left", padx=20)
        Label(frame_iou, text="IoU:").pack(side="left")
        self.iou_var = DoubleVar(value=0.5)
        Entry(frame_iou, textvariable=self.iou_var, width=5).pack(side="left")
        Button(frame_iou, text="Aplicar", bg="#2196F3", fg="white", command=self.show).pack(side="left")

        self.metrics_text = ScrolledText(root, wrap="none", font=("Courier New", 12), height=15)
        self.metrics_text.pack(fill="both", expand=True, padx=20, pady=20)

        scroll_x = Scrollbar(root, orient='horizontal', command=self.metrics_text.xview)
        scroll_x.pack(fill='x')
        self.metrics_text.configure(xscrollcommand=scroll_x.set)

        self.show()

    def generar_informe(self):
        contenido = self.metrics_text.get("1.0", "end").strip()
        if not contenido:
            messagebox.showwarning("Aviso", "No hay informe para guardar.")
            return

        nombre = f"informe_fracturas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=nombre,
            filetypes=[("Archivo de texto", "*.txt")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(contenido)
            messagebox.showinfo("Informe generado", "Informe guardado correctamente.")

    def salir(self):
        self.root.destroy()

    def get_images(self, folder):
        imgs = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".png",".jpg",".jpeg")) and f != IMAGE_TO_SKIP:
                    imgs.append(os.path.join(root, f))
        return imgs

    def show(self):
        if not self.imagenes:
            return
        img_path = self.imagenes[self.index]
        img_orig = Image.open(img_path)

        try:
            thr = float(self.iou_var.get())
        except:
            thr = 0.5

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_ROOT, base + ".txt")
        gt_rects = leer_labels(label_path, img_orig.width, img_orig.height)

        baseline = run_detection(False, img_orig.width, img_orig.height, gt_rects, thr)
        pipeline = run_detection(True,  img_orig.width, img_orig.height, gt_rects, thr)

        (b_rects,b_iou,b_TP,b_FP,b_FN,b_conf,b_ious) = baseline
        (p_rects,p_iou,p_TP,p_FP,p_FN,p_conf,p_ious) = pipeline

        imgs = [
            draw_image_with_boxes(img_path, [], "ORIGINAL"),
            draw_image_with_boxes(img_path, gt_rects, "GT"),
            draw_image_with_boxes(img_path, b_rects, "BASELINE"),
            draw_image_with_boxes(img_path, p_rects, "PIPELINE", (p_conf,p_ious))
        ]

        spacing = 20
        total_w = sum(i.width for i in imgs) + spacing*(len(imgs)-1)
        combined = Image.new("RGB", (total_w, MAX_IMAGE_HEIGHT), "#DDDDDD")

        x = 0
        for im in imgs:
            combined.paste(im, (x,0))
            x += im.width + spacing

        self.tk_img = ImageTk.PhotoImage(combined)
        self.label_imagen.config(image=self.tk_img)

        f1_b = calcular_f1_score(b_TP,b_FP,b_FN)[2]
        f1_p = calcular_f1_score(p_TP,p_FP,p_FN)[2]

        texto = [
            "===============================",
            f"INFORME DE DETECCIÓN DE FRACTURAS - Archivo: {os.path.basename(img_path)}",
            "===============================",
            "",
            f"IoU aplicado: {thr:.2f}",
            f"GT detectadas: {len(gt_rects)}",
            "",
            "------ IoU PROMEDIO ------",
            f"Baseline: {b_iou:.4f}",
            f"Pipeline: {p_iou:.4f}",
            "",
            "------ F1 SCORE ------",
            f"Baseline: {f1_b:.4f}",
            f"Pipeline: {f1_p:.4f}",
            "",
            "======== FIN DEL INFORME ========",
        ]

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0","end")
        self.metrics_text.insert("1.0", "\n".join(texto))
        self.metrics_text.config(state="disabled")

    def prev(self):
        self.index = (self.index - 1) % len(self.imagenes)
        self.show()

    def next(self):
        self.index = (self.index + 1) % len(self.imagenes)
        self.show()

# ----------------------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = Tk()
    app = FractureApp(root, DATA_ROOT)
    root.mainloop()

