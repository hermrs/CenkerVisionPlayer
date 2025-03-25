#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CenkerVision - YOLO tabanlı video oynatıcı
Bu uygulama, videolarda nesne tespiti yapmak için YOLO modelini kullanır.
"""

import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import queue

class CenkerVision:
    def __init__(self, root):
        self.root = root
        self.root.title("CenkerVision - YOLO Tabanlı Video Oynatıcı")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Değişkenler
        self.cap = None
        self.video_path = None
        self.is_playing = False
        self.model = None
        self.detect_objects = False
        self.frame_count = 0
        self.current_frame = 0
        self.play_thread = None
        self.stop_thread = False
        self.custom_models = []  # Özel modelleri saklamak için
        self.conf_threshold = 0.25  # Varsayılan confidence threshold değeri
        self.iou_threshold = 0.45  # Varsayılan IOU threshold değeri
        self.display_mode = "normal"  # normal, confidence, boxes_only, censored
        self.frame_queue = queue.Queue(maxsize=5)  # Frame'leri saklamak için queue
        self.processing = False  # İşleme durumu
        self.seek_lock = threading.Lock()  # Video karelerini güvenli şekilde sıçratmak için kilit
        
        # FPS sayacı için değişkenler
        self.fps = 0
        self.frame_times = []
        self.fps_update_interval = 1.0  # FPS güncelleme aralığı (saniye)
        self.last_fps_update = time.time()
        
        # Özel modelleri saklamak için dizini kontrol et ve oluştur
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Mevcut özel modelleri yükle
        self.load_custom_models()
        
        # Ana çerçeve
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video görüntüleme alanı
        self.video_frame = ttk.Frame(main_frame, borderwidth=2, relief="sunken")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        self.canvas.create_text(500, 350, text="Video burada görüntülenecek", fill="white", font=('Arial', 14))
        
        # Alt panel çerçevesi
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Sol kontrol paneli
        control_frame = ttk.LabelFrame(bottom_frame, text="Temel Kontroller")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Dosya seçme butonu
        self.browse_btn = ttk.Button(control_frame, text="Video Seç", command=self.browse_video)
        self.browse_btn.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)
        
        # Oynat/Duraklat butonu
        self.play_btn = ttk.Button(control_frame, text="Oynat", command=self.toggle_play)
        self.play_btn.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)
        
        # İleri/Geri butonları
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)
        
        self.prev_frame_btn = ttk.Button(btn_frame, text="◀ 10 Kare", command=lambda: self.jump_frames(-10))
        self.prev_frame_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.next_frame_btn = ttk.Button(btn_frame, text="10 Kare ▶", command=lambda: self.jump_frames(10))
        self.next_frame_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Orta panel - YOLO ayarları
        yolo_frame = ttk.LabelFrame(bottom_frame, text="YOLO Ayarları")
        yolo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # YOLO modeli seçme
        model_frame = ttk.Frame(yolo_frame)
        model_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="YOLO Modeli:").pack(side=tk.LEFT, padx=5)
        
        default_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        all_models = default_models + self.custom_models
        
        self.model_var = tk.StringVar(value="yolov8n.pt")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=all_models, width=30)
        self.model_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Özel model yükleme butonu
        self.add_model_btn = ttk.Button(model_frame, text="Özel Model Ekle", command=self.add_custom_model)
        self.add_model_btn.pack(side=tk.RIGHT, padx=5)
        
        # Nesne tespiti onay kutusu
        detect_frame = ttk.Frame(yolo_frame)
        detect_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.detect_var = tk.BooleanVar(value=False)
        self.detect_checkbox = ttk.Checkbutton(detect_frame, text="Nesne Tespiti", 
                                            variable=self.detect_var, command=self.toggle_detection)
        self.detect_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Eşik değerleri ve görüntüleme modları
        threshold_frame = ttk.Frame(yolo_frame)
        threshold_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Confidence Threshold
        conf_frame = ttk.Frame(threshold_frame)
        conf_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=5)
        self.conf_value_label = ttk.Label(conf_frame, text="25%")
        self.conf_value_label.pack(side=tk.RIGHT, padx=5)
        
        self.conf_slider = ttk.Scale(conf_frame, from_=0, to=100, 
                                 orient=tk.HORIZONTAL, command=self.update_conf_threshold)
        self.conf_slider.set(25)  # 0.25 -> 25%
        self.conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # IOU Threshold (NMS)
        iou_frame = ttk.Frame(threshold_frame)
        iou_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        ttk.Label(iou_frame, text="Overlap (IOU) Threshold:").pack(side=tk.LEFT, padx=5)
        self.iou_value_label = ttk.Label(iou_frame, text="45%")
        self.iou_value_label.pack(side=tk.RIGHT, padx=5)
        
        self.iou_slider = ttk.Scale(iou_frame, from_=0, to=100, 
                                orient=tk.HORIZONTAL, command=self.update_iou_threshold)
        self.iou_slider.set(45)  # 0.45 -> 45%
        self.iou_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Görüntüleme modları
        display_frame = ttk.Frame(yolo_frame)
        display_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(display_frame, text="Görüntüleme Modu:").pack(side=tk.LEFT, padx=5)
        
        self.display_mode_var = tk.StringVar(value="normal")
        self.display_modes = [
            ("Normal", "normal"),
            ("Sadece Kutular", "boxes_only"),
            ("Güven Skorları", "confidence"),
            ("Sansürlü", "censored")
        ]
        
        display_subframe = ttk.Frame(display_frame)
        display_subframe.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        for text, mode in self.display_modes:
            ttk.Radiobutton(display_subframe, text=text, variable=self.display_mode_var, 
                          value=mode, command=self.update_display_mode).pack(side=tk.LEFT, padx=10)
        
        # İlerleme çubuğu çerçevesi
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # İlerleme çubuğu
        self.time_label_start = ttk.Label(slider_frame, text="0:00")
        self.time_label_start.pack(side=tk.LEFT, padx=5)
        
        self.progress_slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                      orient=tk.HORIZONTAL, command=self.slider_changed)
        self.progress_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_slider.bind("<ButtonRelease-1>", self.slider_released)
        
        self.time_label_end = ttk.Label(slider_frame, text="0:00")
        self.time_label_end.pack(side=tk.LEFT, padx=5)
        
        # Durum çubuğu
        self.status_label = ttk.Label(main_frame, text="Hazır", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # UI güncelleme zamanlayıcısını başlat
        self.check_queue()
        
        # Pencere kapatıldığında
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # Arayüz oluşturulduktan sonra modeli yükle
        self.root.after(100, lambda: self.load_yolo_model("yolov8n.pt"))
    
    def check_queue(self):
        """Frame queue'yu kontrol et ve görüntüle"""
        try:
            if not self.frame_queue.empty():
                frame, current_frame = self.frame_queue.get_nowait()
                self.update_ui(frame, current_frame)
                self.frame_queue.task_done()
        except queue.Empty:
            pass
        finally:
            # 10ms sonra tekrar kontrol et
            self.root.after(10, self.check_queue)
    
    def load_custom_models(self):
        """Özel modelleri yükle"""
        if os.path.exists(self.models_dir):
            self.custom_models = [f for f in os.listdir(self.models_dir) if f.endswith('.pt')]
    
    def add_custom_model(self):
        """Özel model ekleme"""
        file_path = filedialog.askopenfilename(
            title="YOLO Model Dosyası Seç (.pt)", 
            filetypes=[("YOLO Model", "*.pt"), ("Tüm Dosyalar", "*")]
        )
        
        if not file_path:
            return
        
        # Dosyayı models dizinine kopyala
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(self.models_dir, file_name)
        
        try:
            # Eğer aynı isimde bir dosya varsa kullanıcıya sor
            if os.path.exists(dest_path):
                overwrite = messagebox.askyesno(
                    "Dosya Zaten Var",
                    "{} isimli model zaten mevcut. Üzerine yazmak istiyor musunuz?".format(file_name)
                )
                if not overwrite:
                    return
            
            # Dosyayı kopyala (aslında taşıma işlemi değil)
            import shutil
            shutil.copy2(file_path, dest_path)
            
            # Özel modeller listesini güncelle
            if file_name not in self.custom_models:
                self.custom_models.append(file_name)
            
            # Combobox'ı güncelle
            all_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"] + self.custom_models
            self.model_combo['values'] = all_models
            
            # Yeni eklenen modeli seç
            self.model_var.set(file_name)
            self.load_yolo_model(file_name)
            
            messagebox.showinfo("Model Eklendi", "{} modeli başarıyla eklendi.".format(file_name))
            
        except Exception as e:
            messagebox.showerror("Model Ekleme Hatası", 
                               "Model eklenirken bir hata oluştu: {}".format(e))
    
    def update_conf_threshold(self, value):
        """Confidence threshold değerini güncelle"""
        value = float(value)
        self.conf_threshold = value / 100.0  # 0-100 -> 0-1
        self.conf_value_label.config(text="{}%".format(int(value)))
        
        # Değişikliği log'a yazdır
        print(f"Confidence threshold değeri güncellendi: {self.conf_threshold:.2f}")
        
        # Eğer video durdurulmuşsa ve mevcut bir kare varsa, güncellenmiş değerlerle yeniden işle
        if self.cap is not None and not self.is_playing and hasattr(self, 'current_processed_frame'):
            self.status_label.config(text=f"Confidence threshold: {self.conf_threshold:.2f}, yeniden işleniyor...")
            self.root.update()
            processed_frame = self.process_frame(self.current_processed_frame)
            self.update_ui(processed_frame, self.current_frame)
            self.status_label.config(text=f"Confidence: {self.conf_threshold:.2f}, IOU: {self.iou_threshold:.2f}")
    
    def update_iou_threshold(self, value):
        """IOU threshold değerini güncelle"""
        value = float(value)
        self.iou_threshold = value / 100.0  # 0-100 -> 0-1
        self.iou_value_label.config(text="{}%".format(int(value)))
        
        # Değişikliği log'a yazdır
        print(f"IOU threshold değeri güncellendi: {self.iou_threshold:.2f}")
        
        # Eğer video durdurulmuşsa ve mevcut bir kare varsa, güncellenmiş değerlerle yeniden işle
        if self.cap is not None and not self.is_playing and hasattr(self, 'current_processed_frame'):
            self.status_label.config(text=f"IOU threshold: {self.iou_threshold:.2f}, yeniden işleniyor...")
            self.root.update()
            processed_frame = self.process_frame(self.current_processed_frame)
            self.update_ui(processed_frame, self.current_frame)
            self.status_label.config(text=f"Confidence: {self.conf_threshold:.2f}, IOU: {self.iou_threshold:.2f}")
    
    def update_display_mode(self):
        """Görüntüleme modunu güncelle"""
        self.display_mode = self.display_mode_var.get()
    
    def load_yolo_model(self, model_name):
        """YOLO modelini yükle"""
        try:
            # status_label kullanılabilirliğini kontrol et
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Model yükleniyor: {}".format(model_name))
                self.root.update()  # UI'yi hemen güncelle
            
            # Özel model yolunu kontrol et
            if model_name in self.custom_models:
                model_path = os.path.join(self.models_dir, model_name)
                self.model = YOLO(model_path)
            else:
                self.model = YOLO(model_name)
                
            print("{} modeli başarıyla yüklendi.".format(model_name))
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text="{} modeli başarıyla yüklendi.".format(model_name))
                
        except Exception as e:
            error_msg = "Model yüklenirken hata oluştu: {}".format(e)
            print(error_msg)
            messagebox.showerror("Model Hatası", error_msg)
            self.model = None
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Model yükleme hatası!")
    
    def on_model_change(self, event=None):
        """Model değiştiğinde"""
        self.load_yolo_model(self.model_var.get())
    
    def browse_video(self):
        """Video dosyası seçme diyalogu"""
        video_path = filedialog.askopenfilename(
            title="Video Dosyası Seç", 
            filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov"), ("Tüm Dosyalar", "*")]
        )
        
        if video_path:
            self.video_path = video_path
            self.load_video(video_path)
    
    def load_video(self, video_path):
        """Videoyu yükle ve hazırla"""
        if self.cap is not None:
            self.stop_play_thread()
            self.cap.release()
        
        self.status_label.config(text="Video yükleniyor...")
        self.root.update()
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Hatası", "Video dosyası açılamadı!")
            self.status_label.config(text="Video yükleme hatası!")
            return
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_slider.config(to=self.frame_count)
        self.current_frame = 0
        
        # İlk kareyi göster
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        
        # Video süresini hesapla ve göster
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = self.frame_count / fps
        self.time_label_end.config(text=self.format_time(duration))
        
        self.play_btn.config(text="Oynat")
        self.is_playing = False
        
        video_name = os.path.basename(video_path)
        self.status_label.config(text="Video hazır: {} ({} kare)".format(video_name, self.frame_count))
    
    def toggle_play(self):
        """Video oynatmayı başlat/durdur"""
        if self.cap is None:
            return
        
        if self.is_playing:
            self.stop_play_thread()
            self.play_btn.config(text="Oynat")
            self.is_playing = False
            self.status_label.config(text="Durduruldu")
        else:
            self.is_playing = True
            self.play_btn.config(text="Duraklat")
            self.status_label.config(text="Oynatılıyor...")
            
            # Frame queue'yu temizle
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            
            # Yeni bir thread oluştur
            self.stop_thread = False
            self.play_thread = threading.Thread(target=self.play_video)
            self.play_thread.daemon = True
            self.play_thread.start()
    
    def play_video(self):
        """Videoyu ayrı bir thread'de oynat"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1.0 / fps if fps > 0 else 0.033  # Varsayılan ~30fps
        
        while self.is_playing and not self.stop_thread:
            start_time = time.time()
            
            # Kilit kullanarak güvenli okuma
            with self.seek_lock:
                ret, frame = self.cap.read()
                if not ret:
                    # Video sonuna gelindi, başa sar
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    self.root.after(0, self.update_ui_stopped)
                    break
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Frame'i işle ve queue'ya ekle
            try:
                # İşlenmiş frame'i queue'ya ekle
                if not self.processing:
                    self.processing = True
                    processed_frame = self.process_frame(frame)
                    self.frame_queue.put((processed_frame, self.current_frame), block=False)
                    self.processing = False
            except queue.Full:
                # Queue dolu, önceki frame'leri atla
                pass
            
            # FPS hesaplama
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            
            # Her 30 frame'de bir veya 1 saniyede bir FPS'i güncelle
            current_time = time.time()
            if len(self.frame_times) >= 30 or (current_time - self.last_fps_update) >= self.fps_update_interval:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_frame_time
                self.frame_times = []
                self.last_fps_update = current_time
            
            # Slider konumunu güncelle (UI thread'indeki düşük öncelikli işlem)
            self.root.after_idle(lambda cf=self.current_frame: self.progress_slider.set(cf))
            
            # İlerleme bilgisini güncelle
            if self.current_frame % 10 == 0:  # Her 10 karede bir güncelle
                current_time = self.current_frame / fps
                status_text = "Oynatılıyor: {}/{} ({}) - FPS: {:.1f}".format(
                    self.format_time(current_time),
                    self.format_time(self.frame_count / fps),
                    self.current_frame,
                    self.fps
                )
                self.root.after_idle(lambda t=status_text: self.status_label.config(text=t))
            
            # FPS kontrolü
            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_time - elapsed)
            time.sleep(sleep_time)
    
    def process_frame(self, frame):
        """Frame işleme"""
        # İşlenmemiş kareyi sakla, threshold değiştiğinde kullanmak için
        self.current_processed_frame = frame.copy()
        
        if self.detect_var.get() and self.model is not None:
            # Eşik değerleri kullanım bilgisini yazdır
            print(f"YOLO çalıştırılıyor - Model: {self.model_var.get()}, Conf: {self.conf_threshold:.2f}, IOU: {self.iou_threshold:.2f}")
            
            # YOLO ile nesne tespiti
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Algılanan nesne sayısını yazdır
            box_count = len(results[0].boxes)
            print(f"Algılanan nesne sayısı: {box_count}")
            
            # Görüntüleme moduna göre işlem yap
            if self.display_mode == "normal":
                # Varsayılan görüntüleme - hem kutular hem class isimleri hem de conf değerleri
                annotated_frame = results[0].plot()
                display_frame = annotated_frame
            
            elif self.display_mode == "boxes_only":
                # Sadece kutuları göster, isimleri ve conf değerlerini gösterme
                annotated_frame = frame.copy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                display_frame = annotated_frame
            
            elif self.display_mode == "confidence":
                # Sadece kutuları ve conf değerlerini göster
                annotated_frame = frame.copy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                conf_values = results[0].boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    conf = conf_values[i]
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "{:.2f}".format(conf), (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                display_frame = annotated_frame
            
            elif self.display_mode == "censored":
                # Tespit edilen nesneleri sansürle/bulanıklaştır
                annotated_frame = frame.copy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    # Tespit edilen bölgeyi bulanıklaştır
                    roi = annotated_frame[y1:y2, x1:x2]
                    if roi.size > 0:  # ROI'nin boş olmadığından emin ol
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        annotated_frame[y1:y2, x1:x2] = blurred_roi
                
                display_frame = annotated_frame
            
            else:
                # Bilinmeyen mod, varsayılan görüntülemeyi kullan
                display_frame = results[0].plot()
        else:
            display_frame = frame
            
        return display_frame
    
    def update_ui(self, frame, current_frame):
        """UI elemanlarını güncelle"""
        # OpenCV BGR formatını RGB'ye çevir
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Canvas boyutunu al
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Görüntü boyutlarını al
        h, w, _ = rgb_image.shape
        
        # En-boy oranını koruyarak yeniden boyutlandır
        ratio = min(canvas_width/w, canvas_height/h)
        new_width = int(w * ratio)
        new_height = int(h * ratio)
        
        # Yeniden boyutlandır
        resized = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # PIL Image'e çevir
        pil_img = Image.fromarray(resized)
        
        # PhotoImage oluştur
        self.photo = ImageTk.PhotoImage(image=pil_img)
        
        # Canvas'ı temizle
        self.canvas.delete("all")
        
        # FPS göstergesini ekle
        fps_text = f"FPS: {self.fps:.1f}"
        self.canvas.create_text(10, 10, text=fps_text, fill="white", 
                              font=('Arial', 12, 'bold'), anchor=tk.NW)
        
        # Görüntüyü canvas'a yerleştir (ortalanmış)
        x_position = (canvas_width - new_width) // 2
        y_position = (canvas_height - new_height) // 2
        self.canvas.create_image(x_position, y_position, anchor=tk.NW, image=self.photo)
        
        # Zaman etiketini güncelle
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        current_time = current_frame / fps
        self.time_label_start.config(text=self.format_time(current_time))
    
    def update_ui_stopped(self):
        """Video durduğunda UI güncellemeleri"""
        self.play_btn.config(text="Oynat")
        self.is_playing = False
        self.status_label.config(text="Video tamamlandı")
    
    def stop_play_thread(self):
        """Oynatma thread'ini durdur"""
        if self.play_thread is not None and self.play_thread.is_alive():
            self.stop_thread = True
            self.play_thread.join(1.0)  # 1 saniye bekle
            
            # Frame queue'yu temizle
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
    
    def display_frame(self, frame):
        """Tek bir frame gösterme (oynatma dışındaki durumlar için)"""
        processed_frame = self.process_frame(frame)
        self.update_ui(processed_frame, self.current_frame)
        
        # Eşik değerlerini durum çubuğunda göster
        if self.detect_var.get() and self.model is not None:
            self.status_label.config(text=f"Confidence: {self.conf_threshold:.2f}, IOU: {self.iou_threshold:.2f}")
    
    def toggle_detection(self):
        """Nesne tespitini açıp kapama"""
        if self.detect_var.get() and self.model is None:
            self.load_yolo_model(self.model_var.get())
    
    def slider_changed(self, value):
        """İlerleme çubuğu değiştiğinde"""
        value = float(value)
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            current_time = value / fps
            self.time_label_start.config(text=self.format_time(current_time))
    
    def slider_released(self, event):
        """İlerleme çubuğu bırakıldığında"""
        if self.cap is None:
            return
            
        # Önce oynatmayı durdur
        was_playing = self.is_playing
        if was_playing:
            self.stop_play_thread()
        
        value = self.progress_slider.get()
        target_frame = int(value)
        
        # Video'yu güvenli şekilde yeni konuma taşı
        with self.seek_lock:
            # Büyük sıçramalar için videoyu yeniden açmak daha güvenlidir
            if abs(target_frame - self.current_frame) > 100:
                # Video dosyasını kapat ve yeniden aç
                video_path = self.video_path
                self.cap.release()
                self.cap = cv2.VideoCapture(video_path)
                
                # Pozisyonu ayarla
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.current_frame = target_frame
            else:
                # Küçük sıçramalar için normal pozisyon ayarlaması
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.current_frame = target_frame
            
            # Güncel kareyi oku
            ret, frame = self.cap.read()
            if ret:
                # UI'yi güncelle
                self.display_frame(frame)
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Zaman bilgisini güncelle
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                current_time = self.current_frame / fps
                self.status_label.config(text="Konum: {}/{} ({}) - Conf: {:.2f}, IOU: {:.2f}".format(
                    self.format_time(current_time),
                    self.format_time(self.frame_count / fps),
                    self.current_frame,
                    self.conf_threshold,
                    self.iou_threshold
                ))
        
        # Eğer önceden oynatılıyorsa, tekrar başlat
        if was_playing:
            self.root.after(100, self.toggle_play)
    
    def safe_set_frame_position(self, position):
        """Video pozisyonunu güvenli bir şekilde ayarla"""
        # Güvenli sınırlar içinde kal
        position = max(0, min(position, self.frame_count - 1))
        
        # Videoyu güvenli şekilde kapat ve yeniden aç
        video_path = self.video_path
        with self.seek_lock:
            self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.current_frame = position
            
            # Güncel kareyi oku
            ret, frame = self.cap.read()
            if ret:
                # UI'yi güncelle
                self.display_frame(frame)
                return True
        return False
    
    def jump_frames(self, frames):
        """İleri veya geri belirli sayıda kare atla"""
        if self.cap is None:
            return
        
        # Önce oynatmayı durdur
        was_playing = self.is_playing
        if was_playing:
            self.stop_play_thread()
        
        target_frame = min(max(0, self.current_frame + frames), self.frame_count - 1)
        
        # Geriye doğru hareket veya büyük atlamalar için güvenli kare ayarını kullan
        if frames < 0 or abs(frames) > 30:
            success = self.safe_set_frame_position(target_frame)
        else:
            # İleri doğru küçük hareketler için normal ayarlama
            with self.seek_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.current_frame = target_frame
                ret, frame = self.cap.read()
                success = False
                if ret:
                    self.display_frame(frame)
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    success = True
        
        if success:
            # Slider'ı güncelle
            self.progress_slider.set(self.current_frame)
            
            # Zaman etiketini güncelle
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            current_time = self.current_frame / fps
            self.time_label_start.config(text=self.format_time(current_time))
            
            self.status_label.config(text="{} kare {} yönüne atlandı".format(
                abs(frames), "ileri" if frames > 0 else "geri"))
        
        # Eğer önceden oynatılıyorsa, tekrar başlat
        if was_playing:
            self.root.after(100, self.toggle_play)
    
    def format_time(self, seconds):
        """Saniye değerini MM:SS formatına çevir"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return "{}:{:02d}".format(mins, secs)
    
    def close_app(self):
        """Uygulama kapatılırken temizlik"""
        self.stop_play_thread()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CenkerVision(root)
    root.mainloop()


if __name__ == "__main__":
    main()
