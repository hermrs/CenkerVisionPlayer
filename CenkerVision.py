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
import torch  
import yaml
import platform
from datetime import datetime
from typing import Dict, Any
from memory_bank import MemoryBank

# Sabit değişkenler
DEBUG_MODE = False # Hata ayıklama modu
SIMPLE_MODE = False # Basit mod - YOLO işlemini atlar sadece videoyu gösterir
FORCE_CPU = False # M1 Mac'in GPU/MPS desteğini etkinleştir
MAX_FRAME_RATE = 100 # Maksimum FPS değeri - CPU kullanımını optimize etmek için

# ByteTrack varsayılan ayarları
BYTETRACK_CONFIG = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.3,  
    "track_low_thresh": 0.15,  
    "new_track_thresh": 0.3,  
    "track_buffer": 25,        # Buffer'ı küçülterek bellek kullanımını azalt
    "match_thresh": 0.8,
    "fuse_score": True,
    "min_box_area": 10,        # Çok küçük kutuları filtrele
    "max_track_per_frame": 50  # Maksimum takip edilen nesne sayısını sınırla
}

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
        self.is_webcam = False # Webcam kullanılıp kullanılmadığını belirtir
        
        # Takip (tracking) değişkenleri
        self.enable_tracking = False  # Takip özelliği açık/kapalı
        self.tracker_config = BYTETRACK_CONFIG.copy()  # Varsayılan takip ayarları
        self.tracker_type = "bytetrack"  # tracker türü (bytetrack, botsort vb.)
        self.tracker_config_path = self.get_tracker_config_path()
        
        # Debug modu
        self.debug_mode = DEBUG_MODE
        self.simple_mode = SIMPLE_MODE
        self.force_cpu = FORCE_CPU
        
        # Başlık güncelleme
        mode_info = []
        if self.simple_mode:
            mode_info.append("BASİT MOD")
        if self.debug_mode:
            mode_info.append("HATA AYIKLAMA")
        if self.force_cpu:
            mode_info.append("CPU MODU")
        
        if mode_info:
            self.root.title(f"CenkerVision - YOLO Tabanlı Video Oynatıcı [{', '.join(mode_info)}]")
        
        # Cihaz seçimi (CPU veya MPS/GPU)
        self.device = self.get_device()
        print(f"Kullanılacak cihaz: {self.device}")
        
        # FPS sayacı için değişkenler
        self.fps = 0
        self.frame_times = []
        self.fps_update_interval = 1.0  # FPS güncelleme aralığı (saniye)
        self.last_fps_update = time.time()
        
        # Basit mod UI kontrolü
        if self.simple_mode:
            print("BASİT MOD ETKİN: YOLO işlemi atlanacak ve sadece video gösterilecek")
        
        # M1/Metal kullanım bilgisi
        if not self.force_cpu and self.device == "mps":
            print("⚡ M1/M2 Metal Performance Shaders (MPS) etkin - GPU hızlandırma kullanılıyor")
            print("   MPS destekli PyTorch sürümü:", torch.__version__)
        
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
        
        # Webcam seçme butonu
        self.webcam_btn = ttk.Button(control_frame, text="Webcam Kullan", command=self.select_webcam_source)
        self.webcam_btn.pack(side=tk.TOP, padx=5, pady=5, fill=tk.X)
        
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
        
        # ByteTrack takip etkinleştirme onay kutusu
        self.track_var = tk.BooleanVar(value=self.enable_tracking)
        self.track_checkbox = ttk.Checkbutton(detect_frame, text="Nesne Takibi (ByteTrack)", 
                                           variable=self.track_var, command=self.toggle_tracking)
        self.track_checkbox.pack(side=tk.LEFT, padx=5)
        
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
        
        # Memory Bank'ı başlat
        self.memory_bank = MemoryBank()
        self._initialize_memory_bank()
        
        self.locked_object_timer = 0.0  # Kitlenme dörtgeni içindeki süre (saniye)
        self.locked_object_last_time = None  # Son frame zamanı
        self.locked_object_present = False  # Dörtgende obje var mı?
        self.locked_object_show_text = "Obje yok"  # Ekrana yazılacak metin
    
    def _initialize_memory_bank(self):
        """Memory Bank'ı başlatır ve proje dokümantasyonunu oluşturur."""
        try:
            # Memory Bank'ı başlat
            result = self.memory_bank.initialize(
                goal="CenkerVision: Gelişmiş nesne takip ve analiz sistemi"
            )
            
            if "error" in result:
                print(f"Memory Bank initialization error: {result['error']}")
                return
            
            # Proje özetini güncelle
            self.memory_bank.update_document(
                "projectbrief",
                f"""# CenkerVision Projesi

## Amaç
CenkerVision, gelişmiş nesne takip ve analiz yeteneklerine sahip bir bilgisayarlı görü sistemidir.

## Özellikler
- YOLOv8 tabanlı nesne tespiti
- Çoklu nesne takibi
- Gerçek zamanlı analiz
- Kullanıcı dostu arayüz

## Teknik Detaylar
- Python tabanlı uygulama
- OpenCV ve YOLOv8 entegrasyonu
- Çoklu iş parçacığı desteği
"""
            )
            
            # Teknik bağlamı güncelle
            self.memory_bank.update_document(
                "techContext",
                f"""# Teknik Bağlam

## Kullanılan Teknolojiler
- Python {platform.python_version()}
- OpenCV
- YOLOv8
- PyQt6

## Sistem Gereksinimleri
- CUDA destekli GPU (önerilen)
- En az 8GB RAM
- Python 3.8 veya üzeri
"""
            )
            
        except Exception as e:
            print(f"Error initializing Memory Bank: {e}")
    
    def update_memory_bank(self, event_type: str, details: str):
        """Memory Bank'ı günceller."""
        try:
            # Aktif bağlamı güncelle
            self.memory_bank.update_document(
                "activeContext",
                f"""# Aktif Bağlam

## Son Olay
- Tip: {event_type}
- Detaylar: {details}
- Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            )
        except Exception as e:
            print(f"Error updating Memory Bank: {e}")
    
    def search_memory_bank(self, query: str) -> Dict[str, Any]:
        """Memory Bank'ta arama yapar."""
        try:
            return self.memory_bank.query(query)
        except Exception as e:
            print(f"Error searching Memory Bank: {e}")
            return {"error": str(e)}
    
    def check_queue(self):
        """Frame queue'yu kontrol et ve görüntüle"""
        print("DEBUG: check_queue called") # DEBUG
        try:
            if not self.frame_queue.empty():
                frame, current_frame = self.frame_queue.get_nowait()
                print(f"DEBUG: Got frame from queue. Type: {type(frame)}, Current Frame No: {current_frame}") # DEBUG
                if frame is not None:
                    print(f"DEBUG: Frame from queue shape: {frame.shape if hasattr(frame, 'shape') else 'No shape'}") # DEBUG
                
                if frame is not None and len(frame.shape) == 3:  # Geçerli bir frame mi?
                    self.update_ui(frame, current_frame)
                else:
                    print("Geçersiz frame alındı, atlanıyor")
                
                self.frame_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Queue işleme hatası: {str(e)}")
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
    
    def get_device(self):
        """Kullanılacak cihazı belirle (MPS, CUDA veya CPU)"""
        if self.force_cpu:
            print("CPU kullanımı manuel olarak zorlandı.")
            return "cpu"
            
        if torch.backends.mps.is_available():
            try:
                # MPS kullanılabilirliğini daha detaylı kontrol et
                test_tensor = torch.zeros(1).to("mps")
                device = "mps"
                print("Apple Silicon MPS (Metal Performance Shaders) kullanılıyor")
                print("MPS cihazı hazır:", torch.backends.mps.is_built())
                return "mps"
            except Exception as e:
                print(f"MPS kullanılabilir ancak bir hata oluştu: {e}")
                print("CPU'ya düşüyor...")
                return "cpu"
        elif torch.cuda.is_available():
            device = "cuda"
            print("NVIDIA CUDA GPU kullanılıyor")
            return "cuda"
        else:
            device = "cpu"
            print("CPU kullanılıyor")
            return "cpu"
    
    def load_yolo_model(self, model_path):
        try:
            # status_label kullanılabilirliğini kontrol et
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Model yükleniyor: {model_path} ({self.device})")
                self.root.update()  # UI'yi hemen güncelle
            
            # Özel model yolunu kontrol et
            if model_path in self.custom_models:
                model_path = os.path.join(self.models_dir, model_path)
                self.model = YOLO(model_path)
            else:
                self.model = YOLO(model_path)
                
            # Modeli seçilen cihaza taşı
            if self.device != "cpu":
                try:
                    print(f"Model {self.device} cihazına taşınıyor...")
                    self.model.to(self.device)
                    # Model başarıyla GPU'ya taşındı
                    print(f"Model başarıyla {self.device} cihazına taşındı!")
                except Exception as e:
                    print(f"Model {self.device} cihazına taşınırken hata: {e}")
                    print("Güvenli mod: Model CPU'da kalacak")
                    self.device = "cpu" # Cihazı CPU'ya çevir
                
            print(f"{model_path} modeli başarıyla yüklendi. (Cihaz: {self.device})")
            
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"{model_path} modeli başarıyla yüklendi. (Cihaz: {self.device})")
                
            # Model yüklendiğinde Memory Bank'ı güncelle
            self.update_memory_bank(
                "Model Yüklendi",
                f"YOLOv8 modeli başarıyla yüklendi: {model_path}"
            )
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.update_memory_bank(
                "Model Yükleme Hatası",
                f"YOLOv8 modeli yüklenirken hata oluştu: {str(e)}"
            )
    
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
            self.is_webcam = False # Dosya seçildiğinde webcam olmadığını belirt
            self.load_video(video_path)
    
    def select_webcam_source(self):
        """Webcam kaynağı seçme"""
        # Basit bir input dialog ile webcam ID'si sorulabilir veya varsayılan 0 kullanılır.
        # Şimdilik varsayılan olarak 0 kullanalım.
        # Gelişmiş: Mevcut kameraları listeleyip seçtirme eklenebilir.
        webcam_id_str = tk.StringVar(value="0")
        dialog = tk.Toplevel(self.root)
        dialog.title("Webcam ID Seçin")
        dialog.geometry("300x100")
        ttk.Label(dialog, text="Webcam ID (genellikle 0):").pack(pady=5)
        entry = ttk.Entry(dialog, textvariable=webcam_id_str)
        entry.pack(pady=5)
        
        def on_ok():
            try:
                webcam_id = int(webcam_id_str.get())
                dialog.destroy()
                self.video_path = f"Webcam {webcam_id}" # Bilgilendirme amaçlı
                self.is_webcam = True
                self.load_video(webcam_id)
            except ValueError:
                messagebox.showerror("Hata", "Lütfen geçerli bir sayı girin.")

        ttk.Button(dialog, text="Tamam", command=on_ok).pack(pady=5)
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)

    def load_video(self, source):
        """Videoyu veya webcam'i yükle ve hazırla"""
        if self.cap is not None:
            self.stop_play_thread()
            self.cap.release()
        
        self.status_label.config(text="Kaynak yükleniyor...")
        self.root.update()
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            if self.is_webcam:
                messagebox.showerror("Webcam Hatası", f"Webcam ID {source} açılamadı!")
                self.status_label.config(text="Webcam açma hatası!")
                print(f"DEBUG: Failed to open webcam ID {source}") # DEBUG
            else:
                messagebox.showerror("Video Hatası", "Video dosyası açılamadı!")
                self.status_label.config(text="Video yükleme hatası!")
            self.cap = None # Hata durumunda cap'i None yap
            return
        
        if self.is_webcam:
            print(f"DEBUG: Webcam ID {source} opened successfully: {self.cap.isOpened()}") # DEBUG
            self.frame_count = float('inf') # Webcam için sonsuz frame
            self.progress_slider.config(to=100, state=tk.DISABLED) # Slider'ı devre dışı bırak
            self.time_label_start.config(text="Canlı")
            self.time_label_end.config(text="Canlı")
            self.prev_frame_btn.config(state=tk.DISABLED)
            self.next_frame_btn.config(state=tk.DISABLED)
            self.status_label.config(text=f"Webcam {source} aktif.")
            # Webcam aktif olduğunda video_path'i de ayarlayalım ki slider_released gibi yerlerde doğru çalışsın
            if isinstance(source, int): # Eğer source bir int ise (webcam ID)
                self.video_path = f"Webcam_{source}" 
        else:
            self.video_path = source # Video dosyası için yolu sakla
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_slider.config(to=self.frame_count, state=tk.NORMAL) # Slider'ı etkinleştir
            self.prev_frame_btn.config(state=tk.NORMAL)
            self.next_frame_btn.config(state=tk.NORMAL)
            
            # Video süresini hesapla ve göster
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                duration = self.frame_count / fps
                self.time_label_end.config(text=self.format_time(duration))
            else:
                self.time_label_end.config(text="N/A")
            video_name = os.path.basename(source)
            self.status_label.config(text="Video hazır: {} ({} kare)".format(video_name, self.frame_count))

        self.current_frame = 0
        
        # İlk kareyi göster (hem video hem webcam için)
        ret, frame = self.cap.read()
        if ret:
            if self.is_webcam:
                print(f"DEBUG: Initial webcam frame read success. Shape: {frame.shape}") # DEBUG
            self.display_frame(frame)
        else:
            if self.is_webcam:
                print("DEBUG: Failed to read initial frame from webcam.") # DEBUG
        
        self.play_btn.config(text="Oynat")
        self.is_playing = False
    
    def toggle_play(self):
        """Video oynatmayı başlat/durdur"""
        print("DEBUG: toggle_play called") # DEBUG
        if self.cap is None:
            print("DEBUG: toggle_play - self.cap is None, returning") # DEBUG
            return
        
        if self.is_playing:
            self.stop_play_thread()
            self.play_btn.config(text="Oynat")
            self.is_playing = False
            self.status_label.config(text="Durduruldu")
            print("DEBUG: toggle_play - Video stopped") # DEBUG
        else:
            self.is_playing = True
            self.play_btn.config(text="Duraklat")
            self.status_label.config(text="Oynatılıyor...")
            print("DEBUG: toggle_play - Video playing") # DEBUG
            
            # Frame queue'yu temizle
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            
            # Yeni bir thread oluştur
            self.stop_thread = False
            self.play_thread = threading.Thread(target=self.play_video)
            self.play_thread.daemon = True
            print("DEBUG: toggle_play - Starting play_thread") # DEBUG
            self.play_thread.start()
    
    def play_video(self):
        """Videoyu ayrı bir thread'de oynat"""
        print("DEBUG: play_video thread started") # DEBUG
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1.0 / fps if fps > 0 else 0.033  # Varsayılan ~30fps
        
        # Maksimum frame rate ile sınırlandır
        if MAX_FRAME_RATE > 0 and fps > MAX_FRAME_RATE:
            target_frame_time = 1.0 / MAX_FRAME_RATE
            print(f"FPS {fps} -> {MAX_FRAME_RATE} olarak sınırlandırıldı (performans optimizasyonu)")
        else:
            target_frame_time = frame_time
            
        frame_counter = 0  # Hata ayıklama için sayaç
        last_debug_time = time.time()  # Hata ayıklama için zaman
        
        while self.is_playing and not self.stop_thread:
            print("DEBUG: play_video loop iteration") # DEBUG
            start_time = time.time()
            frame_counter += 1
            
            # Periyodik hata ayıklama çıktısı
            if self.debug_mode and (time.time() - last_debug_time > 2.0):  # Her 2 saniyede bir
                print(f"Debug: Frame sayacı = {frame_counter}, FPS = {self.fps:.1f}")
                last_debug_time = time.time()
                frame_counter = 0  # Sayacı sıfırla
            
            try:
                # Kilit kullanarak güvenli okuma
                with self.seek_lock:
                    ret, frame = self.cap.read()
                    if self.is_webcam: # Debugging for webcam
                        print(f"DEBUG: play_video cap.read() ret: {ret}") # DEBUG
                        if ret:
                            print(f"DEBUG: play_video webcam frame shape: {frame.shape}") # DEBUG

                    if not ret:
                        if self.is_webcam:
                            # Webcam'de bu durum hata anlamına gelebilir, durdur
                            print("Webcam bağlantısı kesildi veya hata.")
                            self.root.after(0, self.stop_play_thread) # Oynatmayı durdur
                            self.root.after(0, lambda: self.status_label.config(text="Webcam hatası/bağlantı kesildi."))
                            break
                        else:
                            # Video sonuna gelindi, başa sar
                            print("Video sonuna gelindi, başa sarılıyor...")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.current_frame = 0
                            self.root.after(0, self.update_ui_stopped)
                            break
                    
                    if not self.is_webcam: # Webcam için POS_FRAMES güncellenmeyebilir/anlamsız olabilir
                        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    else:
                        # Webcam için frame sayısını kendimiz artıralım (gösterim amaçlı)
                        self.current_frame +=1 
                
                # Basit modda direkt frame'i queue'ya ekle (YOLO işlemesini atla)
                if self.simple_mode:
                    try:
                        print("DEBUG: play_video - Simple mode: Attempting to put frame on queue") # DEBUG
                        frame_to_queue = frame.copy()  # Savunma amaçlı kopya
                        self.frame_queue.put((frame_to_queue, self.current_frame), block=True, timeout=1)
                        print("DEBUG: play_video - Simple mode: Frame put on queue SUCCESS") # DEBUG
                    except queue.Full:
                        print("DEBUG: play_video - Simple mode: Frame queue FULL, frame dropped") # DEBUG
                else:
                    # Frame'i işle ve queue'ya ekle
                    try:
                        print(f"DEBUG: play_video - YOLO mode: Processing frame {self.current_frame}") # DEBUG
                        processed_frame = self.process_frame(frame)
                        print(f"DEBUG: play_video - YOLO mode: Frame {self.current_frame} processed. Attempting to put on queue") # DEBUG
                        frame_to_queue = processed_frame.copy() # Emin olmak için processed_frame'in kopyasını al
                        self.frame_queue.put((frame_to_queue, self.current_frame), block=True, timeout=1)
                        print(f"DEBUG: play_video - YOLO mode: Processed frame {self.current_frame} put on queue SUCCESS") # DEBUG
                    except queue.Full:
                        print(f"DEBUG: play_video - YOLO mode: Frame queue FULL for frame {self.current_frame}, frame dropped") # DEBUG
                    except Exception as e:
                        if str(e):  # Sadece boş olmayan hataları yazdır
                            print(f"Frame işleme hatası (in play_video): {str(e)}")
                        try:
                            print(f"DEBUG: play_video - YOLO mode: Error processing, attempting to put ORIGINAL frame {self.current_frame} on queue") # DEBUG
                            original_frame_to_queue = frame.copy()  # Savunma amaçlı kopya
                            self.frame_queue.put((original_frame_to_queue, self.current_frame), block=True, timeout=1)
                            print(f"DEBUG: play_video - YOLO mode: Original frame {self.current_frame} put on queue after error SUCCESS") # DEBUG
                        except queue.Full:
                            print(f"DEBUG: play_video - YOLO mode: Frame queue FULL for original frame {self.current_frame} after error, frame dropped") # DEBUG
                
                # FPS hesaplama
                frame_time = time.time() - start_time
                self.frame_times.append(frame_time)
                
                # Her 30 frame'de bir veya 1 saniyede bir FPS'i güncelle
                current_time = time.time()
                if len(self.frame_times) >= 30 or (current_time - self.last_fps_update) >= self.fps_update_interval:
                    if self.frame_times:  # Boş liste kontrolü
                        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                        self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    self.frame_times = []
                    self.last_fps_update = current_time
                
                # Slider konumunu güncelle (UI thread'indeki düşük öncelikli işlem)
                # Sadece video dosyası için slider'ı güncelle
                if not self.is_webcam:
                    self.root.after_idle(lambda cf=self.current_frame: self.progress_slider.set(cf))
                
                # İlerleme bilgisini güncelle
                if self.current_frame % 10 == 0:  # Her 10 karede bir güncelle
                    mode_text = "[BASİT MOD]" if self.simple_mode else ""
                    if self.is_webcam:
                        status_text = f"Oynatılıyor: Webcam - Kare: {self.current_frame} - FPS: {self.fps:.1f} {mode_text}"
                    elif fps > 0: # fps sıfır değilse
                        current_time_val = self.current_frame / fps
                        total_duration_val = self.frame_count / fps
                        status_text = f"Oynatılıyor: {self.format_time(current_time_val)}/{self.format_time(total_duration_val)} ({self.current_frame}) - FPS: {self.fps:.1f} {mode_text}"
                    else: # fps sıfırsa (bazı video formatları için)
                        status_text = f"Oynatılıyor: Kare: {self.current_frame} - FPS: {self.fps:.1f} {mode_text}"
                    self.root.after_idle(lambda t=status_text: self.status_label.config(text=t))
                
                # Kare hızını kontrol et ve sınırlandır (CPU kullanımını azaltmak için)
                elapsed = time.time() - start_time
                sleep_time = max(0.001, target_frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                if str(e):  # Sadece boş olmayan hataları yazdır
                    print(f"Video oynatma hatası: {str(e)}")
                # Hata durumunda kısa bir süre bekleyip devam et
                time.sleep(0.1)
    
    def process_frame(self, frame):
        """Frame işleme"""
        # İşlenmemiş kareyi sakla, threshold değiştiğinde kullanmak için
        try:
            original_frame_for_display = frame.copy() # Görüntüleme için orijinal kareyi sakla
            self.current_processed_frame = frame.copy() # Yeniden işleme için orijinal kareyi sakla
            
            # Ölçekleme oranlarını başlangıçta 1.0 olarak ayarla (ölçekleme yapılmadığında)
            scale_ratio_w, scale_ratio_h = 1.0, 1.0
            original_h, original_w = frame.shape[:2]

            if self.detect_var.get() and self.model is not None:
                try:
                    # Hata ayıklama için
                    if self.debug_mode:
                        print(f"Frame boyutu: {frame.shape}")
                    
                    # Kareyi 720p'ye yeniden boyutlandır (yüksek çözünürlüklü videolar için performans artışı)
                    h, w = frame.shape[:2]
                    target_h = 720
                    if h > target_h:
                        ratio = target_h / h
                        target_w = int(w * ratio)
                        # Hız için INTER_LINEAR kullan
                        resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        if self.debug_mode:
                            print(f"Frame boyutlandırıldı: {w}x{h} -> {target_w}x{target_h}")
                        # Boyutlandırılmış kareyi kullan
                        process_frame = resized_frame
                        
                        # Ölçekleme oranlarını hesapla (orijinal'den işlenen frame'e)
                        scale_ratio_w = original_w / target_w
                        scale_ratio_h = original_h / target_h
                        
                        if self.debug_mode:
                            print(f"Ölçekleme oranları: W={scale_ratio_w:.2f}, H={scale_ratio_h:.2f}")
                    else:
                        # Zaten küçük boyuttaysa orijinal kareyi kullan
                        process_frame = frame
                    
                    # MPS ile ilgili hataları yakalamak için önce CPU'da deneyelim
                    if self.debug_mode:
                        print(f"İşlem öncesi model cihazı: {self.device}")
                    
                    # YOLO ile nesne tespiti / takibi işlemini gerçekleştir
                    t_start = time.time()
                    
                    # Model çalıştırma işlemini try/except bloğuna al
                    try:
                        # Eşik değerleri kullanım bilgisini yazdır
                        if self.debug_mode:
                            print(f"YOLO çalıştırılıyor - Model: {self.model_var.get()}, Conf: {self.conf_threshold:.2f}, IOU: {self.iou_threshold:.2f}, Cihaz: {self.device}")
                        
                        # Takip modu etkinse track() metodunu kullan, değilse predict() metodunu kullan
                        if self.enable_tracking:
                            # ByteTrack ile nesne takibi
                            tracker_path = self.tracker_config_path if self.tracker_config_path else "bytetrack.yaml"
                            try:
                                # Güvenli tensor dönüşümleri için try-except bloğu ekle
                                results = self.model.track(
                                    process_frame, 
                                    persist=True,  # Takip ID'lerini sonraki frameler için sakla
                                    conf=self.conf_threshold, 
                                    iou=self.iou_threshold,
                                    tracker=tracker_path  # ByteTrack yapılandırması
                                )
                                if self.debug_mode:
                                    print(f"ByteTrack kullanılıyor: {tracker_path}")
                            except Exception as track_error:
                                # ByteTrack hatası durumunda normal predict() metodunu kullan
                                error_str = str(track_error)
                                if error_str:
                                    print(f"ByteTrack hatası, normal tespit kullanılıyor: {error_str}")
                                else:
                                    print("ByteTrack hatası, normal tespit kullanılıyor")
                                
                                # Hata durumunda takibi devre dışı bırak
                                self.enable_tracking = False
                                self.track_var.set(False)
                                
                                # Takip olmadan normal predict kullan
                                results = self.model(process_frame, conf=self.conf_threshold, iou=self.iou_threshold)
                        else:
                            # Standart nesne tespiti (takip olmadan)
                            results = self.model(process_frame, conf=self.conf_threshold, iou=self.iou_threshold)
                        
                        inference_time = time.time() - t_start
                        
                        if self.debug_mode:
                            print(f"YOLO çıkarım süresi: {inference_time*1000:.1f} ms")
                    except Exception as e:
                        # Ana hata yakalama
                        print(f"Model çalıştırma hatası: {str(e)}")
                        # Hata durumunda CPU'ya geç
                        self.device = 'cpu'
                        if self.debug_mode:
                            print("Model CPU'ya taşındı")
                        
                        # CPU'da tekrar dene
                        try:
                            t_start = time.time()
                            # Takibi devre dışı bırak ve sadece tespit kullan
                            self.enable_tracking = False
                            self.track_var.set(False)
                            
                            # CPU'da model çalıştır
                            results = self.model(process_frame, conf=self.conf_threshold, iou=self.iou_threshold)
                            inference_time = time.time() - t_start
                            
                            if self.debug_mode:
                                print(f"CPU'da YOLO çıkarım süresi: {inference_time*1000:.1f} ms")
                        except Exception as cpu_error:
                            print(f"CPU'da da hata oluştu: {str(cpu_error)}")
                            # Bu durumda nesne tespitini devre dışı bırak ve orijinal kareyi döndür
                            self.detect_var.set(False)
                            return original_frame_for_display
                    
                    # Algılanan nesne/takip sayısını yazdır
                    if results and results[0].boxes is not None:
                        box_count = len(results[0].boxes)
                        
                        # Takip bilgisi varsa yazdır (ID'ler)
                        if self.enable_tracking and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                            try:
                                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                                if self.debug_mode:
                                    print(f"Takip edilen nesneler: {len(track_ids)}, ID'ler: {track_ids}")
                            except Exception as e:
                                print(f"Takip ID'leri alınırken hata: {str(e)}")
                                # Hata durumunda takibi devre dışı bırak
                                self.enable_tracking = False
                                self.track_var.set(False)
                        
                        if self.debug_mode:
                            print(f"Algılanan nesne sayısı: {box_count}")
                        
                        # Görüntüleme moduna göre işlem yap
                        if self.display_mode == "normal":
                            # Varsayılan görüntüleme - hem kutular hem class isimleri hem de conf değerleri
                            try:
                                # YOLO.plot() metodu ile xyxy değiştiremediğimiz için manuel çizim yapalım
                                if scale_ratio_w != 1.0 or scale_ratio_h != 1.0:
                                    # Orijinal görüntüyü kullan
                                    annotated_frame = original_frame_for_display.copy()
                                    
                                    # Kutu koordinatlarını al
                                    boxes = results[0].boxes.xyxy.cpu().numpy()
                                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                                    conf_values = results[0].boxes.conf.cpu().numpy()
                                    
                                    # Kutular için renk paleti
                                    color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                                                   (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                                    
                                    # Sınıf isimlerini al (varsa)
                                    class_names = results[0].names if hasattr(results[0], 'names') else {}
                                    
                                    # Kutuları ölçeklendir ve çiz
                                    for i, box in enumerate(boxes):
                                        # Koordinatları ölçeklendir
                                        x1 = int(box[0] * scale_ratio_w)
                                        y1 = int(box[1] * scale_ratio_h)
                                        x2 = int(box[2] * scale_ratio_w)
                                        y2 = int(box[3] * scale_ratio_h)
                                        
                                        # Sınıf ve güven değeri
                                        cls_id = classes[i]
                                        conf = conf_values[i]
                                        label = ""
                                        
                                        # Sınıf adını ekle
                                        if cls_id in class_names:
                                            label = f"{class_names[cls_id]} {conf:.2f}"
                                        else:
                                            label = f"Class:{cls_id} {conf:.2f}"
                                        
                                        # Renk seç
                                        color = color_palette[cls_id % len(color_palette)]
                                        
                                        # Kutuyu çiz
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        # Metin arka planı
                                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                        cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), 
                                                    (x1+text_size[0], y1), color, -1)
                                        
                                        # Metni ekle
                                        cv2.putText(annotated_frame, label, (x1, y1-5),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                        
                                        # Takip ID'sini ekle (varsa)
                                        if self.enable_tracking and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                                            try:
                                                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                                                
                                                # Takip ID'si yazısını ekle
                                                id_text = f"ID:{track_ids[i]}"
                                                # Farklı bir konuma ekle (kutunun sağ üst köşesi)
                                                cv2.putText(annotated_frame, id_text, (x2-50, y1-5),
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                            except Exception as id_error:
                                                print(f"Takip ID'sini çizerken hata: {str(id_error)}")
                                    
                                    display_frame = annotated_frame
                                else:
                                    # Ölçekleme yoksa normal plot metodunu kullan
                                    annotated_frame = results[0].plot(img=original_frame_for_display.copy())
                                    display_frame = annotated_frame
                            except Exception as plot_error:
                                print(f"Plot hatası: {str(plot_error)}")
                                display_frame = original_frame_for_display
                        
                        elif self.display_mode == "boxes_only":
                            # Sadece kutuları göster, isimleri ve conf değerlerini gösterme
                            annotated_frame = original_frame_for_display.copy()
                            
                            try:
                                # Kutu koordinatlarını al ve orijinal boyuta ölçeklendir
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                
                                if scale_ratio_w != 1.0 or scale_ratio_h != 1.0:
                                    # Kutuları orijinal boyuta ölçeklendir
                                    boxes_scaled = boxes.copy()
                                    boxes_scaled[:, 0] *= scale_ratio_w  # x1
                                    boxes_scaled[:, 1] *= scale_ratio_h  # y1
                                    boxes_scaled[:, 2] *= scale_ratio_w  # x2
                                    boxes_scaled[:, 3] *= scale_ratio_h  # y2
                                    boxes = boxes_scaled
                                
                                # Takip etkinse, takip ID'lerini göster
                                if self.enable_tracking and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                                    try:
                                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                                        
                                        for i, box in enumerate(boxes):
                                            x1, y1, x2, y2 = box.astype(int)
                                            # Her takip ID'si için farklı renk kullan
                                            color = self.get_color_for_id(track_ids[i])
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                            # Takip ID'sini kutu üzerine ekle
                                            cv2.putText(annotated_frame, f"ID:{track_ids[i]}", (x1, y1-10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    except Exception as id_error:
                                        print(f"Takip ID'lerini işlerken hata: {str(id_error)}")
                                        # Hata durumunda takip olmadan devam et
                                        for box in boxes:
                                            x1, y1, x2, y2 = box.astype(int)
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                else:
                                    # Takip yoksa sadece kutuları çiz
                                    for box in boxes:
                                        x1, y1, x2, y2 = box.astype(int)
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            except Exception as box_error:
                                print(f"Kutuları işlerken hata: {str(box_error)}")
                                # Hata durumunda orijinal kareyi göster
                            
                            display_frame = annotated_frame
                        
                        elif self.display_mode == "confidence":
                            # Sadece kutuları ve conf değerlerini göster
                            annotated_frame = original_frame_for_display.copy()
                            
                            try:
                                # Kutu koordinatlarını al ve orijinal boyuta ölçeklendir
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                conf_values = results[0].boxes.conf.cpu().numpy()
                                
                                if scale_ratio_w != 1.0 or scale_ratio_h != 1.0:
                                    # Kutuları orijinal boyuta ölçeklendir
                                    boxes_scaled = boxes.copy()
                                    boxes_scaled[:, 0] *= scale_ratio_w  # x1
                                    boxes_scaled[:, 1] *= scale_ratio_h  # y1
                                    boxes_scaled[:, 2] *= scale_ratio_w  # x2
                                    boxes_scaled[:, 3] *= scale_ratio_h  # y2
                                    boxes = boxes_scaled
                                
                                # Takip etkinse, takip ID'lerini de göster
                                if self.enable_tracking and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                                    try:
                                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                                        
                                        for i, box in enumerate(boxes):
                                            x1, y1, x2, y2 = box.astype(int)
                                            conf = conf_values[i]
                                            # Her takip ID'si için farklı renk kullan
                                            color = self.get_color_for_id(track_ids[i])
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                            # Takip ID'si ve conf değerini göster
                                            cv2.putText(annotated_frame, f"ID:{track_ids[i]} {conf:.2f}", (x1, y1-10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    except Exception as id_error:
                                        print(f"Conf modunda ID'leri işlerken hata: {str(id_error)}")
                                        # Takip ID'leri alınamazsa conf değeri ile devam et
                                        for i, box in enumerate(boxes):
                                            x1, y1, x2, y2 = box.astype(int)
                                            conf = conf_values[i]
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(annotated_frame, "{:.2f}".format(conf), (x1, y1-10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    # Takip yoksa sadece conf değerlerini göster
                                    for i, box in enumerate(boxes):
                                        x1, y1, x2, y2 = box.astype(int)
                                        conf = conf_values[i]
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(annotated_frame, "{:.2f}".format(conf), (x1, y1-10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            except Exception as conf_error:
                                print(f"Confidence değerlerini işlerken hata: {str(conf_error)}")
                                # Hata durumunda orijinal kareyi göster
                            
                            display_frame = annotated_frame
                        
                        elif self.display_mode == "censored":
                            # Tespit edilen nesneleri sansürle/bulanıklaştır
                            annotated_frame = original_frame_for_display.copy()
                            
                            try:
                                # Kutu koordinatlarını al ve orijinal boyuta ölçeklendir
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                
                                if scale_ratio_w != 1.0 or scale_ratio_h != 1.0:
                                    # Kutuları orijinal boyuta ölçeklendir
                                    boxes_scaled = boxes.copy()
                                    boxes_scaled[:, 0] *= scale_ratio_w  # x1
                                    boxes_scaled[:, 1] *= scale_ratio_h  # y1
                                    boxes_scaled[:, 2] *= scale_ratio_w  # x2
                                    boxes_scaled[:, 3] *= scale_ratio_h  # y2
                                    boxes = boxes_scaled
                                
                                for box in boxes:
                                    x1, y1, x2, y2 = box.astype(int)
                                    # Sınırları çerçeve içinde tut
                                    h, w = annotated_frame.shape[:2]
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w-1, x2), min(h-1, y2)
                                    
                                    # Geçerli koordinatlar kontrolü
                                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
                                        continue
                                        
                                    # Tespit edilen bölgeyi bulanıklaştır
                                    roi = annotated_frame[y1:y2, x1:x2]
                                    if roi.size > 0:  # ROI'nin boş olmadığından emin ol
                                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                                        annotated_frame[y1:y2, x1:x2] = blurred_roi
                            except Exception as blur_error:
                                print(f"Bulanıklaştırma hatası: {str(blur_error)}")
                                # Hata durumunda orijinal kareyi göster
                            
                            display_frame = annotated_frame
                        
                        else:
                            # Bilinmeyen mod, varsayılan görüntülemeyi kullan
                            try:
                                # Kutu koordinatlarını düzelt
                                if scale_ratio_w != 1.0 or scale_ratio_h != 1.0:
                                    # Manuel çizim yap (plot() metodu yerine)
                                    annotated_frame = original_frame_for_display.copy()
                                    
                                    # Kutu koordinatlarını al
                                    boxes = results[0].boxes.xyxy.cpu().numpy()
                                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                                    conf_values = results[0].boxes.conf.cpu().numpy()
                                    
                                    # Kutular için renk paleti
                                    color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                                                   (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                                    
                                    # Sınıf isimlerini al (varsa)
                                    class_names = results[0].names if hasattr(results[0], 'names') else {}
                                    
                                    # Kutuları ölçeklendir ve çiz
                                    for i, box in enumerate(boxes):
                                        # Koordinatları ölçeklendir
                                        x1 = int(box[0] * scale_ratio_w)
                                        y1 = int(box[1] * scale_ratio_h)
                                        x2 = int(box[2] * scale_ratio_w)
                                        y2 = int(box[3] * scale_ratio_h)
                                        
                                        # Sınıf ve güven değeri
                                        cls_id = classes[i]
                                        conf = conf_values[i]
                                        label = ""
                                        
                                        # Sınıf adını ekle
                                        if cls_id in class_names:
                                            label = f"{class_names[cls_id]} {conf:.2f}"
                                        else:
                                            label = f"Class:{cls_id} {conf:.2f}"
                                        
                                        # Renk seç
                                        color = color_palette[cls_id % len(color_palette)]
                                        
                                        # Kutuyu çiz
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        # Metin arka planı
                                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                        cv2.rectangle(annotated_frame, (x1, y1-text_size[1]-5), 
                                                    (x1+text_size[0], y1), color, -1)
                                        
                                        # Metni ekle
                                        cv2.putText(annotated_frame, label, (x1, y1-5),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                        
                                    display_frame = annotated_frame
                                else:
                                    # Ölçekleme yoksa normal plot metodunu kullan
                                    display_frame = results[0].plot(img=original_frame_for_display.copy())
                            except Exception as plot_error:
                                print(f"Varsayılan plot hatası: {str(plot_error)}")
                                display_frame = original_frame_for_display
                
                except Exception as e:
                    if str(e):  # Boş hata mesajlarını gösterme
                        print(f"Genel bir hata oluştu: {str(e)}")
                    # Hata durumunda orijinal görüntüyü göster
                    display_frame = original_frame_for_display
                    # Eğer sürekli hata alınıyorsa nesne tespitini kapat
                    if hasattr(self, 'detect_var'):
                        print("Nesne tespiti geçici olarak devre dışı bırakılıyor...")
                        self.detect_var.set(False)
            else:
                display_frame = original_frame_for_display
                
            # Kitlenme dörtgeni koordinatları (örnek: %25 yatay, %10 dikey boşluk)
            lock_left = int(original_w * 0.25)
            lock_top = int(original_h * 0.10)
            lock_right = int(original_w * 0.75)
            lock_bottom = int(original_h * 0.90)

            # Dörtgeni çiz (kırmızı) - `display_frame` üzerine çizilmeli
            if display_frame is not None and hasattr(display_frame, 'shape') and len(display_frame.shape) == 3:
                cv2.rectangle(display_frame, (lock_left, lock_top), (lock_right, lock_bottom), (0, 0, 255), 2)
            
            # --- Kitlenme dörtgeni sayaç mantığı ---
            object_in_lock = False
            # Nesne tespiti açıksa ve kutular varsa
            if self.detect_var.get() and self.model is not None:
                try:
                    results = getattr(self, 'last_results', None)  # Sonuçlar varsa kullan
                except Exception:
                    results = None
                # Son frame'de kutular varsa
                if 'results' in locals() and results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    # Her kutu için kontrol et
                    for box in boxes:
                        x1, y1, x2, y2 = box.astype(int)
                        # Kutu, kitlenme dörtgeninin içinde mi? (merkez noktası ile basit kontrol)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if lock_left <= cx <= lock_right and lock_top <= cy <= lock_bottom:
                            object_in_lock = True
                            break
            # Sayaç güncelle
            now = time.time()
            if object_in_lock:
                if self.locked_object_last_time is not None:
                    self.locked_object_timer += now - self.locked_object_last_time
                self.locked_object_present = True
                self.locked_object_show_text = f"Süre: {self.locked_object_timer:.1f} sn"
            else:
                self.locked_object_timer = 0.0
                self.locked_object_present = False
                self.locked_object_show_text = "Obje yok"
            self.locked_object_last_time = now
            # --- sayaç mantığı sonu ---

            return display_frame
            
        except Exception as e:
            # En son çare - herhangi bir hata durumunda orijinal frame'i döndür
            print(f"Process frame'de kritik hata: {str(e)}")
            return frame
    
    def update_ui(self, frame, current_frame):
        """UI elemanlarını güncelle"""
        try:
            print(f"DEBUG: update_ui called. Frame shape: {frame.shape if frame is not None else 'None'}, Current Frame No: {current_frame}") # DEBUG
            # Null kontrol
            if frame is None:
                print("Boş frame, UI güncelleme atlanıyor")
                return
                
            # Boyut kontrolü
            if len(frame.shape) != 3:
                print(f"Geçersiz frame boyutu: {frame.shape}")
                return
                
            # OpenCV BGR formatını RGB'ye çevir
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Canvas boyutunu al
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Geçerli canvas boyutu kontrolü
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas henüz hazır değil, atla
                return
            
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
            if self.simple_mode:
                fps_text += " [BASİT MOD]"
            if self.force_cpu:
                fps_text += " [CPU]"
                
            self.canvas.create_text(10, 10, text=fps_text, fill="white", 
                                font=('Arial', 12, 'bold'), anchor=tk.NW)
            
            # --- Kitlenme dörtgeni sayaç/metin ekle ---
            lock_text = getattr(self, 'locked_object_show_text', "Obje yok")
            self.canvas.create_text(10, 40, text=lock_text, fill="yellow", font=('Arial', 16, 'bold'), anchor=tk.NW)
            # --- ---
            
            # Görüntüyü canvas'a yerleştir (ortalanmış)
            x_position = (canvas_width - new_width) // 2
            y_position = (canvas_height - new_height) // 2
            self.canvas.create_image(x_position, y_position, anchor=tk.NW, image=self.photo)
            
            # Zaman etiketini güncelle
            if self.cap is not None:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                current_time = current_frame / fps
                self.time_label_start.config(text=self.format_time(current_time))
                
        except Exception as e:
            print(f"UI güncelleme hatası: {str(e)}")
    
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
        if self.simple_mode:
            # Basit modda, direkt frame'i göster
            self.update_ui(frame, self.current_frame)
        else:
            processed_frame = self.process_frame(frame)
            self.update_ui(processed_frame, self.current_frame)
        
        # Eşik değerlerini durum çubuğunda göster
        if self.detect_var.get() and self.model is not None and not self.simple_mode:
            self.status_label.config(text=f"Confidence: {self.conf_threshold:.2f}, IOU: {self.iou_threshold:.2f}")
    
    def toggle_detection(self):
        """Nesne tespitini açıp kapama"""
        if self.detect_var.get():
            if self.simple_mode:
                # Basit moddan çıkıp nesne tespitini aç
                print("Basit moddan nesne tespiti moduna geçiliyor...")
                self.simple_mode = False
                # Başlık güncelleme
                self.update_title()
            
            if self.model is None:
                self.load_yolo_model(self.model_var.get())
        else:
            # Nesne tespiti kapalı - basit mod otomatik devreye girmez
            print("Nesne tespiti kapatıldı")
    
    def update_title(self):
        """Pencere başlığını güncelle"""
        mode_info = []
        if self.simple_mode:
            mode_info.append("BASİT MOD")
        if self.debug_mode:
            mode_info.append("HATA AYIKLAMA")
        if self.force_cpu:
            mode_info.append("CPU MODU")
        
        if mode_info:
            self.root.title(f"CenkerVision - YOLO Tabanlı Video Oynatıcı [{', '.join(mode_info)}]")
        else:
            self.root.title("CenkerVision - YOLO Tabanlı Video Oynatıcı")
    
    # Yeni buton ekle - basit mod ve tam mod arası geçiş için
    def add_simple_mode_button(self):
        """Basit mod butonu ekle"""
        # Bu metodu __init__ içinde çağırabilirsiniz
        self.simple_mode_var = tk.BooleanVar(value=self.simple_mode)
        self.simple_mode_checkbox = ttk.Checkbutton(
            self.video_frame, 
            text="Basit Mod (Sadece Video)", 
            variable=self.simple_mode_var,
            command=self.toggle_simple_mode
        )
        self.simple_mode_checkbox.pack(side=tk.TOP, padx=5, pady=5)
    
    def toggle_simple_mode(self):
        """Basit mod geçişi"""
        new_mode = self.simple_mode_var.get()
        if new_mode != self.simple_mode:
            # Modu değiştir
            self.simple_mode = new_mode
            
            if self.simple_mode:
                print("Basit mod etkinleştirildi - YOLO işlemi atlanacak")
                # Basit modda nesne tespitini devre dışı bırak
                self.detect_var.set(False)
            else:
                print("Tam mod etkinleştirildi - YOLO işlemi etkin olabilir")
            
            # Başlık güncelleme
            self.update_title()
            
            # Eğer video oynatılıyorsa, durdur ve yeniden başlat
            if self.is_playing:
                was_playing = True
                self.stop_play_thread()
            else:
                was_playing = False
            
            # Video oynatılıyorduysa yeniden başlat
            if was_playing:
                self.root.after(100, self.toggle_play)
    
    def slider_changed(self, value):
        """İlerleme çubuğu değiştiğinde"""
        value = float(value)
        if self.cap is not None and not self.is_webcam: # Sadece video dosyaları için
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                current_time = value / fps
                self.time_label_start.config(text=self.format_time(current_time))
            else:
                 self.time_label_start.config(text=f"Kare: {int(value)}")
    
    def slider_released(self, event):
        """İlerleme çubuğu bırakıldığında"""
        if self.cap is None or self.is_webcam: # Webcam için bu fonksiyonu atla
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
        if self.cap is None or self.is_webcam: # Webcam için bu fonksiyonu atla
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
        if seconds == float('inf') or seconds < 0:
            return "N/A"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return "{}:{:02d}".format(mins, secs)
    
    def close_app(self):
        """Uygulama kapatılırken temizlik"""
        self.stop_play_thread()
        if self.cap is not None:
            self.cap.release()
            self.cap = None # cap'i None olarak ayarla
        self.root.destroy()

    def get_tracker_config_path(self):
        """ByteTrack konfigürasyon dosyasının yolunu al"""
        # Önce projede yerel olarak kontrol et
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trackers", "bytetrack.yaml")
        
        if os.path.exists(local_path):
            return local_path
            
        # Yerel dosya yoksa, ultralytics paketindeki kopyayı kullanmaya çalış
        try:
            from ultralytics.cfg import get_cfg
            # Ultralytics v8+ için
            return get_cfg('trackers/bytetrack.yaml')
        except:
            # Dosya bulunamadıysa None döndür, varsayılan ayarlar kullanılacak
            return None
            
    def toggle_tracking(self):
        """Nesne takip özelliğini aç/kapat"""
        # Takibi etkinleştir veya devre dışı bırak
        self.enable_tracking = self.track_var.get()
        
        if self.enable_tracking:
            print("ByteTrack nesne takibi etkinleştirildi")
            # Eğer nesne tespiti açık değilse, otomatik olarak aç
            if not self.detect_var.get():
                self.detect_var.set(True)
                self.toggle_detection()
                
            # Takip özelliği için bellek optimizasyonu
            if torch.cuda.is_available():
                # CUDA belleğini temizle
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                # MPS optimizasyonu için belleği serbest bırak
                import gc
                gc.collect()
                
            # Takip yapılandırmasını zorla yeniden yükle
            try:
                self.tracker_config_path = self.get_tracker_config_path()
                if self.tracker_config_path and os.path.exists(self.tracker_config_path):
                    print(f"ByteTrack yapılandırması yüklendi: {self.tracker_config_path}")
                    # Yapılandırma dosyasını oku
                    with open(self.tracker_config_path, 'r') as f:
                        config_content = f.read()
                        print(f"Yapılandırma içeriği:\n{config_content}")
                else:
                    print("ByteTrack yapılandırma dosyası bulunamadı, varsayılan ayarlar kullanılacak")
            except Exception as config_error:
                print(f"Yapılandırma dosyası okuma hatası: {str(config_error)}")
        else:
            print("Nesne takibi devre dışı bırakıldı")
            
            # Takip devre dışı bırakıldığında belleği temizle
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        
        # Durum çubuğunu güncelle
        track_status = "Etkin ✓" if self.enable_tracking else "Devre Dışı ✗"
        self.status_label.config(text=f"ByteTrack Takip: {track_status}")

    def get_color_for_id(self, track_id):
        """Takip ID'si için renkli bir renk döndür"""
        # Her ID için farklı bir renk döndür (basit hash fonksiyonu ile)
        colors = [
            (0, 255, 0),    # Yeşil
            (255, 0, 0),    # Mavi
            (0, 0, 255),    # Kırmızı
            (255, 255, 0),  # Camgöbeği
            (255, 0, 255),  # Mor
            (0, 255, 255),  # Sarı
            (128, 0, 0),    # Koyu mavi
            (0, 128, 0),    # Koyu yeşil
            (0, 0, 128),    # Koyu kırmızı
            (128, 128, 0),  # Koyu camgöbeği
            (128, 0, 128),  # Koyu mor
            (0, 128, 128)   # Koyu sarı
        ]
        return colors[track_id % len(colors)]


def main():
    root = tk.Tk()
    app = CenkerVision(root)
    root.mainloop()


if __name__ == "__main__":
    main()
