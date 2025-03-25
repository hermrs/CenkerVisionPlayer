# CenkerVision - YOLO Tabanlı Video Oynatıcı

CenkerVision, YOLO nesne tespit teknolojisini kullanarak videolarda gerçek zamanlı nesne tespiti yapabilen bir video oynatıcıdır.

## Özellikler

- Video oynatma ve duraklatma
- İleri/geri kare atlama özellikleri
- Video dosyası seçme
- Farklı YOLO modellerini seçme imkanı (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
- Özel (custom) eğitilmiş YOLO modelleri ekleme ve kullanma
- Nesne tespitini açma/kapama
- Confidence (güven) eşiği ayarlama (0-100%)
- IOU (Intersection over Union) eşiği ayarlama (0-100%) 
- Farklı görüntüleme modları:
  - Normal: Standart YOLOv8 görüntüleme
  - Sadece Kutular: Sınıf ve güven değerleri olmadan sadece kutuları gösterir
  - Güven Skorları: Sınıflar olmadan kutuları ve güven değerlerini gösterir
  - Sansürlü: Tespit edilen nesneleri bulanıklaştırır
- Video ilerleme çubuğu

## Gereksinimler

- Python 3.8 veya daha yüksek
- OpenCV
- Tkinter (çoğu Python kurulumunda varsayılan olarak gelir)
- Pillow (PIL)
- Ultralytics (YOLOv8)
- NumPy

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. YOLO modellerini indirmek için internet bağlantısı gerekmektedir. İlk çalıştırmada, seçilen model otomatik olarak indirilecektir.

## Kullanım

Uygulamayı başlatmak için:

```bash
python CenkerVision.py
```

1. "Video Seç" butonuna tıklayarak bir video dosyası seçin.
2. "Oynat" butonuna tıklayarak videoyu başlatın.
3. "Nesne Tespiti" onay kutusunu işaretleyerek YOLO nesne tespitini etkinleştirin.
4. İleri/geri butonlarıyla 10'ar kare atlayabilirsiniz.
5. YOLO Modeli açılır listesinden farklı modeller seçebilirsiniz.

### Özel Model Ekleme

1. "Özel Model Ekle" butonuna tıklayarak kendi eğittiğiniz YOLO modelini (.pt uzantılı) seçin.
2. Seçilen model otomatik olarak models klasörüne kopyalanır ve kullanıma hazır hale gelir.
3. Model eklendikten sonra model listesinde görünecek ve seçilebilecektir.

### Eşik Değerleri Ayarlama

- **Confidence Threshold**: Tespit edilen nesnelerin minimum güven skorunu belirler. Daha yüksek değerler daha az sayıda ancak daha kesin tespitler yapar.
- **Overlap (IOU) Threshold**: Non-maximum suppression (NMS) için IOU eşiği. Aynı nesne için çakışan kutuların filtrelenmesini kontrol eder.

### Görüntüleme Modları

- **Normal**: Standart YOLOv8 görüntüleme - sınıf isimleri ve güven skorları ile birlikte.
- **Sadece Kutular**: Sınıf isimleri ve güven skorları olmadan sadece sınırlayıcı kutuları gösterir.
- **Güven Skorları**: Kutuları ve güven skorlarını gösterir, sınıf isimlerini göstermez.
- **Sansürlü**: Tespit edilen nesneleri bulanıklaştırır.

## YOLO Modelleri Hakkında

- **YOLOv8n**: En hızlı, en hafif ancak doğruluk açısından en düşük model
- **YOLOv8s**: Hafif ve hızlı bir model, temel kullanım için uygun
- **YOLOv8m**: Orta seviyede dengeli model
- **YOLOv8l**: Daha yüksek doğruluk ancak daha yavaş çalışır
- **YOLOv8x**: En yüksek doğruluk ancak en yavaş model

Daha büyük modeller daha doğru tespit sağlarken, daha fazla sistem kaynağı kullanır ve daha yavaş çalışır.

## Not

- Bu uygulama, yüksek performans için güçlü bir grafik kartına sahip sistemlerde daha iyi çalışır.
- YOLOv8 modelleri ilk kez kullanıldığında otomatik olarak indirilecektir.
- Özel modeller "models" klasöründe saklanır.
Below is the translated version:

---

# CenkerVision - YOLO Based Video Player

CenkerVision is a video player capable of performing real-time object detection in videos using YOLO object detection technology.

## Features

- Video playback and pause functionality
- Forward/backward frame skipping capabilities
- Ability to select a video file
- Option to choose from different YOLO models (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
- Adding and using custom-trained YOLO models
- Toggle object detection on/off
- Setting the confidence threshold (0-100%)
- Setting the IOU (Intersection over Union) threshold (0-100%)
- Various viewing modes:
  - **Normal:** Standard YOLOv8 display with class names and confidence scores.
  - **Boxes Only:** Displays only bounding boxes without class names and confidence scores.
  - **Confidence Scores:** Displays bounding boxes along with confidence scores, without class names.
  - **Censored:** Blurs the detected objects.
- Video progress bar

## Requirements

- Python 3.8 or higher
- OpenCV
- Tkinter (included by default with most Python installations)
- Pillow (PIL)
- Ultralytics (YOLOv8)
- NumPy

## Installation

1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. An internet connection is needed to download the YOLO models. On the first run, the selected model will be automatically downloaded.

## Usage

To start the application:

```bash
python CenkerVision.py
```

1. Click the "Select Video" button to choose a video file.
2. Click the "Play" button to start the video.
3. Enable YOLO object detection by checking the "Object Detection" checkbox.
4. Use the forward/backward buttons to skip 10 frames at a time.
5. Choose different models from the YOLO Model dropdown list.

### Adding a Custom Model

1. Click the "Add Custom Model" button to select your own trained YOLO model (.pt extension).
2. The selected model will be automatically copied to the "models" folder and made ready for use.
3. After adding, the model will appear in the model list and can be selected.

### Adjusting Threshold Values

- **Confidence Threshold:** Determines the minimum confidence score required for detected objects. Higher values lead to fewer but more precise detections.
- **Overlap (IOU) Threshold:** The IOU threshold used for non-maximum suppression (NMS). Controls the filtering of overlapping boxes for the same object.

### Viewing Modes

- **Normal:** Standard YOLOv8 display – with class names and confidence scores.
- **Boxes Only:** Displays only bounding boxes without class names and confidence scores.
- **Confidence Scores:** Displays bounding boxes and confidence scores without class names.
- **Censored:** Blurs the detected objects.

## About YOLO Models

- **YOLOv8n:** The fastest and lightest model, but with the lowest accuracy.
- **YOLOv8s:** A light and fast model, suitable for basic usage.
- **YOLOv8m:** A balanced model with intermediate performance.
- **YOLOv8l:** Offers higher accuracy but runs slower.
- **YOLOv8x:** Provides the highest accuracy but is the slowest model.

Larger models provide more accurate detections while consuming more system resources and operating more slowly.

## Note

- This application performs best on systems equipped with a powerful graphics card for high performance.
- YOLOv8 models will be automatically downloaded on their first use.
- Custom models are stored in the "models" folder.

---
