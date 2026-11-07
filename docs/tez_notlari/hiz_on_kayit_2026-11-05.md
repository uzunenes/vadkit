# Ön kayıt: hızlandırma (iş planı C, 5 Kasım 2026)

**Bugüne kadar ölçülenler:** `paper/ieee_access/kanit/hiz_20261105/`. Ortam: RTX 3090 + EPYC 7542, batch 1, Avenue.

| Adım | Sonuç |
|---|---|
| Sıralı çalışma | 13.5 FPS |
| C0: akış GPU aşamalarıyla paralel | 18.2–20.0 FPS (paylaşımlı makinede çalıştırmadan çalıştırmaya oynuyor). Çıktı birebir aynı (80 kare) |
| C1: poz modeli de paralel | Çıktı aynı (120 kare), ama **kazanç yok**: 18.6–18.8 FPS. İki YOLO aynı anda çalışınca dedektör yavaşlıyor. Seçenek kapalı kalacak |

**Profil:**
- YOLO11n tek karede 10.8 ms; bunun 8.5 ms'si "inference". FP16'da 12.8 ms, yani FP16 kazanç getirmiyor.
- MobileCLIP2-S0, 7 kırpma: FP32'de 17.1 ms, FP16'da 19.9 ms.
- Kırpma döngüsü yalnız 2.2 ms.

Küçük ağlarda süreyi Python ve kernel çağırma yükü belirliyor.

## C2: TensorRT (FP16) motorları

**Ne denenecek:** YOLO11n, YOLO11n-pose ve MobileCLIP2-S0 görüntü kodlayıcısı için TensorRT FP16 motorları. YOLO'lar Ultralytics'in `export(format="engine")` aracıyla, kodlayıcı ONNX üzerinden üretilecek; kodlayıcı sabit batch kovalarıyla çalışacak.

**Adım 1, yalnız hız:** Bileşen başına ve uçtan uca FPS ölçülür. Uçtan uca kazanç %20'nin altındaysa C2 burada bırakılır ve sonuç raporlanır.

**Adım 2, doğruluk:** FP16 sayısal sonucu değiştirdiği için doğruluk kontrolü gerekiyor.
- Ped1 (geliştirme seti) özellikleri TensorRT arka ucuyla yeniden çıkarılır.
- Nesne dalının nedensel mikro AUC'si FP32 sonucuyla karşılaştırılır.
- Fark **≤ 0.5 puan** ise TensorRT arka ucu "eşdeğer" sayılır ve makalede hız bu arka uçla verilir. Ped2, Avenue ve ShT de yeniden koşulup raporlanır.
- Fark daha büyükse FP32 hızı ana sonuç olarak kalır; TensorRT sonucu ayrı satırda, AUC farkıyla birlikte verilir.

**Seçim:** Hiçbir seçimde test etiketi kullanılmaz. Ped1 yalnız geliştirme seti olarak kullanılır; seçim kuralı yukarıdaki sabit eşiktir.
