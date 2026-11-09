# Ön kayıt: yerel akışlı ızgara dalı, birleşik uçtan uca hız ölçümü ve nitel şekil (8 Kasım 2026)

Sonuç görülmeden yazıldı. Ölçümler bu belge commit edildikten sonra vast.ai'de yapılır ve sonuçları ne olursa olsun
raporlanır. Bu belge yeni bir yöntem seçimi yapmaz: makaledeki doğruluk sayıları (arşivlenmiş `evaluate.py`
skorları, `kanit/r2_20261105`) değişmez.

## 1. Neden

Makalede iki eksik vardı:
- Izgara dalı akış sınıfının parçası değildi. Bu yüzden nesne + ızgara yapılandırmasının uçtan uca gecikmesi
  ölçülmemişti.
- Gecikme ölçümleri farklı ve paylaşımlı sunucularda yapılmıştı. Bir oturumda GPU 18 ölçümün 13'ünden önce boşta
  değildi.

Yapılanlar:
- Izgara dalı `vadkit/grid/stream.py` (`GridBranch`, `fit_grid`) olarak akış mimarisine alındı.
  `StreamingVAD(..., grid=...)` kanalları I, Fs, Fd, ovel, opose, oapp sırasıyla kalibre eder ve toplar.
- Tek bir ölçüm betiği yazıldı: `tools/bench_e2e.py`.

## 2. Izgara dalının arşivle denkliği (kontrol, seçim değil)

- `tools/grid_check.py` ızgara durumunu eğitim önbelleğinden `evaluate.py` ile aynı kurallarla yeniden fit eder:
  - adım 8, havuz 48, istatistikler t = 15::3, alfa 0.5, I referansları parite katlarıyla t = 15::15
  - Fs ve Fd için `fit_channel`
- Ardından şunları raporlar:
  - test karelerinde (t ≥ gap) arşivlenmiş `pool48_mc4` / `mc5` ham skorlarından en büyük mutlak fark
  - kanal başına nedensel AUC: yeniden fit ve arşiv, ikisi de kendi referansıyla Fisher kalibre, medyan 9, ısınma 4
  - akış == toplu farkı
- Veri setleri Ped2 ve Avenue'dür. Ped2 bu oturumda CPU'da, ikisi de vast'ta GPU ile koşulur.
- **Bu belgeden önce yapılan pilot (CPU, Ped2, `kanit/birlesik_20261108/pilot_ped2_cpu.json`).** Kodun
  çalıştığını görmek için Ped2'de CPU ile bir yeniden fit yapıldı. Sonuçlar:
  - I için referans uzunluğu arşivden farklı çıktı: 262'ye karşı 154.
    - Arşivlenen `mc4` skorları 3 Ekim 12:06'da üretilmiş.
    - `evaluate.py`'nin I referans adımını 15'ten 9'a çeviren `34af7eb` commit'i aynı gün 13:15'te geldi.
    - Bu yüzden `GridConfig.ref_every = 15` yapıldı. Bu bir kod düzeltmesidir, bir ayar seçimi değildir.
  - Pilotta (I referansı ::9 ile) nedensel AUC farkları şöyle çıktı: I +0,0006, Fs +0,0016, Fd −0,00003,
    I+Fs+Fd −0,0003.
  - Akış ile toplu hesap arasındaki en büyük fark 6e-5 oldu.
  - Aşağıdaki ölçüt bu pilot görüldükten sonra yazıldı, ama pilotun hiçbir sonucu bir seçimi değiştirmedi.
- **Bit eşitliği beklenmez:**
  - UBM alt örneklemi cihaza bağlı bir üreteçle çekilir.
  - GPU'da `index_add_` float32 sırası değişebilir.
  - OpenCV DIS sürümü farklı olabilir.
- **Kabul ölçütü:** her kanalda ve I+Fs+Fd füzyonunda |ΔAUC| ≤ 0,005. Ölçüt tutmazsa:
  - akış uygulaması "arşivi yeniden üretmiyor" diye raporlanır;
  - hız sayıları yine raporlanır, çünkü gecikme durum değerlerine bağlı değildir;
  - doğruluk tabloları arşivden kalır.
- `tools/stream_eval.py` Avenue test videolarının tamamını birleşik akış sınıfından geçirir ve şunları raporlar:
  - nesne ve nesne + ızgara nedensel mikro/makro AUC'si
  - akış skorunun çevrim dışı yeniden hesapla farkı

  Bu bir tutarlılık kontrolüdür ve Ek'te raporlanır. Ana tablolardaki sayıların yerine geçmez. Nesne durumu bu
  oturumda yeniden çıkarılan özniteliklerden fit edilir (aşağıya bakın). Arşivle fark çıkarsa olduğu gibi raporlanır.

## 3. Birleşik hız ölçümü (`tools/bench_e2e.py`)

**Donanım**
- vast.ai, 1× RTX 3090, doğrulanmış (verified) teklif, en az 16 etkin CPU çekirdeği.
- İmaj `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`.
- Tek oturum, tek makine. Bütün koşular bu oturumdadır ve başka oturumlarla birleştirilmez.

**Durumlar**
- Nesne durumu `vadkit/configs/r2/base.yaml` ile bu oturumda çıkarılan eğitim özniteliklerinden fit edilir. Arşivlenmiş
  `data/r2_feat/base` yalnızca dizüstündedir.
- Izgara durumu §2'deki GPU fit'idir.
- Gecikme, durumun sayısal değerlerine bağlı değildir. Bileşen sayıları ve boyutları aynıdır.

**Koşullar**
- Veri setleri: Ped2 ve Avenue.
- Arka uçlar:
  - PyTorch: `r2/base.yaml`
  - TensorRT FP16 YOLO11n ve YOLO11n-pose: `r2/base_trt.yaml`
- Modlar:
  - `object`
  - `object+grid`: ızgara dalı ayrı bir iş parçacığında, kendi DIS akışıyla aynı kareyi işler
- Ortak ayarlar:
  - nesne akışı paralel (`--parallel-flow`), ızgara paralel (`--parallel-grid`)
  - batch 1, test kareleri zaman sırasında
  - 20 ısınma + 3 × 100 kare
- Her (veri seti, arka uç) için modlar şu sırayla birer süreçte koşulur: object, object+grid, object+grid, object,
  object, object+grid (ABBAAB). Her mod için 3 koşu yapılır.

**Saat**
- Ölçüm, JPEG okumadan çevrimiçi füzyon skorunun Python sayısı olarak dönmesine kadar sürer. Bu nedenle GPU işi
  bitmiştir.
- Aşama başına senkronizasyon yapılmaz.

**GPU boşta kuralı**
- Her koşudan önce 5 s boyunca 100 ms'de bir `nvidia-smi` örneklenir. Hesaplama süreçleri ve `/proc/loadavg` da
  kaydedilir.
- Bütün örnekler ≤ %5 ve listede yabancı süreç yoksa koşu başlar. Değilse 60 s beklenir; en çok 3 deneme yapılır.
- Bütün denemeler JSON'a yazılır.
- Boşta GPU hiç görülmezse koşu yine yapılır ve `idle_ok: false` diye işaretlenir.
- Koşu sırasında GPU kullanımı bilgi amaçlı örneklenir.

**Raporlanacaklar**
- (veri seti, arka uç, mod) başına:
  - kare başına medyan, p95, p99 ve en büyük gecikme (ms), 300 karenin tamamı üzerinden
  - FPS = 1000 / (koşu medyanlarının medyanı), ayrıca koşular arası aralık
  - boşta denemeleri, yük ortalamaları ve ortam bilgisi (GPU, CPU, sürümler, betik sha256)
- Izgara dalının eklediği süre, aynı oturumdaki iki modun medyan farkı olarak verilir.

**Makaleye giriş kuralı**
- Bu oturumun Avenue TensorRT değerleri, nesne ve nesne + ızgara için makalenin birleşik hız cümlesi olur.
- Ped2 değerleri ve PyTorch değerleri Ek tabloya gider.
- Eski oturumların tabloları (Tablo hız, aşama dökümü; Ek gecikme tablosu) kendi oturum tanımlarıyla Ek'te kalır.
- "Izgara dalı akış sınıfında değil" ve "nesne + ızgara gecikmesi ölçülmedi" cümleleri kaldırılır.
- GPU boşta değil uyarısı yalnızca hâlâ geçerli olduğu ölçümde (eski üçüncü oturum, poz olmadan ölçüm, Ek) kalır.
- Yeni ölçümde `idle_ok` false olan bir koşu olursa bu açıkça yazılır.

## 4. Nitel şekil (Şekil 3) seçim kuralı

- **Veri seti:** Avenue. Poz modeli Ped2'de normal kişi kutularının yalnızca %1'inde anahtar nokta veriyor, bu
  yüzden poz katmanı orada boş kalır. Bu gerekçe, sonuçlar görülmeden önce
  `e1_poz_kapsama_kapisi_on_kayit_2026-11-06.md`'den bilinmektedir.
- **Video:** `tools/stream_eval.py`'nin birleşik akış skoru (nesne + ızgara, çevrimiçi) ile video başına nedensel AUC
  hesaplanır. Yalnızca iki sınıfı da içeren videolar alınır. AUC'ye göre artan sıralamada (eşitlikte ad sırası)
  `len // 2` sıradaki video seçilir. Bu, eski `fig_src/qualitative.py` kuralıdır.
- **Kare:** o videonun ilk etiketli olayı içinde çevrimiçi skorun en yüksek olduğu karedir.
- **Gösterilenler:**
  - üst panel: renkli kare, YOLO kutuları (dedektör düz, hareket kesikli), eşleşen kişilerin iskeletleri, en
    yüksek −log p'li nesne, ızgara haritası (I, Fs ve Fd'nin 48×48 yerel ortalamalarından o karede baskın kanalın
    haritası)
  - alt panel: kare ekseninde altı kanalın −log p_j eğrileri, çevrimiçi füzyon skoru, etiketli olay gölgesi, seçili
    kare ve her karede baskın kanal şeridi
- Eşik çizgisi çizilmez, çünkü etiketsiz eşikler kalibre değildir (makale Limitations).
- Seçim kuralı değiştirilmez. Görsel beğenilmese de bu kare kullanılır.
