# E1: poz kapsama kapısı, ön kayıt (6 Kasım 2026)

**Kaynak:** KISS uzman paneli (iş akışı `kiss-levers-design`, 6 Kasım), öneri E1.

**Kural (etiketsiz):** Veri setinin normal eğitim videolarında poz kapsamasını hesapla:

  c = (poz_ok > 0 olan kişi kutusu sayısı) / (kişi kutusu sayısı, cls == 0)

- c ≥ 0.5 ise poz kanalı (`opose`) füzyona girer.
- c < 0.5 ise poz kanalı ve YOLO11n-pose modeli kullanılmaz.
- Kural veri seti düzeyindedir. Eşik sabittir (0.5); sahne başına sürüm yoktur.
- NIW, Fisher kalibrasyonu, toplam füzyon ve medyan 9 değişmez.

**Ölçülen kapsama** (`data/r2_feat/base/<ds>/training/*.npz`, yalnız normal eğitim verisi; bu dosya yazılmadan önce hesaplandı):

| | kişi kutusu | poz bulunan | c | kapı |
|---|---|---|---|---|
| Ped1 | 61,600 | 6,846 | 0.111 | kapalı |
| Ped2 | 28,505 | 279 | 0.010 | kapalı |
| Avenue | 112,856 | 100,023 | 0.886 | açık |
| ShT | 166,469 | 68,626 | 0.412 | kapalı |

Eşik (0.412, 0.886) aralığında herhangi bir değer olsaydı da karar aynı çıkardı.

**Dürüstlük notu (makalede yazılacak):**
- Bu fikir, test etiketleriyle değerlendirilmiş S1 ablation'ından çıktı. S1'de L-opose satırları Ped1 ve ShT'de D1'den anlamlı biçimde yüksekti.
- Bu yüzden Ped1, Ped2 ve ShT'deki etki **post hoc**'tur; benimsenirse † ile işaretlenir.
- Gerçekten doğrulayıcı olan kısım, kuralın pozu tuttuğu Avenue'daki öngörü (b) ile ızgaralı yapılandırmadaki koruma koşuludur (c).
- Bu dosya, Avenue S1 sonucu (L-opose satırı) görülmeden commit'lenmiştir.

**Sınamalar:**
- **(a) Yeniden üretim.** Kapılı nesne dalı Ped1, Ped2 ve ShT'de S1'in L-opose satırına, Avenue'da S1'in D1 satırına 1e-6 içinde eşit olmalı. Değilse durulur.
- **(b) Doğrulayıcı öngörü (Avenue).** S1'deki L-opose (pozsuz) online micro AUC'si D1'den (pozlu) düşük çıkar. Nokta tahmini ölçüttür; bootstrap aralığı ayrıca raporlanır.
- **(c) Koruma (nesne + ızgara).** Ped1, Ped2 ve ShT'de poz kapalıyken nesne + ızgaranın online micro AUC'si, mevcut nesne + ızgara satırından en fazla 0.005 aşağıda kalır. Nokta tahmini ölçüttür; eşleştirilmiş video bootstrap (2000 tekrar, seed 0) raporlanır.

**Karar:**
- (a), (b) ve (c) birlikte tutarsa kapı varsayılan olur. Makalede † ile ve bu ön kayda atıfla yazılır. Ped1, Ped2 ve ShT kazançları post hoc gözlem olarak anlatılır.
- Poz kapalı veri setlerinde ölçülen parametre sayısı ve hız da raporlanır.
- (b) veya (c) tutmazsa üç kanal her yerde kalır ve kapı olumsuz bir post hoc gözlem olarak raporlanır. Başka eşik veya sahne başına sürüm raporlanmaz.

**KISS-1 ile ilişki:** E1 temel sistemin (üç kanal) üzerinde değerlendirilir. KISS-1 benimsenirse, E1'in KISS-1 üzerindeki sonucu da betimleyici olarak verilir.
