# KISS-1: hız büyüklüğü kanalı, ön kayıt (6 Kasım 2026)

**Değişiklik:** Nesne dalına dördüncü bir nitelik kanalı eklenir: `omag`, yani kutu içi ortalama L1 akış büyüklüğü (c1, 8 yön kutusunun toplamı; `vadkit` kind `magnitude`).
- Bu kanal kayıtlı `vel` dizilerinden hesaplanır; yeni ağ, yeni özellik çıkarma veya yeni hiperparametre yoktur.
- Yoğunluk NIW karışımıdır, normal veride seçilir.
- Kalibrasyon Fisher, füzyon toplam, düzleştirme nedensel medyan 9; hepsi mevcut sistemdeki gibi.
- Kanal, nitelik listesinin sonuna eklenir. Böylece mevcut üç kanalın uydurması bit bit aynı kalır.

**Dürüstlük notu (makalede yazılacak):** Bu değişikliği test etiketleriyle değerlendirilmiş hız pilotu düşündürdü (`kanit/velpilot_20261104/summary.txt`). Pilotta c1 tek başına Ped2'de c8'den yüksek çıktı, Avenue'da düşük. Bu yüzden değişiklik etiketsiz bir seçim değildir. Benimsenirse makalede † (post hoc, test sonuçlarından esinlenmiş) olarak işaretlenir.

**Veri:** Arşivlenmiş `base` özellikleri. Dört veri seti; ShT'de sahne başına. Izgara kanalları (I, Fs, Fd) için arşivlenmiş skor dosyaları kullanılır, ayarlar `runs_r2/run.sh` ile aynıdır.

**Kontroller:**
- Temel satır (üç kanal) `res/base/<ds>_obj.json` ve `<ds>_grid.json` içindeki füzyon online AUC'sini 1e-6 içinde yeniden üretmelidir; üretmezse koşu durur.
- KISS-1'deki ilk üç kanal, temel satırdaki kanallarla bit bit aynı olmalıdır.

**Ölçüt:** Online micro AUC; ayrıca macro AUC ve standart protokol. KISS-1 ile temel arasında eşleştirilmiş video bootstrap: 2000 tekrar, seed 0. Bu, nesne dalı ve nesne + ızgara için ayrı ayrı yapılır.

**Hipotez H-K1:** KISS-1, nesne dalının online micro AUC'sini dört veri setinin en az üçünde artırır ve bunlardan en az ikisinde fark güven aralığı sıfırı dışlar.

**Karar:**
- H-K1 tutarsa KISS-1 yeni varsayılan olur. Makalede † ile ve bu ön kayda atıfla raporlanır.
- Tutmazsa sistem değişmez ve sonuç olumsuz olarak raporlanır.
- Ara durum yoktur. Tek varyant denenir; başka c1/c8 bileşimleri bu turda denenmez.
