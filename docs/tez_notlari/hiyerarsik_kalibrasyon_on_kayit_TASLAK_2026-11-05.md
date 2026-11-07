# Ön kayıt TASLAĞI: ShT ızgara dalı için hiyerarşik Bayesçi kalibrasyon (5 Kasım 2026; yeni kolların hiçbir test sonucu görülmeden yazıldı)

> **TASLAK.** Ana ajan gözden geçirip commit eder. Commit hash'i `kanit/hb_20261105/README.md`'ye yazılır ve makalede atıf olarak verilir. Commit'ten önce yeni kolların hiçbiri test verisinde koşulmaz.
> Bu çalışma ShT test sonucunu gördükten sonra doğdu. Bu yüzden makalede bütünüyle † ile işaretlenir.

## 0. Seçilen tasarım (yargıç kararlarıyla)

Çalışma üç aşamalı. Üç aşamanın hepsi bu belgede, Aşama 1'in testi açılmadan kayda geçer. Bir aşamanın test sonucu, sonraki aşamanın tasarımını değiştirmez.

| Aşama | Ne | Kaynak | Hesap |
|---|---|---|---|
| 1 | **HB-POT:** Fisher kalibratörünün üstel kuyruğu, ξ ≥ 0 olan GPD kuyruğuna genişletilir. Sahnelerin kuyruk parametreleri ampirik Bayes ile veri setine büzülür | calib-eb; robust-tail'in C3 kestiricisi ikincil kol | Yalnız CPU, arşiv `ref/` dizileri; hemen koşabilir |
| 2 | **Hücre → sahne → veri seti NIW hiyerarşisi.** Önce referans protokolü (yarı model yerine LOVO), sonra hiyerarşi | cell-hier + yargıç eklemeleri | GPU'da AE geçişi + CPU |
| 3 | **Kontrol:** I kanalı için AE çapraz uydurma | yargıç önerisi | GPU, ayrı kullanıcı onayı |

**Yargıç kararlarından alınanlar:**
- T2 sabit birincil. T2 ile T2p arasında seçim yapılmaz.
- ξ ≤ 0 kırpması yerine sonsalın ξ ≥ 0'a kesilmesi kullanılır.
- T2 − T1 karşıtlığı (hiyerarşinin kendisi) eklendi.
- Havuz %99 noktası üstündeki sahne payları kapıya bağlantı ölçüsü olarak eklendi.
- B0L − B0 (referans protokolü) ayrı bir doğrulayıcı hipotez oldu.
- Fd için C'nin sahne düzeyi kovaryans tabanı alındı.
- B−nV (n = V_s) kolu eklendi.
- AE çapraz uydurma kontrolü eklendi.
- E1 ve E2 açıklayıcı çözümlemeleri eklendi.
- Bootstrap 2000 tekrar, tohum 0.

**Bırakılanlar:**
- PIT skoru (C2), havuzlanmış kalibratör (C4), kare ortak ölçeği (C5) ve robust-t ECM (C1).
- B'nin "+2.0 puan" eşiği.
- I kanalını füzyondan çıkarmak. Testten esinlenen bir değişiklik olur; ancak † ile ve sonradan ek olarak raporlanabilir.

**Yargıçlardan ayrılan tek nokta:** Yargıçlar T1 için C3 kestiricisini önerdi. T2 − T1 farkının yalnız havuzlamayı ölçmesi için T1, T2'nin τ → ∞ limiti olarak tanımlandı. C3 ayrı bir kol olarak (T1cv) raporlanır.

## 1. Neden

### Bilinen test sonuçları

Bunlar zaten görüldü ve bu çalışmada hiçbir seçimde kullanılmaz.

- **ShT** (12 sahne, 107 test videosu; model ve kalibrasyon sahne başına):
  - Nesne dalının mikro AUC'si 79.5. Izgara dalı eklenince 70.2'ye düşüyor: fark −9.3 [−14.0, −4.8]. Eşli video bootstrap, 2000 tekrar, tohum 0.
  - Makro AUC aynı geçişte 81.9'dan 82.3'e çıkıyor.
  - Kaynaklar: `kanit/r2_20261105/boot/shanghaitech.json`, `kanit/r2_20261105/res/base/shanghaitech_{obj,grid}.json`.
- **Izgara kanalları tek başına (mikro / makro):** I 55.8 / 66.3, Fs 73.0 / 78.3, Fd 72.7 / 79.9. Üç kanalın toplamı tek başına 66.0.
- **Tek sahneli setler** (ızgara+nesne − nesne):
  - Ped1 +5.6 [2.8, 8.2]
  - Ped2 +2.7 [−0.6, 7.3]
  - Avenue +4.6 [−0.6, 10.9]
  - Kaynak: `boot/{ped1,ped2,avenue}.json`.
- **Nesne dalının kalibrasyonu:** ShT'de eşik sahne başına seçildiğinde iyi kalibre. α = %5'te test-normal FAR/α 0.72 (`kurulum_olcumleri_sonuc_2026-11-05.md`).
- **Daha az normal video ile:** Videoların %25'i kullanıldığında ShT mikro 0.795'ten 0.726'ya düşüyor; makro ise yalnız 0.819'dan 0.810'a (aynı dosya).

**Okuma (sınanacak hipotez):** Mikro AUC düşüyor, makro düşmüyor. Bu, kaybın sahne içi sıralamadan değil, sahneler arası ölçek uyumsuzluğundan geldiğini düşündürüyor. Hipotez şu: ızgara kanallarının sahne başına kalibrasyonu sahneler arasında karşılaştırılabilir değil. Referanslar küçük ve gürültülü; sahne düzeyinde kaymalar var.

### Normal veride ön tanı (etiketsiz)

- **Ne kullanıldı:** Yalnız `ref/` dizileri, sahne başına LOVO. Test etiketi ve test skoru kullanılmadı.
- **Kaynaklar şu an scratchpad'de:**
  - `/tmp/claude-1000/-home-lei-Desktop-msc/e8dc9422-76c8-42e7-babd-ff39d7dc137d/scratchpad/gridref/diag_out.txt`
  - `/tmp/claude-1000/-home-lei-Desktop-msc/e8dc9422-76c8-42e7-babd-ff39d7dc137d/scratchpad/gridref/diag_tail_out.txt`
  - `/tmp/claude-1000/-home-lei-Desktop-msc/e8dc9422-76c8-42e7-babd-ff39d7dc137d/scratchpad/angleC/tail_c_out.txt`
- **Commit'ten önce** bu dosyalar betikleriyle birlikte `kanit/hb_20261105/on_tani/`'ya kopyalanmalı. Aşağıdaki sayılar ancak o zaman repodan kaynaklı olur.

**Bulgular:**
- **Referans boyutları:** Sahne başına I 248–7677, Fs/Fd 769–23365 değer. Görev metnindeki "154–850" ShT için geçerli değil.
- **%5 ve %1 düzeyleri sahneler arasında kabaca tutarlı:**
  - LOVO FAR/α, %5'te 0.55–1.82.
  - %1'de I 1.03–2.47, Fs 0.63–1.83, Fd 0.95–3.51.
  - Referanslar sahneler arasında havuzlanınca çok daha kötü: bir sahne dışarıda bırakılarak ölçülen FAR/%5, 0.00–4.43.
  - Sonuç: gövde ve u_s sahne başına kalmalı.
- **Sorun uç kuyrukta, özellikle Fd'de:**
  - Fd'nin üst %10'una uydurulan GPD şekli ξ̂, 11/12 sahnede 0.16–0.96 (sahne 03'te −0.19). Mevcut kalibratör ise ξ = 0 varsayıyor.
  - z > 10 olan tutulmuş normal kare sayısı ve beklenen değeri: Fd 190 / 4.09, Fs 22 / 4.09, I 11 / 1.34.
  - Fd'de en büyük z 75.6.
- **Az videolu sahnede kuyruk kararsız:**
  - Fd sahne 07'de (8 video), %99 noktasının kalibre değeri video bootstrap'ında [3.15, 34.59] aralığında; nominal değer 4.61.
  - u'nun değişim katsayısı 1.31.
- **Sahne başına GPD kuyruğunun etkisi (C3; ξ video-ayrık 5 katla seçildi):**
  - Fd'de z > 10 sayısı 190'dan 27'ye, en büyük z 75.6'dan 17.1'e iniyor.
  - %1 düzeyi düzelmiyor: Fd 0.93–3.94.
  - I+Fs+Fd toplamında, havuz %99 noktası üstündeki sahne paylarının yayılımı artıyor: SS 0.43'ten 0.57 yüzde puana.
- **Bağımlılık:** Video arası varyans payı η² üç kanalda 0.04–0.72. Örnekleme birimi kare değil, video olmalı.

**Çıkarım (spekülatif):** Referanslar kendi içinde tutarlı görünüyor. Bu yüzden −9.3'ün büyük kısmı, referanslarda görünmeyen bir **referans → test uyumsuzluğundan** gelebilir. İki olası kaynak var:
- (i) I referansları AE için örneklem içi: AE bütün eğitim videolarıyla eğitildi (`evaluate.py:168-211`).
- (ii) Referanslar yarı veriyle uydurulan modelden, test skorları tam veriyle uydurulan modelden geliyor (`aean/channels.py:398-417`). Bu fark, sahnenin video sayısı J_s'ye bağlı.

Hangi aşama neyi hedefliyor:
- Aşama 1: yalnız kuyruk.
- Aşama 2: (ii) ve hücre istatistikleri.
- Aşama 3: (i).
- E1–E2 (§6): kaybın nereden geldiğini etiketle açıklar.

## 2. Ortak çerçeve (bütün aşamalarda değişmez)

- **Kanallar:** I (`frx_bayes`), Fs (`flow_ubm_m`), Fd (`flow_rx_m`). Kaynak, arşivdeki `scores.npz` dosyaları; yollar `vad_bench/vast/runs_r2/run.sh` içindeki `grid()` şablonlarında.
- **Nesne dalı:** Tur 2 `base` yapılandırması (ShT için `base_sht.yaml`, `--scenes`). Özellikler `/workspace/r2/feat/base/<ds>`. Nesne dalının `fisher` kalibratörü değişmez. 79.5 / 70.2 bu yapılandırmadan geldi.
- **Füzyon ve zaman:** Toplam füzyon, nedensel medyan 9; her videonun ilk 4 karesi atılır.
- **Ölçüler:** Nedensel mikro AUC (havuz) ve makro AUC (video başı ortalama).
  - Eşli video bootstrap: `vadkit/eval/protocol.py::paired_bootstrap`, 2000 tekrar, tohum 0, yüzdelik CI.
  - Makro AUC için yalnız nokta kestirimi verilir.
- **Tek yapılandırma:** Her veri setinde aynı yordam, aynı ızgaralar ve aynı seçim kuralı. Değerler yalnız o setin normal eğitim verisinden seçilir. Test anında her adım kare başına donmuş bir dönüşüm, ardından nedensel medyan. Yöntem kesinlikle nedenseldir.
- **Monotonluk notu (Aşama 1 için hata kontrolü):**
  - Her kol, sahne başına azalmayan bir dönüşüm. Kollar arasında gövde aynı; kuyruk her kolda kesin artan.
  - Nedensel medyan bu dönüşümlerle yer değiştirir. İstisna, çift uzunluklu başlangıç pencereleri: ısınmadan sonraki 2., 4., 6. ve 8. kareler.
  - Bu yüzden Aşama 1 değiştiremez: tek kanalın makro AUC'sini; tek sahneli setlerde tek kanalın mikro AUC'sini.
  - ShT'de tek kanalın mikro AUC'si yalnız sahneler arası karşılaştırılabilirlik yoluyla değişebilir. Füzyon ise ek olarak uç kuyruktaki kanal ağırlıkları yoluyla değişebilir.

## 3. Aşama 1: HB-POT (kalibrasyon düzeyinde hiyerarşi)

### 3.1 Model

Her kanal ayrı uydurulur. Gösterim:
- s = 1..S sahneler.
- R_s: sahne s'nin referansı, n_s değer.
- J_s: sahne s'nin eğitim videosu sayısı.

**Gövde** (değişmez, `vadkit/calibration/fisher.py`):
```
u_s = Q_0.9(R_s)
x ≤ u_s:  z_s(x) = −log[(n_s − #{r < x} + 1) / (n_s + 1)]
```

**Kuyruk** (GPD; ξ = 0 mevcut kalibratör). y = x − u_s > 0 için:
```
ξ_s > 0:  z_s(x) = ln 10 + (1/ξ_s) · log(1 + ξ_s·y/σ_s)
ξ_s = 0:  z_s(x) = ln 10 + y/σ_s
```
T0'da ξ_s = 0 ve σ_s = β_s (aşım ortalaması). Bu, `FisherCalibrator`'ın bit bit aynısıdır.

**Boyutsuz parametreler:**
```
d_s = Q_0.9(R_s) − Q_0.5(R_s)
θ_s = (ξ_s, ρ_s),   ρ_s = log(σ_s / d_s)
```

**Sahne kestirimi:**
- θ̂_s: aşımlar üzerinde GPD MLE. Konum 0, ξ ∈ [−0.5, 1.5] ile sınırlı.
- Σ_s: video-küme bootstrap ile. B = 500, tohum 0; her tekrarda u_s, d_s ve θ̂_s yeniden uydurulur.
- Küçük örneklem düzeltmesi: Σ_s × J_s/(J_s − 1).

**Hiyerarşi** (normal–normal ampirik Bayes):
```
θ_s ~ N(μ, T) · 1(ξ_s ≥ 0),      T = diag(τ_ξ², τ_ρ²)
θ̂_s | θ_s ~ N(θ_s, Σ_s)
```
(μ, T) REML ile kestirilir. Kesme bu adımda yok sayılır; bu bir yaklaşıklıktır.
```
ℓ_R(μ, T) = −½ Σ_s [ log|T + Σ_s| + (θ̂_s − μ̂_T)ᵀ (T + Σ_s)⁻¹ (θ̂_s − μ̂_T) ] − ½ log|Σ_s (T + Σ_s)⁻¹|
μ̂_T = GLS ortalaması
```

**Sonsal:**
```
V_s = (T⁻¹ + Σ_s⁻¹)⁻¹
θ̃_s = V_s (T⁻¹ μ̂ + Σ_s⁻¹ θ̂_s)
```
- Skaler halde θ̃_s = (1 − a_s)·θ̂_s + a_s·μ̂, burada a_s = Σ_s/(Σ_s + τ²).
- Σ_s yaklaşık 1/J_s ile orantılı. Bu yüzden az videolu sahne veri setine daha çok çekilir.

**ξ ≥ 0 kısıtı:** Sonsal ξ ≥ 0 yarı düzlemine kesilir. Kesik Gauss'un modu:
- ξ̃_s ≥ 0 ise θ* = θ̃_s.
- Değilse ξ* = 0 ve ρ* = ρ̃_s − (V_s[ρ,ξ] / V_s[ξ,ξ]) · ξ̃_s.
- Her iki durumda σ* = d_s · e^{ρ*}.

Kısıtın gerekçesi: ξ < 0 sonlu bir uç nokta verir ve uç noktanın ötesinde z = ∞ olur. Ayrıca kısıt, kuyruğun mevcut üstel kuyruktan hafif olmasını engeller.

**Hocanın hattıyla ilişki:**
- Şahin (Tez No 545380) Denk. 18–21 ve Arısoy doktora tezi Denk. 2.51–2.52: yerelden globale MAP büzmesi, ağırlığı örnek sayısına bağlı. Burada aynı yapı bir düzey yukarıda kullanılıyor: sahne kalibrasyonu → veri seti. τ sabit değil, marjinal olabilirlikten kestiriliyor.
- Eşlenik MAP biçimi: Kayabol EUSIPCO 2015 Denk. 11–12.
- T2p kolu, Arısoy Denk. 2.22–2.23'teki Student-t sonsal tahmininin benzeri.
- Kalibrasyon düzeyindeki bu hiyerarşi bizim uzantımız. Onlarınmış gibi sunulmaz.

### 3.2 Kollar

| Kol | Tanım | Rol |
|---|---|---|
| T0 | ξ = 0, σ = β_s | mevcut yöntem, karşılaştırıcı |
| T1 | havuzsuz: T⁻¹ = 0 (τ → ∞), aynı kesik mod | hiyerarşinin etkisini ayırmak |
| **T2** | EB sonsalının kesik modu | **ShT'de birincil** |
| T2p | sonsal tahmin (aşağıda) | ikincil |
| T3 | tam havuz: T = 0, bütün sahnelerde μ̂ (ölçek d_s ile sahneye taşınır) | ikincil |
| T1cv | C3: ξ ∈ {0, .05, .1, .2, .3, .5, .7}, sahne başına video-ayrık 5 katta dışarıda tutulmuş aşım olabilirliğiyle seçilir; σ = β_s(1 − ξ) | ikincil (ön tanının kestiricisi) |

**T2p** için sonsal tahmin:
```
P̃_s(R > u_s + y) = 0.1 · (1/M) Σ_m (1 + ξ_m·y/σ_m)^(−1/ξ_m)
(ξ_m, ρ_m) ~ N(θ̃_s, V_s), ξ ≥ 0'a kesik;   M = 256, tohum 0
z = −log P̃
```

**İndirgenme:**
- T1cv'de ξ = 0 seçilirse yöntem T0'ın tam aynısı olur.
- T1 sınırda (ξ* = 0) yalnız yaklaşık olarak T0'dır.
- S = 1'de (Ped1, Ped2, Avenue) havuzlanacak sahne yoktur ve T2 ≡ T1. Bütün setlerde tek yapılandırma kullanılır (`fisher_hb`); tek sahneli setlerde bu T1'e iner ve T0'a karşı raporlanır.

### 3.3 Neye ne uydurulur

- **Kullanılan veri:**
  - Yalnız `ref/<key>` dizileri.
  - Eğitim kare sayılarından yeniden kurulan video kimlikleri; dizi boyutlarının eşit olduğu assert ile denetlenir.
- **Video ve kare sırası:**
  - I referansı: sıralı eğitim videoları, kareler t = 15, 24, … (`evaluate.py:203-211`).
  - Fs/Fd: önce çift, sonra tek indisli videolar, kareler [::3] (`aean/channels.py:406-415`).
- **Körlük:** Kapı, `ref/` anahtarlarından ayrı bir dosyaya kopyalanmış dizilerle çalışır. Test anahtarları ve `gt` hiç açılmaz.
- **Sabit düğmeler:**
  - q = 0.9 (değişmez), B = 500, M = 256, ξ sınırları [−0.5, 1.5].
  - C3 ızgarası ve 5 kat.
  - τ ve μ REML'den gelir.
  - Başka ayar yoktur ve test verisiyle hiçbir şey seçilmez.

### 3.4 Normal veri kapısı

Kapı önce koşar. `gate.json` test değerlendirmesinden önce commit edilir.

**LOVO yordamı.** Her sahne s ve her eğitim videosu v için:
1. Gövde, u_s, d_s ve θ̂_s, v olmadan yeniden uydurulur.
2. Σ_s^(−v) = Σ_s · J_s/(J_s − 1) alınır.
3. (μ, T), sahne s'nin kestirimi değiştirilerek yeniden kestirilir.
4. v'nin kareleri puanlanır.

T1cv'de ξ, iç 5 katla kalan videolarda seçilir.

**Ölçüler** (her kanal, her kol):
- **Kuyruk log-skoru:** Aşımlarda log 0.1 + log g(y). Gövde terimleri kollar arasında aynı olduğu için düşer.
  - Önce sahne içinde kare ortalaması, sonra sahneler arası ağırlıksız ortalama alınır.
  - CI: sahne içinde tutulan videolar üzerinde küme bootstrap, 1000 tekrar, tohum 0.
- **Sahne başına FAR/α:** α ∈ {%5, %1, %0.1}.
- **z > 10 sayısı:** nominal olasılık e^−10.
- **Sahneler arası tutarlılık:** Hizalı I+Fs+Fd toplamı (kareler t = 15 + 9j) havuzlanır. Havuzun %99 ve %99.9 noktaları üstünde kalan sahne paylarının sahneler arası SS'si hesaplanır.

**Kapı hipotezleri:**
- **G1 (Fd):** T2 − T0 kuyruk log-skoru > 0 ve CI alt sınırı > 0. Neredeyse kesin geçer; yalnız akıl sağlığı kontrolüdür.
- **G2 (Fd):** T2 ile z > 10 olan kare sayısı ≤ 12, yani beklenen 4.09'un yaklaşık 3 katı. Ön tanıda bu sayı T0'da 190, C3'te 27 idi; tutmayabilir.
- **G3 (I, Fs):**
  - z > 10 sayısı T0'ı aşmaz (I'da 11, Fs'de 22).
  - %1 FAR/α, her kanalda en az 11/12 sahnede [0.5, 2.5] aralığında kalır.
- **G4 (kapı ile mikro AUC arasındaki bağ):** Havuz %99 noktası üstündeki sahne paylarının SS'si T2'de T0'dan küçüktür (ön tanıda T0 0.43 yüzde puan). C3 bu ölçüyü kötüleştirmişti (0.57).

Kapı tutmazsa da test bir kez koşulur ve sonuç olumsuz olarak raporlanır. ShT'de başka ayar denenmez.

### 3.5 Test değerlendirmesi

Değerlendirme tek koşudur.

- **Tek süreç:** Nesne durumu bir kez uydurulur. T0, T1, T2, T2p, T3 ve T1cv aynı nesne skorlarıyla hesaplanır.
- **Yeniden üretim kontrolü:** T0 kolu, ızgara+nesne ve nesne için aşağıdaki mikro değerleri 1e-4 içinde üretmeli. Betikte assert vardır: üretmezse diğer kollar yazdırılmadan hata aranır.

  | Set | ızgara+nesne | nesne |
  |---|---|---|
  | ShT | 0.7020 | 0.7954 |
  | Ped1 | 0.7641 | 0.7086 |
  | Ped2 | 0.9653 | 0.9385 |
  | Avenue | 0.8738 | 0.8278 |

- **Setler:** ShT birincil, 107 video. Ped1, Ped2 ve Avenue koruma için.

### 3.6 Hipotezler (Aşama 1)

**Doğrulayıcı hipotez:**
- **H1 (birincil):** mikro(nesne + ızgara_T2) − mikro(nesne + ızgara_T0) > 0, %95 CI alt sınırı > 0.

**İkincil hipotezler** (düzeltilmemiş %95 CI ile raporlanır):
- **H1b (hiyerarşinin kendisi):** T2 − T1, H1 ile aynı biçimde.
- **H2 (zarar giderildi mi):** mikro(nesne + ızgara_T2) − mikro(nesne).
  - Zayıf başarı: CI üst sınırı ≥ 0.
  - Güçlü başarı: CI alt sınırı > 0.
  - Öncül beklenti: güçlü başarı olası değil.
- **H3 (mekanizma):**
  - (a) Her kolda tek kanal makro AUC değişimi |Δ| < 0.005. Aşılırsa bu bir uygulama hatasıdır.
  - (b) Fd tek başına: mikro T2 − T0 için CI alt sınırı > 0.
  - (c) Tahmin: I tek başına mikro |Δ| < 0.02. Yani Aşama 1 I'yı onaramaz.
- **H4 (tek sahne koruması):**
  - Ped1, Ped2 ve Avenue'nun her birinde mikro(nesne + ızgara_T1) − mikro(nesne + ızgara_T0) nokta kestirimi ≥ −0.01.
  - Ped1'de ızgaranın nesneye kazancı anlamlı kalır (CI alt sınırı > 0).
  - Tek kanal mikro ve makro değişimi |Δ| < 0.005 (hata kontrolü).
- **H5 (makro koruması):** ShT'de nesne + ızgara_T2 makro AUC'si ≥ 0.8125 (= 0.8225 − 0.01). Nokta kestirimi.

**Karar kuralı:** T2, makalenin ızgara kalibratörü olur (†, commit hash'iyle) yalnız H1 ve H4 birlikte tutarsa. Aksi halde T0 kalır ve Aşama 1 olumsuz sonuç olarak raporlanır.

**Öncül beklenti (spekülasyon):** Aşama 1 esas olarak Fd'nin uç kuyruğunu düzeltir; mikro AUC'ye etkisi küçük olur. H1 tutmasa bile, kapıdaki Fd kuyruk düzeltmesi normal veri bulgusu olarak yazılır.

## 4. Aşama 2: hücre → sahne → veri seti NIW hiyerarşisi

### 4.1 Model

**Kanallar:** k ∈ {I, Fd}.
- I: C = 1; AE artığı piksel başına, 16 px hücre.
- Fd: C = 9; 8 yönlü log1p akış histogramı + hücre maksimumu, 8 px hücre.
- Fs hiyerarşik olarak modellenmez; onda yalnız referans protokolü değişir (§4.3).

**Etkin sayı** (yalnız normal eğitim verisi):
- Her hücre (s, g) için kare özeti y_{v,t} alınır: I'da hücredeki piksellerin ortalaması, Fd'de özellik vektörü.
- Bu özet üzerinde tek yönlü rastgele etki ICC'si ρ_{s,g} hesaplanır: bileşenler üzerinden ortalama, [0.01, 1] aralığına kırpılır. Toplam varyans < 1e-8 ise ρ = 1 alınır.
```
deff_{s,g} = 1 + (m̄ − 1) · ρ_{s,g}           m̄ = ANOVA ortalama grup boyu
n_{s,g}    = F_{s,g} / deff_{s,g}            (V_s ≤ n_{s,g} ≤ F_{s,g})
```
- Her gözleme n_{s,g} / (hücredeki gözlem sayısı) ağırlığı verilir. Bu ağırlık hücre ortalamasını ve kovaryansını değiştirmez; yalnız verinin önsel karşısındaki ağırlığını belirler.
- **Gerekçe:** Sabit kamerada her hücre aynı sayıda kare görür. Kare sayısıyla, Şahin Denk. 21'deki sayıya bağlı α sahne içinde sabit kalır ve bilgi taşımaz.

**Düzey 3 → 2** (yalnız S > 1). Sahne s'nin veri seti önseli, sahne s dışarıda bırakılarak kurulur:
```
μ_D = ort_{s'≠s} x̄_{s'}
Σ_D = ort_{s'≠s} [ Σ̂_{s'} + (x̄_{s'} − μ_D)(x̄_{s'} − μ_D)ᵀ ]
μ̃_s = (κ1·μ_D + n_s·x̄_s) / (κ1 + n_s)
Σ̃_s = [ (ν1 − C − 1)·Σ_D + n_s·Σ̂_s + (κ1·n_s/(κ1 + n_s))·(x̄_s − μ_D)(x̄_s − μ_D)ᵀ ] / (ν1 + n_s − C − 1)
```
S = 1 ise μ̃_s = x̄_s ve Σ̃_s = Σ̂_s alınır. Bunlar mevcut global istatistiklerdir (`aean/scoring.py:142-145`).

**Düzey 2 → 1.** Hücre önseli NIW(μ̃_s, κ0, (ν0 − C − 1)·Σ̃_s, ν0), n = n_{s,g}. Sonsal:
```
κ_n = κ0 + n      ν_n = ν0 + n      μ_n = (κ0·μ̃_s + n·x̄) / κ_n
Ψ_n = (ν0 − C − 1)·Σ̃_s + n·Σ̂ + (κ0·n/κ_n)·(x̄ − μ̃_s)(x̄ − μ̃_s)ᵀ
```

**Skor:** Sonsal tahmin dağılımı çok değişkenli t'dir (Arısoy Denk. 2.22–2.23; `aean/niw.py`).
```
dof = ν_n − C + 1
Λ   = Ψ_n·(κ_n + 1)/(κ_n·dof) + ε_s·I,    ε_s = 1e-3 · tr(Σ̃_s)/C   (sahne düzeyi mutlak taban)
δ   = (x − μ_n)ᵀ Λ⁻¹ (x − μ_n)
s   = ½ (dof + C) · log(1 + δ/dof)
```
- Log-determinant terimi yoktur (`scoring.py:264-266` gibi).
- Sahne düzeyi taban, hep sıfır olan Fd hücrelerinde kovaryansın çökmesini önler.
- Kare skoru değişmez: 48 px yerel ortalamanın maksimumu.

**Mevcut yöntemin içerilmesi:** S = 1, κ0 = ν0 − C − 1 = n·α/(1 − α) ve Gauss skoru ile model, `BackgroundStats.finalize`'a iner (α = 0.5 → κ0 = n). Tasarımcının numpy kontrolünde en büyük fark 4e-15 idi; birim testinde yeniden doğrulanır.

**Bayesçi çıkarım olarak durumu:** Yukarıdan aşağı bir eklenti yaklaşıklığıdır. Üst düzeyin sonsal ortalaması alt düzeyin önsel merkezi olur; üst düzeyin belirsizliği aşağı aktarılmaz. Tam çıkarım (Gibbs ya da VB) 12 sahneyle gerekçelendirilemiyor. Bu, sınırlılık olarak yazılır.

### 4.2 Hiperparametre seçimi (normal veri)

- **Izgaralar** (`aean/niw.py:106-107` ile aynı):
  - κ ∈ {0.1, 1, 3, 10, 30, 100, 300, 1000, 1e4}.
  - ν = C + 1 + e, e ∈ {1, 3, 10, 30, 100, 1000}.
  - Hücre düzeyi için ek "bağlı" seçenek: κ0 = ν0 − C − 1 = r·n_{s,g}, r ∈ {1/9, 1/3, 1, 3, 9}. r = 1 ile Gauss skoru mevcut yöntemdir, yani seçim mevcut yöntemi de döndürebilir.
- **Ölçüt:**
```
J(θ) = Σ_s (1/|H_s|) · Σ_{v ∈ V_s} Σ_{x ∈ H_v} log p(x | V_s ∖ v, θ)
```
  - H_v: v'nin referans kareleri. I'da sabit 4×4 piksel kafesi, Fd'de bütün hücreler.
  - Her sahne eşit ağırlıklıdır.
  - Seçim iki adımlıdır: önce (κ1, ν1), sonra (κ0, ν0).
  - Eşitlik r = 1 lehine bozulur.
- **Veri seti başına tek θ:** ShT sahnelerinde ortak. Aynı yordam her veri setinde uygulanır.
- **Kullanılan kareler:**
  - I istatistikleri: t = 15, 18, …
  - Fd istatistikleri: t ≥ 2. 0–1 kareleri, 2. karenin nedensel olmayan kopyası (`aean/channels.py:88`).
  - LOVO, yeterli istatistiklerden çıkarmayla kesin ve ucuz hesaplanır.

### 4.3 Referans protokolü

- **B0 (mevcut, yeniden hesaplanır):** Videolar pariteye göre 2 kata ayrılır; her kat diğer yarının modeliyle puanlanır (yarı model).
- **LOVO:** Her eğitim videosunun referans kareleri, o video çıkarılmış modelle puanlanır.
  - I: Hücre istatistikleri LOVO. AE tüm videolarla eğitilmiş kalır, yani örneklem içidir (Aşama 3 bunu sınar).
  - Fs: UBM bütün normal videolarda bir kez uydurulur; hücre MAP uyarlaması LOVO. Böylece UBM örneklem içi kalır.
  - Fd: Hücre istatistikleri LOVO.
- **Test kareleri** her zaman bütün videolarla uydurulan modelle puanlanır.
- **Kalibratör:** Sahne başına T0 ya da Aşama 1'in T2'si.

### 4.4 Kollar

Bütün kollar aynı koşuda yeniden hesaplanır.

| Kol | Tanım | Rol |
|---|---|---|
| B0 | α = 0.5, Gauss skoru, parite referansı | yeniden üretim, karşılaştırıcı |
| B0L | B0 + LOVO referansları (I, Fs, Fd) | referans protokolünün etkisi (H6) |
| **B** | 3 düzey + etkin sayı + t skoru + LOVO referansı; θ olabilirlikle seçilir | **hiyerarşinin etkisi (H7)** |
| B−L3 | düzey 3 yok | ablasyon |
| B−t | Gauss skoru | ablasyon |
| B−nV | n = V_s (ICC yerine) | basit seçenek |
| B0L×T2, B×T2 | Aşama 1 kalibratörüyle | 2×2 (kalibratör × hücre modeli), betimsel |
| Nesne | ızgara yok | referans |

### 4.5 Test etiketinden önceki kontroller

Bu kontrollerin sonuçları commit edildikten sonra test etiketine geçilir.

- **Yeniden üretim (etiketsiz):** Yeniden hesaplanan B0 ile arşivdeki `scores.npz` karşılaştırılır: en büyük mutlak fark ve video başına Spearman korelasyonu, hem test skorları hem referanslar için.
  - I'nın yakın çıkması beklenir.
  - Fs ve Fd, DIS sürümü ve atılan 0–1 kareleri yüzünden kayabilir.
  - Karşılaştırıcı yeniden hesaplanan B0'dır; arşiv sayıları yanında verilir.
- **G5 (mekanizma, normal veri):** İç içe LOVO kullanılır: her tutulan videonun kalibratörü, diğer videoların LOVO referanslarıyla uydurulur. I ve Fd ayrı ayrı; B, B0L'ye karşı karşılaştırılır.
  - (a) En büyük sahne %1 FAR/α ≤ 2.0 ve sahneler arası aralık B0L'dekinden en az %25 dar.
  - (b) z > 10 olan kare sayısı ≤ 5·n·e^−10 ve B0L'deki sayının en fazla yarısı.
  - Not: Normal veri içindeki LOVO, referans → test kaymasını göremez. G5 bir mekanizma kaydıdır, test sonucunun öngörücüsü değil.
- **Betimsel (etiketsiz):** Sahne başına, test karelerinin o sahnenin %5 eşiğini aşma payı; B ve B0 için.

### 4.6 Hipotezler (Aşama 2)

**Doğrulayıcı hipotezler.** İki hipotez var; Bonferroni düzeltmesiyle her biri %97.5 CI ile sınanır.
- **H6 (referans protokolü):** mikro(nesne + ızgara_B0L) − mikro(nesne + ızgara_B0), ikisi de T0 ile. %97.5 CI alt sınırı > 0.
- **H7 (hücre hiyerarşisi):** mikro(nesne + ızgara_B) − mikro(nesne + ızgara_B0L), T0 ile. %97.5 CI alt sınırı > 0. Nokta kestirimi de raporlanır.

**İkincil hipotezler ve korumalar:**
- **H8 (zarar):** mikro(nesne + ızgara_son) − mikro(nesne). "Son", Aşama 1'de T2 benimsendiyse B×T2, değilse B×T0'dır.
  - Zayıf başarı: CI üst sınırı ≥ 0.
  - Güçlü başarı: CI alt sınırı > 0.
  - Bugünkü değer: −9.3 [−14.0, −4.8].
- **H9 (tek sahne koruması):** Ped1, Ped2 ve Avenue'da, ızgara tek başına ve ızgara + nesne için, B − B0 ve B0L − B0 farkları: nokta kestirimi ≥ −0.01 ve CI üst sınırı > 0.
- **H10 (makro koruması):** ShT'de nesne + ızgara_son makro AUC'si ≥ makro(nesne + ızgara_B0, yeniden hesaplanmış) − 0.01.

**Karar kuralı:**
- B0L referans protokolü olur, yalnız H6 tutarsa ve B0L − B0 için H9 tutarsa.
- B makalenin ızgara yöntemi olur, yalnız H7 tutarsa ve B − B0 için H9 tutarsa.
- Aksi halde sonuç olumsuz olarak raporlanır ve B0 kalır. Benimsenen her şey † taşır.

**Öncül beklenti (spekülasyon):**
- En umutlu hipotez H6. J_s'ye bağlı yarı model / tam model asimetrisini doğrudan kaldırıyor.
- H7, Fd için olası. I için düzey 3 muhtemelen κ1 → 0'a gider, çünkü 12 farklı AE'nin artıkları değiştirilebilir değil.
- H8'de güçlü başarı olası değil.

## 5. Aşama 3: I kanalı için AE çapraz uydurma

Bu aşama bir kontroldür ve ayrı GPU onayı gerektirir.

- **Yordam:**
  - Her ShT sahnesinde eğitim videoları pariteye göre ikiye bölünür.
  - Her yarıda AE aynı mimari ve ayarlarla yeniden eğitilir. Toplam 12 × 2 = 24 eğitim.
  - I referansları karşı yarının AE'siyle ve LOVO hücre istatistikleriyle üretilir.
  - Test skorları tam veriyle eğitilmiş AE'den gelir; değişmez.
- **Etiketsiz kontrol:** Sahne başına, I referansının Q0.9 değerinde örneklem içi / çapraz uydurma oranı. Beklenti (spekülasyon): oran < 1, yani örneklem içi artık daha küçük.
- **H11 (doğrulayıcı, %95):** mikro(nesne + I_cf + Fs + Fd) − mikro(nesne + I + Fs + Fd), ikisi de B0L protokolü ve T0 ile. CI alt sınırı > 0. I tek başına mikro değişimi de raporlanır.
- **Yorum ve benimseme:**
  - H11 tutarsa: "I'nın örneklem içi referansı ShT kaybının bir nedenidir".
  - Benimsemenin ek koşulları: tek sahneli setler için de çapraz uydurulmuş AE'ler eğitilir (3 × 2 = 6 eğitim daha) ve H9 ölçütü bu setlerde tutar.

## 6. Açıklayıcı çözümlemeler

Bu çözümlemeler test etiketi kullanır, en son raporlanır ve hiçbir seçimde kullanılmaz.

- **E1 (havuz AUC'nin ayrıştırılması):** Havuz AUC, sahne içi ve sahneler arası çift terimlerine ayrılır.
```
AUC = w·AUC_iç + (1 − w)·AUC_arası
w   = Σ_s A_s·N_s / (A·N)
```
  A_s ve N_s, sahne s'nin anomali ve normal kare sayılarıdır (ısınmadan sonra). Ayrıştırma nesne, nesne + ızgara (T0, T2, B0, B0L, B) ve her ızgara kanalı için yapılır.
- **E2 (referans → test kayması):** Test videolarının normal karelerinde (gt = 0), sahne ve ızgara kanalı başına FAR/α.
  - Eşik z > −ln α; gövdede bu, referansın (1 − α) kantiline karşılık gelir.
  - α ∈ {%5, %1}; T0 ve T2; düzeltilmemiş kare skorları.
  - Sonuçlar normal veri LOVO değerleriyle (ön tanı C) karşılaştırılır.
  - Bu, referans → test kaymasının doğrudan ölçüsüdür.
  - Beklenti (spekülasyon): I'da test-normal FAR/α'nın sahneler arası yayılımı, LOVO'dakinden (%1'de 1.03–2.47) belirgin biçimde geniş olur.

## 7. Başarısızlıkta raporlanacaklar

- Bütün kolların bütün sayıları raporlanır, olumsuz olanlar dahil: mikro, makro, farklar ve CI'lar, kapılar ve etiketsiz kontroller.
- Başarısız aşamada mevcut yöntem kalır. Makalede ShT ızgara zararı, E1–E2 açıklamasıyla sınırlılık olarak yazılır.
- Test sonucu görüldükten sonra ShT'de yeni ayar denenmez. Yeni bir fikir yeni bir ön kayıt gerektirir ve † taşır.
- Uygulama hatası yüzünden yeniden koşu, yalnız şu kontrollerden biri tetiklerse yapılır: T0/B0 yeniden üretimi, H3a ya da H4'teki |Δ| < 0.005 kontrolü. Yeniden koşu not edilir ve ilk sonuç da saklanır.

## 8. Uygulama planı

### Aşama 1 (CPU; GPU gerekmez)

| Dosya | İş |
|---|---|
| `vadkit/calibration/gpd.py` (yeni) | `GPDTailCalibrator(xi, sigma)`: log1p ile; ξ = 0'da `FisherCalibrator` ile bit bit aynı. `PredictiveGPDCalibrator` (T2p): kuyruk için arama tablosu |
| `vadkit/calibration/hier.py` (yeni) | `gpd_fit(exc, xi_bounds)`, `video_bootstrap(ref, vid, B, seed)`, `eb_reml(theta_hat, Sigma)`, `truncated_mode(theta, V)`, `cv_xi(ref, vid, grid, folds)`, `fit_groups(refs, ids, mode) -> {sahne: Calibrator}`; mode ∈ {t0, t1, t1cv, t2, t2p, t3} |
| `vadkit/plugins.py` | `("calibrator", "fisher_gpd")` ve yeni tür `("group_calibrator", "fisher_hb")` |
| `vadkit/grid/precomputed.py` | `calibrated()` iki geçişe bölünür: önce `{kanal: {sahne: (ref, ids)}}` toplanır, sonra sahne başına `build(...).fit` ya da `fit_groups` çağrılır. Grup kalibratörü video kimliği olmadan çağrılırsa hata verir; sessizce geri düşmez |
| `vadkit/cli.py` | `grid_calibrator` anahtarı, `calibrator`'dan ayrı (nesne dalı `fisher` kalır). `--extra-ref-ids` seçeneği |
| `vadkit/eval/protocol.py` | `paired_bootstrap(..., level=0.95)`: geriye uyumlu; Aşama 2'nin %97.5'i için |
| `tools/grid_ref_ids.py` (yeni) | Scratchpad `diag.py`'deki kimlik mantığı. Çıktı `/workspace/hb/refids/` (yeni dizin); boyutlar assert ile denetlenir |
| `tools/grid_refs_only.py` (yeni) | Arşiv dosyalarından yalnız `ref/` anahtarlarını `/workspace/hb/refs/`'e kopyalar (kör kapı için) |
| `tools/hb_gate.py` (yeni) | §3.4'teki kapı → `kanit/hb_20261105/gate.json` |
| `vad_bench/vast/runs_hb/boot_hb.py` (yeni) | `boot_grid.py` deseninde. Tek süreç, bütün kollar, T0 yeniden üretim assert'i, eşli bootstrap, tek kanal mikro/makro, E1 ve E2 |
| `tests/vadkit/test_calibration_hier.py` (yeni) | Testler aşağıda |

`tests/vadkit/test_calibration_hier.py` şunları sınar:
- ξ = 0'da `FisherCalibrator` ile bit bit eşitlik.
- Sentetik GPD'nin parametrelerinin geri kazanılması.
- τ → ∞'da T1, τ → 0'da T3.
- S = 1'de T1.
- u noktasında süreklilik ve monotonluk.
- Kesik mod.
- Kapı betiğinin test anahtarı açmaması: erişimde hata veren sahte bir npz ile.

**Hesap:**
- İş yükü: 36 (sahne, kanal) × 500 bootstrap GPD uydurması, artı 323 × 3 LOVO yeniden uydurması. Tahmin: dakikalardan birkaç saate kadar CPU (ölçülmedi).
- Test anında maliyet: T2'de kare ve kanal başına bir log1p; T2p'de arama tablosu.
- Test değerlendirmesi vast'ta `CUDA_VISIBLE_DEVICES=""` ile koşar, çünkü r2 özellikleri orada.

### Aşama 2 (GPU'da AE geçişi + CPU)

| Dosya | İş |
|---|---|
| `aean/niw.py` | `NIWCells.finalize(..., prior=None)` (varsayılan değişmez; nesne dalı etkilenmez), `effective_counts()`, `scene_prior(stats_by_scene, kappa1, nu1, leave_out)`, `select_hier_lovo()` |
| `aean/scoring.py` | `HierBackground`: μ_n, Λ'nın Cholesky çarpanı, dof; t skoru ve ε_s tabanı |
| `aean/channels.py` | `CellRXHier` (Fd); `CellUBM` için video başına yeterli istatistik ve LOVO uyarlaması (Fs) |
| `tools/grid_dump.py` (yeni, GPU) | Video başına hücre yeterli istatistikleri (n, s1, s2) ve ICC momentleri (I, Fd). I referans ve test kareleri için float16 artık haritaları. `feat_flow_*`. `gt.json` açılmaz |
| `tools/grid_hier.py` (yeni, CPU) | Seçim, LOVO ve bütün kollar → `/workspace/gridhier/<ds>/{sc}/scores_hier.npz` (`'<key>/<video>'`, `'ref/<key>'`) ve `hyper.json` |
| vadkit | Aşama 1'den sonra değişiklik yok; skorlar `PrecomputedChannels` şablonlarıyla okunur |
| testler | Aşağıda |

Aşama 2 testleri şunları sınar:
- Bağlı r = 1, S = 1 ve Gauss skoru ile `BackgroundStats` ve `CellRX.maps` sonuçlarının 1e-6 içinde üretilmesi.
- S = 1'de düzey 3'ün ortadan kalkması.
- Çıkarmayla hesaplanan LOVO'nun yeniden uydurmayla aynı sonucu vermesi.
- Skor dosyalarında `gt/` anahtarı bulunmaması.

**Hesap** (tasarımcının tahminleri, ölçülmedi):
- ShT sahne önbelleklerinin `prepare_shanghaitech.py` ile yeniden kurulması: 30 dk'dan az CPU.
- Yaklaşık 364 bin kare için DIS: 48 çekirdekte 30–60 dk.
- Yaklaşık 164 bin kare için AE geçişi: yaklaşık 40 GPU-dk. Dayanak: 14.14 ms/kare, 3090 Ti, sahne 03 `results.json`.
- Disk: yaklaşık 11 GB float16.
- CPU'da seçim ve puanlama: 1 saatten az.
- Maliyet: 1 $'dan az.
- Tasarımcının kontrolüne göre ShT sahne önbellekleri ve akış özellikleri vast makinesinde yok.
- Ne zaman: r2 bitip GPU boşaldıktan sonra ya da yeni bir makinede. Makine iş bitince silinir.

### Aşama 3

- 24 AE eğitimi. Arşiv günlüklerine göre sahne başına yaklaşık 5–7 dk; bu süre doğrulanmadı.
- Toplam tahmin yaklaşık 2.5 GPU-saat, artı Aşama 2'deki I geçişi.
- Ayrı kullanıcı onayı gerekir.

## 9. Sıra ve lojistik

1. Bu taslak gözden geçirilir. Ön tanı çıktıları `kanit/hb_20261105/on_tani/`'ya kopyalanır, commit edilir ve hash kaydedilir.
2. Aşama 1'in kodu ve testleri dizüstünde yazılır.
3. Yalnız `ref/` anahtarlarını içeren dosyalar ve video kimlikleri vast'ta yeni bir dizine (`/workspace/hb/`) yazılır, sonra dizüstüne alınır. Kapı koşulur ve `gate.json` commit edilir.
4. Aşama 1'in testi vast'ta CPU ile koşulur; sonuç commit edilir.
5. Aşama 2, GPU boşalınca başlar. Önce etiketsiz kontroller ve G5 commit edilir, sonra test koşulur.
6. Aşama 3, kullanıcı onay verirse koşulur.

**Uyarılar:**
- r2 makinesi kapatılmadan önce ya Aşama 1'in testi orada bitmeli ya da r2 `base` özellikleri ve arşiv `scores.npz` dosyaları yedeklenmeli. Arşiv skorlarının `veri-2026-11-02` release'inde olup olmadığı doğrulanmadı.
- `/workspace/checkpoints` altında hiçbir şey değiştirilmez.
- Veri silmeden önce kullanıcıya sorulur.

## 10. Riskler

1. **Kayıp referans → test kaymasındaysa:** −9.3'ün esas kaynağı buysa Aşama 1 etkisiz kalır. Bu öngörüldü (H3c).
2. **Gerçek anomaliler de sıkışır:** Fd kuyruğunu sıkıştırmak gerçek Fd anomalilerini de sıkıştırır. Ped1'in anlamlı +5.6'sı aşınabilir (H4).
3. **12 sahneyle EB kırılgan:**
   - 6–10 kümeli bootstrap Σ_s'yi küçük kestirir; J/(J−1) düzeltmesi bunu yalnız kısmen giderir.
   - ξ̂'nın normal yaklaşıklığı −0.5 civarında zayıf.
   - u_s ve d_s havuzlanmıyor; Fd sahne 07'de u'nun değişim katsayısı 1.31.
   - Bağımlı karelerin aşımları kümelenmiyor (declustering yok).
4. **Kanallar bağımsız değil:** Fisher toplamı bağımsızlık varsayar ve bu çalışma bunu düzeltmez. Kurulum ölçümünde analitik eşik, ShT ızgara + nesne için %5'te FAR/α 5.3 vermişti.
5. **Olabilirlik tespiti izlemeyebilir:**
   - Bu repoda κ0'ın çapraz doğrulamayla seçilmesi 1–3 değerlerini seçti ve Ped2'yi kötüleştirdi (`evaluate.py:136-138`).
   - Nesne dalında κ0 1e4'e kaçtı.
6. **Tek sahneli setler zıt büzme yönleri istedi** (`izgara_denetimi_2026-11-04.md` §2a):
   - Ped2: FRX-Bayes 0.893, hücre RX 0.804.
   - Avenue: FRX-Bayes 0.718, hücre RX 0.740.
   - Bu yüzden H9 risk altında.
7. **Ağır kuyruk için öncül sonuçlar karışık:** FRX-t 14 koşunun yalnız 5'inde önde; Fd'de çarpık-t üç setin üçünde de geride (`izgara_denetimi_2026-11-04.md` §2c).
8. **Yeniden hesaplama kayması:** Önbellek ve DIS sürümü değişebilir. Bu yüzden Aşama 2'nin karşılaştırıcısı yayımlanan 70.2 değil, yeniden hesaplanan B0.
9. **Çatallı yollar:** Çalışma ShT test sonucundan doğdu. Bu yüzden † ve commit hash'i zorunlu. Kol sayısı fazla; iddia yalnız H1, H6, H7 ve H11 üzerinden kurulur.
10. **Yenilik denetlenmedi:** Kameralar arası hiyerarşik POT kalibrasyonu kullanan bir VAD çalışması bilmiyoruz, ama literatür taranmadı. Yenilik iddia edilmez.

## 11. Kaynaklar

**Hocanın hattı.** Denklem numaraları `kanit/literature/hoca_hatti_kaynaklar_2026-11-04.json` dosyasından:
- Şahin YL tezi (Tez No 545380): Denk. 18–21.
- Kayabol EUSIPCO 2015: Denk. 11–12.
- Arısoy doktora tezi:
  - Denk. 2.13–2.18 (NIW sonsalı)
  - Denk. 2.22–2.23 (Student-t tahmini)
  - Denk. 2.51–2.52 (sayıya bağlı ağırlık)
- Aytekin YL tezi (Tez No 767348): Denk. 2.69.
- Kayabol vd. GRSL 2022: Denk. 1–3.
- Arısoy vd. AEAN (arXiv 2101.08827): 16×16 bloklar.

**Genel istatistik:**
- `refs.bib`'de var: Pickands 1975 (POT), Efron & Tibshirani 1994 (bootstrap), Reynolds vd. 2000 (GMM-UBM, Fs).
- `refs.bib`'de yok ve bu oturumda doğrulanmadı; atıftan önce doğrulanmalı:
  - Balkema & de Haan 1974
  - Davison & Smith 1990 (JRSS-B)
  - Hosking & Wallis 1997 (bölgesel frekans analizi)
  - Cooley, Nychka & Naveau 2007 (JASA)
  - Efron & Morris 1975
