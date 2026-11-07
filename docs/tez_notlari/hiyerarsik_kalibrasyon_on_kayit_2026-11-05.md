# Ön kayıt: ızgara dalı için hiyerarşik Bayesçi kalibrasyon, Aşama 1 (5 Kasım 2026, test sonucu görülmeden)

**Kaynaklar:**
- Ayrıntılı tasarım ve türetmeler: `hiyerarsik_kalibrasyon_on_kayit_TASLAK_2026-11-05.md`. 7 ajanlı iş akışı: kod haritası, 3 tasarım, 2 yargıç, sentez.
- Bu belge onun **bağlayıcı, sadeleştirilmiş** hâli. Çelişki olursa bu belge geçerlidir.

## 1. Neden

**Bilinen sonuç:** ShT'de ızgara dalı nesne dalına eklenince mikro AUC 79.5'ten 70.2'ye düşüyor. Eşli bootstrap farkı −9.3, %95 CI [−14.0, −4.8]. Makro AUC ise 81.9'dan 82.3'e hafifçe yükseliyor.

**Normal veride, etiket kullanmadan yapılan ön tanı:**
- Sahne başına kalibrasyon %5 ve %1 düzeylerinde kabaca tutarlı.
- Uç kuyruk ise tutarlı değil. Fd'nin kuyruğu ağır: GPD ξ̂ 12 sahnenin 11'inde 0.16 ile 0.96 arasında.
- Mevcut kalibratör üstel kuyruk (ξ = 0) varsayıyor. Bu yüzden normal karelerde z > 10 olan kare sayısı Fd'de 190; beklenen 4.1, yani 46 kat fazla.
- Küçük sahnelerde kuyruk parametreleri gürültülü. Kuyruk parametresi β'nın video bootstrap'iyle varyasyon katsayısı Fd'de 0.73'e kadar çıkıyor.

## 2. Yöntem: HB-POT

Gövde değişmez. Sahne başına 90. yüzdelik u_s'nin altında ampirik kuyruk kullanılır (`FisherCalibrator`).

**Kuyruk:** u_s'nin üstünde genelleştirilmiş Pareto (GPD):
- ξ = 0 iken mevcut üstel kuyruğa iner.
- Sahne parametreleri θ_s = (ξ_s, log σ_s/d_s); burada d_s = Q90 − Q50.
- Önce GPD MLE ile kestirilir. Belirsizlik Σ_s, video küme bootstrap'iyle bulunur: B = 500, tohum 0, ardından J_s/(J_s − 1) düzeltmesi.

**Hiyerarşi** (Şahin ve Arısoy'daki yerelden globale büzmenin bir düzey yukarısı):
- θ_s ~ N(μ, T) ve θ̂_s | θ_s ~ N(θ_s, Σ_s).
- (μ, T) yalnız normal referanslardan REML ile kestirilir.
- Sonsalın ξ ≥ 0'a kesik modu alınır.
- Az videolu sahne, veri seti ortalamasına daha çok çekilir.
- Kalibrasyon düzeyindeki bu hiyerarşi bizim uzantımız. Hocanın kaynaklarınınmış gibi sunulmaz.

**Kollar** (yalnız üçü; başka kol yok):

| Kol | Tanım |
|---|---|
| T0 | Mevcut kalibratör (ξ = 0, σ = aşım ortalaması). `FisherCalibrator`'ın bit bit aynısı |
| T1 | Sahne başına GPD, havuzlama yok |
| **T2** | EB sonsalının kesik modu. **Birincil kol** |

**Diğer setler:** Tek sahneli setlerde (Ped1, Ped2, Avenue) T2 ≡ T1 olur. Bütün setlerde aynı yordam kullanılır.

**Kapsam:**
- Nesne dalının kalibratörü değişmez.
- Sabit değerler: q = 0.9, B = 500, ξ ∈ [−0.5, 1.5] sınırları, tohum 0.
- Test verisiyle hiçbir şey seçilmez.

## 3. Önce normal veri kapısı (etiketsiz)

**Yordam:** Her eğitim videosu dışarıda bırakılarak (LOVO) kalibratör yeniden uydurulur ve o video puanlanır. Ölçülenler:
- kuyruk log-skoru,
- sahne başına FAR/α (%5, %1, %0.1),
- z > 10 sayısı,
- havuzlanmış I+Fs+Fd toplamında, havuzun %99 noktasının üstüne çıkan sahne paylarının sahneler arası standart sapması.

**Kapı hipotezleri:**
- **G1:** Fd'de T2'nin kuyruk log-skoru T0'dan büyük ve CI alt sınırı 0'ın üstünde.
- **G2:** Fd'de T2 ile z > 10 olan kare sayısı ≤ 12 (T0'da 190).
- **G3:** I ve Fs'de z > 10 sayısı T0'ı aşmıyor, ve %1 FAR/α en az 11/12 sahnede [0.5, 2.5] aralığında.
- **G4:** Sahneler arası standart sapma T2'de T0'dan küçük (T0'da 0.43 yüzde puanı).

**Sıra:**
1. Kapı sonucu (`gate.json`), test değerlendirmesinden **önce** commit edilir.
2. Kapı tutmasa da test bir kez koşulur ve sonuç raporlanır.
3. ShT'de başka ayar denenmez.

## 4. Test değerlendirmesi (tek koşu)

**Kurulum:** Tur 2 `base` nesne dalı, toplam füzyon, nedensel medyan 9, warmup 4. Ölçüler mikro ve makro AUC; eşli video bootstrap 2000 tekrar, tohum 0.

**Yeniden üretim kontrolü:** T0 kolu şu değerleri 1e-4 hassasiyetle üretmeli:

| Set | ızgara + nesne | nesne |
|---|---|---|
| ShT | 0.7020 | 0.7954 |
| Ped1 | 0.7641 | 0.7086 |
| Ped2 | 0.9653 | 0.9385 |
| Avenue | 0.8738 | 0.8278 |

Üretemezse diğer kollar açılmadan hata aranır.

## 5. Hipotezler

**Birincil:**
- **H1 (ShT):** mikro(nesne + ızgara_T2) − mikro(nesne + ızgara_T0) > 0, ve %95 CI alt sınırı 0'ın üstünde.

**İkincil** (düzeltilmemiş %95 CI ile raporlanır):
- **H1b, hiyerarşinin kendisi:** T2 − T1, H1 ile aynı biçimde.
- **H2, zarar giderildi mi:** mikro(nesne + ızgara_T2) − mikro(nesne).
  - Zayıf başarı: CI üst sınırı ≥ 0.
  - Güçlü başarı: CI alt sınırı > 0.
- **H3, mekanizma ve hata kontrolü:**
  - Tek kanalın makro değişimi |Δ| < 0.005 olmalı. Aşılırsa bu bir uygulama hatasıdır.
  - Fd tek başına mikro T2 − T0 için CI alt sınırı 0'ın üstünde.
- **H4, tek sahne koruması:** Ped1, Ped2 ve Avenue'da mikro(nesne + ızgara_T1) − mikro(nesne + ızgara_T0) ≥ −0.01.
- **H5, makro koruması:** ShT'de nesne + ızgara_T2 makro AUC'si ≥ 0.8125.

**Karar kuralı:**
- T2, yalnız H1 ve H4 birlikte tutarsa makalenin ızgara kalibratörü olur; † ve commit hash'iyle işaretlenir.
- Aksi halde T0 kalır ve Aşama 1 olumsuz sonuç olarak raporlanır.

**Öncül beklenti (spekülasyon):** Etki büyük olasılıkla küçük olacak. Aşama 1 esas olarak Fd'nin uç kuyruğunu düzeltir; I kanalının ShT'deki zayıflığını onaramaz.

## 6. Sonraki aşama (ayrı ön kayıt)

**Aşama 2:** I ve Fd için hücre → sahne → veri seti NIW hiyerarşisi. Bu, Şahin denklem 21'deki veriye bağlı α'nın genellemesi. Tasarım taslakta §4'te.

**Kaynak ihtiyacı:** GPU'da AE geçişi ve ShT önbelleklerinin yeniden kurulması gerekiyor.

**Sıra:** Aşama 1'in test sonucu açılmadan önce ayrı bir belge olarak ön kayda girer. Böylece Aşama 1'in sonucu Aşama 2'nin tasarımını etkileyemez.
