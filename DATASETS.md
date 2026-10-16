# Hamsokhan (هم‌سخن) — Persian Speech Datasets Landscape
> **مجموعه دادگان صوتی زبان فارسی و تفکیک وظایف ۳ لایه‌ای در پروژه هم‌سخن**

---

## ۱. جدول مرجع و مشخصات جامع دیتاست‌ها

| نام دیتاست (Dataset) | نرخ نمونه‌برداری (Sample Rate) | حجم (ساعت) | گویندگان (Speakers) | شناسه گوینده (speaker_id) | کیفیت آکوستیک (DNSMOS / MOS) | مجوز (License) | زمان‌بندی (Timestamps) | سبک گفتار (Conversational) | فرمت داده (Format) | لینک مستقیم (Direct Link) |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **YodaLingua-Farsi** | 24 kHz | 72 h | 678 | ✅ | ✅ > 3.0 | CC BY 4.0 | ❌ بدون تراز کلمه | ❌ رسمی | MP3 / Parquet | [HuggingFace](https://huggingface.co/datasets/YodaLingua/YodaLingua-Farsi) |
| **ManaTTS** | 44.1 kHz | 114+ h | 1 | ❌ | استودیویی عالی | CC0 | ⚠️ تراز در پایپ‌لاین | ❌ رسمی استودیویی | Audio / Parquet | [HuggingFace](https://huggingface.co/datasets/hezarai/mana-tts) |
| **ParsVoice** | 16 kHz | 2,177.9 h | 1,803 | ✅ | ⚠️ متادیتا بدون DNSMOS | بررسی مجزا | ❌ | ❌ رسمی / تلفنی | FLAC / Parquet | [HuggingFace](https://huggingface.co/datasets/hezarai/parsvoice) |
| **Persian-Speech-Dataset** | 16 kHz | ≈ 13 h | 87 | ✅ | ❌ | MIT | ❌ | ❌ با تگ احساسات | Audio / Parquet | [HuggingFace](https://huggingface.co/datasets/SLT-KNTU/Persian-Speech-Dataset) |
| **Synthetic Farsi Diarization** | 16 kHz | ≈ 60 h | Synthetic | ⚠️ زمانی | MIT | ✅ مرز دقیق گوینده | ⚠️ سنتتیک | Audio / Parquet | [GitHub / HF](https://huggingface.co/datasets) |
| **PersianAudiobook** | 16 kHz | 218.56 h | نامشخص | ❌ | ❌ | Source-dependent | ⚠️ start_sec / end_sec | ❌ کتاب گویا | FLAC / Parquet | [HuggingFace](https://huggingface.co/datasets/Kamal/PersianAudiobook) |
| **Persian-Farsi-Speech (Thomcles)** | **24 / 44.1 kHz** | **417 h** | Multi | ✅ | ✅ MOS > 3.5 | CC BY-SA 4.0 | ❌ | ❌ پاکسازی‌شده | Parquet / WAV | [HuggingFace](https://huggingface.co/datasets/Thomcles/Persian-Farsi-Speech) |
| **GPTInformal-Persian** | 44.1 kHz | 6+ h | 1 | ❌ | ✅ کیفیت بالا | CC BY-NC 4.0 | ⚠️ Segment-level | ✅ **محاوره‌ای خالص** | Audio / Parquet | [HuggingFace](https://huggingface.co/datasets/MahtaFetrat/GPTInformal-Persian) |
| **PersianVox** | **16 kHz** | **~150 h** | Multi (Web) | ✅ | ⚠️ Prosody score | Research Only | ✅ **Turn timestamps** | ✅ **محاوره‌ای پادکست** | FLAC / Parquet | [HuggingFace](https://huggingface.co/datasets/saeedzou/persianvox_all) |
| **persian-voice-v1 (vhdm)** | 16 / 48 kHz | 100+ h | 1,000+ | ✅ | ⚠️ اصلاح GPT-4o | CC0 / MPL | ❌ | ❌ Common Voice | MP3 / Parquet | [HuggingFace](https://huggingface.co/datasets/vhdm/persian-voice-v1) |
| **Filimo-Persian-ASR** | 16 kHz | 50+ h | Multi | ❌ | ⚠️ نویزی | Research | ⚠️ ساب‌تایتل سریال | ✅ محاوره‌ای عاطفی | WAV / Parquet | [HuggingFace](https://huggingface.co/datasets/PerSets/filimo-persian-asr) |
| **Neyshekar (نیشکر)** | 16 / 48 kHz | ~50 h | 500+ | ✅ | ❌ | Open Access (CC0) | ❌ | ⚠️ تنوع لهجه‌ها | WAV / Parquet | [HuggingFace](https://huggingface.co/datasets/shekar-ai/neyshekar) |
| **persian-tts-youtube-clean** | **24 / 44.1 kHz** | **32 h** | 1 | ✅ | ✅ > 3.2 | CC BY 4.0 | ⚠️ Sentence-level | ✅ **محاوره‌ای روان** | WAV / Parquet | [HuggingFace](https://huggingface.co/datasets/ErfanRou/persian-tts-youtube-clean-32h) |

---

## ۲. تفکیک وظایف مهندسی در ۳ لایه معماری (3-Layer Taxonomy)

با توجه به محدودیت‌های محاسباتی کارت **NVIDIA T4** و الزامات فرکانسی وکودر **Vocos (24kHz)**، تمام دادگان بالا در ۳ لایه تخصصی دسته‌بندی می‌شوند:

```
[مخزن داده‌های خام فارسی]
 │
 ├──► لایه ۱: سنتز آکوستیک (≥ 24kHz, MOS > 3.2) ──► Thomcles (417h) + ManaTTS (114h) + YouTube Clean (32h)
 │
 ├──► لایه ۲: نوبت‌گیری و وقفه (Turn-Taking) ──────► PersianVox (150h) + Synthetic Diarization (60h)
 │
 └──► لایه ۳: حذفیات از سنتز (16kHz / نویزی) ──────► ParsVoice, PersianAudiobook, Filimo (رد برای سنتز)
```

### لایه ۱: داده‌های طلایی سنتز آکوستیک و وکودر Vocos (Acoustic Flow Matching & Vocoder)
* **داده‌ها:**
  - **Persian-Farsi-Speech (Thomcles - 417h):** داده اصلی پیش‌آموزش و انطباق آکوستیک با فرکانس بالای ۲۴ کیلوهرتز و امتیاز MOS بالاتر از ۳.۵.
  - **ManaTTS (114h):** ضبط تک‌گوینده استودیویی فوق‌العاده شفاف تحت لایسنس CC0 جهت تضمین پایداری هویت صوتی و کالیبراسیون لحن پایه.
  - **persian-tts-youtube-clean (32h):** گفتار محاوره‌ای، روان و امروزی بدون نویز موسیقی برای طبیعی‌سازی لحن خروجی.
* **پیش‌پردازش اجباری:**
  - ریسمپل به **۲۴,۰۰۰ هرتز مونو (Mono 24kHz)**.
  - نرمال‌سازی بلندی صدا بر اساس استاندارد **ITU-R BS.1770 به -20 dBFS**.
  - برش هوشمند VAD با حفظ **۱۶۰ms نویز محیطی در ابتدا** و **۲۴۰ms در انتها** به همراه فید کسینوسی ۱۵ms.
  - کوانتیزه‌سازی به ماکرو-پکت‌های **۱۶۰ میلی‌ثانیه‌ای** (دقیقاً ۱۵ فریم مِل Vocos).

---

### لایه ۲: داده‌های ویژه نوبت‌گیری، دیالوگ و قطع کلام (Turn-Taking Head & Barge-in)
* **داده‌ها:**
  - **PersianVox (150h):** استخراج‌شده از پادکست‌ها با متادیتای زمانی نوبت گفتار. منبع اصلی آموزش هد باینری نوبت‌گیری (Turn-Taking Head).
  - **Synthetic Farsi Diarization (60h):** دارای برچسب‌های زمانی صریح شروع و پایان برای کالیبراسیون مرزهای تغییر گوینده (Speaker Switch).
* **پیش‌پردازش اجباری:**
  - تولید برچسب‌های نرم شیب‌دار: `[0.15, 0.35, 0.65, 1.0, 1.0, 1.0]` در بازه ۳۲۰ms منتهی به پایان کلام.
  - برچسب‌گذاری مکث‌های تفکر درون‌جمله‌ای کمتر از ۷۲۰ms («اممم...»، «خب...») با **0.0 (Hard Negatives)**.
  - ماسک کردن دنباله سکوت با **-100.0** (`ignore_index`).

---

### لایه ۳: حذفیات مهندسی از آموزش سنتز آکوستیک (Pruned Workloads)
* **داده‌ها:**
  - **ParsVoice (2,177h) و PersianAudiobook (218h):** به دلیل نرخ نمونه‌برداری **۱۶ کیلوهرتز**، وارد پایپ‌لاین سنتز صوت نمی‌شوند (استفاده از آن‌ها رزولوشن خروجی Vocos را به کیفیت تلفنی کاهش می‌دهد). این داده‌ها صرفاً برای آموزش انکودر متنی/ASR در صورت نیاز مجازند.
  - **Filimo-Persian-ASR (50h):** به دلیل نویز محیطی بالا، موسیقی فیلم و تداخل همزمان صداها، برای سنتز رد صلاحیت شده و صرفاً متن‌های محاوره‌ای آن استخراج می‌شود.

---

## ۳. تحلیل متنی و گویندگان

* **YodaLingua-Farsi:** منبع غنی از ۶۷۸ گوینده برای آموزش امبدینگ گوینده (Speaker Embedding).
* **Neyshekar (نیشکر):** ۵۰ ساعت صوت محاوره‌ای با لایسنس آزاد و تنوع لهجه‌های اقوام ایرانی (کردی، لری، آذری، اصفهانی، مشهدی) برای تست استحکام مدل در برابر تنوع لهجه‌ها.
* **persian-voice-v1 (vhdm):** نسخه بازنویسی‌شده Common Voice با GPT-4o برای جلوگیری از خطاهای املایی کاربران در تطبیق متن و صوت.
