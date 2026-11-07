# Hamsokhan (هم‌سخن) — Kaggle Dual T4 Training & Optimization Guide

> **Hardware Target:** 2x NVIDIA Tesla T4 (Dual-GPU, 16GB VRAM each, Turing sm_75)  
> **Orchestrator:** PyTorch Distributed Data Parallel (DDP) via Hugging Face `accelerate`  
> **Model Repository:** [huggingface.co/Abolfazl27Aflhami88/Hamsokhan](https://huggingface.co/Abolfazl27Aflhami88/Hamsokhan)  
> **Dataset Shards:** [huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards](https://huggingface.co/datasets/Abolfazl27Aflhami88/hamsokhan-shards)

---

## ۱. مرور خط لوله آموزش کگل (Kaggle Training Architecture)

آموزش مدل‌های یکپارچه گفتار‌به‌گفتار به طور معمول نیازمند خوشه‌های پردازشی گران‌قیمت نظیر ۸ کارت H100 یا A100 است. سامانه **هم‌سخن** به گونه‌ای معماری شده است که فرآیند تنظیم دقیق (Fine-Tuning) لایه‌های LoRA، پروجکتور علّی، هد نوبت‌گیری و سینت‌سایزر صوتی را به صورت پایدار بر روی **دو کارت گرافیک ارزان‌قیمت NVIDIA Tesla T4 کگل** به اتمام برساند.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Kaggle Dual T4 Execution                             │
│                                                                                  │
│   [GPU 0: T4 16GB] (Rank 0)                     [GPU 1: T4 16GB] (Rank 1)        │
│   • DDP Local Batch (FP16)                      • DDP Local Batch (FP16)         │
│   • Forward / Backward                          • Forward / Backward             │
│   • Master Checkpoint Thread (Background)       • Worker Gradient Sync           │
│                           │                              │                       │
│                           └──────────────┬───────────────┘                       │
│                                          ▼                                       │
│                        Gradient All-Reduce via NCCL                              │
│                     (NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1)                      │
│                                          │                                       │
│                                          ▼                                       │
│           [Asynchronous Non-blocking Thread (Daemon)]                             │
│           • Snapshot weights to CPU RAM                                          │
│           • Serialize .safetensors (Zero-Copy)                                   │
│           • Auto-Upload to Hugging Face Model Hub (Every 500 steps)              │
│           • Local Rolling Pruning (Delete step_<N-2> to save disk)               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## ۲. مهار محدودیت‌های سخت‌افزاری کگل (T4 Constraints & Solutions)

کارت‌های تسلا T4 در پلتفرم کگل چالش‌های سخت‌افزاری ویژه‌ای دارند که در پایپلاین هم‌سخن با تکنیک‌های زیر مهار شده‌اند:

### ۲.۱. سقف فضای دیسک ۲۰ گیگابایتی (`/kaggle/working`)
دانلود دیتاست‌های خام صوتی به سرعت دیسک ۲۰ گیگابایتی کگل را پر کرده و منجر به خطای مرگبار `No space left on device` می‌شود.
* **راهکار:** پارتیشن موقت `/kaggle/tmp` دارای دیسک جداگانه با سهمیه بالاتری است. با بازتنظیم متغیرهای سیستمی:
  ```python
  os.environ["HF_HOME"] = "/kaggle/tmp/hf_home"
  os.environ["HF_DATASETS_CACHE"] = "/kaggle/tmp/hf_datasets_cache"
  os.environ["TORCH_HOME"] = "/kaggle/tmp/torch_cache"
  os.environ["TMPDIR"] = "/kaggle/tmp"
  ```
  کل فضای دانلود و کش‌های واسط به مسیر موقت منتقل شده و فضای `/kaggle/working` منحصراً به فایل‌های شارد نهایی و کدها اختصاص می‌یابد.

### ۲.۲. فقدان سخت‌افزاری Bfloat16 در کارت‌های معماری تورینگ (T4)
کارت‌های سری Turing از فرمت محاسباتی `bfloat16` پشتیبانی سخت‌افزاری نمی‌کنند. در صورتی که آموزش روی این فرمت اجرا شود، سرعت محاسبات به دلیل شبیه‌سازی نرم‌افزاری به شدت افت کرده یا با شکست مواجه می‌شود.
* **راهکار:** قفل کامل محاسبات بر روی `fp16` با تنظیمات اختصاصی گرادیان اسکیلر (`GradScalerKwargs`):
  ```python
  scaler_kwargs = GradScalerKwargs(
      init_scale=2**14,      # مقیاس اولیه پایدار برای جلوگیری از سرریز
      growth_interval=1000,  # افزایش آرام مقیاس جهت جلوگیری از تولید NaN
  )
  ```
  همچنین متغیر محیطی `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` مانع از تکه‌تکه‌شدن قطعات حافظه در کارت‌های ۱۶ گیگابایتی می‌شود.

### ۲.۳. بن‌بست ارتباطی NCCL در DDP کگل
در برخی موارد، ارتباط نقطه به نقطه کارت‌های گرافیک در کرنل لینوکس کگل دچار قفل پیوسته (Deadlock) می‌شود.
* **راهکار:** غیرفعال‌سازی صریح ارتباطات هم‌سطح پی‌سی‌آی و InfiniBand در سطح محیط سیستم:
  ```python
  os.environ["NCCL_P2P_DISABLE"] = "1"
  os.environ["NCCL_IB_DISABLE"] = "1"
  ```

### ۲.۴. سقف زمانی ۱۲ ساعته سشن‌های کگل (Session Timeout)
سشن‌های پردازشی کگل پس از حداکثر ۱۲ ساعت به صورت خودکار نابود می‌شوند.
* **راهکار:** پیاده‌سازی سیستم آپلود پیوسته و ناهمگام در ماژول `train_hamsokhan_ddp.py`:
  - یک ریسه پس‌زمینه (Background Thread) به موازات آموزش فعال است.
  - هر ۵۰۰ گام، وزن‌های آموزش‌پذیر مدل (LoRA + Projector + Synthesizer) را روی حافظه RAM ذخیره کرده و به مخزن هاگینگ‌فیس `Abolfazl27Aflhami88/Hamsokhan` ارسال می‌کند.
  - چک‌پوینت‌های قدیمی از حافظه محلی حذف می‌شوند تا سهمیه دیسک اشغال نماند.

---

## ۳. راهنمای گام‌به‌گام نوت‌بوک کگل (`hamsokhan.ipynb`)

نوت‌بوک مرجع پروژه از ۵ گام مجزا و مستقل تشکیل شده است:

### گام ۱: آماده‌سازی محیط و نصب وابستگی‌ها
- تنظیم متغیرهای محیطی برای جداسازی کش‌ها در `/kaggle/tmp`.
- تایید در دسترس بودن هر دو کارت گرافیک NVIDIA T4 با متد `torch.cuda.device_count() == 2`.
- کلون شاخه `mahdi` مخزن گیت‌هاب `Abolfazl-Afkhami/Hamsokhan`.
- نصب پیش‌نیازهای اختصاصی صوتی و مدل‌سازی (`accelerate`, `safetensors`, `librosa`, `soundfile`).

### گام ۲: احراز هویت در هاگینگ‌فیس هاب و بازیابی چک‌پوینت
- خواندن ایمن توکن کاربر (`HF_TOKEN`) از بخش Kaggle Secrets.
- لاگین در هاب و ایجاد خودکار مخازن مدل (`Abolfazl27Aflhami88/Hamsokhan`) و دیتاست (`Abolfazl27Aflhami88/hamsokhan-shards`).
- جستجوی آخرین چک‌پوینت ثبت‌شده در هاب جهت از سرگیری خودکار (`resume_from_checkpoint`).

### گام ۳: شاردینگ باینری و همگام‌سازی کش
- بررسی وجود شاردهای پیش‌ساخته در هاب.
- در صورت عدم وجود شاردها، اجرای اسکریپت `scripts/kaggle_dataset_builder.py`:
  - دانلود و پالایش داده‌های صوتی فارسی (Thomcles, ManaTTS, Common Voice, Diarization).
  - تبدیل به شاردهای فشرده `.safetensors` بدون کپی حافظه (Zero-Copy mmap) با دسته‌بندی `train`, `val`, `test`.
  - آپلود خودکار شاردها به هاب جهت استفاده‌های آتی بدون نیاز به بازسازی.

### گام ۴: آغاز آموزش موازی (Multi-GPU DDP Training)
- فراخوانی `accelerate launch` با تنظیمات توزیع‌شده برای دو کارت T4:
  ```bash
  accelerate launch --multi_gpu --num_processes 2 train_hamsokhan_ddp.py \
      --shard_dir /kaggle/working/cached_shards \
      --output_dir /kaggle/working/ckpts \
      --batch_size 4 \
      --gradient_accumulation_steps 4 \
      --lr_backbone 5e-5 \
      --lr_synthesizer 2e-4 \
      --max_steps 15000
  ```
- فعال‌سازی ریسه پشتیبان‌گیر خودکار در هاگینگ‌فیس هاب.

### گام ۵: ارزیابی قطعی و ترویج مدل پروداکشن (Evaluation & Promotion)
- استخراج بهترین مدل ذخیره‌شده (`best_model/trainable_hamsokhan.safetensors`).
- ارزیابی سخت‌گیرانه روی شاردهای آزمون مستقل (`test shards`):
  - اندازه‌گیری خطای بازسازی فرکانسی (`Mel L1 Loss`) و گرادیان طیفی (`SpecGrad`).
  - ارزیابی دقت کلاسیفایر نوبت‌گیری (`Turn-Taking Precision/Recall`).
  - اندازه‌گیری تاخیر زمانی رسیدن به نخستین فریم صوتی (`TTFA`).
- تولید و رندر یک فایل صوتی ۲۴ کیلوهرتز سنتزشده به زبان فارسی در خروجی نوت‌بوک.
- آپلود نهایی وزن‌های تاییدشده به همراه گزارش ارزیابی کامل (`evaluation_report.json`) در هاگینگ‌فیس هاب.
