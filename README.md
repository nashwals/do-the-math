![Maths Quiz Presentation in Colourful Fun Style ](https://github.com/user-attachments/assets/df0883c5-9d70-4fb4-9ff1-ab300038c70e)

# DO THE MATH!

> Project Tugas Besar Sistem Teknologi Multimedia (IF25-40305)

---

## Deskripsi Project

**DO THE MATH!** adalah sebuah **filter interaktif berbasis multimedia** yang menggabungkan _computer vision_, _gesture recognition_, dan _audio processing_ untuk menciptakan pengalaman belajar matematika yang menyenangkan.

Filter ini akan:

- Mengeluarkan **soal matematika dasar** (penjumlahan, pengurangan, perkalian, pembagian) dalam bentuk suara.
- Mengizinkan pengguna **menjawab dengan menggambar angka di udara** menggunakan jari mereka, yang akan terdeteksi melalui kamera.
- Memberikan **umpan balik langsung** berupa efek _confetti_ dan suara tepuk tangan ketika jawaban benar.

---

## Video Demo

Video demo project ini dapat diakses melalui link berikut:
> [Video Demo DO THE MATH!](https://drive.google.com/file/d/1urlYUDikPjhKtki3ZWOaJ9RDNbnePHWT/view?usp=sharing)

---

## Cara Menjalankan Program

### 📋 Prerequisites

Pastikan sistem Anda memiliki:

- **Python 3.9, 3.10, atau 3.11**
- **Webcam** (built-in atau external)
- **Speaker/Headphone** untuk audio
- **Koneksi Internet** (untuk download dependencies)

---

### 📥 Step 1: Download Repository

#### **Windows:**

```powershell
# Clone repository menggunakan Git
git clone https://github.com/nashwals/do-the-math.git

# Masuk ke folder project
cd do-the-math
```

**Atau download manual:**

1. Kunjungi [https://github.com/nashwals/do-the-math](https://github.com/nashwals/do-the-math)
2. Klik tombol **Code** → **Download ZIP**
3. Extract file ZIP ke folder yang diinginkan
4. Buka Command Prompt atau PowerShell, navigasi ke folder tersebut

#### **Mac/Linux:**

```bash
# Clone repository menggunakan Git
git clone https://github.com/nashwals/do-the-math.git

# Masuk ke folder project
cd do-the-math
```

**Atau download manual:**

1. Kunjungi [https://github.com/nashwals/do-the-math](https://github.com/nashwals/do-the-math)
2. Klik tombol **Code** → **Download ZIP**
3. Extract file ZIP ke folder yang diinginkan
4. Buka Terminal, navigasi ke folder tersebut

---

### 🔧 Step 2: Setup Virtual Environment dengan UV

#### **Windows:**

```powershell
# Install UV (jika belum terinstall)
pip install uv

# Buat virtual environment dengan UV
uv venv

# Aktivasi virtual environment
.venv\Scripts\activate

# Verifikasi UV environment aktif
uv pip list
```

#### **Mac/Linux:**

```bash
# Install UV (jika belum terinstall)
pip install uv

# Buat virtual environment dengan UV
uv venv

# Aktivasi virtual environment
source .venv/bin/activate

# Verifikasi UV environment aktif
uv pip list
```

---

### 📦 Step 3: Install Dependencies

#### **Windows:**

```powershell
# Install semua dependencies dari requirements.txt
uv pip install -r requirements.txt
```

#### **Mac/Linux:**

```bash
# Install semua dependencies dari requirements.txt
uv pip install -r requirements.txt
```

---

### ▶️ Step 4: Jalankan Program

#### **Windows:**

```powershell
# Pastikan virtual environment masih aktif (ada tulisan (venv) di prompt)
# Jika belum, jalankan: .\venv\Scripts\activate

# Jalankan program
python main.py
```

#### **Mac/Linux:**

```bash
# Pastikan virtual environment masih aktif (ada tulisan (venv) di prompt)
# Jika belum, jalankan: source venv/bin/activate

# Jalankan program
python3 main.py
```

---

### 🎮 Step 5: Cara Bermain

1. **Intro Screen:**

   - Program akan menampilkan intro screen dengan instruksi
   - Tekan **SPACE** untuk memulai quiz
   - Tekan **Q** untuk keluar

2. **Gameplay:**

   - Soal matematika akan muncul di kanan atas dengan audio
   - Gunakan gesture untuk berinteraksi:
     - **1 Jari (Telunjuk)** → Menggambar angka jawaban di udara
     - **4 Jari (Tanpa Jempol)** → Submit jawaban
     - **5 Jari (Semua)** → Hapus canvas

3. **Feedback:**

   - **Jawaban Benar** → Confetti effect + suara tepuk tangan
   - **Jawaban Salah** → Notifikasi merah dengan jawaban yang benar

4. **Final Score:**
   - Setelah 15 soal selesai, skor akhir akan ditampilkan
   - Tekan **R** untuk restart quiz
   - Tekan **Q** untuk keluar

---

### 📂 Struktur Project

```
do-the-math/
├── main.py                    # Program utama
├── gesture_tracking.py        # Modul gesture recognition
├── digit_recognition.py       # Modul digit recognition
├── quiz_manager.py            # Modul quiz management
├── audio_manager.py           # Modul audio playback
├── confetti_effect.py         # Modul confetti particle system
├── requirements.txt           # Dependencies
├── models/
│   └── mnist-8.onnx          # Pre-trained ONNX model
├── data/
│   ├── soal_*.png            # 15 gambar soal
│   └── audio_soal_*.wav      # 15 audio soal
├── assets/
│   ├── audio/
|   │   ├── opening.wav       # Intro audio
|   │   ├── correct.wav       # Sound effect benar
|   │   ├── wrong.wav         # Sound effect salah
|   │   └── applause.wav      # Sound effect tepuk tangan
|   └── icons/
|       ├── one.png           # Icon 1 jari
|       ├── four.png          # Icon 4 jari
|       └── five.png          # Icon 5 jari
└── report/                   # Folder Laporan
```

---

## Referensi

- _Canvas Reference:_ [YouTube Short](https://youtube.com/shorts/_jz-gwRbofQ?si=2l0iqb3cMH95I1BX)
- _Math Problem Reference:_ [YouTube Short](https://youtube.com/shorts/2rGmck478cM?si=skYg6DbHSEiaDUgr)
  > _Project ini merupakan kombinasi dari kedua referensi di atas._

---

## Anggota Kelompok

| Nama                            | NIM       | GitHub ID |
| ------------------------------- | --------- | --------- |
| **Tawakkal Rabbani Muhammad**   | 122140029 | TawakkalM |
| **Naufal Harris Nurkhoirulloh** | 122140040 | Harisskh  |
| **Nashwa Putri Laisya**         | 122140180 | nashwals  |

---

# Weekly Logbook

## Week - 1

#### 28 Oktober 2025

- Diskusi dan pemilihan ide topik Tugas Besar yang akan dikerjakan

## Week - 2

#### 7 November 2025

- Membuat dataset audio soal matematika
- Membuat dataset gambar soal matematika

## Week - 3

#### 10 - 12 November 2025

- Membuat program untuk hand gesture tracking
- Membuat program untuk menggambar di canvas/udara

## Week - 4

#### 18 - 20 November 2025

- Membuat program digit recognizer untuk inferensi mnist onnx

## Week - 5

#### 24 - 28 November 2025

- Membuat program tampilan quiz

#### 29 - 30 November 2025

- Fiksasi program secara keseluruhan

## Week - 6

#### 08 - 12 Desember 2025
- Membuat laporan project
- Membuat video demo