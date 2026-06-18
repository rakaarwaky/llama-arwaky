# 🦙 llama-arwaky

> **GGUF Model Picker + llama-server launcher** — dioptimalkan untuk **AMD RX 6800 XT** + ROCm

---

## 🤖 Agent Context

File ini berisi konteks untuk AI agent / coding assistant agar memahami proyek ini.

### Ringkasan Proyek

`llama-arwaky` adalah TUI (Text User Interface) berbasis Python untuk:
1. **Menemukan** semua file `.gguf` secara rekursif di folder
2. **Memilih** model lewat menu interaktif di terminal
3. **Menjalankan** `llama-server` dengan konfigurasi optimal untuk hardware Raka

### Hardware Target

| Komponen | Spesifikasi |
|---|---|
| CPU | AMD Ryzen 5600G (6 core / 12 thread) |
| GPU | AMD Radeon RX 6800 XT — **16 GB VRAM** |
| ROCm | 7.2.4 |
| Backend | `llama-server` build b9656 |

### Struktur File

```
llama-arwaky/
├── llama-tui          # Script utama (Python 3, no .py extension)
├── config.yaml        # Konfigurasi server & GPU
├── setup_env.sh       # Export PATH & LD_LIBRARY_PATH
├── agent.md           # File ini — konteks untuk AI agent
└── .gitignore         # Exclude .gguf dan binary llama
```

> ⚠️ **Folder `llama-b9656/`** dan file `*.gguf` tidak di-commit (di-ignore) karena ukurannya sangat besar.

### Cara Kerja

```
llama-tui
  └─► scan *.gguf (rekursif, skip mmproj*)
  └─► tampilkan menu TUI
  └─► pilih model → launch llama-server
        ├─ auto-detect mmproj-*.gguf (untuk multimodal)
        ├─ set HSA_OVERRIDE_GFX_VERSION=10.3.0 (ROCm RX6800XT)
        └─ stream stdout llama-server ke terminal
```

### Konfigurasi (`config.yaml`)

- **`ngl: 999`** → semua layer ke GPU (full offload)
- **`no_kv_offload: true`** → KV cache tetap di CPU untuk hemat VRAM
- **`flash_attn: true`** → Flash Attention aktif
- **`cache_type_k/v: q4_0`** → KV cache dikuantisasi ke 4-bit
- **`reasoning_budget: 0`** → matikan reasoning budget (untuk model non-thinking)

### Dependensi

```bash
pip install pyyaml
```

### Cara Pakai

```bash
# 1. Setup PATH (sekali saja)
source setup_env.sh

# 2. Jalankan TUI
python3 llama-tui
# atau jika sudah chmod +x:
./llama-tui
```

### ROCm Environment

Script secara otomatis mengeset:
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0   # Diperlukan untuk RX 6800 XT di ROCm
LD_LIBRARY_PATH=<llama-b9656>:<rocm-lib>
```

---

## 🧠 Catatan untuk Agent

- **Jangan ubah** `HSA_OVERRIDE_GFX_VERSION` — ini krusial untuk RX 6800 XT
- **`llama-server` path** ada di `llama-tui` baris 17 — sesuaikan jika pindah folder
- **`llama-b9656/`** adalah binary release dari [llama.cpp](https://github.com/ggerganov/llama.cpp), bukan source code
- File `llama-tui` sengaja tanpa ekstensi `.py` agar bisa dijalankan langsung sebagai executable
- Bahasa campuran Indonesia/Inggris adalah intentional (proyek personal Raka)
