# 🧠 Windows C++ GGML Memory Optimization Framework

Production-ready **memory management & security framework** for running GGML-based LLMs on **Windows + C++**.  
This repository provides optimized allocators, secure IPC, GPU memory pooling, TPM2 logging, and Grafana monitoring – all tailored for **high-performance inference** and **sustainable operations**.

---

## ✨ Features

- **Memory Optimization**
  - Arena Allocator (RAII + Large Page fallback)
  - VRAM Pool with async free + CUDA Graph acceleration
  - NUMA-aware allocation
  - Heterogeneous Memory Management (HMM with `cudaMallocManaged`)

- **Security**
  - Zero-Copy IPC with **AES-256-GCM**
  - TPM 2.0 PCR Extend logging
  - DLL SafeLoad + Code Signing automation

- **Operations & Monitoring**
  - Hot Reload (zero-downtime model swap)
  - Rollback Agent (automatic failover)
  - Grafana plugin (latency, throughput, VRAM, carbon metrics)
  - ESG JSON report generation

- **CI/CD**
  - GitHub Actions (Windows 2022 runner)
  - Dr.Memory leak detection integrated

---

## 📦 Installation

### 1. Environment Setup (One-click)

```powershell
# Run as Administrator
.\install-all.ps1
````

* Installs **Chocolatey, vcpkg, CUDA, CMake, Hyperscan, Dr.Memory**
* Configures **Large Pages + Lock Pages privilege**
* Sets **DLL security policy**
* Creates **self-signed Code Signing certificate**
* Verifies installation with diagnostic script

### 2. Build

```powershell
cmake -B build -A x64 `
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
    -DUSE_CUDA=ON
cmake --build build --config Release --parallel
```

---

## 📊 Monitoring

Grafana dashboard included:

* Latency (ms)
* Throughput (tokens/sec)
* VRAM usage (MB)
* Carbon emission (gCO₂/h)

```bash
docker-compose up -d
```

Access Grafana at: **[http://localhost:3000](http://localhost:3000)**

---

## 📖 Documentation

* [Part 1 – Production-Ready & Benchmarks](./WindowsC++GGML_Memory_TechDoc_Part1_v13.md)
* [Part 2 – Security & Memory Enhancements](./WindowsC++GGML_Memory_TechDoc_Part2_v13.md)
* [Part 3 – Ops, Monitoring & ESG](./WindowsC++GGML_Memory_TechDoc_Part3_v13.md)
* [Environment Setup Guide](./WindowsC++GGML_EnvSetup_InstallGuide_v13.md)
* [Comprehensive Manual](./WindowsC++GGML_Memory_TechDoc_Manual_v13.md)
* [Glossary](./Glossary.md)

---

## 🔐 Security Notes

* TPM 2.0 required for full PCR logging
* Dr.Memory CI must pass for all pull requests
* Windows API & CUDA calls must use enforced error handling macros

---

## 🌱 ESG Integration

The system outputs **ESG JSON reports** (latency, throughput, energy, renewable percentage).
These can be integrated into **GRI/SASB frameworks** for sustainability disclosure.

---

## 🤝 Contributing

1. Fork the repo & create a branch
2. Run `drmemory` tests before submitting PR
3. Include benchmark results when possible
4. Follow [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## 📜 License

Apache 2.0 – see [LICENSE](./LICENSE)

---

## 🧪 Benchmark Snapshot

| Model      | Baseline Latency | Optimized | Memory Saved | Throughput Gain |
| ---------- | ---------------- | --------- | ------------ | --------------- |
| LLaMA-7B   | 142ms            | 95ms      | -38%         | +49%            |
| LLaMA-13B  | 218ms            | 136ms     | -41%         | +60%            |
| Falcon-40B | 612ms            | 385ms     | -33%         | +59%            |

---

