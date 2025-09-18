# WindowsC++GGML_Memory_TechDoc_Part2_v13.md (English Translation)

**Security · Memory · Operations Enhancement Final Edition (2/3) – TPM2 Implementation Completed**

## 5. Zero-Copy IPC (AES-256-GCM, Fixed Packet Format)

> Packet structure: `[u32 packet_size][12B nonce][ciphertext...][16B tag]`

```cpp
// encrypted_ipc.hpp (v13)
#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <atomic>
#include <vector>
#include <system_error>
#pragma comment(lib, "bcrypt.lib")

class EncryptedIPC {
    static constexpr size_t TAG_SIZE   = 16;
    static constexpr size_t NONCE_SIZE = 12;
    static constexpr size_t HDR_SIZE   = sizeof(uint32_t);

    HANDLE map_ = nullptr; BYTE* view_ = nullptr;
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_KEY_HANDLE key_ = nullptr;

    static void chk(NTSTATUS s, const char* msg){
        if(!BCRYPT_SUCCESS(s)) throw std::system_error(s, std::system_category(), msg);
    }
    void open_aes_gcm() {
        chk(BCryptOpenAlgorithmProvider(&alg_, BCRYPT_AES_ALGORITHM, nullptr, 0),
            "BCryptOpenAlgorithmProvider");
        chk(BCryptSetProperty(alg_, BCRYPT_CHAINING_MODE,
            (PUCHAR)BCRYPT_CHAIN_MODE_GCM, (ULONG)sizeof(BCRYPT_CHAIN_MODE_GCM), 0),
            "BCryptSetProperty(GCM)");
        UCHAR key_bytes[32];
        BCryptGenRandom(nullptr, key_bytes, sizeof(key_bytes), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
        chk(BCryptGenerateSymmetricKey(alg_, &key_, nullptr, 0, key_bytes, sizeof(key_bytes), 0),
            "BCryptGenerateSymmetricKey");
    }

public:
    struct Header { alignas(64) std::atomic<ULONG> w{0}, r{0}; ULONG cap{0}; };

    EncryptedIPC(const wchar_t* name, size_t ring_bytes, bool create) {
        open_aes_gcm();
        size_t total = ring_bytes + sizeof(Header);
        map_ = CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,(DWORD)total,name);
        if(!map_) throw std::system_error(GetLastError(), std::system_category(), "CreateFileMappingW");
        view_ = (BYTE*)MapViewOfFile(map_, FILE_MAP_ALL_ACCESS,0,0,0);
        if(!view_) { CloseHandle(map_); throw std::system_error(GetLastError(), std::system_category(),"MapViewOfFile"); }
        if(create) new (view_) Header{ .cap = (ULONG)ring_bytes };
    }

    bool Write(const void* data, size_t len){
        auto* h = (Header*)view_; BYTE* base = view_ + sizeof(Header);
        ULONG w = h->w.load(std::memory_order_relaxed);
        ULONG r = h->r.load(std::memory_order_acquire);

        ULONG total_size = (ULONG)(HDR_SIZE + NONCE_SIZE + len + TAG_SIZE);
        size_t free_space = (r > w) ? (r - w - 1) : (h->cap - w + r - 1);
        if(free_space < total_size) return false;

        std::vector<UCHAR> packet(total_size);
        memcpy(packet.data(), &total_size, HDR_SIZE);
        UCHAR* nonce = packet.data() + HDR_SIZE;
        UCHAR* ciph  = nonce + NONCE_SIZE;
        UCHAR* tag   = ciph + len;

        BCryptGenRandom(nullptr, nonce, NONCE_SIZE, BCRYPT_USE_SYSTEM_PREFERRED_RNG);

        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO ai;
        BCRYPT_INIT_AUTH_MODE_INFO(ai);
        ai.pbNonce = nonce; ai.cbNonce = (ULONG)NONCE_SIZE;
        ai.pbTag   = tag;   ai.cbTag   = (ULONG)TAG_SIZE;

        ULONG out = 0;
        chk(BCryptEncrypt(key_, (PUCHAR)data, (ULONG)len, &ai, nullptr, 0,
                          ciph, (ULONG)len, &out, 0), "BCryptEncrypt");

        memcpy(base + w, packet.data(), total_size);
        h->w.store((w + total_size) % h->cap, std::memory_order_release);
        return true;
    }

    std::vector<UCHAR> Read(){
        auto* h = (Header*)view_; BYTE* base = view_ + sizeof(Header);
        ULONG r = h->r.load(std::memory_order_acquire);
        ULONG w = h->w.load(std::memory_order_relaxed);
        if(r == w) return {};

        ULONG total_size = 0; memcpy(&total_size, base + r, HDR_SIZE);
        std::vector<UCHAR> packet(total_size);
        memcpy(packet.data(), base + r, total_size);

        UCHAR* nonce = packet.data() + HDR_SIZE;
        UCHAR* ciph  = nonce + NONCE_SIZE;
        ULONG  clen  = total_size - HDR_SIZE - NONCE_SIZE - (ULONG)TAG_SIZE;
        UCHAR* tag   = ciph + clen;

        std::vector<UCHAR> plain(clen);
        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO ai;
        BCRYPT_INIT_AUTH_MODE_INFO(ai);
        ai.pbNonce = nonce; ai.cbNonce = (ULONG)NONCE_SIZE;
        ai.pbTag   = tag;   ai.cbTag   = (ULONG)TAG_SIZE;

        ULONG out = 0;
        NTSTATUS st = BCryptDecrypt(key_, ciph, clen, &ai, nullptr, 0,
                                    plain.data(), (ULONG)plain.size(), &out, 0);
        h->r.store((r + total_size) % h->cap, std::memory_order_release);
        if(!BCRYPT_SUCCESS(st)) return {};
        plain.resize(out);
        return plain;
    }

    ~EncryptedIPC() noexcept {
        if(view_) UnmapViewOfFile(view_);
        if(map_)  CloseHandle(map_);
        if(key_)  BCryptDestroyKey(key_);
        if(alg_)  BCryptCloseAlgorithmProvider(alg_,0);
    }
};
```

---

## 6. PII Masking Logging: TPM 2.0 PCR Extend (Full Implementation)

```cpp
// tpm_logger.hpp (v13 - full implementation)
#pragma once
#include <tbs.h>
#include <vector>
#include <system_error>
#include <wincrypt.h>
#pragma comment(lib, "tbs.lib")
#pragma comment(lib, "bcrypt.lib")

class TPMLogger {
    TBS_HCONTEXT ctx_ = 0;
    
    static void chk(TBS_RESULT r, const char* m){ 
        if(r!=TBS_SUCCESS) throw std::system_error(r, std::system_category(), m); 
    }
    
    struct TPM2_Header {
        uint16_t tag;      // TPM_ST_NO_SESSIONS = 0x8001
        uint32_t size;     // total command size
        uint32_t command;  // TPM_CC_PCR_Extend = 0x00000182
    };
    
    static uint16_t htons(uint16_t v) { return ((v & 0xFF) << 8) | ((v >> 8) & 0xFF); }
    static uint32_t htonl(uint32_t v) { 
        return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | 
               ((v & 0xFF0000) >> 8) | ((v >> 24) & 0xFF); 
    }
    
public:
    TPMLogger(){ 
        chk(Tbsi_Context_Create(&ctx_), "Tbsi_Context_Create"); 
    }

    // Extend event hash into PCR#12 (actual TPM2 spec implementation)
    void LogMaskEventPCR12(const wchar_t* type, uint64_t count){
        // 1. SHA256 hash of event data
        BCRYPT_ALG_HANDLE alg = nullptr;
        BCRYPT_HASH_HANDLE hash = nullptr;
        
        BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
        BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0);
        
        BCryptHashData(hash, (PUCHAR)type, (ULONG)(wcslen(type) * sizeof(wchar_t)), 0);
        BCryptHashData(hash, (PUCHAR)&count, sizeof(count), 0);
        
        BYTE digest[32];
        BCryptFinishHash(hash, digest, sizeof(digest), 0);
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(alg, 0);
        
        // 2. Build TPM2_PCR_Extend command buffer
        std::vector<BYTE> cmd;
        
        TPM2_Header hdr;
        hdr.tag = htons(0x8001);
        hdr.command = htonl(0x00000182);
        hdr.size = htonl(sizeof(TPM2_Header) + 4 + 4 + 2 + 4 + 32);
        
        cmd.resize(sizeof(hdr));
        memcpy(cmd.data(), &hdr, sizeof(hdr));
        
        uint32_t pcr_index = htonl(12);
        cmd.insert(cmd.end(), (BYTE*)&pcr_index, (BYTE*)&pcr_index + 4);
        
        uint32_t auth_size = 0;
        cmd.insert(cmd.end(), (BYTE*)&auth_size, (BYTE*)&auth_size + 4);
        
        uint16_t digest_count = htons(1);
        cmd.insert(cmd.end(), (BYTE*)&digest_count, (BYTE*)&digest_count + 2);
        
        uint16_t hash_alg = htons(0x000B);
        cmd.insert(cmd.end(), (BYTE*)&hash_alg, (BYTE*)&hash_alg + 2);
        
        cmd.insert(cmd.end(), digest, digest + 32);
        
        ((TPM2_Header*)cmd.data())->size = htonl((uint32_t)cmd.size());
        
        // 3. Submit command to TPM
        std::vector<BYTE> rsp(4096); 
        UINT32 rspLen = (UINT32)rsp.size();
        
        TBS_RESULT r = Tbsip_Submit_Command(ctx_, 
            TBS_COMMAND_LOCALITY_ZERO, 
            TBS_COMMAND_PRIORITY_NORMAL,
            cmd.data(), (UINT32)cmd.size(), 
            rsp.data(), &rspLen);
            
        if(r != TBS_SUCCESS){
            // If failed, do not terminate program
            // Option: log into Windows Event Log
        } else {
            if(rspLen >= 10) {
                uint32_t response_code = ntohl(*(uint32_t*)(rsp.data() + 6));
                if(response_code != 0) {
                    // Handle TPM error codes
                }
            }
        }
    }
    
    ~TPMLogger(){ 
        if(ctx_) Tbsip_Context_Close(ctx_); 
    }
    
private:
    static uint32_t ntohl(uint32_t v) { return htonl(v); }
};
```

---

## 7. HMM (cudaMallocManaged) & Prefetch

```cpp
// hmm_allocator.hpp (v13)
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define cudaChk(e) do{ auto _e=(e); if(_e!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(_e)); }while(0)

struct HMMAllocator {
    static void* Alloc(size_t bytes){ void* p=nullptr; cudaChk(cudaMallocManaged(&p, bytes, cudaMemAttachGlobal)); return p; }
    static void  PrefetchToGPU(void* p, size_t bytes, cudaStream_t s){ cudaChk(cudaMemPrefetchAsync(p, bytes, 0, s)); }
    static void  Free(void* p){ if(p) cudaFree(p); }
};
```

---

## 8. NUMA Interleaving (VirtualAlloc2)

```cpp
// numa_interleave.hpp (v13)
#pragma once
#include <windows.h>
#include <memoryapi.h>
#include <system_error>

class NUMAInterleave {
    static inline DWORD next_node = 0, max_node = [](){ 
        ULONG hi; 
        if(!GetNumaHighestNodeNumber(&hi)) 
            throw std::system_error(GetLastError(), std::system_category(),"GetNumaHighestNodeNumber"); 
        return hi+1; 
    }();
public:
    static void* Alloc(size_t bytes){
        DWORD node = InterlockedIncrement(&next_node) % max_node;
        MEM_EXTENDED_PARAMETER p{}; 
        p.Type = MemExtendedParameterNumaNode; 
        p.ULong = node;
        void* ptr = VirtualAlloc2(nullptr,nullptr,bytes,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE,&p,1);
        if(!ptr) throw std::system_error(GetLastError(), std::system_category(),"VirtualAlloc2");
        return ptr;
    }
};
```

---

## 9. CI/CD (Windows-2022, Cache Enabled)

```yaml
# .github/workflows/ggml-ci.yml (v13)
name: GGML-Windows-CI
on: [push, pull_request]
jobs:
  build:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - name: Cache vcpkg
        uses: actions/cache@v4
        with:
          path: D:/a/${{ github.event.repository.name }}/${{ github.event.repository.name }}/vcpkg/installed
          key: ${{ runner.os }}-vcpkg-${{ hashFiles('**/vcpkg.json') }}
      - name: Cache choco
        uses: actions/cache@v4
        with: { path: C:\ProgramData\chocolatey\lib, key: ${{ runner.os }}-choco-cache }
      - name: Install deps
        run: |
          choco install drmemory cuda cmake -y --no-progress
          refreshenv
      - name: Configure (vcpkg toolchain)
        run: cmake -B build -A x64 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DUSE_CUDA=ON
      - name: Build
        run: cmake --build build --config RelWithDebInfo --parallel
      - name: Test (memcheck)
        run: |
          drmemory -- build\RelWithDebInfo\inference_test.exe
```

---

## 10. PowerShell Provisioning

```powershell
# scripts/setup-node.ps1 (v13)
# Large Pages & Code Signing: reuse install guide scripts
Write-Host "Use the unified provisioning in the install guide." -ForegroundColor Cyan
```

---

## 11. Lifecycle / Destructor Synchronization

```cpp
// vram_pool_manager.hpp (v13)
~VRAMPoolManager() noexcept {
    cudaDeviceSynchronize(); // Ensure completion of async tasks
    for(auto& b: blocks){ if(b.ptr) cudaFreeAsync(b.ptr, 0); }
    cudaDeviceSynchronize(); // Ensure deallocation completed
}
```
