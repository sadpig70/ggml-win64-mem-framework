# WindowsC++GGML_EnvSetup_InstallGuide_v13.md (English Translation)

**Environment Setup & Installation Guide (Improved) – Enhanced Automation Scripts**


## 📦 Package Managers

### 1. Chocolatey

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### 2. vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

---

## 🔧 Development Tools Installation

```powershell
choco install cuda --version=12.3 -y --no-progress
choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y
choco install git -y
choco install drmemory -y
choco install nvml -y  # NVIDIA Management Library
```

### Hyperscan (via vcpkg)

```powershell
vcpkg install hyperscan:x64-windows
```

> **CMake Integration**:
> `-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`
> `find_package(hs REQUIRED)` → vcpkg auto-detects install path

---

## ⚙️ System Configuration

### Large Pages + Lock Pages Privilege (Full Automation)

```powershell
# setup-largepages.ps1 (v13 - fully automated)
function Enable-LargePages {
    param(
        [string]$UserName = $env:USERNAME
    )
    
    # 1. Enable Large System Cache
    $key = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    $currentValue = (Get-ItemProperty $key -Name "LargeSystemCache" -ErrorAction SilentlyContinue).LargeSystemCache
    
    if ($currentValue -ne 1) {
        Set-ItemProperty $key -Name "LargeSystemCache" -Value 1
        Write-Host "✅ LargeSystemCache enabled" -ForegroundColor Green
        $rebootRequired = $true
    } else {
        Write-Host "✅ LargeSystemCache already enabled" -ForegroundColor Green
    }
    
    # 2. Grant Lock Pages in Memory privilege (automated with secedit)
    Write-Host "Setting 'Lock Pages in Memory' privilege for $UserName..." -ForegroundColor Cyan
    
    $tempFile = "$env:TEMP\secpol.cfg"
    $dbFile = "$env:TEMP\secpol.sdb"
    
    secedit /export /cfg $tempFile /quiet
    
    $content = Get-Content $tempFile
    $sid = (New-Object System.Security.Principal.NTAccount($UserName)).Translate([System.Security.Principal.SecurityIdentifier]).Value
    
    $privilegeLine = "SeLockMemoryPrivilege = *$sid"
    $found = $false
    
    for ($i = 0; $i -lt $content.Length; $i++) {
        if ($content[$i] -match "SeLockMemoryPrivilege") {
            if ($content[$i] -notmatch $sid) {
                $content[$i] = $content[$i].TrimEnd() + ",*$sid"
            }
            $found = $true
            break
        }
    }
    
    if (-not $found) {
        for ($i = 0; $i -lt $content.Length; $i++) {
            if ($content[$i] -match "\[Privilege Rights\]") {
                $content = $content[0..$i] + $privilegeLine + $content[($i+1)..($content.Length-1)]
                break
            }
        }
    }
    
    Set-Content $tempFile $content
    secedit /configure /db $dbFile /cfg $tempFile /quiet
    
    Remove-Item $tempFile -Force
    Remove-Item $dbFile -Force
    
    Write-Host "✅ Lock Pages in Memory privilege granted to $UserName" -ForegroundColor Green
    
    if ($rebootRequired) {
        Write-Host "⚠️ REBOOT REQUIRED for changes to take effect" -ForegroundColor Yellow
    }
    
    return $rebootRequired
}

# Run
$needReboot = Enable-LargePages
```

### DLL Security Policy

```powershell
# setup-dll-security.ps1 (v13)
function Set-DLLSecurity {
    $regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager"
    
    Set-ItemProperty -Path $regPath -Name "CWDIllegalInDllSearch" -Value 0xFFFFFFFF -Type DWord
    Write-Host "✅ DLL search path hardened (CWDIllegalInDllSearch = 0xFFFFFFFF)" -ForegroundColor Green
    
    Set-ItemProperty -Path $regPath -Name "SafeDllSearchMode" -Value 1 -Type DWord
    Write-Host "✅ SafeDllSearchMode enabled" -ForegroundColor Green
}

Set-DLLSecurity
```

---

## 🔐 Code Signing (Improved Automation)

```powershell
# setup-codesigning.ps1 (v13 - enhanced error handling)
function New-CodeSigningCertificate {
    param(
        [string]$Subject = "CN=GGML-Test",
        [string]$Password = "YourPassword123!",
        [string]$ExportPath = ".\ggml-test-cert.pfx"
    )
    
    try {
        $existingCert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | 
                        Where-Object { $_.Subject -eq $Subject }
        
        if ($existingCert) {
            Write-Host "⚠️ Certificate already exists. Using existing certificate." -ForegroundColor Yellow
            $cert = $existingCert[0]
        } else {
            $cert = New-SelfSignedCertificate `
                -Type CodeSigning `
                -Subject $Subject `
                -KeyUsage DigitalSignature `
                -KeyAlgorithm RSA `
                -KeyLength 2048 `
                -CertStoreLocation "Cert:\CurrentUser\My" `
                -NotAfter (Get-Date).AddYears(3)
            
            Write-Host "✅ Certificate created: $($cert.Thumbprint)" -ForegroundColor Green
        }
        
        $pwd = ConvertTo-SecureString -String $Password -Force -AsPlainText
        Export-PfxCertificate -Cert $cert -FilePath $ExportPath -Password $pwd | Out-Null
        Write-Host "✅ Certificate exported to: $ExportPath" -ForegroundColor Green
        
        $cerPath = $ExportPath.Replace('.pfx', '.cer')
        Export-Certificate -Cert $cert -FilePath $cerPath | Out-Null
        
        Import-Certificate -FilePath $cerPath -CertStoreLocation "Cert:\LocalMachine\Root" | Out-Null
        Import-Certificate -FilePath $cerPath -CertStoreLocation "Cert:\LocalMachine\TrustedPublisher" | Out-Null
        
        Write-Host "✅ Certificate installed in Trusted Root & Publisher stores" -ForegroundColor Green
        
        return $cert
        
    } catch {
        Write-Host "❌ Error creating certificate: $_" -ForegroundColor Red
        return $null
    }
}
```

---

## 🧪 Integrated Installation Verification Script

```powershell
# verify-installation.ps1 (v13 - detailed diagnostics)
Write-Host "`n=== GGML-Windows Installation Verification ===" -ForegroundColor Cyan

$results = @()

# CUDA check
try {
    $cudaVersion = & nvcc --version 2>$null | Select-String "release" | Out-String
    if ($cudaVersion) {
        $version = $cudaVersion -match "release (\d+\.\d+)" | Out-Null; $matches[1]
        $results += [PSCustomObject]@{Component="CUDA"; Status="✅"; Version=$version}
        
        $nvmlTest = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($nvmlTest) {
            $results += [PSCustomObject]@{Component="NVML"; Status="✅"; Version="OK"}
        } else {
            $results += [PSCustomObject]@{Component="NVML"; Status="⚠️"; Version="nvidia-smi not found"}
        }
    } else {
        $results += [PSCustomObject]@{Component="CUDA"; Status="❌"; Version="Not found"}
    }
} catch {
    $results += [PSCustomObject]@{Component="CUDA"; Status="❌"; Version="Error: $_"}
}

# ... (continues with Hyperscan, Large Pages, Lock Pages, DLL Security, Code Signing, Dr.Memory)
```

---

## 🔄 One-Click Installation Script

```powershell
# install-all.ps1 (v13 - fully automated)
#Requires -RunAsAdministrator

Write-Host "GGML Windows Environment Setup" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

Write-Host "`n[1/6] Installing package managers..." -ForegroundColor Yellow
& .\setup-chocolatey.ps1
& .\setup-vcpkg.ps1

Write-Host "`n[2/6] Installing development tools..." -ForegroundColor Yellow
choco install cuda cmake git drmemory nvml -y --no-progress
vcpkg install hyperscan:x64-windows

Write-Host "`n[3/6] Configuring system settings..." -ForegroundColor Yellow
& .\setup-largepages.ps1
& .\setup-dll-security.ps1

Write-Host "`n[4/6] Setting up code signing..." -ForegroundColor Yellow
& .\setup-codesigning.ps1

Write-Host "`n[5/6] Building GGML..." -ForegroundColor Yellow
cmake -B build -A x64 `
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
    -DUSE_CUDA=ON
cmake --build build --config Release --parallel

Write-Host "`n[6/6] Verifying installation..." -ForegroundColor Yellow
& .\verify-installation.ps1

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
```

---

## ❗ Troubleshooting

1. Always run PowerShell as **Administrator**
2. **Reboot required** after enabling Large Pages & Lock Pages privilege
3. `vcpkg integrate install` is mandatory
4. Reinstall choco/vcpkg in case of package conflicts
5. Check CUDA toolkit version if mismatch occurs

---

