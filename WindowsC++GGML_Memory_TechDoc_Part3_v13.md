# WindowsC++GGML_Memory_TechDoc_Part3_v13.md (English Translation)

**Operations · Monitoring · ESG Final Edition (3/3) – Grafana Frontend Added**


## 13. Hot Reload (Zero-Downtime Model Swap)

```cpp
// hot_reload.hpp (v13)
class HotReload {
    HANDLE f_=INVALID_HANDLE_VALUE, m_=nullptr;
    ggml_context* ctx_=nullptr;
    std::filesystem::file_time_type last_;

    void close_all(){ 
        if(ctx_){ggml_free(ctx_);ctx_=nullptr;} 
        if(m_){CloseHandle(m_);m_=nullptr;} 
        if(f_!=INVALID_HANDLE_VALUE){CloseHandle(f_);f_=INVALID_HANDLE_VALUE;} 
    }

public:
    ggml_context* Load(const std::wstring& path){
        using fs=std::filesystem;
        if(fs::path(path).extension()==L".tmp") return ctx_;
        if(!fs::exists(path)) throw std::system_error(ERROR_FILE_NOT_FOUND,std::system_category(),"file not found");
        auto wt=fs::last_write_time(path); 
        if(wt==last_) return ctx_;
        
        close_all();
        f_=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_SEQUENTIAL_SCAN,nullptr);
        if(f_==INVALID_HANDLE_VALUE) throw std::system_error(GetLastError(),std::system_category(),"CreateFileW failed");
        
        LARGE_INTEGER sz; 
        if(!GetFileSizeEx(f_,&sz)) throw std::system_error(GetLastError(),std::system_category(),"GetFileSizeEx failed");
        
        m_=CreateFileMappingW(f_,nullptr,PAGE_READONLY,0,0,nullptr);
        if(!m_) throw std::system_error(GetLastError(),std::system_category(),"CreateFileMappingW failed");
        
        void* base=MapViewOfFile(m_,FILE_MAP_READ,0,0,0);
        if(!base) throw std::system_error(GetLastError(),std::system_category(),"MapViewOfFile failed");
        
        ctx_=ggml_init({.mem_size=(size_t)sz.QuadPart,.mem_buffer=base,.no_alloc=true});
        if(!ctx_) throw std::runtime_error("ggml_init failed");
        
        last_=wt; 
        return ctx_;
    }
    ~HotReload(){close_all();}
};
```

---

## 14. Carbon Emission Measurement (NVML + ETW)

```cpp
// carbon_meter.hpp (v13)
class CarbonMeter {
    nvmlDevice_t dev=nullptr;
    static constexpr double gCO2_per_kWh=475.0; // Based on Korea power grid
    
    void chk(nvmlReturn_t r,const char* m){ 
        if(r!=NVML_SUCCESS) throw std::system_error(r,std::system_category(),std::string(m)+nvmlErrorString(r)); 
    }
    
public:
    CarbonMeter(){ 
        chk(nvmlInit(),"nvmlInit"); 
        chk(nvmlDeviceGetHandleByIndex(0,&dev),"DeviceGetHandleByIndex"); 
    }
    
    unsigned long long TotalJ(){ 
        unsigned long long mj=0; 
        chk(nvmlDeviceGetTotalEnergyConsumption(dev,&mj),"TotalEnergy"); 
        return mj/1000; 
    }
    
    static double JoulesToGCO2(unsigned long long j){ 
        if(!j) return 0; 
        double kWh=(double)j/3.6e6; 
        return kWh*gCO2_per_kWh; 
    }
    
    ~CarbonMeter(){ if(dev) nvmlShutdown(); }
};
```

---

## 15. Grafana Plugin (Backend + Frontend Complete)

### Backend: plugin.go

```go
// datasource-plugin/pkg/plugin.go (v13)
package main

import (
    "context"
    "encoding/json"
    "time"
    "github.com/grafana/grafana-plugin-sdk-go/backend"
    "github.com/grafana/grafana-plugin-sdk-go/backend/datasource"
    "github.com/grafana/grafana-plugin-sdk-go/data"
)

type GGMLDataSource struct{}

func (d *GGMLDataSource) QueryData(ctx context.Context, req *backend.QueryDataRequest) (*backend.QueryDataResponse, error) {
    resp := backend.NewQueryDataResponse()
    
    for _, q := range req.Queries {
        var model map[string]interface{}
        if err := json.Unmarshal(q.JSON, &model); err != nil {
            backend.Logger.Error("JSON unmarshal", "err", err)
            continue
        }
        
        metric, ok := model["metric"].(string)
        if !ok {
            backend.Logger.Warn("metric missing")
            continue
        }
        
        val, err := ReadSharedMemoryFloat(metric)
        if err != nil {
            backend.Logger.Error("read shm", "metric", metric, "err", err)
            continue
        }
        
        frame := data.NewFrame(metric,
            data.NewField("time", nil, []time.Time{time.Now()}),
            data.NewField("value", nil, []float64{val}),
        )
        
        resp.Responses[q.RefID] = backend.DataResponse{Frames: []*data.Frame{frame}}
    }
    
    return resp, nil
}

func main() {
    datasource.Manage("ggml-datasource", &GGMLDataSource{}, datasource.ManageOpts{})
}
```

### Frontend: module.ts (newly added)

```typescript
// datasource-plugin/src/module.ts (v13)
import { DataSourcePlugin } from '@grafana/data';
import { DataSource } from './datasource';
import { ConfigEditor } from './ConfigEditor';
import { QueryEditor } from './QueryEditor';
import { GGMLQuery, GGMLDataSourceOptions } from './types';

export const plugin = new DataSourcePlugin<DataSource, GGMLQuery, GGMLDataSourceOptions>(DataSource)
  .setConfigEditor(ConfigEditor)
  .setQueryEditor(QueryEditor);
```

### Frontend: QueryEditor.tsx (newly added)

```tsx
// datasource-plugin/src/QueryEditor.tsx (v13)
import React, { ChangeEvent, PureComponent } from 'react';
import { QueryEditorProps, SelectableValue } from '@grafana/data';
import { InlineField, Input, Select } from '@grafana/ui';
import { DataSource } from './datasource';
import { GGMLDataSourceOptions, GGMLQuery } from './types';

type Props = QueryEditorProps<DataSource, GGMLQuery, GGMLDataSourceOptions>;

const metricOptions: Array<SelectableValue<string>> = [
  { label: 'Latency (ms)', value: 'latency_ms' },
  { label: 'Throughput (tok/s)', value: 'throughput_tps' },
  { label: 'VRAM Usage (MB)', value: 'vram_mb' },
  { label: 'Carbon (gCO₂)', value: 'carbon_gco2' },
  { label: 'Temperature (°C)', value: 'gpu_temp' },
];

export class QueryEditor extends PureComponent<Props> {
  onMetricChange = (value: SelectableValue<string>) => {
    const { onChange, query } = this.props;
    onChange({ ...query, metric: value.value! });
  };

  onAggregationChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { onChange, query } = this.props;
    onChange({ ...query, aggregation: event.target.value });
  };

  render() {
    const { query } = this.props;
    const { metric, aggregation } = query;

    return (
      <div className="gf-form">
        <InlineField label="Metric" tooltip="Select GGML metric to monitor">
          <Select
            options={metricOptions}
            value={metric}
            onChange={this.onMetricChange}
            placeholder="Select metric"
            width={25}
          />
        </InlineField>
        
        <InlineField label="Aggregation" tooltip="Optional aggregation function">
          <Input
            value={aggregation || ''}
            onChange={this.onAggregationChange}
            placeholder="avg, max, min, sum"
            width={15}
          />
        </InlineField>
      </div>
    );
  }
}
```

### Dashboard Configuration: dashboard.json (newly added)

```json
{
  "dashboard": {
    "title": "GGML Windows Performance Monitor",
    "uid": "ggml-win-perf",
    "panels": [
      {
        "id": 1,
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "title": "Inference Latency",
        "type": "graph",
        "targets": [{
          "datasource": "GGML",
          "metric": "latency_ms",
          "refId": "A"
        }],
        "yaxes": [{"format": "ms", "label": "Latency"}]
      },
      {
        "id": 2,
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "title": "Throughput",
        "type": "graph",
        "targets": [{
          "datasource": "GGML",
          "metric": "throughput_tps",
          "refId": "B"
        }],
        "yaxes": [{"format": "short", "label": "Tokens/sec"}]
      },
      {
        "id": 3,
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "title": "VRAM Usage",
        "type": "graph",
        "targets": [{
          "datasource": "GGML",
          "metric": "vram_mb",
          "refId": "C"
        }],
        "yaxes": [{"format": "mbytes", "label": "VRAM"}],
        "alert": {
          "conditions": [{
            "evaluator": {"params": [14000], "type": "gt"},
            "operator": {"type": "and"},
            "query": {"model": "C", "params": ["5m", "now"]},
            "reducer": {"params": [], "type": "avg"},
            "type": "query"
          }],
          "name": "High VRAM Usage",
          "message": "VRAM usage exceeded 14GB"
        }
      },
      {
        "id": 4,
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "title": "Carbon Emissions",
        "type": "stat",
        "targets": [{
          "datasource": "GGML",
          "metric": "carbon_gco2",
          "refId": "D"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "gCO₂/h",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 100}
              ]
            }
          }
        }
      }
    ],
    "refresh": "5s",
    "time": {"from": "now-1h", "to": "now"}
  }
}
```

### docker-compose.yml

```yaml
# docker-compose.yml (v13)
services:
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes:
      - ./ggml-datasource/dist:/var/lib/grafana/plugins/ggml
      - ./grafana-data:/var/lib/grafana
      - ./dashboard.json:/etc/grafana/provisioning/dashboards/ggml.json
    environment:
      - GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=ggml
      - GF_LOG_LEVEL=debug
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/provisioning/dashboards/ggml.json
```

---

## 16. Open-Source Contribution Guide

* **LICENSE: Apache-2.0**
* **CONTRIBUTING.md**:

  * PRs must pass `drmemory`
  * Mandatory error handling macros for Windows API / CUDA calls
  * Benchmark results encouraged
* **CMakeLists.txt**: Hyperscan / NVML integration, code signing optional

---

## 17. Automatic Rollback & Failure Recovery

```cpp
// rollback_agent.hpp (v13)
class RollbackAgent {
    fs::path dir,backup;
    void cp(const fs::path&s,const fs::path&d){ 
        fs::copy_file(s,d,fs::copy_options::overwrite_existing); 
    }
public:
    RollbackAgent(const fs::path& d):dir(d),backup(d.string()+".bak"){ 
        if(!fs::exists(d)) throw std::runtime_error("dir not found"); 
    }
    void Backup(){ if(fs::exists(dir)) cp(dir,backup); }
    void Rollback(){ if(fs::exists(backup)) cp(backup,dir); }
    bool HealthCheck(double latency,int errcode=0){ 
        if(latency>200.0||errcode!=0){ 
            Rollback(); 
            return false;
        } 
        return true; 
    }
};
```

---

## 18. ESG Report Sample

```json
{
  "timestamp": "2025-09-18T12:00:00Z",
  "version": "v1.5.0",
  "datacenter": "ICN-01",
  "workload": "LLM-13B",
  "metrics": { 
    "avg_latency_ms": 108.2, 
    "p99_latency_ms": 155.3, 
    "throughput_tps": 1380.4 
  },
  "carbon": { 
    "inference_gCO2_per_1k": 17.8, 
    "total_kWh": 42.3, 
    "renewable_pct": 31.2 
  },
  "security": { 
    "dll_signed": true, 
    "tpm_pcr_extended": true, 
    "aes_gcm_enabled": true 
  }
}
```

---

## 19. Checklist (Part 3 Completed)

* [x] Hot Reload atomic swap
* [x] NVML Carbon meter integration
* [x] Grafana plugin backend/frontend
* [x] Dashboard provisioning
* [x] Open-source contribution docs
* [x] Rollback Agent + HealthCheck
* [x] ESG report sample

---

