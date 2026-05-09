# 02 Data and Leakage Contract

## 1. 已知数据结构

Phase 0 audit 显示，当前美国 `DA.nc` 实际采用 multi-channel arrays，而不是每个语义变量都是独立 NetCDF data variable：

```text
input[T, 12, H, W]
target[T, 4, H, W]
height = 256
width  = 640
```

合作方代码中的语义变量仍然有效，但必须通过 channel mapping 解析：

```text
sm_surface_forecast
sm_rootzone_forecast
mwrtm_vegopacity
tb_h_obs
tb_v_obs
base_valid_mask
sm_surface_analysis
sm_rootzone_analysis
```

`specs/hydroda_dataset_contract.yaml` 必须反映实际 multi-channel structure，并保留 semantic aliases。

---

## 2. Dataset sample contract

`HydroDADataset.__getitem__` 应返回：

```python
{
  "x": Tensor[C, H, W],
  "forecast": Tensor[2, H, W],
  "analysis": Tensor[2, H, W],
  "increment": Tensor[2, H, W],
  "obs_mask": Tensor[1, H, W],
  "loss_mask": Tensor[1, H, W],
  "region_mask": Tensor[1, H, W],
  "time_code": Tensor[T],
  "date": str,
  "year": int,
  "month": int,
  "day": int,
  "country_id": str,
  "region_id": str,
  "regime_id": str,
  "split_id": str,
  "patch_id": str,
}
```

其中：

```text
forecast/raw analysis 必须保留 raw physical units。
increment_raw = analysis_raw - forecast_raw。
```

如果模型使用 normalized target，应额外保存 target stats，并保证 stats 只来自 source train 或 allowed support。

---

## 3. Mask 语义绝不能混淆

当前已知 `input` channel 11 是动态 binary mask，暂定命名为：

```text
base_valid_mask
```

在没有进一步证据前，不要把它强行解释为：

```text
obs_mask
region_mask
static_land_mask
```

推荐 Phase 1 暂定：

```text
base_valid_mask = input[:, 11, :, :]
loss_mask       = base_valid_mask
metric_mask     = base_valid_mask
region_mask     = external artifact from Phase 2
obs_mask        = None unless independently recovered
```

`region_mask` 必须由 geolocation/region artifact 生成，不得由 `base_valid_mask` 冒充。

---

## 4. Normalization contract

默认：

```text
input_stats      来自 source train only
increment_stats  来自 source train only
```

K>0 adaptation 可以使用 target support labels 计算 adaptation loss，但不要用 support stats 改变全局 normalization，除非该行为被明确记录为一个方法变体。

禁止：

```text
使用 target query years 计算任何 normalization statistics
```

---

## 5. Reconstruction invariant

任何 dataset / model / metric 实现都必须满足：

```text
analysis ≈ forecast + increment
pred_analysis = forecast + pred_increment
```

Phase 1 必须写测试：

```text
max_abs((analysis - forecast) - increment) < tolerance
```

---

## 6. NetCDF audit 必须报告

```text
dims / coords / data_vars
time range / frequency / missing dates
input / target channel names and order
shape of each variable
NaN / Inf count and ratio
mask unique values and valid ratio
finite forecast / analysis overlap
estimated memory cost
coordinate availability: lat/lon or grid only
```

如果 `DA.nc` 没有 lat/lon，不能直接断言 geolocation 不可恢复；必须继续递归检查：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP
```

如果整个目录级 geolocation lookup 仍失败，区域 mask 才能进入 development-only fallback 模式，并在报告中明确说明。
