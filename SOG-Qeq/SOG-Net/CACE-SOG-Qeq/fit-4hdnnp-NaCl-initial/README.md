# fit-4hdnnp-NaCl-initial

本目录与 `fit-4hdnnp-NaCl` 使用相同的 CACE+Qeq+SOG 训练流程，唯一区别是 **SOG 参数按 BSA（双边级数近似）与 Ji 等 2026 论文一致地初始化**。

- **数据**：使用 `../fit-4hdnnp-NaCl/NaCl.xyz`，无需在本目录下再放一份数据。
- **脚本**：`fit-cace-SOG.py`。在 `cace_nnp.to(device)` 之后调用  
  `charge_eq.init_sog_from_bsa(r_cut=cutoff, b=2.0)`，  
  用 1/r 的 BSA 实空间高斯和参数覆盖 `sog_log_alpha` 与 `sog_weights` 的初值。
- **BSA 实现**：在 `cace/modules/charge_eq.py` 的 `ChargeEq.init_sog_from_bsa(r_cut, b=2.0)` 中实现，详见 `../fit-4hdnnp-NaCl/SOG_initialization.md`。

运行方式与 `fit-4hdnnp-NaCl` 相同，例如：

```bash
python fit-cace-SOG.py
```

损失等输出会写入 `loss_data/N_18_BSA_*`，以与未做 BSA 初始化的 `N_18_*` 区分。
