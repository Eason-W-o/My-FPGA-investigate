# 模型对照结论

所有指标均在同一 Set5 图像、Y 通道、Bicubic ×2 降采样口径下计算。

- **Bicubic**：PSNR 32.6398 dB，SSIM 0.92782，算力 0.0000 GMAC/帧。
- **Original FSRCNN-s**：PSNR 33.8217 dB，SSIM 0.93852，算力 2.0409 GMAC/帧。
- **Team lightweight model INT8**：PSNR 34.0190 dB，SSIM 0.93785，算力 1.4681 GMAC/帧。

原版 FSRCNN-s 用作软件画质参考；团队 INT8 轻量模型用作 FPGA 实现目标，二者不可混称。
