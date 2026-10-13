# 视网膜血管图像分割系统

基于 PySide6 的眼底图像处理与算法比较桌面程序，支持灰度预览、窗宽窗位、绿色通道、高斯与自适应中值滤波、双图同步操作、ROI、Otsu 和分水岭分割，以及 Sobel、Prewitt、Roberts、Canny、Laplacian、Scharr 比较与结果导出。

## 安装与启动

推荐 Python 3.11。进入仓库根目录后执行：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

打开 `data/retina.jpg` 可运行自带示例。`docs/ui-preview.png` 为界面预览，`docs/validation.json` 保留原项目的历史算法与交互验证记录。

## 验证


```bash
python -m pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest -q
# 可选：重新生成真实示例图像的验证记录
QT_QPA_PLATFORM=offscreen python scripts/verify_demo.py
```

最终版冻结为 `v1.0.0`。39 项现有测试通过，覆盖处理算法、GUI 交互、图像读写和导出。该项目用于教学与算法比较，尚未提供基于真实血管标注数据集的分割精度评估。冻结信息及文件校验值见 `FINAL_VERSION.json`。

## 版权与项目介绍

Copyright (c) 2026 cin330。原创部分以 [AGPL-3.0](LICENSE) 发布，使用、修改及分发须保留署名并遵守许可。第三方内容保留原有权利。

[三个项目的完整中文介绍](PROJECT_INTRODUCTION.md) · [版权声明](COPYRIGHT.md) · [第三方组件](THIRD_PARTY.md)
