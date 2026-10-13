# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="视网膜血管图像分割系统")
    parser.add_argument("image", nargs="?", help="启动时加载的图像路径")
    parser.add_argument("--example", action="store_true", help="启动时加载示例眼底图")
    parser.add_argument("--smoke-test", action="store_true", help="无显示器启动并验证窗口构造后退出")
    args = parser.parse_args(argv)
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # 延迟导入让 --help 无需先安装图形依赖。
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from retinal_vessel_system.config import STYLE
    from retinal_vessel_system.gui.main_window import MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName("视网膜血管图像分割系统")
    app.setOrganizationName("RetinalVesselLab")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if args.image:
        window.load_image(args.image)
    elif args.example:
        window.load_example()
    if args.smoke_test:
        QTimer.singleShot(600, window.close)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
