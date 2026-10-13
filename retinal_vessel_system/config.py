# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALGORITHM_NAMES = {
    "preview": "预处理预览",
    "otsu": "Otsu 自动阈值",
    "watershed": "自适应分水岭",
    "sobel": "Sobel",
    "prewitt": "Prewitt",
    "roberts": "Roberts",
    "canny": "Canny",
    "laplacian": "Laplacian",
    "scharr": "Scharr",
}
COMPARISON_ALGORITHMS = tuple(k for k in ALGORITHM_NAMES if k != "preview")

STYLE = """
QMainWindow, QDialog { background: #FFFFFF; }
QWidget { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
          font-size: 13px; color: #111827; }
QFrame#sidebar, QFrame#parameters { background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 8px; }
QFrame#imageCard { background: white; border: 1px solid #E5E7EB; border-radius: 8px; }
QLabel#brand { font-size: 19px; font-weight: 700; }
QLabel#sectionTitle { font-size: 15px; font-weight: 600; }
QLabel#muted { color: #6B7280; }
QLabel#value { color: #2563EB; font-weight: 600; }
QLabel#navHeading { color: #64748B; font-size: 12px; font-weight: 600; }
QPushButton#navButton { text-align: left; border: none; background: transparent; padding: 4px 9px; }
QPushButton#navButton:hover { background: #F1F5F9; }
QPushButton#navButton:checked { background: #EAF2FF; color: #2563EB; font-weight: 600; }
QPushButton#navButton:disabled { color: #9CA3AF; }
QWidget#scrollBody { background: #F8FAFC; }
QTabWidget::pane { border: 1px solid #E5E7EB; border-radius: 5px; }
QTabBar::tab { background: #F1F5F9; border: none; padding: 7px 15px; }
QTabBar::tab:selected { background: #FFFFFF; color: #2563EB; }
QScrollBar:vertical { background: #F1F5F9; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QPushButton { background: white; border: 1px solid #DDE2E9; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #EFF6FF; border-color: #93C5FD; }
QPushButton:checked { background: #EFF6FF; color: #2563EB; border-color: #93C5FD; }
QPushButton:disabled { color: #9CA3AF; background: #F3F4F6; border-color: #E5E7EB; }
QPushButton#primary { background: #2563EB; border-color: #2563EB; color: white; }
QPushButton#primary:hover { background: #1D4ED8; }
QPushButton#primary:disabled { background: #CBD5E1; border-color: #CBD5E1; }
QComboBox { background: white; border: 1px solid #DDE2E9; border-radius: 6px; padding: 7px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #EFF6FF; selection-color: #2563EB; }
QSlider::groove:horizontal { height: 5px; background: #E2E8F0; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2563EB; border-radius: 2px; }
QSlider::handle:horizontal { background: #2563EB; width: 13px; margin: -4px 0; border-radius: 6px; }
QScrollArea { border: none; background: transparent; }
QGroupBox { border: 1px solid #E5E7EB; border-radius: 6px; margin-top: 14px; padding-top: 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QStatusBar { background: #F8FAFC; border-top: 1px solid #E5E7EB; }
QToolTip { background: #111827; color: white; padding: 5px; border: none; }
"""

