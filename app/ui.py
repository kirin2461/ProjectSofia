"""Qt6-интерфейс лаунчера: заставка с прогрессом, окно с веб-UI, трей."""

import logging
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import bootstrap, paths
from app.envfile import upsert_env
from app.llm import ModelInfo, scan_models
from app.runtime import STAGES, Runtime

logger = logging.getLogger(__name__)

ACCENT = "#38e8ff"
ACCENT_2 = "#7b5cff"
BG = "#070a14"
PANEL = "#0f1424"
LINE = "#1d2540"
MUTED = "#8a96b0"
OK = "#3ddc97"
ERR = "#ff5d5d"

STYLE = f"""
QWidget {{ color: #e8eef9; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 13px; }}
QMainWindow, QDialog, QMessageBox {{ background: {BG}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 9px;
               padding: 7px 16px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton#primary {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {ACCENT}, stop:1 {ACCENT_2}); color: #071018;
                        border: none; font-weight: 600; }}
QPushButton#link {{ background: transparent; border: none; color: {MUTED}; padding: 4px 6px; }}
QPushButton#link:hover {{ color: #e8eef9; }}
QMenuBar {{ background: {BG}; border-bottom: 1px solid {LINE}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {PANEL}; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 6px 22px 6px 24px; border-radius: 6px; }}
QMenu::item:selected {{ background: #1c2a4a; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 5px 8px; }}
QTableWidget {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px;
                gridline-color: {LINE}; }}
QHeaderView::section {{ background: {BG}; color: {MUTED}; border: none; padding: 6px; }}
QToolTip {{ background: {PANEL}; color: #e8eef9; border: 1px solid {LINE}; padding: 4px; }}
"""


def paint_orb(p: QPainter, rect: QRectF, a: QColor, b: QColor, glow: float = 1.0):
    """Сфера Софии: свечение + ядро-градиент (как в веб-интерфейсе)."""
    c = rect.center()
    r = rect.width() / 2
    halo = QRadialGradient(c, r)
    inner = QColor(a)
    inner.setAlphaF(max(0.0, min(1.0, 0.45 * glow)))
    outer = QColor(a)
    outer.setAlphaF(0.0)
    halo.setColorAt(0.35, inner)
    halo.setColorAt(1.0, outer)
    p.setPen(Qt.NoPen)
    p.setBrush(halo)
    p.drawEllipse(rect)

    core = rect.adjusted(r * 0.3, r * 0.3, -r * 0.3, -r * 0.3)
    cr = core.width() / 2
    grad = QRadialGradient(core.left() + cr * 0.64, core.top() + cr * 0.56, cr * 1.4)
    grad.setColorAt(0.0, QColor("#ffffff"))
    grad.setColorAt(0.2, a)
    grad.setColorAt(0.62, b)
    grad.setColorAt(1.0, QColor("#0b1020"))
    p.setBrush(grad)
    p.drawEllipse(core)


def make_icon() -> QIcon:
    """Иконка приложения — та же сфера."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        if size <= 24:  # на мелких размерах свечение превращается в «кашу»
            grad = QRadialGradient(size * 0.36, size * 0.32, size * 0.7)
            grad.setColorAt(0.0, QColor("#dffbff"))
            grad.setColorAt(0.45, QColor(ACCENT))
            grad.setColorAt(1.0, QColor(ACCENT_2))
            p.setPen(Qt.NoPen)
            p.setBrush(grad)
            p.drawEllipse(QRectF(1, 1, size - 2, size - 2))
        else:
            grow = size * 0.18
            paint_orb(p, QRectF(-grow, -grow, size + 2 * grow, size + 2 * grow),
                      QColor(ACCENT), QColor(ACCENT_2))
        p.end()
        icon.addPixmap(pix)
    return icon


def dark_title_bar(widget: QWidget):
    """Тёмная системная рамка окна (Windows 10 1809+ / 11)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        on = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (новый / старый номер)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
        caption = ctypes.c_int(0x00140A07)  # DWMWA_CAPTION_COLOR, BGR для #070a14
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), 4)
    except (AttributeError, OSError):
        pass


# ═══════════════════════════════════════════════════════════
# Фоновый запуск
# ═══════════════════════════════════════════════════════════
class StartWorker(QObject):
    progress = Signal(str, int)  # текст, процент (-1 = без процентов)
    stage = Signal(int)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, base: Path):
        super().__init__()
        self.base = base
        self.runtime: Optional[Runtime] = None

    def run(self):
        self.runtime = Runtime(self.base, self.report, self.stage.emit)
        try:
            self.runtime.prepare()
        except Exception as e:  # noqa: BLE001
            logger.error("Ошибка запуска:\n%s", traceback.format_exc())
            self.failed.emit(str(e) or e.__class__.__name__)
            return
        self.finished.emit()

    def report(self, text: str, percent: Optional[int]):
        logger.info("%s", text)
        self.progress.emit(text, -1 if percent is None else int(percent))


# ═══════════════════════════════════════════════════════════
# Заставка
# ═══════════════════════════════════════════════════════════
class OrbWidget(QWidget):
    """Анимированная сфера: дыхание + вращающиеся кольца, пока идёт работа."""

    def __init__(self, size: int = 120, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.a = QColor(ACCENT)
        self.b = QColor(ACCENT_2)
        self.busy = True
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def set_colors(self, a: str, b: str):
        self.a, self.b = QColor(a), QColor(b)
        self.update()

    def set_busy(self, busy: bool):
        self.busy = busy
        self.update()

    def _tick(self):
        self._t += 0.016
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        scale = 1.0 + 0.035 * math.sin(self._t * 2 * math.pi / 4)
        pad = w * (1 - scale) / 2
        paint_orb(p, QRectF(pad, pad, w - 2 * pad, w - 2 * pad), self.a, self.b,
                  glow=0.8 + 0.2 * math.sin(self._t * 1.6))
        if self.busy:
            for i, (radius, speed, col) in enumerate(
                    ((0.40, 150, self.a), (0.47, -100, self.b))):
                pen = QPen(col, 1.6)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                r = w * radius
                rect = QRectF(w / 2 - r, w / 2 - r, 2 * r, 2 * r)
                start = int((self._t * speed + i * 120) * 16) % (360 * 16)
                p.drawArc(rect, start, 110 * 16)
                p.drawArc(rect, start + 180 * 16, 60 * 16)
        p.end()


class StepRow(QWidget):
    """Строка шага запуска: значок состояния + название."""

    GLYPHS = {"wait": "○", "run": "◌", "done": "✓", "fail": "✕"}
    MARK = {"wait": "#4a5574", "run": ACCENT, "done": OK, "fail": ERR}
    TEXT = {"wait": "#5d6886", "run": "#e8eef9", "done": MUTED, "fail": "#ffc9c9"}

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.mark = QLabel()
        self.mark.setFixedWidth(18)
        self.mark.setAlignment(Qt.AlignCenter)
        self.title = QLabel(title)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self.mark)
        lay.addWidget(self.title, 1)
        self.set_state("wait")

    def set_state(self, state: str):
        self.state = state
        weight = 600 if state == "run" else 400
        self.mark.setText(self.GLYPHS[state])
        self.mark.setStyleSheet(f"color: {self.MARK[state]}; font-size: 14px; font-weight: 700;")
        self.title.setStyleSheet(f"color: {self.TEXT[state]}; font-weight: {weight};")


class Splash(QWidget):
    """Окно запуска без рамки: сфера, шаги, прогресс, журнал по запросу."""

    retry = Signal()

    def __init__(self, base: Path, icon: QIcon):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Window)
        self.base = base
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("София — запуск")
        self.setWindowIcon(icon)
        self.setFixedWidth(470)
        self._drag: Optional[QPoint] = None
        self._stage = -1

        card = QFrame(self)
        card.setObjectName("card")
        card.setStyleSheet(f"""
            QFrame#card {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                           stop:0 #0d1326, stop:1 {BG});
                           border: 1px solid {LINE}; border-radius: 18px; }}
            QLabel {{ background: transparent; }}
            QProgressBar {{ background: {PANEL}; border: none; border-radius: 2px; }}
            QProgressBar::chunk {{ border-radius: 2px; background: qlineargradient(
                           x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_2}, stop:1 {ACCENT}); }}
            QPlainTextEdit {{ background: #060912; border: 1px solid {LINE}; border-radius: 8px;
                              font-family: "Cascadia Code", Consolas; font-size: 11px;
                              color: {MUTED}; padding: 4px; }}
        """)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 180))
        card.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addWidget(card)

        self.orb = OrbWidget(112)
        name = QLabel("СОФИЯ")
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet("font-size: 22px; font-weight: 600; letter-spacing: 7px;")
        self.sub = QLabel("Просыпаюсь…")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setStyleSheet(f"color: {MUTED};")

        self.rows = [StepRow(t) for t in STAGES]
        steps = QVBoxLayout()
        steps.setSpacing(8)
        for row in self.rows:
            steps.addWidget(row)

        self.status = QLabel(" ")
        self.status.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setFixedHeight(4)
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 0)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setFixedHeight(130)
        self.log.hide()

        self.btn_details = QPushButton("Подробнее ▾")
        self.btn_details.setObjectName("link")
        self.btn_details.clicked.connect(self._toggle_log)
        self.btn_log = QPushButton("Журнал")
        self.btn_retry = QPushButton("Повторить")
        self.btn_retry.setObjectName("primary")
        self.btn_quit = QPushButton("Выход")
        self.btn_quit.setObjectName("link")
        self.btn_log.clicked.connect(lambda: open_path(paths.log_path(self.base)))
        self.btn_retry.clicked.connect(self.retry.emit)
        self.btn_quit.clicked.connect(QApplication.quit)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_details)
        buttons.addStretch(1)
        for b in (self.btn_quit, self.btn_log, self.btn_retry):
            buttons.addWidget(b)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(32, 28, 32, 20)
        lay.setSpacing(6)
        lay.addWidget(self.orb, alignment=Qt.AlignHCenter)
        lay.addSpacing(6)
        lay.addWidget(name)
        lay.addWidget(self.sub)
        lay.addSpacing(18)
        lay.addLayout(steps)
        lay.addSpacing(14)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.log)
        lay.addSpacing(4)
        lay.addLayout(buttons)
        self.reset()

    # ── Перетаскивание окна без рамки ───────────────────────
    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e):  # noqa: N802
        self._drag = None

    # ── Состояние ───────────────────────────────────────────
    def _toggle_log(self):
        show = not self.log.isVisible()
        self.log.setVisible(show)
        self.btn_details.setText("Скрыть ▴" if show else "Подробнее ▾")
        self.adjustSize()

    def _status_style(self, color: str):
        self.status.setStyleSheet(f"color: {color}; font-size: 12px;")

    def on_stage(self, index: int):
        self._stage = index
        for i, row in enumerate(self.rows):
            row.set_state("done" if i < index else "run" if i == index else "wait")
        if index >= len(self.rows):
            self.sub.setText("Готова!")
            self.orb.set_busy(False)

    def on_progress(self, text: str, percent: int):
        self.status.setText(text)
        self.log.appendPlainText(text)
        if percent >= 0:
            self.bar.setRange(0, 100)
            self.bar.setValue(percent)
        elif self.bar.maximum() != 0:
            self.bar.setRange(0, 0)

    def on_failed(self, message: str):
        if 0 <= self._stage < len(self.rows):
            self.rows[self._stage].set_state("fail")
        self.sub.setText("Что-то пошло не так")
        self.status.setText(message)
        self._status_style("#ffc9c9")
        self.log.appendPlainText(f"ОШИБКА: {message}")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.orb.set_busy(False)
        self.orb.set_colors(ERR, "#ff9f1c")
        self.btn_log.show()
        self.btn_retry.show()

    def reset(self):
        self.btn_log.hide()
        self.btn_retry.hide()
        self._stage = -1
        for row in self.rows:
            row.set_state("wait")
        self.bar.setRange(0, 0)
        self.sub.setText("Просыпаюсь…")
        self.status.setText(" ")
        self._status_style(MUTED)
        self.orb.set_colors(ACCENT, ACCENT_2)
        self.orb.set_busy(True)

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        self.adjustSize()
        screen = self.screen().availableGeometry()
        self.move(screen.center() - self.rect().center())


# ═══════════════════════════════════════════════════════════
# Главное окно
# ═══════════════════════════════════════════════════════════
class SofiaPage(QWebEnginePage):
    """Внешние ссылки открываются в системном браузере."""

    def __init__(self, local_prefix: str, parent=None):
        super().__init__(parent)
        self.local_prefix = local_prefix

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):  # noqa: N802
        if is_main_frame and not url.toString().startswith(self.local_prefix):
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class MainWindow(QMainWindow):
    def __init__(self, runtime: Runtime, icon: QIcon):
        super().__init__()
        self.runtime = runtime
        self.icon = icon
        self.quitting = False
        self._tray_hint_shown = False

        self.setWindowTitle("София")
        self.setWindowIcon(icon)
        self.resize(1100, 780)

        self.view = QWebEngineView(self)
        page = SofiaPage(runtime.url.rstrip("/"), self.view)
        page.setBackgroundColor(QColor(BG))  # без белой вспышки до загрузки
        self.view.setPage(page)
        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        self.setCentralWidget(self.view)
        self.view.load(QUrl(runtime.url))

        self._build_menu()
        self._build_tray()
        dark_title_bar(self)

    # ── Меню ────────────────────────────────────────────────
    def _build_menu(self):
        bar = self.menuBar()

        m_file = bar.addMenu("&Файл")
        m_file.addAction("Папка данных", lambda: open_path(self.runtime.base / "data"))
        m_file.addAction("Образцы голоса", lambda: open_path(
            self.runtime.base / "data" / "voice_samples"))
        m_file.addAction("Библиотека (книги)", lambda: open_path(
            self.runtime.base / "data" / "library"))
        m_file.addSeparator()
        m_file.addAction("Перезапустить", self.restart)
        m_file.addAction("Выход", self.quit_app)

        self.m_model = bar.addMenu("&Модель")
        self._fill_model_menu()

        m_tools = bar.addMenu("&Инструменты")
        m_tools.addAction("Обновить страницу", self.view.reload)
        m_tools.addAction("Открыть в браузере",
                          lambda: QDesktopServices.openUrl(QUrl(self.runtime.url)))
        m_tools.addSeparator()
        m_tools.addAction("Диагностика…", self.show_diagnostics)
        m_tools.addAction("Журнал приложения",
                          lambda: open_path(paths.log_path(self.runtime.base)))
        m_tools.addAction("Журнал llama-server",
                          lambda: open_path(self.runtime.base / "logs" / "llama-server.log"))

    def _fill_model_menu(self):
        menu = self.m_model
        menu.clear()
        group = QActionGroup(menu)
        group.setExclusive(True)
        current = self.runtime.model
        for model in [m for m in self.runtime.models if not m.embedding]:
            act = QAction(str(model), menu)
            act.setCheckable(True)
            act.setToolTip(str(model.path))
            act.setChecked(current is not None and model.path == current.path)
            act.triggered.connect(lambda _=False, m=model: self.switch_model(m))
            group.addAction(act)
            menu.addAction(act)
        if not self.runtime.models:
            menu.addAction("(моделей не найдено)").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Автовыбор под железо", self.auto_model)
        menu.addAction("Добавить GGUF-файл…", self.add_model)
        menu.addAction("Папка моделей", lambda: open_path(paths.model_dir(self.runtime.base)))

    # ── Трей ────────────────────────────────────────────────
    def _build_tray(self):
        self.tray = QSystemTrayIcon(self.icon, self)
        self.tray.setToolTip("София")
        menu = QMenu()
        menu.addAction("Открыть Софию", self.show_window)
        menu.addAction("Перезапустить", self.restart)
        menu.addSeparator()
        menu.addAction("Выход", self.quit_app)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray)
        self.tray.show()

    def _on_tray(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_window()

    def show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):  # noqa: N802
        if self.quitting or not QSystemTrayIcon.isSystemTrayAvailable():
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.tray.showMessage("София работает в фоне",
                                  "Откройте её из значка в трее. «Выход» — в меню значка.",
                                  self.icon, 4000)

    def quit_app(self):
        self.quitting = True
        self.tray.hide()
        QApplication.quit()

    def restart(self):
        self.quitting = True
        self.tray.hide()
        app = QApplication.instance()
        app.setProperty("sofia_relaunch", True)
        app.quit()

    # ── Модели ──────────────────────────────────────────────
    def switch_model(self, model: ModelInfo):
        if self.runtime.model and model.path == self.runtime.model.path:
            return
        answer = QMessageBox.question(
            self, "Смена модели",
            f"Переключиться на «{model.name}» ({model.size_gb:.1f} ГБ)?\n"
            "София перезапустится, это займёт 1–2 минуты.")
        if answer != QMessageBox.Yes:
            self._fill_model_menu()
            return
        bootstrap.apply_model_choice(self.runtime.base, model)
        self.restart()

    def auto_model(self):
        upsert_env(paths.env_path(self.runtime.base), {"SOFIA_MODEL_PATH": ""})
        QMessageBox.information(self, "Автовыбор",
                                "Модель будет подобрана заново после перезапуска.")
        self.restart()

    def add_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите GGUF-модель",
                                              str(Path.home()), "GGUF (*.gguf)")
        if not file:
            return
        found = scan_models([Path(file).parent])
        model = next((m for m in found if m.path == Path(file).resolve()), None)
        if model is None or model.embedding:
            QMessageBox.warning(self, "Модель", "Файл не похож на языковую GGUF-модель.")
            return
        self.switch_model(model)

    # ── Диагностика ─────────────────────────────────────────
    def show_diagnostics(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Диагностика")
        dark_title_bar(dlg)
        dlg.resize(640, 380)
        rows = self.runtime.diagnostics()
        table = QTableWidget(len(rows), 3, dlg)
        table.setHorizontalHeaderLabels(["Проверка", "", "Значение"])
        for i, (name, ok, value) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(name))
            mark = QTableWidgetItem("✔" if ok else "✖")
            mark.setForeground(QColor(OK if ok else ERR))
            table.setItem(i, 1, mark)
            table.setItem(i, 2, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()
        lay = QVBoxLayout(dlg)
        lay.addWidget(table)
        close = QPushButton("Закрыть")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=Qt.AlignRight)
        dlg.exec()


def open_path(path: Path):
    path = Path(path)
    if not path.exists():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# ═══════════════════════════════════════════════════════════
# Сборка приложения
# ═══════════════════════════════════════════════════════════
class Controller(QObject):
    """Связывает заставку, фоновый запуск и главное окно."""

    def __init__(self, app: QApplication, base: Path):
        super().__init__()
        self.app = app
        self.base = base
        self.icon = make_icon()
        app.setWindowIcon(self.icon)
        self.splash = Splash(base, self.icon)
        self.splash.retry.connect(self.start)
        self.window: Optional[MainWindow] = None
        self.runtime: Optional[Runtime] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[StartWorker] = None

    def start(self):
        if self.runtime is not None:
            self.runtime.stop()
        self.splash.reset()
        self.splash.show()
        self.worker = StartWorker(self.base)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.splash.on_progress)
        self.worker.stage.connect(self.splash.on_stage)
        self.worker.finished.connect(self._on_ready)
        self.worker.failed.connect(self.splash.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _on_ready(self):
        self.runtime = self.worker.runtime
        self.window = MainWindow(self.runtime, self.icon)
        self.window.show()
        self.splash.hide()

    def show_main(self):
        if self.window is not None:
            self.window.show_window()
        else:
            self.splash.show()
            self.splash.raise_()
            self.splash.activateWindow()

    def shutdown(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        if self.runtime is not None:
            self.runtime.stop()


def apply_style(app: QApplication):
    app.setStyleSheet(STYLE)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
