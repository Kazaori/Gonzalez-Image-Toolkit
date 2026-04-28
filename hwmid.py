import sys
import math
import os
import numpy as np
import cv2
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QGroupBox, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QGraphicsScene, QGraphicsView, QDialog)
from matplotlib import pyplot as plt

# ==================== 核心算法模块 ====================

def resource_path(relative_path):
    """
    资源路径解析。
    提供对 PyInstaller 单文件打包模式下临时解压目录（_MEIPASS）的兼容支持。
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def cv2_to_qimage(img):
    """
    矩阵格式转换。
    将底层计算使用的 NumPy 矩阵转换为 PyQt 前端渲染所需的 QImage 格式，并处理通道排列。
    """
    if img is None: return None
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Indexed8)
        qimg.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        return qimg
    else:
        h, w, ch = img.shape
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(cvt.data, w, h, ch * w, QtGui.QImage.Format_RGB888)

def plot_histogram_on_ax(ax, img, title, mode='gray'):
    """
    基于 Matplotlib 的直方图数据绘制函数。
    对应《数字图像处理（第四版）》第 3.3 节 直方图处理的基本概念。
    """
    ax.set_title(title, fontdict={'fontsize': 11})
    if mode == 'gray':
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ax.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
    elif mode == 'rgb' and img.ndim == 3:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256],[0, 256])
            ax.plot(hist, color=col, alpha=0.8, linewidth=1.5)
        ax.set_xlim([0, 256])

def show_histograms_comparision(img_orig, img_proc, title1="原图直方图 (Original)", title2="处理后直方图 (Processed)"):
    """
    构建直方图可视化面板。
    根据图像通道深度自适应生成 1x2 (单灰度) 或 2x2 (灰度与 RGB 分离) 的统计分布对比图。
    """
    if img_orig is None or img_proc is None: return
    
    plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    is_color = img_orig.ndim == 3 or img_proc.ndim == 3
    
    if is_color:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plot_histogram_on_ax(axes[0, 0], img_orig, title1 + " - 灰度", mode='gray')
        plot_histogram_on_ax(axes[0, 1], img_proc, title2 + " - 灰度", mode='gray')
        plot_histogram_on_ax(axes[1, 0], img_orig, title1 + " - RGB通道", mode='rgb')
        plot_histogram_on_ax(axes[1, 1], img_proc, title2 + " - RGB通道", mode='rgb')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_histogram_on_ax(axes[0], img_orig, title1, mode='gray')
        plot_histogram_on_ax(axes[1], img_proc, title2, mode='gray')
    
    plt.tight_layout()
    plt.show()

def histogram_equalize(img):
    """
    全局直方图均衡化。对应《数字图像处理（第四版）》3.3.1节。
    通过累积分布函数(CDF)平坦化概率密度，实现自适应全局对比度提升。
    注：彩色图像在 YUV 色彩空间仅针对明度(Y)分量处理，以维持色相恒定。
    """
    if img.ndim == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return cv2.equalizeHist(img)

def clahe_enhance(img, clip=2.0, tile=(8,8)):
    """
    限制对比度的自适应直方图均衡化 (CLAHE)。对应《数字图像处理（第四版）》3.3.3节。
    基于局部空间网格限制累积函数的斜率，以抑制均质区域的噪声放大现象。
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    if img.ndim == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return clahe.apply(img)

def gamma_correction(img, gamma=1.0):
    """
    幂律（伽马）变换。对应《数字图像处理（第四版）》3.2.3节。
    基于 s = c * r^gamma 的指数映射。通过预计算 256 级查找表 (LUT) 加速离散灰度级映射。
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def add_noise(img, mode='gaussian', var=10, amount=0.05):
    """
    空间退化/加噪模型。对应《数字图像处理（第四版）》5.2.1节 噪声的空间和频率特性。
    通过合成高斯(Gaussian)、脉冲(Salt-Pepper)等概率密度函数(PDF)扰动图像矩阵。
    """
    if img is None: return img
    out = img.copy().astype(np.float32)
    shape = img.shape
    if mode == 'gaussian':
        out = out + np.random.normal(0, math.sqrt(var), shape)
    elif mode == 'salt_pepper':
        num_salt = np.ceil(amount * img.size * 0.5)
        coords =[np.random.randint(0, i - 1, int(num_salt)) for i in shape[:2]]
        out[tuple(coords)] = 255
        num_pepper = np.ceil(amount * img.size * 0.5)
        coords =[np.random.randint(0, i - 1, int(num_pepper)) for i in shape[:2]]
        out[tuple(coords)] = 0
    elif mode == 'speckle':
        out = out + out * np.random.normal(0, math.sqrt(var), shape)
    elif mode == 'poisson':
        vals = 2 ** np.ceil(np.log2(len(np.unique(out))))
        out = np.random.poisson(out * vals) / float(vals)
    return np.clip(out, 0, 255).astype(np.uint8)

def frequency_filter(img, mode='lowpass', ftype='gaussian', cutoff=30, order=2):
    """
    频率域滤波。对应《数字图像处理（第四版）》4.7节至4.8节。
    对二维傅里叶频域构建滤波器传递函数 H(u,v)，执行频域相乘后再通过逆变换重建空间域图像。
    """
    is_color = (img.ndim == 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if is_color else img.astype(np.float32)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    dft_shift = np.fft.fftshift(np.fft.fft2(gray))
    u, v = np.meshgrid(np.arange(-ccol, cols - ccol), np.arange(-crow, rows - crow))
    D = np.sqrt(u**2 + v**2)
    
    if ftype == 'ideal':
        H = np.zeros_like(D)
        if mode == 'lowpass': H[D <= cutoff] = 1
        else: H[D > cutoff] = 1
    elif ftype == 'gaussian':
        if mode == 'lowpass': H = np.exp(-(D**2) / (2 * (cutoff**2)))
        else: H = 1 - np.exp(-(D**2) / (2 * (cutoff**2)))
    else: 
        if mode == 'lowpass': H = 1 / (1 + (D / (cutoff + 1e-10))**(2 * order))
        else: H = 1 / (1 + (cutoff / (D + 1e-10))**(2 * order))
        
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_shift * H)))
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR) if is_color else img_back

# ==================== GUI 组件定义 ====================

class SaveConfigDialog(QDialog):
    """
    图像导出参数配置对话框。
    用于设定图像的打印分辨率(DPI)及联合图像专家组(JPEG)标准的压缩品质。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出设置")
        self.resize(320, 150)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        lay_dpi = QHBoxLayout()
        lay_dpi.addWidget(QtWidgets.QLabel("分辨率 (DPI):"))
        self.spin_dpi = QtWidgets.QSpinBox()
        self.spin_dpi.setRange(72, 1200); self.spin_dpi.setValue(300)
        lay_dpi.addWidget(self.spin_dpi)
        layout.addLayout(lay_dpi)
        
        lay_quality = QHBoxLayout()
        lay_quality.addWidget(QtWidgets.QLabel("图像质量 (1-100):"))
        self.spin_quality = QtWidgets.QSpinBox()
        self.spin_quality.setRange(1, 100); self.spin_quality.setValue(100)
        lay_quality.addWidget(self.spin_quality)
        layout.addLayout(lay_quality)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.button(QtWidgets.QDialogButtonBox.Ok).setText("确定")
        btn_box.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

class SyncGraphicsView(QGraphicsView):
    """
    基于 QGraphicsView 的交互式渲染视口。
    屏蔽了物理滚动条显示，重写了鼠标点击事件序列以支持活动窗口的无缝拦截与焦点切换。
    """
    viewClicked = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # 隐藏滚动条以维持界面的视觉纯净，同时保持拖拽响应能力
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        self.setStyleSheet("border: 2px solid transparent; background-color: #2e2e2e;")
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

    def set_image(self, qpixmap):
        self.pixmap_item.setPixmap(qpixmap)
        self.scene.setSceneRect(QtCore.QRectF(qpixmap.rect()))

    def mousePressEvent(self, event):
        self.viewClicked.emit(self)
        super().mousePressEvent(event)

# ==================== 主系统界面 ====================

class FusedImageTool(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字图像处理实验系统 (基于冈萨雷斯第4版)")
        self.resize(1300, 800)
        
        icon_full_path = resource_path('icon.ico')
        if os.path.exists(icon_full_path):
            self.setWindowIcon(QtGui.QIcon(icon_full_path))
            
        self.img_first_loaded = None
        self.history =[]
        self.img_base = None
        self.img_proc = None
        
        self.sync_mode = True
        self.is_syncing_scroll = False
        
        self.init_ui()
        
        self.active_view = self.view_orig
        self.update_active_view_style()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        
        # ---------------- 顶部控制台 ----------------
        btn_layout = QHBoxLayout()
        btn_load = QtWidgets.QPushButton("📂 加载图像")
        btn_save = QtWidgets.QPushButton("💾 保存结果")
        btn_pipeline = QtWidgets.QPushButton("📌 固化为原图 (进入下一步)")
        btn_pipeline.setStyleSheet("color: #0078D7; font-weight: bold; border: 1px solid #0078D7; border-radius: 4px; padding: 5px;")
        
        btn_undo = QtWidgets.QPushButton("⬅️ 撤销 (回到上一步)")
        btn_reset_all = QtWidgets.QPushButton("⏮️ 回到初始图像")
        
        btn_load.clicked.connect(self.load_image)
        btn_save.clicked.connect(self.save_image)
        btn_pipeline.clicked.connect(self.set_as_original)
        btn_undo.clicked.connect(self.undo_history)
        btn_reset_all.clicked.connect(self.reset_to_first)
        
        for btn in[btn_load, btn_save, btn_pipeline, btn_undo, btn_reset_all]:
            btn_layout.addWidget(btn)
        left_layout.addLayout(btn_layout)
        
        # ---------------- 视图控制矩阵 ----------------
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QtWidgets.QLabel("视图控制:"))
        
        self.btn_sync_toggle = QtWidgets.QPushButton("🔗 同步操控: 开启")
        self.btn_sync_toggle.setStyleSheet("color: #2E8B57; font-weight: bold;")
        
        btn_zoom_in = QtWidgets.QPushButton("🔍 放大")
        btn_zoom_out = QtWidgets.QPushButton("🔎 缩小")
        btn_zoom_fit = QtWidgets.QPushButton("🖥️ 适应窗口")
        btn_zoom_11 = QtWidgets.QPushButton("🖼️ 图像原始大小")
        
        self.btn_sync_toggle.clicked.connect(self.toggle_sync_mode)
        btn_zoom_in.clicked.connect(lambda: self.zoom_views(1.2))
        btn_zoom_out.clicked.connect(lambda: self.zoom_views(1 / 1.2))
        btn_zoom_fit.clicked.connect(self.fit_views)
        btn_zoom_11.clicked.connect(self.actual_size_views)
        
        for btn in[self.btn_sync_toggle, btn_zoom_in, btn_zoom_out, btn_zoom_fit, btn_zoom_11]:
            zoom_layout.addWidget(btn)
        zoom_layout.addStretch()
        left_layout.addLayout(zoom_layout)
        
        # ---------------- 图像渲染视窗 ----------------
        display_layout = QHBoxLayout()
        grp_orig = QGroupBox("当前原图 (基准)")
        lay_orig = QVBoxLayout(); self.view_orig = SyncGraphicsView(); lay_orig.addWidget(self.view_orig); grp_orig.setLayout(lay_orig)
        
        grp_proc = QGroupBox("处理结果")
        lay_proc = QVBoxLayout(); self.view_proc = SyncGraphicsView(); lay_proc.addWidget(self.view_proc); grp_proc.setLayout(lay_proc)
        
        # 通过设置 stretch=1，强制左右两个 GroupBox 平均分配显示区域，彻底杜绝尺寸挤占现象
        display_layout.addWidget(grp_orig, 1)
        display_layout.addWidget(grp_proc, 1)
        left_layout.addLayout(display_layout)
        
        self.view_orig.viewClicked.connect(self.on_view_clicked)
        self.view_proc.viewClicked.connect(self.on_view_clicked)
        
        self.view_orig.horizontalScrollBar().valueChanged.connect(self.on_hscroll_orig)
        self.view_proc.horizontalScrollBar().valueChanged.connect(self.on_hscroll_proc)
        self.view_orig.verticalScrollBar().valueChanged.connect(self.on_vscroll_orig)
        self.view_proc.verticalScrollBar().valueChanged.connect(self.on_vscroll_proc)
        
        btn_compare_hist = QtWidgets.QPushButton("📊 显示直方图面板")
        btn_compare_hist.clicked.connect(self.on_compare_hist)
        left_layout.addWidget(btn_compare_hist)
        
        # ---------------- 算法算子配置区 ----------------
        right_layout = QVBoxLayout()
        tabs = QTabWidget(); tabs.setFixedWidth(360)
        
        def create_reset_btn(callback):
            btn = QtWidgets.QPushButton("🔄️ 恢复本页默认参数")
            btn.clicked.connect(callback)
            return btn

        # [模块 1: 直方图与亮度映射]
        tab1 = QtWidgets.QWidget(); lay1 = QVBoxLayout(tab1)
        lay1.addWidget(QtWidgets.QLabel("<b>直方图处理</b>"))
        btn_eq = QtWidgets.QPushButton("全局直方图均衡化"); btn_clahe = QtWidgets.QPushButton("CLAHE 局部均衡化")
        btn_eq.clicked.connect(lambda: self.apply_filter(histogram_equalize))
        btn_clahe.clicked.connect(lambda: self.apply_filter(lambda x: clahe_enhance(x, 2.0)))
        lay1.addWidget(btn_eq); lay1.addWidget(btn_clahe)
        
        lay1.addWidget(QtWidgets.QLabel("<hr><b>线性对比度增强</b>"))
        self.spin_alpha = QtWidgets.QDoubleSpinBox(); self.spin_alpha.setRange(0.1, 5.0); self.spin_alpha.setValue(1.2); self.spin_alpha.setSingleStep(0.1)
        self.spin_beta = QtWidgets.QSpinBox(); self.spin_beta.setRange(-100, 100); self.spin_beta.setValue(10)
        lay1.addWidget(QtWidgets.QLabel("对比度增益 (Alpha):")); lay1.addWidget(self.spin_alpha)
        lay1.addWidget(QtWidgets.QLabel("亮度偏移 (Beta):")); lay1.addWidget(self.spin_beta)
        btn_contrast = QtWidgets.QPushButton("应用线性增强")
        btn_contrast.clicked.connect(lambda: self.apply_filter(lambda x: cv2.convertScaleAbs(x, alpha=self.spin_alpha.value(), beta=self.spin_beta.value())))
        lay1.addWidget(btn_contrast)
        
        lay1.addWidget(QtWidgets.QLabel("<hr><b>非线性变换</b>"))
        self.spin_gamma = QtWidgets.QDoubleSpinBox(); self.spin_gamma.setRange(0.1, 5.0); self.spin_gamma.setValue(0.5); self.spin_gamma.setSingleStep(0.1)
        lay1.addWidget(QtWidgets.QLabel("Gamma 值 (<1: 整体提亮, >1: 整体压暗):")); lay1.addWidget(self.spin_gamma)
        btn_gamma = QtWidgets.QPushButton("应用 Gamma 校正")
        btn_gamma.clicked.connect(lambda: self.apply_filter(lambda x: gamma_correction(x, self.spin_gamma.value())))
        lay1.addWidget(btn_gamma)
        lay1.addStretch()
        def reset_t1(): self.spin_alpha.setValue(1.2); self.spin_beta.setValue(10); self.spin_gamma.setValue(0.5)
        lay1.addWidget(create_reset_btn(reset_t1))
        tabs.addTab(tab1, "亮度增强")
        
        #[模块 2: 空间几何变换]
        tab2 = QtWidgets.QWidget(); lay2 = QVBoxLayout(tab2)
        self.chk_flip_h = QtWidgets.QCheckBox("水平镜像 (左右翻转)")
        self.chk_flip_v = QtWidgets.QCheckBox("垂直镜像 (上下翻转)")
        lay2.addWidget(self.chk_flip_h); lay2.addWidget(self.chk_flip_v)
        
        lay2.addWidget(QtWidgets.QLabel("<hr><b>仿射变换</b>"))
        self.spin_rot = QtWidgets.QSpinBox(); self.spin_rot.setRange(-360, 360); self.spin_rot.setValue(90)
        self.spin_scale = QtWidgets.QDoubleSpinBox(); self.spin_scale.setRange(0.1, 5.0); self.spin_scale.setValue(1.0); self.spin_scale.setSingleStep(0.1)
        self.spin_tx = QtWidgets.QSpinBox(); self.spin_tx.setRange(-1000, 1000); self.spin_tx.setValue(0)
        self.spin_ty = QtWidgets.QSpinBox(); self.spin_ty.setRange(-1000, 1000); self.spin_ty.setValue(0)
        lay2.addWidget(QtWidgets.QLabel("旋转角度 (度):")); lay2.addWidget(self.spin_rot)
        lay2.addWidget(QtWidgets.QLabel("缩放比例:")); lay2.addWidget(self.spin_scale)
        lay2.addWidget(QtWidgets.QLabel("平移 X (像素):")); lay2.addWidget(self.spin_tx)
        lay2.addWidget(QtWidgets.QLabel("平移 Y (像素):")); lay2.addWidget(self.spin_ty)
        
        btn_geo = QtWidgets.QPushButton("应用几何变换")
        btn_geo.clicked.connect(self.on_geo)
        lay2.addWidget(btn_geo)
        lay2.addStretch()
        def reset_t2(): self.spin_rot.setValue(90); self.spin_scale.setValue(1.0); self.spin_tx.setValue(0); self.spin_ty.setValue(0); self.chk_flip_h.setChecked(False); self.chk_flip_v.setChecked(False)
        lay2.addWidget(create_reset_btn(reset_t2))
        tabs.addTab(tab2, "几何变换")
        
        # [模块 3: 图像复原与空域滤波]
        tab3 = QtWidgets.QWidget(); lay3 = QVBoxLayout(tab3)
        self.combo_noise = QtWidgets.QComboBox()
        self.combo_noise.addItems(['高斯噪声 (gaussian)', '椒盐噪声 (salt_pepper)', '斑点噪声 (speckle)', '泊松噪声 (poisson)'])
        self.lbl_noise_param = QtWidgets.QLabel("噪声参数:")
        self.spin_var = QtWidgets.QDoubleSpinBox(); self.spin_var.setRange(0, 1000); self.spin_var.setValue(100); self.spin_var.setSingleStep(10)
        
        self.combo_noise.currentIndexChanged.connect(self.on_noise_type_changed)
        lay3.addWidget(QtWidgets.QLabel("噪声类型:")); lay3.addWidget(self.combo_noise)
        lay3.addWidget(self.lbl_noise_param); lay3.addWidget(self.spin_var)
        btn_noise = QtWidgets.QPushButton("添加噪声")
        btn_noise.clicked.connect(self.on_noise)
        lay3.addWidget(btn_noise)
        self.on_noise_type_changed(0) 
        
        lay3.addWidget(QtWidgets.QLabel("<hr><b>空间域滤波 (平滑/去噪)</b>"))
        self.combo_spat = QtWidgets.QComboBox(); self.combo_spat.addItems(['均值滤波 (average)', '高斯滤波 (gaussian)', '中值滤波 (median)'])
        self.spin_ks = QtWidgets.QSpinBox(); self.spin_ks.setRange(1, 31); self.spin_ks.setSingleStep(2); self.spin_ks.setValue(3)
        lay3.addWidget(QtWidgets.QLabel("滤波器类型:")); lay3.addWidget(self.combo_spat)
        lay3.addWidget(QtWidgets.QLabel("核大小 (奇数):")); lay3.addWidget(self.spin_ks)
        btn_spat = QtWidgets.QPushButton("应用空域滤波")
        btn_spat.clicked.connect(self.on_spat)
        lay3.addWidget(btn_spat)
        lay3.addStretch()
        def reset_t3(): self.combo_noise.setCurrentIndex(0); self.spin_var.setValue(100); self.spin_ks.setValue(3)
        lay3.addWidget(create_reset_btn(reset_t3))
        tabs.addTab(tab3, "加噪与空域")
        
        #[模块 4: 频率域增强]
        tab4 = QtWidgets.QWidget(); lay4 = QVBoxLayout(tab4)
        self.combo_freq_mode = QtWidgets.QComboBox(); self.combo_freq_mode.addItems(['低通滤波 (lowpass)', '高通滤波 (highpass)'])
        self.combo_freq_type = QtWidgets.QComboBox(); self.combo_freq_type.addItems(['理想滤波 (ideal)', '高斯滤波 (gaussian)', '巴特沃斯滤波 (butterworth)'])
        self.spin_cut = QtWidgets.QSpinBox(); self.spin_cut.setRange(1, 1000); self.spin_cut.setValue(30)
        self.lbl_freq_order = QtWidgets.QLabel("阶数:")
        self.spin_order = QtWidgets.QSpinBox(); self.spin_order.setRange(1, 10); self.spin_order.setValue(2)
        
        self.combo_freq_type.currentIndexChanged.connect(self.on_freq_type_changed)
        lay4.addWidget(QtWidgets.QLabel("滤波模式:")); lay4.addWidget(self.combo_freq_mode)
        lay4.addWidget(QtWidgets.QLabel("滤波器形状:")); lay4.addWidget(self.combo_freq_type)
        lay4.addWidget(QtWidgets.QLabel("截止频率 D0:")); lay4.addWidget(self.spin_cut)
        lay4.addWidget(self.lbl_freq_order); lay4.addWidget(self.spin_order)
        btn_freq = QtWidgets.QPushButton("应用频域滤波")
        btn_freq.clicked.connect(self.on_freq)
        lay4.addWidget(btn_freq)
        self.on_freq_type_changed(0) 
        
        lay4.addStretch()
        def reset_t4(): self.spin_cut.setValue(30); self.spin_order.setValue(2); self.combo_freq_mode.setCurrentIndex(0); self.combo_freq_type.setCurrentIndex(0)
        lay4.addWidget(create_reset_btn(reset_t4))
        tabs.addTab(tab4, "频率域")
        
        right_layout.addWidget(tabs)
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

    # ==================== 视图架构与事件分发引擎 ====================

    def toggle_sync_mode(self):
        """控制同步模式标志位及按钮 CSS 反馈状态"""
        self.sync_mode = not self.sync_mode
        if self.sync_mode:
            self.btn_sync_toggle.setText("🔗 同步操控: 开启")
            self.btn_sync_toggle.setStyleSheet("color: #2E8B57; font-weight: bold;")
            self.view_orig.setStyleSheet("border: 2px solid transparent; background-color: #2e2e2e;")
            self.view_proc.setStyleSheet("border: 2px solid transparent; background-color: #2e2e2e;")
            
            # 进入同步模式时，强制将另一侧视口位置映射至当前焦点视口
            target_view = self.view_proc if self.active_view == self.view_orig else self.view_orig
            target_view.setTransform(self.active_view.transform())
            target_view.horizontalScrollBar().setValue(self.active_view.horizontalScrollBar().value())
            target_view.verticalScrollBar().setValue(self.active_view.verticalScrollBar().value())
        else:
            self.btn_sync_toggle.setText("🔓 同步操控: 关闭")
            self.btn_sync_toggle.setStyleSheet("color: #D2691E; font-weight: bold;")
            self.update_active_view_style()

    def on_view_clicked(self, view):
        """鼠标左键焦点拦截机制"""
        if not self.sync_mode:
            self.active_view = view
            self.update_active_view_style()

    def update_active_view_style(self):
        """独立操控模式下的高亮绘制器"""
        if self.sync_mode: return
        style_active = "border: 2px solid #00AAFF; background-color: #2e2e2e;"
        style_inactive = "border: 2px solid transparent; background-color: #2e2e2e;"
        if self.active_view == self.view_orig:
            self.view_orig.setStyleSheet(style_active)
            self.view_proc.setStyleSheet(style_inactive)
        else:
            self.view_orig.setStyleSheet(style_inactive)
            self.view_proc.setStyleSheet(style_active)

    def on_hscroll_orig(self, val):
        if self.sync_mode and not self.is_syncing_scroll:
            self.is_syncing_scroll = True
            self.view_proc.horizontalScrollBar().setValue(val)
            self.is_syncing_scroll = False

    def on_hscroll_proc(self, val):
        if self.sync_mode and not self.is_syncing_scroll:
            self.is_syncing_scroll = True
            self.view_orig.horizontalScrollBar().setValue(val)
            self.is_syncing_scroll = False

    def on_vscroll_orig(self, val):
        if self.sync_mode and not self.is_syncing_scroll:
            self.is_syncing_scroll = True
            self.view_proc.verticalScrollBar().setValue(val)
            self.is_syncing_scroll = False

    def on_vscroll_proc(self, val):
        if self.sync_mode and not self.is_syncing_scroll:
            self.is_syncing_scroll = True
            self.view_orig.verticalScrollBar().setValue(val)
            self.is_syncing_scroll = False

    def zoom_views(self, factor):
        """缩放引擎。根据模式标识实施双路或单路派发。"""
        if self.sync_mode:
            self.view_orig.scale(factor, factor)
            self.view_proc.scale(factor, factor)
        else:
            self.active_view.scale(factor, factor)

    def fit_views(self):
        """全图适配引擎。调用底层 SceneRect 进行边界计算。"""
        if self.sync_mode:
            if self.img_base is not None: self.view_orig.fitInView(self.view_orig.sceneRect(), QtCore.Qt.KeepAspectRatio)
            if self.img_proc is not None: self.view_proc.fitInView(self.view_proc.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            if self.active_view == self.view_orig and self.img_base is not None:
                self.view_orig.fitInView(self.view_orig.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif self.active_view == self.view_proc and self.img_proc is not None:
                self.view_proc.fitInView(self.view_proc.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def actual_size_views(self):
        """原始比率引擎。重置 QGraphicsView 的内部仿射矩阵至恒等映射。"""
        if self.sync_mode:
            self.view_orig.resetTransform()
            self.view_proc.resetTransform()
        else:
            self.active_view.resetTransform()

    # ==================== 接口与参数通信 ====================

    def on_noise_type_changed(self, index):
        text = self.combo_noise.currentText()
        if "gaussian" in text:
            self.lbl_noise_param.setText("高斯方差 Var (数值越大噪声越强):")
            self.spin_var.setEnabled(True)
        elif "speckle" in text:
            self.lbl_noise_param.setText("斑点方差 Var (乘性噪声强度):")
            self.spin_var.setEnabled(True)
        elif "salt_pepper" in text:
            self.lbl_noise_param.setText("椒盐污染概率 (百分比 %):")
            self.spin_var.setEnabled(True)
        elif "poisson" in text:
            self.lbl_noise_param.setText("泊松噪声 (基于数据分布分布，无需参数调节)")
            self.spin_var.setEnabled(False)

    def on_freq_type_changed(self, index):
        text = self.combo_freq_type.currentText()
        if "butterworth" in text:
            self.lbl_freq_order.setText("阶数 (仅限巴特沃斯滤波器):")
            self.spin_order.setEnabled(True)
        else:
            self.lbl_freq_order.setText("阶数 (当前滤波器无需此参数):")
            self.spin_order.setEnabled(False)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None: return
            if img.shape[-1] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.img_first_loaded = img.copy()
            self.history.clear()
            self.img_base = img.copy()
            self.img_proc = None
            self.update_views()
            QtCore.QTimer.singleShot(100, self.fit_views)

    def save_image(self):
        if self.img_proc is None:
            QMessageBox.information(self, "提示", "目前没有处理结果可保存！")
            return
            
        dialog = SaveConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            dpi_val = dialog.spin_dpi.value()
            quality_val = dialog.spin_quality.value()
            
            path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "JPEG Files (*.jpg);;PNG Files (*.png);;TIFF Files (*.tif)")
            if path:
                img_rgb = cv2.cvtColor(self.img_proc, cv2.COLOR_BGR2RGB) if self.img_proc.ndim == 3 else self.img_proc
                pil_img = Image.fromarray(img_rgb)
                try:
                    pil_img.save(path, quality=quality_val, dpi=(dpi_val, dpi_val))
                    QMessageBox.information(self, "成功", f"图像已成功保存！\n分辨率: {dpi_val} DPI")
                except Exception as e:
                    QMessageBox.warning(self, "保存失败", str(e))

    def set_as_original(self):
        if self.img_proc is not None:
            self.history.append(self.img_base.copy())
            self.img_base = self.img_proc.copy()
            self.img_proc = None
            self.update_views()

    def undo_history(self):
        if len(self.history) > 0:
            self.img_base = self.history.pop()
            self.img_proc = None
            self.update_views()
        else:
            QMessageBox.information(self, "提示", "已到达历史记录尽头，无法继续撤销。")

    def reset_to_first(self):
        if self.img_first_loaded is not None:
            self.history.clear()
            self.img_base = self.img_first_loaded.copy()
            self.img_proc = None
            self.update_views()
            self.fit_views()

    def apply_filter(self, func):
        if self.img_base is None:
            QMessageBox.information(self, "提示", "请先加载图像！")
            return
        self.img_proc = func(self.img_base)
        self.update_views()

    def update_views(self):
        if self.img_base is not None:
            self.view_orig.set_image(QtGui.QPixmap.fromImage(cv2_to_qimage(self.img_base)))
        if self.img_proc is not None:
            self.view_proc.set_image(QtGui.QPixmap.fromImage(cv2_to_qimage(self.img_proc)))
        else:
            self.view_proc.scene.clear()
            self.view_proc.pixmap_item = QtWidgets.QGraphicsPixmapItem()
            self.view_proc.scene.addItem(self.view_proc.pixmap_item)

    def on_compare_hist(self):
        if self.img_base is None: return
        proc = self.img_proc if self.img_proc is not None else self.img_base
        show_histograms_comparision(self.img_base, proc)

    def on_geo(self):
        def _geo(img):
            if self.chk_flip_h.isChecked(): img = cv2.flip(img, 1)
            if self.chk_flip_v.isChecked(): img = cv2.flip(img, 0)
            
            deg, sc = self.spin_rot.value(), self.spin_scale.value()
            tx, ty = self.spin_tx.value(), self.spin_ty.value()
            
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, sc)
            
            cos_a = np.abs(M[0, 0])
            sin_a = np.abs(M[0, 1])
            new_w = int((h * sin_a) + (w * cos_a))
            new_h = int((h * cos_a) + (w * sin_a))
            
            M[0, 2] += (new_w / 2.0) - (w / 2.0) + tx
            M[1, 2] += (new_h / 2.0) - (h / 2.0) + ty
            
            return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        self.apply_filter(_geo)

    def on_noise(self):
        text = self.combo_noise.currentText()
        mode = text.split('(')[-1].strip(')')
        var = self.spin_var.value()
        amount = var / 100.0 if "salt_pepper" in mode else var
        self.apply_filter(lambda x: add_noise(x, mode=mode, var=var, amount=amount))

    def on_spat(self):
        text = self.combo_spat.currentText()
        k = self.spin_ks.value()
        if k % 2 == 0: k += 1
        def _spat(img):
            if "average" in text: return cv2.blur(img, (k, k))
            elif "gaussian" in text: return cv2.GaussianBlur(img, (k, k), 0)
            elif "median" in text: return cv2.medianBlur(img, k)
            return img
        self.apply_filter(_spat)

    def on_freq(self):
        mode_str = self.combo_freq_mode.currentText()
        mode = 'lowpass' if 'lowpass' in mode_str else 'highpass'
        ftype = self.combo_freq_type.currentText().split('(')[-1].strip(')')
        self.apply_filter(lambda x: frequency_filter(x, mode=mode, ftype=ftype, cutoff=self.spin_cut.value(), order=self.spin_order.value()))

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    
    font = QtGui.QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    win = FusedImageTool()
    win.show()
    sys.exit(app.exec_())