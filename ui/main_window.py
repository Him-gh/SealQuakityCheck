import sys
import os
import datetime
import cv2
import json
import queue
# 将项目根目录加入sys.path，以便导入SQC等模块
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
CONF_PATH="./config/config.json"

from SQC import SQC
from utils.LoadImage import ImageLoader

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox, QSplitter, QDialog,
    QScrollArea, QGroupBox, QFormLayout, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from .edit_config import ConfigEditorDialog 

class ImageLabel(QLabel):
    """自定义支持随着窗口自动缩放并保持宽高比的QLabel"""
    def __init__(self, title=""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.setMinimumSize(100, 100)
        self._pixmap = None
        self.setText(title)

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self.update_pixmap()

    def update_pixmap(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled_pixmap = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sqc = None
        self.image_loader = None
        
        self.sed_dir = ""
        self.ir_dir = ""
        
        # 确保log目录存在
        self.log_dir = os.path.join(root_dir, "log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = os.path.join(self.log_dir, "log.txt")

        self.setWindowTitle("Seal Quality Check (SQC)")
        self.resize(1000, 700)
        self.init_ui()

    def closeEvent(self, event):
        if self.image_loader:
            self.image_loader.stop()
        super().closeEvent(event)

    def init_ui(self):
        # 主窗口的Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ====================
        # 1. 左侧：控制面板
        # ====================
        control_panel = QWidget()
        control_panel.setFixedWidth(200)
        control_layout = QVBoxLayout(control_panel)

        self.btn_path = QPushButton("文件路径")
        self.btn_path.clicked.connect(self.select_paths)

        self.btn_load = QPushButton("加载图片")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_check = QPushButton("图片检测")
        self.btn_check.clicked.connect(self.image_check)

        self.btn_info = QPushButton("详细信息")
        self.btn_info.clicked.connect(self.show_details)

        self.btn_criteria = QPushButton("评判标准")
        self.btn_criteria.clicked.connect(self.show_criteria)

        self.btn_edit_config = QPushButton("修改配置")
        self.btn_edit_config.clicked.connect(self.edit_config)

        self.bnt_show_log=QPushButton("查看日志")
        self.bnt_show_log.clicked.connect(self.show_log)

        # 添加按钮到左侧布局，并允许扩展（便于后期添加新功能）
        control_layout.addWidget(self.btn_path)
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_check)
        control_layout.addWidget(self.btn_info)
        control_layout.addWidget(self.btn_criteria)
        control_layout.addWidget(self.btn_edit_config)
        control_layout.addWidget(self.bnt_show_log)
        control_layout.addStretch(1)  # 占位符，将按钮推向顶端

        # ====================
        # 2 & 3. 右侧：图片显示区 和 日志显示区 (使用QSplitter以便上下调整大小)
        # ====================
        right_splitter = QSplitter(Qt.Vertical)

        # (2) 图片显示部分
        image_panel = QWidget()
        image_layout = QHBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        self.sed_image_display = ImageLabel("图片显示部分\n（处理后显示）")
        self.ir_image_display = ImageLabel("图片显示部分\n（处理后显示）")
        
        image_layout.addWidget(self.sed_image_display)
        image_layout.addWidget(self.ir_image_display)
        
        # (3) 日志与处理结果显示部分
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_title = QLabel("日志与处理结果：")
        log_title.setFont(QFont("Arial", 10, QFont.Bold))
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.text_log)

        # 添加到右侧splitter
        right_splitter.addWidget(image_panel)
        right_splitter.addWidget(log_panel)

        # 设置QSplitter的初始比例 (让图片显示区占用尽可能大，例如 7:3 比例)
        right_splitter.setSizes([700, 300])

        # ====================
        # 合并左右两部分
        # ====================
        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_splitter)

    # ====================
    # 功能函数
    # ====================
    def select_paths(self):
        sed_dir = QFileDialog.getExistingDirectory(self, "选择银边检测路径")
        if not sed_dir:
            return
        ir_dir = QFileDialog.getExistingDirectory(self, "选择红外数据路径")
        if not ir_dir:
            return
            
        self.sed_dir = sed_dir
        self.ir_dir = ir_dir
        
        if self.image_loader:
            self.image_loader.stop()
            self.image_loader = None
            
        self.image_loader = ImageLoader(self.sed_dir, self.ir_dir, window_size=0.1)
        self.image_loader.start()
        
        self.append_log(f"已选择路径：\n银边目录：{self.sed_dir}\n红外目录：{self.ir_dir}\n异步加载系统已启动。")

    def load_image(self):
        if not self.image_loader:
            QMessageBox.warning(self, "警告", "请先使用 '文件路径' 按钮选择包含图片的有效路径！")
            return
            
        try:
            matched_pair = self.image_loader.result_queue.get_nowait()
            sed_item, ir_item = matched_pair
            sed_name = sed_item[1]
            ir_name = ir_item[1]
            sed_path = sed_item[2]
            ir_path = ir_item[2]
        except queue.Empty:
            QMessageBox.information(self, "提示", "结果队列目前没有成对的图片可供加载，请稍后再试。")
            return
        
        self.sed_image_display._pixmap = None
        self.sed_image_display.setText(f"银边图片已加载\n{sed_name}\n请执行检测...")
        
        self.ir_image_display._pixmap = None
        self.ir_image_display.setText(f"红外图片已加载\n{ir_name}\n请执行检测...")

        try:
            if self.sqc is None:
                self.sqc = SQC(IR_image_path=ir_path, SED_image_path=sed_path)
            else:
                self.sqc.load_image(IR_image_path=ir_path, SED_image_path=sed_path)
            self.append_log(f"成功加载匹配的一对图片：{sed_name}, {ir_name}")
            
            # Store the current images for logic
            self.current_sed_name = sed_name
            self.current_ir_name = ir_name
        except Exception as e:
            self.append_log(f"加载图片失败: {str(e)}")

    def image_check(self):
        if not self.sqc:
            QMessageBox.warning(self, "警告", "请先加载图片！")
            return
            
        try:
            self.sqc.check_all()
            
            sed_bgr = self.sqc.show_sed_img
            self.display_numpy_image(self.sed_image_display, sed_bgr)
            
            ir_bgr = self.sqc.show_ir_and_ocr_img
            self.display_numpy_image(self.ir_image_display, ir_bgr)
            
            # 两个字典的合并
            # combined_result = {**self.sqc.check_SED_result, **self.sqc.check_IR_and_OCR_result}
            
            status_sed = self.sqc.check_SED_result.get("status_sed", "NG")
            status_ir = self.sqc.check_IR_and_OCR_result.get("status_ir_and_ocr", "NG")
            
            status = "OK" if status_sed == "OK" and status_ir == "OK" else "NG"
            
            sed_name = getattr(self, "current_sed_name", "UnknownSED")
            ir_name = getattr(self, "current_ir_name", "UnknownIR")
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"<{sed_name}><{ir_name}>--<{status}>--<{current_time}>"
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
                
            self.append_log(log_line)
            self.append_log(f"检测结果: {status}")
            self.append_log(f"SED结果: {self.sqc.check_SED_result}")
            self.append_log(f"IR结果: {self.sqc.check_IR_and_OCR_result}")
            self.append_log("-" * 40)
            
        except Exception as e:
            self.append_log(f"检测时出错: {str(e)}")

    def show_details(self):
        if not self.sqc or not self.sqc.SED_result or not self.sqc.IR_and_OCR_result:
            QMessageBox.information(self, "详细信息", "暂无检测数据，请先执行图片检测！")
            return
            
        combined_details = {**self.sqc.SED_result, **self.sqc.IR_and_OCR_result}
        
        dialog = QDialog(self)
        dialog.setWindowTitle("详细信息")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        
        formatted_info = json.dumps(combined_details, indent=4, ensure_ascii=False)
        text_edit.setText(formatted_info)
        
        font = text_edit.font()
        font.setPointSize(11)
        text_edit.setFont(font)
        
        layout.addWidget(text_edit)
        dialog.exec_()

    def show_criteria(self):
        conf_path=os.path.join(root_dir, CONF_PATH)
        assert os.path.exists(conf_path),f"Invalid conf_path: {conf_path}"
        config = {}
        with open(conf_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.config=config
        self.check_info=config.get("check",{})

        if self.check_info=={}:
            QMessageBox.information(self, "评判标准", "暂无评判标准")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("评判标准")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        
        formatted_info =self.format_check_info(self.check_info)
        # formatted_info = json.dumps(self.check_info, indent=4, ensure_ascii=False)
        text_edit.setText(formatted_info)
        
        font = text_edit.font()
        font.setPointSize(11)
        text_edit.setFont(font)
        
        layout.addWidget(text_edit)
        dialog.exec_()

    def edit_config(self):
        conf_path = os.path.join(root_dir, CONF_PATH)
        if not os.path.exists(conf_path):
            QMessageBox.warning(self, "错误", f"找不到配置文件: {conf_path}")
            return
            
        dialog = ConfigEditorDialog(conf_path, self)
        dialog.exec_()
        
    def show_log(self):
        assert os.path.exists(self.log_file),"无日志"
            
        dialog = QDialog(self)  
        dialog.setWindowTitle("日志查看器")
        dialog.resize(800, 600)
    
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
    
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            text_edit.setPlainText(content)
    
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)  
    
        layout.addWidget(text_edit)
        layout.addWidget(close_btn)
    
        dialog.setLayout(layout)
        dialog.exec_()  

    # ====================
    # 辅助函数
    # ====================
    def append_log(self, text):
        self.text_log.append(text)
        # 滚动到底部
        self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())

    def display_numpy_image(self, target_display, img_bgr):
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        target_display.set_image(pixmap)

    def format_check_info(self,info):
        lines = []
        # Num
        lines.append(f'Num = {info.get("Num", "N/A")}')
        # W: 范围
        w = info.get("W")
        if w and len(w) == 2:
            lines.append(f'{w[0]} <= W <= {w[1]}')
        # H: 范围
        h = info.get("H")
        if h and len(h) == 2:
            lines.append(f'{h[0]} <= H <= {h[1]}')
        # Rect_Ratio
        lines.append(f'Rect_Ratio >= {info.get("Rect_Ratio", "N/A")}')
        # Area: 范围
        area = info.get("Area")
        if area and len(area) == 2:
            lines.append(f'{area[0]} <= Area <= {area[1]}')
        # Angle: 阈值
        lines.append(f'Angle <= {info.get("Angle", "N/A")}')
        # Smoothness
        lines.append(f'Smoothness >= {info.get("Smoothness", "N/A")}')
        # Uniformity
        lines.append(f'Uniformity >= {info.get("Uniformity", "N/A")}')
        # High_Temperature: 范围
        ht = info.get("High_Temperature")
        if ht and len(ht) == 2:
            lines.append(f'{ht[0]} <= High_Temperature <= {ht[1]}')
        # Low_Temperature: 范围
        lt = info.get("Low_Temperature")
        if lt and len(lt) == 2:
            lines.append(f'{lt[0]} <= Low_Temperature <= {lt[1]}')
        return "\n".join(lines)



