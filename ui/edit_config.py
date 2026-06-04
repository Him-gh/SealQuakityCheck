import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QDialog,
    QScrollArea, QGroupBox, QFormLayout, QLineEdit
)



class ConfigEditorDialog(QDialog):
    """配置修改弹窗，通过表单形式修改参数"""
    def __init__(self, conf_path, parent=None):
        super().__init__(parent)
        self.conf_path = conf_path
        self.setWindowTitle("修改配置")
        self.resize(500, 600)
        self.edits = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 使用 ScrollArea 防止配置过多显示不下
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        try:
            with open(self.conf_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置文件失败: {str(e)}")
            self.config_data = {}

        # 动态生成配置表单
        for category, params in self.config_data.items():
            if isinstance(params, dict):
                group_box = QGroupBox(category)
                form_layout = QFormLayout(group_box)
                self.edits[category] = {}
                
                for key, value in params.items():
                    line_edit = QLineEdit()
                    # 使用 json.dumps 将列表、数字等转成对应的文本，方便展示（如字符串带引号，列表用方括号）
                    line_edit.setText(json.dumps(value, ensure_ascii=False))
                    form_layout.addRow(QLabel(key), line_edit)
                    self.edits[category][key] = line_edit
                    
                scroll_layout.addWidget(group_box)
            else:
                # 兼容外层非字典的变量
                if "Root" not in self.edits:
                    self.edits["Root"] = {}
                    self.root_group = QGroupBox("基础配置")
                    self.root_form = QFormLayout(self.root_group)
                    scroll_layout.addWidget(self.root_group)
                
                line_edit = QLineEdit()
                line_edit.setText(json.dumps(params, ensure_ascii=False))
                self.root_form.addRow(QLabel(category), line_edit)
                self.edits["Root"][category] = line_edit

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # 按钮区
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存")
        btn_cancel = QPushButton("取消")
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        btn_save.clicked.connect(self.save_config)
        btn_cancel.clicked.connect(self.reject)

    def save_config(self):
        new_config = {}
        for category, params_edits in self.edits.items():
            if category == "Root":
                for key, line_edit in params_edits.items():
                    try:
                        new_config[key] = json.loads(line_edit.text())
                    except json.JSONDecodeError:
                        new_config[key] = line_edit.text()
            else:
                new_config[category] = {}
                for key, line_edit in params_edits.items():
                    val_str = line_edit.text()
                    try:
                        # 还原为原类型（列表、整数、浮点数、字符串等）
                        val = json.loads(val_str)
                        new_config[category][key] = val
                    except json.JSONDecodeError:
                        QMessageBox.warning(self, "格式错误", f"参数 '{category}->{key}' 格式无效！\n如果需要填字符串请注意外层加双引号。\n您输入的是: {val_str}")
                        return

        try:
            with open(self.conf_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "成功", "配置已保存！\n（注意：部分依赖该配置的组件可能需要重新加载才能生效）")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")