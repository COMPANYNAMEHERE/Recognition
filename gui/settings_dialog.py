# src/gui/settings_dialog.py
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                           QLabel, QSpinBox, QComboBox, QCheckBox)
from utils.config import save_config

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 100)
        self.conf_spin.setValue(int(self.config.get('confidence_threshold', 50)))
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)
        
        # GPU Usage
        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setChecked(self.config.get('use_gpu', True))
        layout.addWidget(self.gpu_check)
        
        # Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'])
        self.model_combo.setCurrentText(self.config.get('model', 'yolov8n'))
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Save location
        save_layout = QHBoxLayout()
        self.save_path = QLabel(self.config.get('save_path', 'Default'))
        save_layout.addWidget(self.save_path)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_save_location)
        save_layout.addWidget(self.browse_button)
        layout.addLayout(save_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def browse_save_location(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Location")
        if directory:
            self.save_path.setText(directory)
            
    def save_settings(self):
        self.config.update({
            'confidence_threshold': self.conf_spin.value() / 100,
            'use_gpu': self.gpu_check.isChecked(),
            'model': self.model_combo.currentText(),
            'save_path': self.save_path.text()
        })
        save_config(self.config)
        self.accept()
