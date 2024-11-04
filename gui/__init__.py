# src/gui/__init__.py
"""
Video Recognition GUI Package

This package contains all the GUI components for the video recognition application.
"""

from .main_window import MainWindow
from .video_widget import VideoWidget

__all__ = [
    'MainWindow',
    'VideoWidget'
]

# Package version
__version__ = '1.0.0'

# Package metadata
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__status__ = 'Development'
