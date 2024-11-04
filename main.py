# src/main.py
import sys
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from gui.main_window import MainWindow
from utils.config import load_config
from pathlib import Path
import os

def setup_logging():
    """Configure application-wide logging with rotation and error tracking"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger for the application
    logger = logging.getLogger('VideoRecognition')
    
    # Add a special handler for errors
    error_handler = logging.FileHandler(log_dir / 'errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(error_handler)
    
    return logger

def show_error_dialog(error_msg: str):
    """Show error dialog with detailed information"""
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Icon.Critical)
    error_box.setWindowTitle("Application Error")
    error_box.setText("A critical error has occurred.")
    error_box.setInformativeText(error_msg)
    error_box.setDetailedText(traceback.format_exc())
    error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    error_box.exec()

def exception_hook(exctype, value, tb):
    """Enhanced global exception handler with user notification"""
    logger = logging.getLogger('VideoRecognition')
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logger.error(f"Uncaught exception:\n{error_msg}")
    
    # Ensure we're in the main thread when showing dialog
    if QApplication.instance() is not None:
        show_error_dialog(str(value))
    
    # Call the default handler
    sys.__excepthook__(exctype, value, tb)

def check_environment():
    """Check and verify required environment setup"""
    logger = logging.getLogger('VideoRecognition')
    
    # Check for required directories
    required_dirs = ['models', 'logs', 'cache', 'output']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.info(f"Creating required directory: {dir_name}")
            dir_path.mkdir(exist_ok=True)
    
    # Check for GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = torch.cuda.get_device_properties(0)
            logger.info(f"GPU detected: {gpu_info.name} ({gpu_info.total_memory / 1024**2:.0f}MB)")
        else:
            logger.info("No GPU detected, running in CPU mode")
    except Exception as e:
        logger.warning(f"Error checking GPU status: {e}")

def setup_application_info():
    """Set up application-wide information"""
    QApplication.setApplicationName("Video Recognition")
    QApplication.setApplicationVersion("1.0.0")
    QApplication.setOrganizationName("YourOrganization")
    QApplication.setOrganizationDomain("yourorganization.com")

def setup_high_dpi():
    """Configure High DPI settings for Qt6"""
    try:
        # Enable High DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        
        if hasattr(Qt.ApplicationAttribute, 'AA_Use96Dpi'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi)
        
        if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        else:
            os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
            
        if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    except Exception as e:
        logger = logging.getLogger('VideoRecognition')
        logger.warning(f"Error setting up high DPI: {e}")

def main():
    """
    Main application entry point with enhanced error handling and initialization
    """
    try:
        # Initialize logging first
        logger = setup_logging()
        logger.info("Starting Video Recognition Application")
        
        # Set up global exception handling
        sys.excepthook = exception_hook
        
        # Check environment and create required directories
        check_environment()
        
        # Create application instance first
        app = QApplication(sys.argv)
        
        # Set up high DPI support
        setup_high_dpi()
        
        # Set up application information
        setup_application_info()
        
        # Load configuration
        try:
            config = load_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            show_error_dialog(f"Error loading configuration: {str(e)}")
            return 1
        
        # Create and show main window with progress handling
        try:
            window = MainWindow(config)
            window.show()
            
            # Log successful initialization
            logger.info("Application GUI initialized successfully")
            
            # Start application event loop
            exit_code = app.exec()
            
            # Log application exit
            logger.info(f"Application exiting with code {exit_code}")
            return exit_code
            
        except Exception as e:
            logger.error("Error initializing main window", exc_info=True)
            show_error_dialog(f"Error initializing application: {str(e)}")
            return 1
        
    except Exception as e:
        # Handle any startup errors
        if 'logger' in locals():
            logger.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        else:
            print(f"Fatal error before logging was initialized: {str(e)}")
            traceback.print_exc()
        
        # Try to show error dialog if QApplication exists
        if QApplication.instance() is not None:
            show_error_dialog(f"Fatal application error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())