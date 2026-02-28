"""
Enhanced configuration - intelligent config management for the Scene Motion Planner
Supports environment detection, auto-optimization, and multi-environment configuration
"""

import os
import json
import platform
try:
    import psutil  # optional dependency
except Exception:
    psutil = None
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
try:
    import yaml  # optional dependency
except Exception:
    yaml = None


class EnhancedConfig:
    """Enhanced configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration"""
        self.config_file = config_file
        self.environment = self._detect_environment()
        self.system_info = self._get_system_info()
        self.load_config()
        self.optimize_for_environment()
    
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        if os.getenv("DOCKER_CONTAINER"):
            return "docker"
        elif os.getenv("CI"):
            return "ci"
        elif os.getenv("VIRTUAL_ENV"):
            return "development"
        else:
            return "production"
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information (with fallback when optional deps are missing)"""
        info: Dict[str, Any] = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }
        # CPU cores
        try:
            info["cpu_count"] = (psutil.cpu_count() if psutil else (os.cpu_count() or 1))
        except Exception:
            info["cpu_count"] = os.cpu_count() or 1
        # Memory GB
        try:
            if psutil:
                info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
            else:
                info["memory_gb"] = 8.0
        except Exception:
            info["memory_gb"] = 8.0
        # Disk free GB
        try:
            import shutil
            info["disk_free_gb"] = round(shutil.disk_usage('/').free / (1024**3), 2)
        except Exception:
            info["disk_free_gb"] = 10.0
        return info
    
    def load_config(self):
        """Load configuration values"""
        
        # basic
        self._load_basic_config()
        
        # model
        self._load_model_config()
        
        # output
        self._load_output_config()
        
        # performance
        self._load_performance_config()
        
        # quality
        self._load_quality_config()
        
        # file management
        self._load_file_config()
        
        # logging
        self._load_log_config()
        
        # API
        self._load_api_config()
        
        # UI
        self._load_ui_config()
        
        # advanced features
        self._load_advanced_config()
        
        # create required directories
        self._create_directories()
    
    def _load_basic_config(self):
        """Load basic configuration"""
        
        self.QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-4046430e513f44c68beec5635a02d97f")
        self.QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-vl-max")
        self.QWEN_API_URL = os.getenv("QWEN_API_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
        self.QWEN_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "30"))
        self.QWEN_MAX_RETRIES = int(os.getenv("QWEN_MAX_RETRIES", "3"))
        
        # project info
        self.PROJECT_NAME = "Scene Motion Planner"
        self.VERSION = "v2.0.0"
        self.DESCRIPTION = "Intelligent system integrating QwenVL scene recognition and MotionLLM motion generation"
    
    def _load_model_config(self):
        """Load model configuration"""
        
        self.MOTION_MODEL_PATH = os.getenv("MOTION_MODEL_PATH", "Motion-Agent/ckpt/motionllm.pth")
        self.LLM_BACKBONE_PATH = os.getenv("LLM_BACKBONE_PATH", "Motion-Agent/gemma-2-2b-it")
        self.MOTION_DEVICE = self._get_optimal_device()
        self.MOTION_WINDOW_SIZE = int(os.getenv("MOTION_WINDOW_SIZE", "196"))
        self.MOTION_DATANAME = os.getenv("MOTION_DATANAME", "t2m")
        
      
        self.MODEL_PRECISION = os.getenv("MODEL_PRECISION", "float16")  # float16/float32
        self.MODEL_COMPILATION = os.getenv("MODEL_COMPILATION", "auto")  # auto/on/off
        self.MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "2048"))  # MB
        
        self.MOTION_AGENT_DIR = os.path.join(os.path.dirname(__file__), "Motion-Agent")
    
    def _load_output_config(self):
        """Load output configuration"""
        # output directories
        self.OUTPUT_BASE_DIR = os.getenv("OUTPUT_BASE_DIR", "./scene_motion_output")
        self.SESSION_DIR = os.getenv("SESSION_DIR", "./saved_sessions")
        self.LOG_DIR = os.getenv("LOG_DIR", "./logs")
        self.TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
        self.CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
        
        # output formats
        self.OUTPUT_FORMATS = ["json", "yaml", "txt", "csv"]
        self.DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_OUTPUT_FORMAT", "json")
        self.INCLUDE_METADATA = os.getenv("INCLUDE_METADATA", "true").lower() == "true"
    
    def _load_performance_config(self):
        """Load performance configuration"""
        # motion generation
        self.DEFAULT_FPS = int(os.getenv("DEFAULT_FPS", "20"))
        self.DEFAULT_RADIUS = int(os.getenv("DEFAULT_RADIUS", "4"))
        self.MAX_MOTION_STEPS = int(os.getenv("MAX_MOTION_STEPS", "15"))
        self.MIN_MOTION_STEPS = int(os.getenv("MIN_MOTION_STEPS", "2"))
        self.DEFAULT_TRANSITION_FRAMES = int(os.getenv("DEFAULT_TRANSITION_FRAMES", "5"))
        
        # parallel processing
        self.MAX_WORKERS = min(int(os.getenv("MAX_WORKERS", "4")), self.system_info["cpu_count"])
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
        self.ENABLE_PARALLEL = os.getenv("ENABLE_PARALLEL", "true").lower() == "true"
        
        # memory management
        self.MAX_MEMORY_USAGE = float(os.getenv("MAX_MEMORY_USAGE", "0.8"))  # 80% of available memory
        self.ENABLE_MEMORY_OPTIMIZATION = os.getenv("ENABLE_MEMORY_OPTIMIZATION", "true").lower() == "true"
    
    def _load_quality_config(self):
        """Load quality configuration"""
        self.QUALITY_THRESHOLDS = {
            "min_frames": int(os.getenv("MIN_FRAMES", "10")),
            "max_frames": int(os.getenv("MAX_FRAMES", "300")),
            "smoothness_threshold": float(os.getenv("SMOOTHNESS_THRESHOLD", "0.1")),
            "stability_threshold": float(os.getenv("STABILITY_THRESHOLD", "0.05")),
            "min_quality_score": float(os.getenv("MIN_QUALITY_SCORE", "0.3")),
            "naturalness_threshold": float(os.getenv("NATURALNESS_THRESHOLD", "0.7")),
            "efficiency_threshold": float(os.getenv("EFFICIENCY_THRESHOLD", "0.6"))
        }
        
        # quality evaluation
        self.ENABLE_QUALITY_CHECK = os.getenv("ENABLE_QUALITY_CHECK", "true").lower() == "true"
        self.QUALITY_CHECK_INTERVAL = int(os.getenv("QUALITY_CHECK_INTERVAL", "5"))  # check every 5 motions
        self.AUTO_QUALITY_IMPROVEMENT = os.getenv("AUTO_QUALITY_IMPROVEMENT", "true").lower() == "true"
    
    def _load_file_config(self):
        """Load file management configuration"""
        self.AUTO_CLEANUP_DAYS = int(os.getenv("AUTO_CLEANUP_DAYS", "7"))
        self.MAX_SESSION_SIZE_MB = int(os.getenv("MAX_SESSION_SIZE_MB", "500"))
        self.SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']
        self.MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "50"))
        
        # compression
        self.ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
        self.COMPRESSION_QUALITY = int(os.getenv("COMPRESSION_QUALITY", "85"))
        
        # backup
        self.ENABLE_BACKUP = os.getenv("ENABLE_BACKUP", "true").lower() == "true"
        self.BACKUP_INTERVAL_HOURS = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
    
    def _load_log_config(self):
        """Load logging configuration"""
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.LOG_MAX_SIZE_MB = int(os.getenv("LOG_MAX_SIZE_MB", "10"))
        self.LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        
        # advanced logging
        self.ENABLE_STRUCTURED_LOGGING = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
        self.LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        self.LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
        self.LOG_TO_DATABASE = os.getenv("LOG_TO_DATABASE", "false").lower() == "true"
    
    def _load_api_config(self):
        """Load API configuration"""
        self.API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
        self.API_RETRY_COUNT = int(os.getenv("API_RETRY_COUNT", "3"))
        self.API_RETRY_DELAY = int(os.getenv("API_RETRY_DELAY", "1"))
        
        # API service
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_WORKERS = int(os.getenv("API_WORKERS", "4"))
        self.API_MAX_REQUESTS = int(os.getenv("API_MAX_REQUESTS", "1000"))
        
        # security
        self.ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
        self.RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        self.ENABLE_AUTHENTICATION = os.getenv("ENABLE_AUTHENTICATION", "false").lower() == "true"
    
    def _load_ui_config(self):
        """Load UI configuration"""
        self.UI_LANGUAGE = os.getenv("UI_LANGUAGE", "en")  # zh/en/ja/ko
        self.SHOW_PROGRESS_BAR = os.getenv("SHOW_PROGRESS_BAR", "true").lower() == "true"
        self.AUTO_SAVE_SESSION = os.getenv("AUTO_SAVE_SESSION", "true").lower() == "true"
        
        # UI theme
        self.UI_THEME = os.getenv("UI_THEME", "auto")  # auto/light/dark
        self.UI_FONT_SIZE = int(os.getenv("UI_FONT_SIZE", "14"))
        self.UI_ANIMATIONS = os.getenv("UI_ANIMATIONS", "true").lower() == "true"
        
        # interaction
        self.ENABLE_KEYBOARD_SHORTCUTS = os.getenv("ENABLE_KEYBOARD_SHORTCUTS", "true").lower() == "true"
        self.ENABLE_MOUSE_GESTURES = os.getenv("ENABLE_MOUSE_GESTURES", "false").lower() == "true"
        self.AUTO_COMPLETION = os.getenv("AUTO_COMPLETION", "true").lower() == "true"
    
    def _load_advanced_config(self):
        """Load advanced feature configuration"""
        # intelligent optimization
        self.ENABLE_AUTO_OPTIMIZATION = os.getenv("ENABLE_AUTO_OPTIMIZATION", "true").lower() == "true"
        self.OPTIMIZATION_INTERVAL_MINUTES = int(os.getenv("OPTIMIZATION_INTERVAL_MINUTES", "30"))
        
        # machine learning
        self.ENABLE_ML_OPTIMIZATION = os.getenv("ENABLE_ML_OPTIMIZATION", "true").lower() == "true"
        self.ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "./ml_models")
        self.ENABLE_FEDERATED_LEARNING = os.getenv("ENABLE_FEDERATED_LEARNING", "false").lower() == "true"
        
        # monitoring
        self.ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        self.MONITORING_INTERVAL_SECONDS = int(os.getenv("MONITORING_INTERVAL_SECONDS", "60"))
        self.ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "true").lower() == "true"
    
    def _get_optimal_device(self) -> str:
        """Get optimal device setting"""
        if os.getenv("FORCE_CPU", "false").lower() == "true":
            return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                # check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory >= 4:  # use GPU if >= 4GB
                    return "cuda:0"
                else:
                    return "cpu"  # insufficient GPU memory, use CPU
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def optimize_for_environment(self):
        """Optimize config based on environment"""
        if self.environment == "docker":
            # Docker optimizations
            self.ENABLE_GPU = False
            self.MAX_WORKERS = 2
            self.ENABLE_MONITORING = False
        
        elif self.environment == "development":
            # Development optimizations
            self.LOG_LEVEL = "DEBUG"
            self.ENABLE_QUALITY_CHECK = True
            self.AUTO_SAVE_SESSION = True
        
        elif self.environment == "production":
            # Production optimizations
            self.LOG_LEVEL = "WARNING"
            self.ENABLE_DEBUG = False
            self.ENABLE_MONITORING = True
        
        # Resource-based optimizations
        if self.system_info["memory_gb"] < 8:
            self.MAX_WORKERS = 2
            self.BATCH_SIZE = 1
            self.ENABLE_MEMORY_OPTIMIZATION = True
        
        if self.system_info["disk_free_gb"] < 10:
            self.AUTO_CLEANUP_DAYS = 3
            self.ENABLE_COMPRESSION = True
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.OUTPUT_BASE_DIR,
            self.SESSION_DIR,
            self.LOG_DIR,
            self.TEMP_DIR,
            self.CACHE_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_session_output_dir(self, session_id: str) -> str:
        """Get session output directory (backward compatible)"""
        output_dir = os.path.join(self.OUTPUT_BASE_DIR, session_id)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_log_file_path(self, log_name: str) -> str:
        """Get log file path (backward compatible)"""
        os.makedirs(self.LOG_DIR, exist_ok=True)
        return os.path.join(self.LOG_DIR, f"{log_name}.log")
    
    def save_config(self, filepath: str = None):
        """Save configuration to file"""
        if filepath is None:
            filepath = "config_backup.yaml"
        
        config_data = {
            "basic": {
                "project_name": self.PROJECT_NAME,
                "version": self.VERSION,
                "environment": self.environment
            },
            "models": {
                "motion_device": self.MOTION_DEVICE,
                "motion_window_size": self.MOTION_WINDOW_SIZE,
                "model_precision": self.MODEL_PRECISION
            },
            "performance": {
                "max_workers": self.MAX_WORKERS,
                "batch_size": self.BATCH_SIZE,
                "enable_parallel": self.ENABLE_PARALLEL
            },
            "quality": self.QUALITY_THRESHOLDS,
            "output": {
                "output_base_dir": self.OUTPUT_BASE_DIR,
                "default_output_format": self.DEFAULT_OUTPUT_FORMAT
            }
        }
        
        # Prefer YAML; fallback to JSON
        try:
            if yaml is not None and (filepath.endswith('.yaml') or filepath.endswith('.yml')):
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                json_path = filepath if filepath.endswith('.json') else filepath.replace('.yaml', '.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
        except Exception:
            # final fallback
            with open(filepath + '.json', 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def load_config_from_file(self, filepath: str):
        """Load configuration from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        # Parse by extension and availability, support YAML/JSON
        config_data: Dict[str, Any] = {}
        try:
            if (filepath.endswith('.yaml') or filepath.endswith('.yml')) and yaml is not None:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
        except Exception:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception:
                config_data = {}
        
        # apply configuration
        for section, values in config_data.items():
            if hasattr(self, section.upper()):
                for key, value in values.items():
                    attr_name = key.upper()
                    if hasattr(self, attr_name):
                        setattr(self, attr_name, value)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "project": f"{self.PROJECT_NAME} {self.VERSION}",
            "environment": self.environment,
            "device": self.MOTION_DEVICE,
            "workers": self.MAX_WORKERS,
            "quality_check": self.ENABLE_QUALITY_CHECK,
            "auto_optimization": self.ENABLE_AUTO_OPTIMIZATION,
            "output_dir": self.OUTPUT_BASE_DIR
        }
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # check directory write permissions
        for directory in [self.OUTPUT_BASE_DIR, self.SESSION_DIR, self.LOG_DIR]:
            if not os.access(os.path.dirname(directory), os.W_OK):
                errors.append(f"Directory not writable: {directory}")
        
        # check API key (supports TEST_MODE)
        if os.getenv("TEST_MODE", "false").lower() == "true":
            # TEST_MODE: skip API key check
            pass
        elif not self.QWEN_API_KEY or self.QWEN_API_KEY == "sk-4046430e513f44c68beec5635a02d97f":
            errors.append("Please set a valid QWEN_API_KEY or enable TEST_MODE=true")
        
        # check model path
        if not os.path.exists(self.MOTION_MODEL_PATH):
            errors.append(f"MotionLLM model file not found: {self.MOTION_MODEL_PATH}")
        
        # check system resources
        if self.system_info["memory_gb"] < 4:
            errors.append("Insufficient system memory, at least 4GB recommended")
        
        return len(errors) == 0, errors


# Global config instance
config = EnhancedConfig()


def validate_environment() -> bool:
    """Validate environment configuration"""
    is_valid, errors = config.validate_config()
    if not is_valid:
        print("❌ Environment validation failed:")
        for error in errors:
            print(f"  - {error}")
    return is_valid

def check_dependencies() -> bool:
    """Check dependencies"""
    required_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"), 
        ("requests", "requests"),
        ("matplotlib", "matplotlib"),
        ("cv2", "opencv-python"),  # opencv-python is imported as cv2
        ("PIL", "Pillow"),         # Pillow is imported as PIL
        ("transformers", "transformers")
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        return False
    
    return True

def show_system_info():
    """Show system information"""
    print("🖥️  System Information:")
    print(f"  Platform: {config.system_info['platform']} {config.system_info['platform_version']}")
    print(f"  Architecture: {config.system_info['architecture']}")
    print(f"  Python: {config.system_info['python_version']}")
    print(f"  CPU cores: {config.system_info['cpu_count']}")
    print(f"  Memory: {config.system_info['memory_gb']}GB")
    print(f"  Disk free: {config.system_info['disk_free_gb']}GB")
    print(f"  Environment: {config.environment}")
    print(f"  Device: {config.MOTION_DEVICE}")
    print(f"  Workers: {config.MAX_WORKERS}")
