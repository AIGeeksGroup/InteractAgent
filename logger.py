#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging system - unified logging utilities
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback

from config import config


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # cyan
        'INFO': '\033[32m',     # green
        'WARNING': '\033[33m',  # yellow
        'ERROR': '\033[31m',    # red
        'CRITICAL': '\033[35m', # magenta
        'RESET': '\033[0m'      # reset
    }
    
    def format(self, record):
        # add color
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class MotionLogger:
    """Motion planner logging system"""
    
    def __init__(self, name: str = "MotionPlanner"):
        """
        Initialize logging system
        
        Args:
            name: logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # performance stats
        self.stats = {
            "start_time": datetime.now(),
            "operations": {},
            "errors": [],
            "warnings": []
        }
    
    def _setup_handlers(self):
        """Setup log handlers"""
        
        # console handler (colored output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # file handler (detailed log)
        log_file = config.get_log_file_path(self.name.lower())
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            config.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # error file handler
        error_log_file = config.get_log_file_path(f"{self.name.lower()}_error")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=config.LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """Debug log"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Info log"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Warning log"""
        self.logger.warning(self._format_message(message, **kwargs))
        self.stats["warnings"].append({
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "kwargs": kwargs
        })
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Error log"""
        error_info = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "kwargs": kwargs
        }
        
        if exception:
            error_info["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
            message += f" - {type(exception).__name__}: {exception}"
        
        self.logger.error(self._format_message(message, **kwargs))
        self.stats["errors"].append(error_info)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Critical error log"""
        if exception:
            message += f" - {type(exception).__name__}: {exception}"
        
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format log message"""
        if kwargs:
            # append kwargs into message
            extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {extra_info}"
        return message
    
    def log_operation_start(self, operation: str, **kwargs):
        """Record operation start"""
        self.stats["operations"][operation] = {
            "start_time": datetime.now(),
            "status": "running",
            "kwargs": kwargs
        }
        self.info(f"Start operation: {operation}", **kwargs)
    
    def log_operation_end(self, operation: str, success: bool = True, **kwargs):
        """Record operation end"""
        if operation in self.stats["operations"]:
            op_info = self.stats["operations"][operation]
            op_info["end_time"] = datetime.now()
            op_info["duration"] = (op_info["end_time"] - op_info["start_time"]).total_seconds()
            op_info["status"] = "success" if success else "failed"
            op_info.update(kwargs)
            
            status_icon = "✅" if success else "❌"
            self.info(f"{status_icon} Operation completed: {operation}", 
                     duration=f"{op_info['duration']:.2f}s", **kwargs)
        else:
            self.warning(f"Operation {operation} start record not found")
    
    def log_motion_generation(self, description: str, motion_id: str, 
                            frames: int, quality: float, duration: float):
        """Record motion generation"""
        self.info("Motion generation completed", 
                 motion_id=motion_id,
                 description=description,
                 frames=frames,
                 quality=f"{quality:.3f}",
                 duration=f"{duration:.2f}s")
    
    def log_scene_analysis(self, image_url: str, analysis_length: int, duration: float):
        """Record scene analysis"""
        self.info("Scene analysis completed",
                 image_url=image_url[:50] + "..." if len(image_url) > 50 else image_url,
                 analysis_length=analysis_length,
                 duration=f"{duration:.2f}s")
    
    def log_api_call(self, api_name: str, status_code: int, duration: float, 
                    request_size: int = 0, response_size: int = 0):
        """Record API call"""
        success = 200 <= status_code < 300
        level = "info" if success else "error"
        
        getattr(self, level)("API call",
                           api=api_name,
                           status=status_code,
                           duration=f"{duration:.2f}s",
                           request_size=request_size,
                           response_size=response_size)
    
    def log_file_operation(self, operation: str, file_path: str, 
                          file_size: int = 0, success: bool = True):
        """Record file operation"""
        level = "info" if success else "error"
        getattr(self, level)("File operation",
                           operation=operation,
                           file=os.path.basename(file_path),
                           size=f"{file_size/1024:.1f}KB" if file_size > 0 else "N/A")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats"""
        current_time = datetime.now()
        # ensure non-negative duration
        time_diff = (current_time - self.stats["start_time"]).total_seconds()
        total_duration = max(0.0, time_diff)  # 避免负数
        
        stats = self.stats.copy()
        stats["total_duration"] = total_duration
        stats["total_operations"] = len(stats["operations"])
        stats["total_errors"] = len(stats["errors"])
        stats["total_warnings"] = len(stats["warnings"])
        
        # compute operation stats
        successful_ops = sum(1 for op in stats["operations"].values() 
                           if op.get("status") == "success")
        stats["success_rate"] = successful_ops / len(stats["operations"]) if stats["operations"] else 0
        
        return stats
    
    def save_stats(self, file_path: Optional[str] = None):
        """Save stats to file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = config.get_log_file_path(f"{self.name.lower()}_stats_{timestamp}")
            file_path = file_path.replace('.log', '.json')
        
        try:
            stats = self.get_stats()
            
            # convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj
            
            stats = convert_datetime(stats)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            self.info(f"Stats saved: {file_path}")
            
        except Exception as e:
            self.error(f"Failed to save stats: {e}")
    
    def print_summary(self):
        """Print session summary"""
        stats = self.get_stats()
        
        print(f"\n📊 Session Summary - {self.name}")
        print("=" * 50)
        print(f"⏱️  Total duration: {stats['total_duration']:.1f} s")
        print(f"🔧 Total operations: {stats['total_operations']}")
        print(f"✅ Success rate: {stats['success_rate']:.1%}")
        print(f"⚠️  Warnings: {stats['total_warnings']}")
        print(f"❌ Errors: {stats['total_errors']}")
        
        if stats["operations"]:
            print(f"\n📋 Operation details:")
            for op_name, op_info in stats["operations"].items():
                status_icon = "✅" if op_info.get("status") == "success" else "❌"
                duration = op_info.get("duration", 0)
                print(f"  {status_icon} {op_name}: {duration:.2f}s")
        
        if stats["errors"]:
            print(f"\n❌ Error list:")
            for error in stats["errors"][-5:]:  # 只显示最近5个错误
                print(f"  - {error['message']}")


# 创建全局日志器实例
motion_logger = MotionLogger("MotionPlanner")
scene_logger = MotionLogger("SceneAnalysis")
api_logger = MotionLogger("APICall")


# Decorators
def log_operation(operation_name: str, logger: Optional[MotionLogger] = None):
    """Operation logging decorator"""
    if logger is None:
        logger = motion_logger
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log_operation_start(operation_name, 
                                     function=func.__name__,
                                     args_count=len(args),
                                     kwargs_count=len(kwargs))
            
            try:
                result = func(*args, **kwargs)
                logger.log_operation_end(operation_name, success=True)
                return result
            except Exception as e:
                logger.log_operation_end(operation_name, success=False)
                logger.error(f"Operation {operation_name} failed", exception=e)
                raise
        
        return wrapper
    return decorator


def log_api_call(api_name: str, logger: Optional[MotionLogger] = None):
    """API call logging decorator"""
    if logger is None:
        logger = api_logger
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # try to get status code from result
                status_code = 200
                if isinstance(result, dict) and "status_code" in result:
                    status_code = result["status_code"]
                
                logger.log_api_call(api_name, status_code, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.log_api_call(api_name, 500, duration)
                logger.error(f"API call {api_name} failed", exception=e)
                raise
        
        return wrapper
    return decorator


# Context manager
class LogContext:
    """Logging context manager"""
    
    def __init__(self, operation: str, logger: Optional[MotionLogger] = None, **kwargs):
        self.operation = operation
        self.logger = logger or motion_logger
        self.kwargs = kwargs
        self.success = False
    
    def __enter__(self):
        self.logger.log_operation_start(self.operation, **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
        else:
            self.logger.error(f"Operation {self.operation} raised an exception", 
                            exception=exc_val)
        
        self.logger.log_operation_end(self.operation, success=self.success)


if __name__ == "__main__":
    # Test logging system
    print("🧪 Testing logging system...")
    
    logger = MotionLogger("Test")
    
    # Test log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test operation logging
    logger.log_operation_start("test_operation", param1="value1")
    import time
    time.sleep(0.1)
    logger.log_operation_end("test_operation", success=True, result="success")
    
    # Test context manager
    with LogContext("context_test", logger):
        logger.info("Running operation inside context")
    
    # Show stats summary
    logger.print_summary()
    
    print("✅ Logging system test completed")



