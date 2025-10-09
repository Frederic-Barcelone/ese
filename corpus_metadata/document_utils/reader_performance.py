#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_performance.py
#
"""
Performance Monitor Module
=========================

Monitors and optimizes performance of document reading operations.
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime
from functools import wraps
import json
from pathlib import Path

from corpus_metadata.document_utils.metadata_config_loader import CorpusConfig

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors performance metrics for document processing.
    
    Tracks metrics like processing time, memory usage, and throughput
    to help identify bottlenecks and optimize performance.
    """
    
    def __init__(self, enable_monitoring: Optional[bool] = None):
        """
        Initialize performance monitor.
        
        Args:
            enable_monitoring: Override for monitoring enable state (for backward compatibility)
        """
        # Load configuration from YAML
        self.corpus_config = CorpusConfig()
        performance_config = self.corpus_config.config.get('performance', {})
        monitoring_config = performance_config.get('monitoring', {})
        
        # Monitoring settings
        self.enabled = enable_monitoring if enable_monitoring is not None else monitoring_config.get('enabled', True)
        self.track_memory = monitoring_config.get('track_memory', True)
        self.track_cpu = monitoring_config.get('track_cpu', True)
        self.track_disk_io = monitoring_config.get('track_disk_io', False)
        self.log_slow_operations = monitoring_config.get('log_slow_operations', True)
        self.slow_operation_threshold = monitoring_config.get('slow_operation_threshold_seconds', 5.0)
        
        # Metrics storage settings
        storage_config = monitoring_config.get('storage', {})
        self.max_metrics_per_operation = storage_config.get('max_metrics_per_operation', 1000)
        self.persist_metrics = storage_config.get('persist_metrics', False)
        self.metrics_file = Path(storage_config.get('metrics_file', './metrics/performance_metrics.json'))
        
        # Reporting settings
        reporting_config = monitoring_config.get('reporting', {})
        self.bottleneck_threshold = reporting_config.get('bottleneck_threshold_seconds', 1.0)
        self.generate_periodic_reports = reporting_config.get('generate_periodic_reports', False)
        self.report_interval_seconds = reporting_config.get('report_interval_seconds', 3600)
        
        # Optimization settings
        optimization_config = performance_config.get('optimization', {})
        self.enable_adaptive_optimization = optimization_config.get('enable_adaptive_optimization', False)
        self.memory_warning_threshold_mb = optimization_config.get('memory_warning_threshold_mb', 1024)
        self.cpu_warning_threshold_percent = optimization_config.get('cpu_warning_threshold_percent', 80)
        
        # Initialize metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=self.max_metrics_per_operation))
        self.start_time = time.time()
        self.operation_times = defaultdict(list)
        self.lock = threading.Lock()
        
        # System resource baseline
        self.baseline_memory = psutil.Process().memory_info().rss if self.track_memory else 0
        self.baseline_cpu = psutil.cpu_percent(interval=0.1) if self.track_cpu else 0
        
        # Disk I/O baseline
        if self.track_disk_io:
            self.baseline_disk_io = psutil.disk_io_counters()
        
        # Load persisted metrics if enabled
        if self.persist_metrics and self.metrics_file.exists():
            self._load_metrics()
        
        # Start periodic reporting if enabled
        if self.generate_periodic_reports:
            self._start_periodic_reporting()
        
        logger.info(f"Performance monitor initialized (enabled: {self.enabled})")
    
    def track_operation(self, operation_name: Optional[str] = None) -> Callable:
        """
        Decorator to track operation performance.
        
        Args:
            operation_name: Name of the operation to track (defaults to function name)
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss if self.track_memory else 0
                start_cpu = psutil.Process().cpu_percent(interval=0) if self.track_cpu else 0
                
                # Track disk I/O if enabled
                if self.track_disk_io:
                    start_disk_io = psutil.disk_io_counters()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss if self.track_memory else 0
                    end_cpu = psutil.Process().cpu_percent(interval=0) if self.track_cpu else 0
                    
                    # Calculate disk I/O delta
                    disk_io_delta = None
                    if self.track_disk_io:
                        end_disk_io = psutil.disk_io_counters()
                        disk_io_delta = {
                            'read_bytes': end_disk_io.read_bytes - start_disk_io.read_bytes,
                            'write_bytes': end_disk_io.write_bytes - start_disk_io.write_bytes
                        }
                    
                    # Record metrics
                    self._record_operation(
                        op_name,
                        end_time - start_time,
                        end_memory - start_memory,
                        end_cpu - start_cpu,
                        disk_io_delta,
                        success,
                        error
                    )
                
                return result
            
            return wrapper
        return decorator
    
    def _record_operation(self,
                         operation: str,
                         duration: float,
                         memory_delta: int,
                         cpu_delta: float,
                         disk_io_delta: Optional[Dict[str, int]],
                         success: bool,
                         error: Optional[str]) -> None:
        """
        Record operation metrics.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            memory_delta: Memory usage change in bytes
            cpu_delta: CPU usage change in percent
            disk_io_delta: Disk I/O changes
            success: Whether operation succeeded
            error: Error message if failed
        """
        with self.lock:
            timestamp = datetime.now()
            
            metric = {
                'timestamp': timestamp.isoformat(),
                'duration': duration,
                'memory_delta': memory_delta,
                'cpu_delta': cpu_delta,
                'disk_io_delta': disk_io_delta,
                'success': success,
                'error': error
            }
            
            self.metrics[operation].append(metric)
            self.operation_times[operation].append(duration)
            
            # Log slow operations
            if self.log_slow_operations and duration > self.slow_operation_threshold:
                logger.warning(f"Slow operation: {operation} took {duration:.2f}s")
            
            # Check resource usage warnings
            if self.track_memory:
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                if memory_mb > self.memory_warning_threshold_mb:
                    logger.warning(f"High memory usage: {memory_mb:.2f} MB")
            
            if self.track_cpu:
                cpu_percent = psutil.Process().cpu_percent(interval=0)
                if cpu_percent > self.cpu_warning_threshold_percent:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Persist metrics if enabled
            if self.persist_metrics:
                self._save_metrics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            stats = {
                'uptime_seconds': time.time() - self.start_time,
                'operations': {},
                'enabled_tracking': {
                    'memory': self.track_memory,
                    'cpu': self.track_cpu,
                    'disk_io': self.track_disk_io
                }
            }
            
            # Calculate statistics for each operation
            for operation, times in self.operation_times.items():
                if times:
                    metrics_list = list(self.metrics[operation])
                    
                    operation_stats = {
                        'count': len(times),
                        'total_time': sum(times),
                        'average_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'success_rate': self._calculate_success_rate(operation)
                    }
                    
                    # Add memory statistics if tracking
                    if self.track_memory and metrics_list:
                        memory_deltas = [m['memory_delta'] for m in metrics_list]
                        operation_stats['memory'] = {
                            'average_delta_mb': sum(memory_deltas) / len(memory_deltas) / (1024 * 1024),
                            'max_delta_mb': max(memory_deltas) / (1024 * 1024),
                            'total_delta_mb': sum(memory_deltas) / (1024 * 1024)
                        }
                    
                    # Add disk I/O statistics if tracking
                    if self.track_disk_io and metrics_list:
                        disk_metrics = [m['disk_io_delta'] for m in metrics_list if m['disk_io_delta']]
                        if disk_metrics:
                            operation_stats['disk_io'] = {
                                'total_read_mb': sum(m['read_bytes'] for m in disk_metrics) / (1024 * 1024),
                                'total_write_mb': sum(m['write_bytes'] for m in disk_metrics) / (1024 * 1024)
                            }
                    
                    stats['operations'][operation] = operation_stats
            
            # System resource usage
            process = psutil.Process()
            stats['system'] = {
                'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
                'memory_delta_mb': (process.memory_info().rss - self.baseline_memory) / (1024 * 1024),
                'cpu_percent': process.cpu_percent(interval=0.1),
                'thread_count': process.num_threads()
            }
            
            # Add disk usage if available
            if self.track_disk_io:
                current_disk_io = psutil.disk_io_counters()
                stats['system']['disk_io'] = {
                    'total_read_mb': (current_disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024 * 1024),
                    'total_write_mb': (current_disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024 * 1024)
                }
            
            return stats
    
    def _calculate_success_rate(self, operation: str) -> float:
        """Calculate success rate for an operation."""
        metrics = self.metrics[operation]
        if not metrics:
            return 0.0
        
        successful = sum(1 for m in metrics if m['success'])
        return successful / len(metrics)
    
    def get_bottlenecks(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold: Time threshold in seconds (uses config default if None)
            
        Returns:
            List of operations exceeding threshold
        """
        threshold = threshold or self.bottleneck_threshold
        bottlenecks = []
        
        with self.lock:
            for operation, times in self.operation_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    if avg_time > threshold:
                        bottleneck_info = {
                            'operation': operation,
                            'average_time': avg_time,
                            'max_time': max(times),
                            'frequency': len(times),
                            'total_time': sum(times)
                        }
                        
                        # Add success rate
                        bottleneck_info['success_rate'] = self._calculate_success_rate(operation)
                        
                        bottlenecks.append(bottleneck_info)
        
        return sorted(bottlenecks, key=lambda x: x['average_time'], reverse=True)
    
    def generate_report(self) -> str:
        """
        Generate a performance report.
        
        Returns:
            Formatted performance report
        """
        stats = self.get_statistics()
        bottlenecks = self.get_bottlenecks()
        
        report = [
            "=== Document Reader Performance Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Uptime: {stats['uptime_seconds']:.2f} seconds",
            f"Configuration: {self.corpus_config.config.get('execution', {}).get('profile', 'default')}",
            "",
            "=== System Resources ===",
            f"Memory Usage: {stats['system']['memory_usage_mb']:.2f} MB",
            f"Memory Delta: {stats['system']['memory_delta_mb']:+.2f} MB",
            f"CPU Usage: {stats['system']['cpu_percent']:.1f}%",
            f"Threads: {stats['system']['thread_count']}",
        ]
        
        if 'disk_io' in stats['system']:
            report.extend([
                f"Disk Read: {stats['system']['disk_io']['total_read_mb']:.2f} MB",
                f"Disk Write: {stats['system']['disk_io']['total_write_mb']:.2f} MB"
            ])
        
        report.extend(["", "=== Operation Statistics ==="])
        
        for op, metrics in stats['operations'].items():
            report.extend([
                f"\n{op}:",
                f"  Count: {metrics['count']}",
                f"  Total Time: {metrics['total_time']:.2f}s",
                f"  Average Time: {metrics['average_time']:.3f}s",
                f"  Min/Max: {metrics['min_time']:.3f}s / {metrics['max_time']:.3f}s",
                f"  Success Rate: {metrics['success_rate']:.1%}"
            ])
            
            if 'memory' in metrics:
                report.extend([
                    f"  Avg Memory Delta: {metrics['memory']['average_delta_mb']:.2f} MB",
                    f"  Max Memory Delta: {metrics['memory']['max_delta_mb']:.2f} MB"
                ])
            
            if 'disk_io' in metrics:
                report.extend([
                    f"  Total Disk Read: {metrics['disk_io']['total_read_mb']:.2f} MB",
                    f"  Total Disk Write: {metrics['disk_io']['total_write_mb']:.2f} MB"
                ])
        
        if bottlenecks:
            report.extend([
                "",
                f"=== Performance Bottlenecks (>{self.bottleneck_threshold}s) ===",
            ])
            for bottleneck in bottlenecks[:10]:  # Top 10 bottlenecks
                report.append(
                    f"- {bottleneck['operation']}: "
                    f"avg {bottleneck['average_time']:.3f}s, "
                    f"max {bottleneck['max_time']:.3f}s, "
                    f"success rate {bottleneck['success_rate']:.1%}"
                )
        
        # Add recommendations if adaptive optimization is enabled
        if self.enable_adaptive_optimization:
            recommendations = self._get_optimization_recommendations()
            if recommendations:
                report.extend(["", "=== Optimization Recommendations ==="])
                report.extend(recommendations)
        
        return '\n'.join(report)
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        stats = self.get_statistics()
        
        # Memory recommendations
        if stats['system']['memory_usage_mb'] > self.memory_warning_threshold_mb:
            recommendations.append(f"• High memory usage detected ({stats['system']['memory_usage_mb']:.0f} MB). Consider:")
            recommendations.append("  - Reducing batch sizes")
            recommendations.append("  - Enabling streaming processing")
            recommendations.append("  - Clearing caches more frequently")
        
        # CPU recommendations
        if stats['system']['cpu_percent'] > self.cpu_warning_threshold_percent:
            recommendations.append(f"• High CPU usage detected ({stats['system']['cpu_percent']:.0f}%). Consider:")
            recommendations.append("  - Reducing parallel processing threads")
            recommendations.append("  - Optimizing regex patterns")
            recommendations.append("  - Disabling non-essential features")
        
        # Operation-specific recommendations
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            recommendations.append("• Slow operations detected:")
            for bottleneck in bottlenecks[:3]:
                if 'ocr' in bottleneck['operation'].lower():
                    recommendations.append(f"  - {bottleneck['operation']}: Consider reducing OCR resolution or page count")
                elif 'pdf' in bottleneck['operation'].lower():
                    recommendations.append(f"  - {bottleneck['operation']}: Consider using faster PDF extraction method")
                elif 'cache' in bottleneck['operation'].lower():
                    recommendations.append(f"  - {bottleneck['operation']}: Consider optimizing cache settings")
        
        return recommendations
    
    def _save_metrics(self) -> None:
        """Save metrics to persistent storage."""
        if not self.persist_metrics:
            return
        
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert metrics to serializable format
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'operations': {}
            }
            
            for operation, metrics_deque in self.metrics.items():
                metrics_data['operations'][operation] = list(metrics_deque)
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_metrics(self) -> None:
        """Load metrics from persistent storage."""
        try:
            with open(self.metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Restore metrics
            for operation, metrics_list in metrics_data.get('operations', {}).items():
                for metric in metrics_list:
                    self.metrics[operation].append(metric)
                    if metric['success']:
                        self.operation_times[operation].append(metric['duration'])
            
            logger.info(f"Loaded metrics from {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def _start_periodic_reporting(self) -> None:
        """Start periodic report generation."""
        def generate_periodic_report():
            while self.generate_periodic_reports:
                time.sleep(self.report_interval_seconds)
                if self.enabled:
                    report = self.generate_report()
                    logger.info(f"\n{report}")
        
        thread = threading.Thread(target=generate_periodic_report, daemon=True)
        thread.start()
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self.lock:
            self.metrics.clear()
            self.operation_times.clear()
            self.start_time = time.time()
            
            # Reset baselines
            self.baseline_memory = psutil.Process().memory_info().rss if self.track_memory else 0
            self.baseline_cpu = psutil.cpu_percent(interval=0.1) if self.track_cpu else 0
            
            if self.track_disk_io:
                self.baseline_disk_io = psutil.disk_io_counters()
            
            logger.info("Performance metrics reset")
    
    def export_metrics(self, output_path: Path) -> None:
        """
        Export metrics to a file.
        
        Args:
            output_path: Path to save metrics
        """
        stats = self.get_statistics()
        report = self.generate_report()
        
        export_data = {
            'report': report,
            'statistics': stats,
            'bottlenecks': self.get_bottlenecks(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def reset_performance_monitor() -> None:
    """Reset the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is not None:
        _performance_monitor.reset_metrics()
    _performance_monitor = None