"""
Performance profiling utilities for measuring processing speed and resource usage.
"""

import time
import psutil
import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from functools import wraps


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.
    """
    operation: str
    duration_seconds: float
    memory_used_mb: float
    cpu_percent: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'duration_seconds': self.duration_seconds,
            'memory_used_mb': self.memory_used_mb,
            'cpu_percent': self.cpu_percent,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class PerformanceProfiler:
    """
    Profiles performance of operations.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process(os.getpid())
    
    def measure(self, 
                operation: str,
                metadata: Optional[Dict] = None) -> 'ProfilerContext':
        """
        Context manager for measuring operation performance.
        
        Args:
            operation: Name of operation being measured
            metadata: Optional metadata about the operation
            
        Returns:
            ProfilerContext for use with 'with' statement
        """
        return ProfilerContext(self, operation, metadata or {})
    
    def add_metric(self, metric: PerformanceMetrics):
        """Add a metric to the collection."""
        self.metrics.append(metric)
    
    def get_metrics(self, operation: Optional[str] = None) -> List[PerformanceMetrics]:
        """
        Get collected metrics.
        
        Args:
            operation: Optional filter by operation name
            
        Returns:
            List of metrics
        """
        if operation:
            return [m for m in self.metrics if m.operation == operation]
        return self.metrics
    
    def get_summary(self, operation: Optional[str] = None) -> Dict:
        """
        Get summary statistics.
        
        Args:
            operation: Optional filter by operation name
            
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.get_metrics(operation)
        
        if not metrics:
            return {}
        
        durations = [m.duration_seconds for m in metrics]
        memories = [m.memory_used_mb for m in metrics]
        cpus = [m.cpu_percent for m in metrics]
        
        return {
            'operation': operation or 'all',
            'count': len(metrics),
            'duration': {
                'total': sum(durations),
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations)
            },
            'memory_mb': {
                'mean': sum(memories) / len(memories),
                'min': min(memories),
                'max': max(memories)
            },
            'cpu_percent': {
                'mean': sum(cpus) / len(cpus),
                'min': min(cpus),
                'max': max(cpus)
            }
        }
    
    def save_metrics(self, output_path: Path, format: str = 'json'):
        """
        Save metrics to file.
        
        Args:
            output_path: Path to output file
            format: Output format ('json' or 'csv')
        """
        if format == 'json':
            self._save_json(output_path)
        elif format == 'csv':
            self._save_csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json(self, output_path: Path):
        """Save metrics as JSON."""
        data = {
            'metrics': [m.to_dict() for m in self.metrics],
            'summary': self.get_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_csv(self, output_path: Path):
        """Save metrics as CSV."""
        if not self.metrics:
            return
        
        fieldnames = ['operation', 'duration_seconds', 'memory_used_mb', 
                     'cpu_percent', 'timestamp']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in self.metrics:
                row = {k: v for k, v in metric.to_dict().items() 
                      if k in fieldnames}
                writer.writerow(row)
    
    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
    
    def print_summary(self, operation: Optional[str] = None):
        """
        Print summary to console.
        
        Args:
            operation: Optional filter by operation name
        """
        summary = self.get_summary(operation)
        
        if not summary:
            print("No metrics collected.")
            return
        
        print("=" * 60)
        print(f"PERFORMANCE SUMMARY: {summary['operation']}")
        print("=" * 60)
        print(f"Operations counted: {summary['count']}")
        print()
        print("Duration (seconds):")
        print(f"  Total: {summary['duration']['total']:.3f}")
        print(f"  Mean:  {summary['duration']['mean']:.3f}")
        print(f"  Min:   {summary['duration']['min']:.3f}")
        print(f"  Max:   {summary['duration']['max']:.3f}")
        print()
        print("Memory (MB):")
        print(f"  Mean:  {summary['memory_mb']['mean']:.2f}")
        print(f"  Min:   {summary['memory_mb']['min']:.2f}")
        print(f"  Max:   {summary['memory_mb']['max']:.2f}")
        print()
        print("CPU (%):")
        print(f"  Mean:  {summary['cpu_percent']['mean']:.2f}")
        print(f"  Min:   {summary['cpu_percent']['min']:.2f}")
        print(f"  Max:   {summary['cpu_percent']['max']:.2f}")
        print("=" * 60)


class ProfilerContext:
    """
    Context manager for profiling operations.
    """
    
    def __init__(self, 
                 profiler: PerformanceProfiler,
                 operation: str,
                 metadata: Dict):
        """
        Initialize context.
        
        Args:
            profiler: Parent profiler instance
            operation: Operation name
            metadata: Operation metadata
        """
        self.profiler = profiler
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        self.start_memory = self.profiler.process.memory_info().rss / (1024 * 1024)
        self.start_cpu = self.profiler.process.cpu_percent()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record metrics."""
        duration = time.time() - self.start_time
        end_memory = self.profiler.process.memory_info().rss / (1024 * 1024)
        end_cpu = self.profiler.process.cpu_percent()
        
        memory_used = end_memory - self.start_memory
        cpu_used = (self.start_cpu + end_cpu) / 2  # Average
        
        metric = PerformanceMetrics(
            operation=self.operation,
            duration_seconds=duration,
            memory_used_mb=memory_used,
            cpu_percent=cpu_used,
            metadata=self.metadata
        )
        
        self.profiler.add_metric(metric)


def profile_function(operation_name: Optional[str] = None):
    """
    Decorator for profiling functions.
    
    Args:
        operation_name: Optional custom operation name
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create global profiler
            if not hasattr(wrapper, '_profiler'):
                wrapper._profiler = PerformanceProfiler()
            
            op_name = operation_name or func.__name__
            
            with wrapper._profiler.measure(op_name):
                result = func(*args, **kwargs)
            
            return result
        
        # Attach profiler to wrapper for access
        wrapper.get_profiler = lambda: getattr(wrapper, '_profiler', None)
        
        return wrapper
    
    return decorator


class BatchProfiler:
    """
    Specialized profiler for batch processing operations.
    """
    
    def __init__(self):
        """Initialize batch profiler."""
        self.profiler = PerformanceProfiler()
        self.batch_start_time = None
        self.items_processed = 0
    
    def start_batch(self):
        """Start batch timing."""
        self.batch_start_time = time.time()
        self.items_processed = 0
    
    def record_item(self, operation: str, metadata: Optional[Dict] = None):
        """
        Record processing of a single item.
        
        Args:
            operation: Operation name
            metadata: Optional metadata
        """
        self.items_processed += 1
        return self.profiler.measure(operation, metadata)
    
    def end_batch(self) -> Dict:
        """
        End batch and get summary.
        
        Returns:
            Batch summary statistics
        """
        if self.batch_start_time is None:
            return {}
        
        total_time = time.time() - self.batch_start_time
        
        summary = self.profiler.get_summary()
        summary['batch'] = {
            'total_time': total_time,
            'items_processed': self.items_processed,
            'items_per_second': self.items_processed / total_time if total_time > 0 else 0,
            'time_per_item': total_time / self.items_processed if self.items_processed > 0 else 0
        }
        
        return summary
    
    def print_batch_summary(self):
        """Print batch summary to console."""
        summary = self.end_batch()
        
        if not summary:
            print("No batch metrics collected.")
            return
        
        print("=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Items processed: {summary['batch']['items_processed']}")
        print(f"Total time: {summary['batch']['total_time']:.2f}s")
        print(f"Items per second: {summary['batch']['items_per_second']:.2f}")
        print(f"Time per item: {summary['batch']['time_per_item']:.3f}s")
        print()
        
        if 'duration' in summary:
            print("Per-operation statistics:")
            print(f"  Mean duration: {summary['duration']['mean']:.3f}s")
            print(f"  Mean memory: {summary['memory_mb']['mean']:.2f} MB")
            print(f"  Mean CPU: {summary['cpu_percent']['mean']:.2f}%")
        
        print("=" * 60)


def measure_throughput(func: Callable,
                       items: List[Any],
                       warmup_iterations: int = 5) -> Dict:
    """
    Measure throughput of a function over a list of items.
    
    Args:
        func: Function to measure
        items: List of items to process
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Throughput statistics
    """
    # Warmup
    for item in items[:warmup_iterations]:
        func(item)
    
    # Measure
    start_time = time.time()
    
    for item in items:
        func(item)
    
    duration = time.time() - start_time
    
    return {
        'total_items': len(items),
        'total_duration': duration,
        'items_per_second': len(items) / duration if duration > 0 else 0,
        'seconds_per_item': duration / len(items) if len(items) > 0 else 0
    }


def compare_performance(func1: Callable,
                       func2: Callable,
                       items: List[Any],
                       func1_name: str = "Function 1",
                       func2_name: str = "Function 2") -> Dict:
    """
    Compare performance of two functions.
    
    Args:
        func1: First function
        func2: Second function
        items: Items to process
        func1_name: Name for first function
        func2_name: Name for second function
        
    Returns:
        Comparison statistics
    """
    print(f"Comparing {func1_name} vs {func2_name}...")
    
    # Measure func1
    stats1 = measure_throughput(func1, items)
    
    # Measure func2
    stats2 = measure_throughput(func2, items)
    
    # Calculate speedup
    speedup = stats2['total_duration'] / stats1['total_duration'] if stats1['total_duration'] > 0 else 0
    
    comparison = {
        func1_name: stats1,
        func2_name: stats2,
        'speedup': speedup,
        'faster': func1_name if speedup > 1 else func2_name
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\n{func1_name}:")
    print(f"  Total time: {stats1['total_duration']:.3f}s")
    print(f"  Items/sec: {stats1['items_per_second']:.2f}")
    print(f"\n{func2_name}:")
    print(f"  Total time: {stats2['total_duration']:.3f}s")
    print(f"  Items/sec: {stats2['items_per_second']:.2f}")
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Faster: {comparison['faster']}")
    print("=" * 60)
    
    return comparison


# Global profiler instance
_global_profiler = None


def get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def reset_global_profiler():
    """Reset global profiler."""
    global _global_profiler
    _global_profiler = None