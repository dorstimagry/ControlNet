#!/usr/bin/env python3
"""ClearML logging utility for experiment tracking.

This module provides a wrapper around ClearML Task for logging metrics, configs, and plots.
If ClearML is not installed, all functions will gracefully degrade and return None.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import PIL

try:
    import clearml
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None


def init_clearml_task(
    task_name: str,
    project: str = "RL_Longitudinal",
    tags: Optional[list[str]] = None,
    task_type: clearml.Task.TaskTypes = clearml.Task.TaskTypes.training,
) -> Optional[Any]:
    """Initialize a ClearML task with date/time in the name.
    
    Args:
        task_name: Base name for the task (will have timestamp appended)
        project: Project name in ClearML (default: "RL_Longitudinal")
        tags: Optional list of tags to add to the task
    
    Returns:
        ClearML Task object if available, None otherwise
    """
    if not CLEARML_AVAILABLE:
        print("[ClearML] ClearML not available - skipping experiment tracking")
        return None
    
    # Format timestamp as YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_task_name = f"{task_name}_{timestamp}"
    
    try:
        task = Task.init(
            project_name=project,
            task_name=full_task_name,
            tags=tags or [],
            task_type=task_type,
        )
        print(f"[ClearML] Initialized task: {full_task_name} in project: {project} (type: {task_type})")
        return task
    except Exception as e:
        print(f"[ClearML] Warning: Failed to initialize task: {e}")
        return None


def log_metrics(
    task: Optional[Any],
    metrics: Dict[str, float],
    step: int,
) -> None:
    """Log metrics to ClearML task.
    
    Args:
        task: ClearML Task object (can be None if ClearML not available)
        metrics: Dictionary of metric names to values
        step: Training/evaluation step number
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    try:
        for metric_name, metric_value in metrics.items():
            task.get_logger().report_scalar(
                title=metric_name.split("/")[0] if "/" in metric_name else "metrics",
                series=metric_name.split("/")[-1] if "/" in metric_name else metric_name,
                value=metric_value,
                iteration=step,
            )
    except Exception as e:
        print(f"[ClearML] Warning: Failed to log metrics: {e}")


def log_config(task: Optional[Any], config: Dict[str, Any]) -> None:
    """Log configuration dictionary to ClearML task.
    
    Args:
        task: ClearML Task object (can be None if ClearML not available)
        config: Configuration dictionary to log
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    try:
        # Log as hyperparameters
        task.connect(config)
    except Exception as e:
        print(f"[ClearML] Warning: Failed to log config: {e}")


def upload_plot(
    task: Optional[Any],
    plot_path: Path,
    title: str,
    description: Optional[str] = None,
) -> None:
    """Upload a plot image file to ClearML task as a matplotlib figure.
    
    Args:
        task: ClearML Task object (can be None if ClearML not available)
        plot_path: Path to the image file (will be loaded as matplotlib figure)
        title: Title for the plot in ClearML
        description: Optional description for the plot (not used in ClearML API)
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    if not plot_path.exists():
        print(f"[ClearML] Warning: Plot file not found: {plot_path}")
        return
    
    try:
        # Load image and convert to matplotlib figure
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        plot_image = PIL.Image.open(plot_path)
        plot_image_np = np.array(plot_image.convert('RGB'))
        
        # Create a matplotlib figure from the image
        fig = Figure(figsize=(plot_image.width / 100, plot_image.height / 100), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(plot_image_np)
        ax.axis('off')  # Remove axes for image display
        
        # Upload as matplotlib figure
        task.get_logger().report_matplotlib_figure(
            title=title,
            series="plots",
            figure=fig,
        )
    except Exception as e:
        print(f"[ClearML] Warning: Failed to upload plot {plot_path}: {e}")


def log_matplotlib_figure(
    task: Optional[Any],
    fig: Any,  # matplotlib.figure.Figure
    title: str,
    description: Optional[str] = None,
) -> None:
    """Log a matplotlib figure directly to ClearML task.
    
    Args:
        task: ClearML Task object (can be None if ClearML not available)
        fig: Matplotlib figure object
        title: Title for the plot in ClearML
        description: Optional description for the plot
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        task.get_logger().report_matplotlib_figure(title=title, series="plots", figure=fig)
        
        # Save figure to bytes buffer
        # buf = BytesIO()
        # fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        # buf.seek(0)
        
        # # Upload to ClearML (report_image doesn't support description parameter)
        # task.get_logger().report_image(
        #     title=title,
        #     series="plots",
        #     image=buf,
        # )
        # buf.close()
    except Exception as e:
        print(f"[ClearML] Warning: Failed to log matplotlib figure: {e}")


def upload_artifact(
    task: Optional[Any],
    file_path: Path,
    artifact_name: str,
    description: Optional[str] = None,
) -> None:
    """Upload a file as an artifact to ClearML task.
    
    Args:
        task: ClearML Task object (can be None if ClearML not available)
        file_path: Path to the file to upload
        artifact_name: Name for the artifact in ClearML
        description: Optional description for the artifact (not used in ClearML API)
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    if not file_path.exists():
        print(f"[ClearML] Warning: Artifact file not found: {file_path}")
        return
    
    try:
        # ClearML upload_artifact doesn't support description parameter
        task.upload_artifact(
            name=artifact_name,
            artifact_object=str(file_path),
        )
    except Exception as e:
        print(f"[ClearML] Warning: Failed to upload artifact {file_path}: {e}")


def close_task(task: Optional[Any]) -> None:
    """Close the ClearML task.
    
    Args:
        task: ClearML Task object (can be None if ClearML not available)
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    try:
        task.close()
        print("[ClearML] Task closed")
    except Exception as e:
        print(f"[ClearML] Warning: Failed to close task: {e}")
