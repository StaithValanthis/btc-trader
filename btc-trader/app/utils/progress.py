def progress_bar(percentage: float, width: int = 20) -> str:
    """
    Generate a text-based progress bar.
    
    Args:
        percentage (float): Current progress percentage (0-100)
        width (int): Width of the progress bar in characters
    
    Returns:
        str: Progress bar string
    """
    filled = int(round(width * percentage / 100))
    return f"[{'█' * filled}{' ' * (width - filled)}] {percentage:.1f}%"