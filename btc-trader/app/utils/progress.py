# app/utils/progress.py
import sys
from structlog import get_logger

logger = get_logger(__name__)

class ProgressBar:
    def __init__(self, total, bar_length=40):
        self.total = total
        self.bar_length = bar_length
        self.current = 0
        self._printed = False

    def update(self, progress):
        """Update the progress bar with proper terminal handling"""
        try:
            self.current = progress
            filled_length = int(self.bar_length * self.current / self.total)
            bar = '█' * filled_length + '─' * (self.bar_length - filled_length)
            percentage = min(100, (self.current / self.total) * 100)
            
            # Write to stderr to avoid conflict with stdout logs
            sys.stderr.write(f"\r[{bar}] {percentage:.1f}%")
            sys.stderr.flush()
            self._printed = True
        except Exception as e:
            logger.error("Progress bar update failed", error=str(e))

    def _render(self):
        """Render the progress bar in the terminal"""
        filled_length = int(self.bar_length * self.current / self.total)
        bar = '█' * filled_length + '─' * (self.bar_length - filled_length)
        percentage = min(100, (self.current / self.total) * 100)
        
        # Write to stderr to avoid conflict with stdout logs
        sys.stderr.write(f"\r[{bar}] {percentage:.1f}%")
        sys.stderr.flush()
        self._printed = True


    def clear(self):
        """Clear the progress bar from terminal"""
        try:
            if self._printed:
                sys.stderr.write('\r' + ' ' * (self.bar_length + 10) + '\r')
                sys.stderr.flush()
                self._printed = False
        except Exception as e:
            logger.error("Progress bar clear failed", error=str(e))

# Add the progress_bar function for backward compatibility
def progress_bar(percentage, bar_length=40):
    """Static method for generating progress bar strings"""
    filled_length = int(bar_length * percentage / 100)
    bar = '█' * filled_length + '─' * (bar_length - filled_length)
    return f"[{bar}] {percentage:.1f}%"