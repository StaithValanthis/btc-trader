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
        """Update the progress bar"""
        self.current = progress
        self._render()

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
        """Clear the progress bar from the terminal"""
        if self._printed:
            sys.stderr.write('\r' + ' ' * (self.bar_length + 10) + '\r')
            sys.stderr.flush()
            self._printed = False