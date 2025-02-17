import sys
import time
from structlog import get_logger

logger = get_logger(__name__)

class ProgressBar:
    def __init__(self, total, bar_length=40):
        self.total = total
        self.bar_length = bar_length
        self.current = 0
        self.has_tty = sys.stderr.isatty()
        self.last_update = 0
        self.completed = False
        self.start_time = time.time()

    def update(self, progress):
        """Update progress with smart display handling"""
        if self.completed:
            return

        self.current = min(progress, self.total)
        now = time.time()

        # Throttle updates to max 1 per second
        if now - self.last_update < 1.0:
            return
        self.last_update = now

        if self.has_tty:
            self._display_progress_bar()
        else:
            self._log_progress()

        # Check completion
        if self.current >= self.total:
            self.completed = True
            if self.has_tty:
                sys.stderr.write("\n")  # Ensure new line after completion
            logger.info("Warmup complete - 100% progress reached", 
                       duration=time.time()-self.start_time)

    def _display_progress_bar(self):
        """Show graphical progress bar in terminals"""
        filled = int(self.bar_length * self.current / self.total)
        bar = '█' * filled + '─' * (self.bar_length - filled)
        percent = min(100, (self.current / self.total) * 100)
        sys.stderr.write(f"\r[{bar}] {percent:.1f}%")
        sys.stderr.flush()

    def _log_progress(self):
        """Show text progress in non-terminal environments"""
        percent = min(100, (self.current / self.total) * 100)
        logger.info(f"Warmup Progress: {percent:.1f}%")

    def reset(self):
        """Reset progress for reuse"""
        self.current = 0
        self.completed = False
        self.start_time = time.time()
        self.last_update = 0

# Add the progress_bar function for backward compatibility
def progress_bar(percentage, bar_length=40):
    """Generate a progress bar string for a given percentage"""
    filled = int(bar_length * percentage / 100)
    bar = '█' * filled + '─' * (bar_length - filled)
    return f"[{bar}] {percentage:.1f}%"