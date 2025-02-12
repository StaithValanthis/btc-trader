# tests/test_progress.py
from app.utils.progress import ProgressBar
import time

def test_progress_bar():
    progress_bar = ProgressBar(total=100)
    for i in range(101):
        progress_bar.update(i)
        time.sleep(0.1)
    progress_bar.clear()
    print("Progress bar test complete!")