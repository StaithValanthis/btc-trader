from app.utils.progress import ProgressBar
import time

progress_bar = ProgressBar(total=100)
for i in range(101):
    progress_bar.update(i)
    time.sleep(0.1)
progress_bar.clear()
print("Progress bar test complete!")