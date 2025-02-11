from app.utils.progress import ProgressBar
import time

def test_progress_bar():
    pb = ProgressBar(total=100)
    for i in range(101):
        pb.update(i)
        time.sleep(0.01)
    pb.clear()
    print("Progress bar test complete!")
