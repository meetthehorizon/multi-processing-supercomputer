import time

for i in range(50):
    sec, min, hour = time.localtime()[5:8]
    print(f"{hour:02d}:{min:02d}:{sec:02d}")
    time.sleep(1)
