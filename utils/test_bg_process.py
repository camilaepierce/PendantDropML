import time

with open("bg_output.txt", "a") as f:
    start_time = time.time()
    f.write(f"Process started at: {start_time}\n")

for i in range(100):
    with open("bg_output.txt", "a") as f:
        f.write(f"Period {i}\n")
    time.sleep(1)

end_time = time.time()
with open("bg_output.txt", "a") as f:
    f.write(f"Process ended at: {end_time}. Took {end_time-start_time} seconds.\n")