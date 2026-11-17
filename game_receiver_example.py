"""What the game side looks like: read the stress value with stress_api.StressClient.

Start stress_cam.py (or stress_simulator.py), then run this in a second terminal.
A real game calls stress.update() once per frame and maps stress.value onto a mechanic.
"""

import time

from stress_api import StressClient

stress = StressClient()  # listens on 127.0.0.1:5005
print("Listening on 127.0.0.1:5005 (Ctrl+C to stop) ...")
try:
    while True:  # stand-in for the game loop
        stress.update()
        jump_power = 1.0 + 0.5 * stress.value  # e.g. jump up to 50% higher when the player is stressed
        if not stress.connected:
            state = "no detector running"
        elif stress.calibrating:
            state = "calibrating"
        else:
            state = f"stress {stress.value:.2f} {stress.level}"
        print(f"{state:<24} -> jump x{jump_power:.2f}   ", end="\r")
        time.sleep(1 / 30)
except KeyboardInterrupt:
    print()
