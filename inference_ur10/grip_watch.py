#!/usr/bin/env python3
"""Watch the 3-Finger's raw status while commanding a move.

Tells you whether gFLT=0x05 is a transient that clears (expected) or a
persistent state (means the gripper genuinely is not accepting motion).

    python3 grip_watch.py <gripper-ip> [mode]

Does NOT move the robot arm. The gripper fingers WILL move -- clear them first.
"""
import sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "./inference_ur10")

from Robotiq3FGripper import Robotiq3FGripper

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.11"
MODE = sys.argv[2] if len(sys.argv) > 2 else "pinch"

FLT = {0x00: "no fault", 0x05: "PRIORITY: action delayed, init must finish",
       0x06: "PRIORITY: activation bit must be set first",
       0x07: "PRIORITY: mode out of range / not allowed",
       0x09: "MINOR: comms chip not ready (booting)",
       0x0A: "MINOR: mode-change interference on scissor (<20s)",
       0x0B: "MINOR: automatic release in progress",
       0x0D: "MAJOR: activation fault -- check for interference",
       0x0E: "MAJOR: mode-change interference on scissor (>20s)",
       0x0F: "MAJOR: automatic release complete -- reset required"}

def dump(tag, st):
    print(f"  {tag:>7} | gFLT=0x{st.gFLT:02X} gACT={st.gACT} gMOD={st.mode_name:<8}"
          f" gGTO={st.gGTO} gIMC={st.gIMC} gSTA={st.gSTA}"
          f" pos={st.pos} DT=({st.gDTA},{st.gDTB},{st.gDTC})"
          f"  {FLT.get(st.gFLT,'?')}")

print(f"connecting to {IP}, mode={MODE}")
g = Robotiq3FGripper(IP, mode=MODE, poll_hz=25)
dump("preinit", g.get_status())

print("\n--- activating ---")
try:
    g.activate()
except Exception as e:
    print("  activate raised:", e)
dump("post-act", g.get_status())

for target, label in [(255, "CLOSE"), (0, "OPEN")]:
    print(f"\n--- commanding {label} (rPRA={target}), watching 3 s at 20 Hz ---")
    g._last_sent = None
    g.move(target)
    t0 = time.monotonic()
    faults, prev = [], None
    while time.monotonic() - t0 < 3.0:
        st = g.get_status()
        key = (st.gFLT, st.gSTA, st.gGTO, st.gIMC, st.pos[0])
        if key != prev:                      # only print on change
            dump(f"{time.monotonic()-t0:5.2f}s", st)
            prev = key
        if st.gFLT: faults.append(st.gFLT)
        time.sleep(0.05)

    st = g.get_status()
    if not faults:
        print(f"  -> {label} clean, no fault at any point. gPOA={st.pos[0]}")
    elif st.gFLT == 0:
        print(f"  -> fault 0x{faults[0]:02X} appeared and CLEARED on its own "
              f"({len(faults)} samples). Transient: the grace-period fix is correct.")
    else:
        print(f"  -> fault 0x{st.gFLT:02X} PERSISTED for the whole 3 s. NOT transient --")
        print("     the gripper is refusing motion. Check gIMC (needs 3) and gGTO (needs 1).")

print()
g.close()