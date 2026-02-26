from src.turbofan.fan_subsystem import FanSubsystem
from src.turbofan.lpc_subsystem import LPCSubsystem
from src.turbofan.hpc_subsystem import HPCSubsystem

def show_map(name, m):
    print(f"\n{name} MAP DOMAIN")
    print(f" speed_vec range : {float(m.speed_vec.min()):.3f} -> {float(m.speed_vec.max()):.3f}")
    print(f" flow_vec  range : {float(m.flow_vec.min()):.3f} -> {float(m.flow_vec.max()):.3f}")

fan = FanSubsystem.from_default_files()
lpc = LPCSubsystem.from_default_files()
hpc = HPCSubsystem.from_default_files()

show_map("FAN", fan.fan_map)
show_map("LPC", lpc.lpc_map)
show_map("HPC", hpc.hpc_map)