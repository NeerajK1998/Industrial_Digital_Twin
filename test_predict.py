from src.turbofan.turbofan_runner import run_turbofan_core_balanced
from ml.predict_live import load_model, predict_from_out

bundle = load_model()

out = run_turbofan_core_balanced(
    throttle_cmd=0.65,
    eff_mod_fan=1.0,
    eff_mod_lpc=1.0,
    eff_mod_hpc=1.0,
    eta_hpt=0.92,
    eta_lpt=0.92,
    BPR=8.0,
)

print(predict_from_out(out, bundle))