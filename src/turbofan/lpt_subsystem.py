from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from .lpt import lpt_core_calc, TurbineGasProps


@dataclass
class LPTSubsystem:
    gas: TurbineGasProps = TurbineGasProps()

    def step_with_pr(
        self,
        P_in: float,
        T_in: float,
        m_gas: float,
        N1_RPM: float,
        PR_turb: float,
        eta_turb: float = 0.9,
    ) -> Dict[str, float]:
        P_out, T_out, torque = lpt_core_calc(
            P_in=P_in,
            T_in=T_in,
            m_gas=m_gas,
            PR_turb=PR_turb,
            eta_turb=eta_turb,
            N1_RPM=N1_RPM,
            gas=self.gas,
        )
        return {
            "P_out": float(P_out),
            "T_out": float(T_out),
            "Torque_LPT": float(torque),
            "PR_LPT": float(PR_turb),
        }

    def solve_for_balance(
        self,
        P_in: float,
        T_in: float,
        m_gas: float,
        N1_RPM: float,
        torque_required: float,
        eta_turb: float = 0.9,
        PR_min: float = 0.1,
        PR_max: float = 0.95,
        tol: float = 1e-2,
        max_iter: int = 60,
    ) -> Dict[str, float]:

        def torque_diff(PR: float) -> float:
            _P, _T, torque = lpt_core_calc(
                P_in=P_in,
                T_in=T_in,
                m_gas=m_gas,
                PR_turb=PR,
                eta_turb=eta_turb,
                N1_RPM=N1_RPM,
                gas=self.gas,
            )
            return float(torque) - float(torque_required)

        low = float(PR_min)
        high = float(PR_max)
        f_low = torque_diff(low)
        f_high = torque_diff(high)

        if f_low * f_high > 0:
            raise RuntimeError(
                f"LPT torque balance not bracketed: f(PR_min)={f_low:.3f}, f(PR_max)={f_high:.3f}. "
                "Adjust PR_min/PR_max."
            )

        mid = 0.5 * (low + high)
        for _ in range(int(max_iter)):
            mid = 0.5 * (low + high)
            f_mid = torque_diff(mid)

            if abs(f_mid) < float(tol):
                break

            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid

        P_out, T_out, torque = lpt_core_calc(
            P_in=P_in,
            T_in=T_in,
            m_gas=m_gas,
            PR_turb=mid,
            eta_turb=eta_turb,
            N1_RPM=N1_RPM,
            gas=self.gas,
        )

        return {
            "P_out": float(P_out),
            "T_out": float(T_out),
            "Torque_LPT": float(torque),
            "PR_LPT": float(mid),
        }