from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .hpt import hpt_core_calc, TurbineGasProps


@dataclass
class HPTSubsystem:
    gas: TurbineGasProps = TurbineGasProps()

    def step_with_pr(
        self,
        P4: float,
        T4: float,
        m_gas: float,
        N2_RPM: float,
        PR_turb: float,
        eta_turb: float = 0.9,
    ) -> Dict[str, float]:
        P45, T45, torque = hpt_core_calc(
            P4=P4,
            T4=T4,
            m_gas=m_gas,
            PR_turb=PR_turb,
            eta_turb=eta_turb,
            N2_RPM=N2_RPM,
            gas=self.gas,
        )
        return {
            "P45": float(P45),
            "T45": float(T45),
            "Torque_HPT": float(torque),
            "PR_HPT": float(PR_turb),
        }

    def solve_for_balance(
        self,
        P4: float,
        T4: float,
        m_gas: float,
        N2_RPM: float,
        torque_required: float,
        eta_turb: float = 0.9,
        PR_min: float = 0.1,
        PR_max: float = 0.9,
        tol: float = 1e-2,
        max_iter: int = 60,
    ) -> Dict[str, float]:
        """
        Find PR_turb in [PR_min, PR_max] such that:
            Torque_HPT(PR) - torque_required = 0
        using bisection.
        """

        def torque_diff(PR: float) -> float:
            _P45, _T45, torque = hpt_core_calc(
                P4=P4,
                T4=T4,
                m_gas=m_gas,
                PR_turb=PR,
                eta_turb=eta_turb,
                N2_RPM=N2_RPM,
                gas=self.gas,
            )
            return float(torque) - float(torque_required)

        low = float(PR_min)
        high = float(PR_max)
        f_low = torque_diff(low)
        f_high = torque_diff(high)

        if f_low * f_high > 0:
            raise RuntimeError(
                f"Torque balance not bracketed: f(PR_min)={f_low:.3f}, f(PR_max)={f_high:.3f}. "
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

        P45, T45, torque = hpt_core_calc(
            P4=P4,
            T4=T4,
            m_gas=m_gas,
            PR_turb=mid,
            eta_turb=eta_turb,
            N2_RPM=N2_RPM,
            gas=self.gas,
        )

        return {
            "P45": float(P45),
            "T45": float(T45),
            "Torque_HPT": float(torque),
            "PR_HPT": float(mid),
        }
