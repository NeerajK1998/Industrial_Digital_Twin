from src.turbofan.combustor import combustor_calc

def main():
    out = combustor_calc(
        P3=1.5e6,
        T3=700.0,
        m_air=40.0,
        throttle_cmd=0.6,
    )
    print(out)

if __name__ == "__main__":
    main()