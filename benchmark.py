"""Reproducible standard-library benchmark for 2D and flattened MAC."""

import sys

import main


def run_benchmarks(sizes=None):
    sizes = sizes or (3, 5, 13, 25, 1000)
    print("Benchmark settings: repeat={}, warmup={}".format(main.REPEAT, main.WARMUP))
    rows = []
    for size in sizes:
        pattern, filt = main.make_x(size), main.make_cross(size)
        flat_pattern, flat_filter = main.flatten(pattern), main.flatten(filt)
        score_2d = main.mac(pattern, filt)
        score_flat = main.mac_flat(flat_pattern, flat_filter)
        if abs(score_2d - score_flat) >= main.EPSILON:
            raise AssertionError("MAC versions disagree at size {}".format(size))
        time_2d = main.measure_ms(main.mac, pattern, filt)
        time_flat = main.measure_ms(main.mac_flat, flat_pattern, flat_filter)
        change = ((time_flat / time_2d) - 1.0) * 100.0 if time_2d else 0.0
        rows.append(("{}×{}".format(size, size), str(size * size),
                     "{:.6f}".format(time_2d), "{:.6f}".format(time_flat),
                     "{:+.1f}%".format(change)))
    main.print_table(("크기", "연산 횟수(N²)", "2차원(ms)", "1차원(ms)", "변화율"), rows)
    print("비고: 평탄화 변환은 MAC 측정 구간에서 제외했습니다. 실제 측정값은 기기 부하에 따라 달라집니다.")


if __name__ == "__main__":
    chosen_sizes = tuple(int(value) for value in sys.argv[1:]) if len(sys.argv) > 1 else None
    run_benchmarks(chosen_sizes)
