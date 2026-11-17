"""Mini NPU Simulator: loop-based MAC pattern classifier (Python 3.8+)."""

import json
import math
import os
import re
import time

EPSILON = 1e-9
REPEATS = 10


def normalize_label(value):
    """Convert documented label spellings to Cross or X; preserve unknowns."""
    if not isinstance(value, str):
        return None
    label = value.strip().casefold()
    if label in ("+", "cross"):
        return "Cross"
    if label == "x":
        return "X"
    return None


def mac(pattern, filter_matrix):
    """Return sum(pattern[r][c] * filter_matrix[r][c]) using loops only."""
    rows = len(pattern)
    if rows == 0 or rows != len(filter_matrix):
        raise ValueError("패턴과 필터의 행 크기가 일치하지 않습니다.")
    total = 0.0
    for r in range(rows):
        if not isinstance(pattern[r], list) or not isinstance(filter_matrix[r], list):
            raise ValueError("각 행은 숫자 목록이어야 합니다.")
        if len(pattern[r]) != rows or len(filter_matrix[r]) != rows:
            raise ValueError("패턴과 필터는 같은 N×N 정사각형이어야 합니다.")
        for c in range(rows):
            try:
                a, b = float(pattern[r][c]), float(filter_matrix[r][c])
            except (TypeError, ValueError):
                raise ValueError("배열에는 숫자만 들어갈 수 있습니다.")
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("배열에는 유한한 숫자만 들어갈 수 있습니다.")
            total += a * b
    return total


def decide(score_cross, score_x):
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def read_matrix(label, n):
    print("{} ({}줄 입력, 각 줄에 숫자 {}개)".format(label, n, n))
    while True:
        matrix = []
        for row in range(n):
            while True:
                raw = input("{}행: ".format(row + 1)).strip().split()
                if len(raw) != n:
                    print("입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(n))
                    continue
                try:
                    values = [float(item) for item in raw]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError
                    matrix.append(values)
                    break
                except ValueError:
                    print("입력 형식 오류: 숫자만 입력하세요. 다시 입력해 주세요.")
        return matrix


def timed_average(pattern, filter_matrix, repeats=REPEATS):
    start = time.perf_counter()
    for _ in range(repeats):
        mac(pattern, filter_matrix)
    return (time.perf_counter() - start) * 1000.0 / repeats


def performance_table(matrices=None):
    print("\n[성능 분석: 평균/{}회, MAC 함수 호출 시간]".format(REPEATS))
    print("크기(N×N)\t평균 시간(ms)\t연산 횟수(N²)")
    for n in (3, 5, 13, 25):
        if matrices and n in matrices:
            pattern, filt = matrices[n]
        else:
            # Deterministic timing input only; never used as mission test data.
            pattern = [[0.0] * n for _ in range(n)]
            filt = [[0.0] * n for _ in range(n)]
        elapsed = timed_average(pattern, filt)
        print("{}×{}\t\t{:.6f}\t\t{}".format(n, n, elapsed, n * n))


def user_mode():
    print("\n=== 사용자 입력 (3×3) ===")
    filter_a = read_matrix("필터 A", 3)
    filter_b = read_matrix("필터 B", 3)
    print("필터 A와 B를 저장했습니다.")
    pattern = read_matrix("패턴", 3)
    start = time.perf_counter()
    score_a, score_b = mac(pattern, filter_a), mac(pattern, filter_b)
    elapsed = (time.perf_counter() - start) * 1000.0
    print("A 점수: {}\nB 점수: {}".format(score_a, score_b))
    print("연산 시간: {:.6f} ms\n판정: {}".format(elapsed, decide(score_a, score_b)))
    performance_table({3: (pattern, filter_a)})


def extract_filter_pair(raw_pair):
    if not isinstance(raw_pair, dict):
        raise ValueError("필터는 Cross/cross/+ 및 X/x 키를 가진 객체여야 합니다.")
    pair = {}
    for label, matrix in raw_pair.items():
        standard = normalize_label(label)
        if standard:
            if standard in pair:
                raise ValueError("같은 표준 라벨의 필터 키가 중복되었습니다: {}".format(standard))
            pair[standard] = matrix
    if set(pair) != {"Cross", "X"}:
        raise ValueError("Cross와 X 필터가 모두 필요합니다.")
    return pair


def json_mode():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    if not os.path.isfile(path):
        print("data.json 파일이 없습니다: {}".format(path))
        print("미션에서 제공한 data.json을 main.py와 같은 폴더에 넣고 다시 실행하세요.")
        return
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print("data.json을 읽지 못했습니다: {}".format(exc))
        return
    if not isinstance(data, dict) or not isinstance(data.get("filters"), dict) or not isinstance(data.get("patterns"), dict):
        print("전체 FAIL: 최상위에 filters와 patterns 객체가 필요합니다.")
        return

    filters = {}
    for size_key in ("size_5", "size_13", "size_25"):
        try:
            filters[size_key] = extract_filter_pair(data["filters"][size_key])
            print("필터 로드 완료: {} (Cross, X)".format(size_key))
        except (KeyError, TypeError, ValueError) as exc:
            print("필터 로드 오류 {}: {} (해당 크기 케이스는 FAIL 처리)".format(size_key, exc))

    total = passed = 0
    failures = []
    timing_inputs = {}
    for case_id, case in data["patterns"].items():
        total += 1
        try:
            match = re.match(r"^size_(\d+)_", str(case_id))
            if not match:
                raise ValueError("케이스 키에서 크기 N을 읽을 수 없습니다.")
            n = int(match.group(1))
            size_key = "size_{}".format(n)
            if size_key not in filters:
                raise ValueError("{} 필터가 없거나 올바르지 않습니다.".format(size_key))
            if not isinstance(case, dict) or "input" not in case or "expected" not in case:
                raise ValueError("케이스에 input과 expected가 필요합니다.")
            pattern = case["input"]
            pair = filters[size_key]
            cross_score = mac(pattern, pair["Cross"])
            x_score = mac(pattern, pair["X"])
            result = decide(cross_score, x_score)
            expected = normalize_label(case["expected"])
            if expected is None:
                raise ValueError("expected 라벨을 Cross/X로 정규화할 수 없습니다.")
            ok = result == expected
            print("\n{} | Cross: {} | X: {} | 판정: {} | expected: {} | {}".format(
                case_id, cross_score, x_score, result, expected, "PASS" if ok else "FAIL"))
            if ok:
                passed += 1
            else:
                reason = "epsilon 동점으로 UNDECIDED" if result == "UNDECIDED" else "판정 결과와 expected 불일치"
                failures.append((case_id, reason))
            timing_inputs.setdefault(n, (pattern, pair["Cross"]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            print("\n{} | FAIL | {}".format(case_id, exc))
            failures.append((case_id, str(exc)))

    performance_table(timing_inputs)
    print("\n[결과 요약] 총 테스트: {} | 통과: {} | 실패: {}".format(total, passed, total - passed))
    if failures:
        print("실패 케이스:")
        for case_id, reason in failures:
            print("- {}: {}".format(case_id, reason))
    elif total == 0:
        print("patterns에 테스트 케이스가 없습니다.")


def main():
    print("=== Mini NPU Simulator ===")
    print("1. 사용자 입력 (3×3)\n2. data.json 분석")
    while True:
        choice = input("선택: ").strip()
        if choice == "1":
            user_mode()
            break
        if choice == "2":
            json_mode()
            break
        print("1 또는 2를 입력해 주세요.")


if __name__ == "__main__":
    main()
