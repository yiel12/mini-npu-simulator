"""Mini NPU Simulator: beginner-friendly, loop-based MAC demo (Python 3.8+)."""

import json
import math
import os
import re
import time
import unicodedata

EPSILON = 1e-9
REPEAT = 10
WARMUP = 1
PERF_SIZES = (3, 5, 13, 25)
LABELS = ("Cross", "X")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_label(value):
    """Map label aliases in one reusable place for consistent parsing and testing."""
    if not isinstance(value, str):
        return None
    label = value.strip().casefold()
    if label in ("+", "cross", "plus"):
        return "Cross"
    if label == "x":
        return "X"
    return None


def parse_size(key, pattern_key=False):
    """Return N from size_N or size_N_idx, rejecting malformed/non-positive sizes."""
    expression = r"^size_(\d+)_\d+$" if pattern_key else r"^size_(\d+)$"
    match = re.match(expression, str(key))
    if not match:
        raise ValueError("크기 키 형식이 올바르지 않습니다: {}".format(key))
    size = int(match.group(1))
    if size < 1:
        raise ValueError("크기 N은 1 이상이어야 합니다.")
    return size


def validate_matrix(matrix, expected_size=None, name="배열", check_values=True):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("{}는 비어 있지 않은 2차원 배열이어야 합니다.".format(name))
    size = len(matrix)
    if expected_size is not None and size != expected_size:
        raise ValueError("{} 행 크기 {}가 기대 크기 {}와 다릅니다.".format(name, size, expected_size))
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("{}는 N×N 정사각형이어야 합니다.".format(name))
        if check_values:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("{}에는 숫자만 들어갈 수 있습니다.".format(name))
                if not math.isfinite(float(value)):
                    raise ValueError("{}에는 유한한 숫자만 들어갈 수 있습니다.".format(name))
    return size


def mac(pattern, filter_matrix):
    """Calculate a 2D MAC using nested loops and per-value validation."""
    size = validate_matrix(pattern, name="패턴", check_values=False)
    validate_matrix(filter_matrix, size, "필터", check_values=False)
    total = 0.0
    for row in range(size):
        for col in range(size):
            left, right = pattern[row][col], filter_matrix[row][col]
            if (isinstance(left, bool) or isinstance(right, bool)
                    or not isinstance(left, (int, float))
                    or not isinstance(right, (int, float))):
                raise ValueError("배열에는 숫자만 들어갈 수 있습니다.")
            left, right = float(left), float(right)
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("배열에는 유한한 숫자만 들어갈 수 있습니다.")
            total += left * right
    return float(total)


def flatten(matrix):
    size = validate_matrix(matrix)
    return [matrix[row][col] for row in range(size) for col in range(size)]


def mac_flat(pattern_flat, filter_flat):
    """Calculate a MAC over pre-flattened lists (conversion is outside timing)."""
    if not pattern_flat or len(pattern_flat) != len(filter_flat):
        raise ValueError("평탄화한 패턴과 필터의 길이가 같고 비어 있지 않아야 합니다.")
    total = 0.0
    for index in range(len(pattern_flat)):
        left, right = pattern_flat[index], filter_flat[index]
        if (isinstance(left, bool) or isinstance(right, bool)
                or not isinstance(left, (int, float))
                or not isinstance(right, (int, float))):
            raise ValueError("배열에는 숫자만 들어갈 수 있습니다.")
        left, right = float(left), float(right)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("배열에는 유한한 숫자만 들어갈 수 있습니다.")
        total += left * right
    return float(total)


def decide(score_cross, score_x):
    if abs(score_cross - score_x) < EPSILON:
        return "UNDECIDED"
    return "Cross" if score_cross > score_x else "X"


def make_cross(size):
    if size < 1:
        raise ValueError("크기는 1 이상이어야 합니다.")
    center = size // 2
    return [[1.0 if row == center or col == center else 0.0
             for col in range(size)] for row in range(size)]


def make_x(size):
    if size < 1:
        raise ValueError("크기는 1 이상이어야 합니다.")
    return [[1.0 if row == col or row + col == size - 1 else 0.0
             for col in range(size)] for row in range(size)]


def measure_ms(function, left, right, repeat=REPEAT, warmup=WARMUP):
    """Measure only the supplied function calls; I/O and preparation stay outside."""
    for _ in range(warmup):
        function(left, right)
    start = time.perf_counter()
    for _ in range(repeat):
        function(left, right)
    return (time.perf_counter() - start) * 1000.0 / repeat


def display_width(value):
    width = 0
    for char in str(value):
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def print_table(headers, rows):
    widths = [max(display_width(headers[i]), *(display_width(row[i]) for row in rows))
              for i in range(len(headers))]

    def line(values):
        print("  ".join(str(value) + " " * (widths[i] - display_width(value))
                         for i, value in enumerate(values)))

    line(headers)
    line(["-" * width for width in widths])
    for row in rows:
        line(row)


def read_line(prompt):
    try:
        return input(prompt)
    except EOFError:
        print("\n입력이 종료되어 현재 작업을 취소합니다.")
        return None


def read_matrix(label, size):
    print("{} ({}줄, 각 줄에 숫자 {}개)".format(label, size, size))
    matrix = []
    for row_index in range(size):
        while True:
            raw = read_line("{}행: ".format(row_index + 1))
            if raw is None:
                return None
            tokens = raw.strip().split()
            if len(tokens) != size:
                print("입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(size))
                print("{}번째 줄을 다시 입력하세요.".format(row_index + 1))
                continue
            try:
                values = [float(token) for token in tokens]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("유한하지 않은 값")
            except ValueError:
                print("입력 형식 오류: 유한한 숫자만 입력하세요.")
                print("{}번째 줄을 다시 입력하세요.".format(row_index + 1))
                continue
            matrix.append(values)
            break
    return matrix


def user_mode():
    print("\n[모드 1] 3×3 필터 비교")
    while True:
        choice = read_line("필터 입력 방식: 1. 직접 입력  2. Cross/X 자동 생성: ")
        if choice is None:
            return
        if choice.strip() in ("1", "2"):
            break
        print("1 또는 2를 선택해 주세요.")
    if choice.strip() == "1":
        filter_a = read_matrix("필터 A", 3)
        if filter_a is None:
            return
        filter_b = read_matrix("필터 B", 3)
        if filter_b is None:
            return
    else:
        filter_a, filter_b = make_cross(3), make_x(3)
        print("생성 필터를 적용했습니다: A=Cross, B=X")
    print("필터 A와 B를 저장했습니다.")
    pattern = read_matrix("패턴", 3)
    if pattern is None:
        return
    start = time.perf_counter()
    score_a, score_b = mac(pattern, filter_a), mac(pattern, filter_b)
    elapsed = (time.perf_counter() - start) * 1000.0
    result = decide(score_a, score_b)
    display = ("판정 불가 (|A-B| < {})".format(EPSILON) if result == "UNDECIDED"
               else ("A" if result == "Cross" else "B"))
    print("A 점수: {}\nB 점수: {}".format(score_a, score_b))
    print("연산 시간: {:.6f} ms\n판정: {}".format(elapsed, display))
    performance_table({3: (pattern, filter_a)})


def extract_filter_pair(raw_pair):
    if not isinstance(raw_pair, dict):
        raise ValueError("필터는 cross/x 라벨을 가진 객체여야 합니다.")
    pair = {}
    for label, matrix in raw_pair.items():
        standard = normalize_label(label)
        if standard:
            if standard in pair:
                raise ValueError("표준 라벨 {}의 필터 키가 중복되었습니다.".format(standard))
            pair[standard] = matrix
    if set(pair) != set(LABELS):
        raise ValueError("Cross와 X 필터가 모두 필요합니다.")
    return pair


def load_filters(data):
    filters = {}
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        raise ValueError("최상위 filters 객체가 필요합니다.")
    for key, raw_pair in raw_filters.items():
        try:
            size = parse_size(key)
            pair = extract_filter_pair(raw_pair)
            for label, matrix in pair.items():
                validate_matrix(matrix, size, "{} {} 필터".format(key, label))
            filters[size] = pair
            print("필터 로드 완료: {} (Cross, X)".format(key))
        except (TypeError, ValueError) as exc:
            print("필터 로드 오류 {}: {}".format(key, exc))
    return filters


def check_case(case_id, case, filters):
    size = parse_size(case_id, pattern_key=True)
    if size not in filters:
        raise ValueError("size_{} 필터가 없거나 올바르지 않습니다.".format(size))
    if not isinstance(case, dict) or "input" not in case or "expected" not in case:
        raise ValueError("케이스에 input과 expected가 필요합니다.")
    pattern = case["input"]
    validate_matrix(pattern, size, "{} 입력".format(case_id))
    expected = normalize_label(case["expected"])
    if expected is None:
        raise ValueError("expected 라벨을 Cross/X로 정규화할 수 없습니다.")
    pair = filters[size]
    score_cross = mac(pattern, pair["Cross"])
    score_x = mac(pattern, pair["X"])
    result = decide(score_cross, score_x)
    ok = result == expected
    delta = abs(score_cross - score_x)
    if ok:
        category, reason = "pass", ""
    elif result == "UNDECIDED":
        category, reason = "undecided", "epsilon 동점 (차이 {:.3e})".format(delta)
    else:
        category, reason = "mismatch", "판정과 expected 불일치 (차이 {:.3e})".format(delta)
    return {"id": case_id, "size": size, "cross": score_cross, "x": score_x,
            "result": result, "expected": expected, "ok": ok,
            "category": category, "reason": reason, "pattern": pattern,
            "filter": pair["Cross"]}


def performance_table(actual_inputs=None):
    print("\n[성능 분석: 평균/{}회, 워밍업 {}회]".format(REPEAT, WARMUP))
    rows = []
    for size in PERF_SIZES:
        pattern, filt = (actual_inputs[size] if actual_inputs and size in actual_inputs
                         else (make_x(size), make_cross(size)))
        elapsed = measure_ms(mac, pattern, filt)
        rows.append(("{}×{}".format(size, size), "{:.6f}".format(elapsed), str(size * size)))
    print_table(("크기(N×N)", "평균 시간(ms)", "연산 횟수(N²)"), rows)


def optimization_table(actual_inputs=None):
    print("\n[최적화 비교: 평탄화 전처리 시간 제외]")
    rows = []
    for size in PERF_SIZES:
        pattern, filt = (actual_inputs[size] if actual_inputs and size in actual_inputs
                         else (make_x(size), make_cross(size)))
        flat_pattern, flat_filter = flatten(pattern), flatten(filt)
        two_d_ms = measure_ms(mac, pattern, filt)
        flat_ms = measure_ms(mac_flat, flat_pattern, flat_filter)
        change = ((flat_ms / two_d_ms) - 1.0) * 100.0 if two_d_ms else 0.0
        rows.append(("{}×{}".format(size, size), "{:.6f}".format(two_d_ms),
                     "{:.6f}".format(flat_ms), "{:+.1f}%".format(change)))
    print_table(("크기", "2차원(ms)", "1차원(ms)", "변화율"), rows)


def run_generator(size):
    cross, x_pattern = make_cross(size), make_x(size)
    print("\n[생성 패턴으로 MAC 판정]")
    rows = []
    for name, pattern, expected in (("Cross", cross, "Cross"), ("X", x_pattern, "X")):
        a, b = mac(pattern, cross), mac(pattern, x_pattern)
        result = decide(a, b)
        rows.append((name, "{:.6g}".format(a), "{:.6g}".format(b), result,
                     "일치" if result == expected else "불일치"))
    print_table(("패턴", "Cross 점수", "X 점수", "판정", "자가 검사"), rows)
    if size <= 25:
        print("\nCross 패턴:")
        for row in cross:
            print(" ".join("1" if value else "0" for value in row))
        print("X 패턴:")
        for row in x_pattern:
            print(" ".join("1" if value else "0" for value in row))
    else:
        print("N>25이므로 출력이 지나치게 길어지지 않게 행렬 대신 1인 칸 수만 표시합니다.")
        print("Cross의 1 개수: {} | X의 1 개수: {}".format(
            sum(map(sum, cross)), sum(map(sum, x_pattern))))
    if size in PERF_SIZES:
        performance_table({size: (x_pattern, cross)})
        optimization_table({size: (x_pattern, cross)})
    else:
        elapsed = measure_ms(mac, x_pattern, cross)
        print("\n[입력 크기 성능] {}×{} | 평균 {:.6f} ms | MAC {}회".format(
            size, size, elapsed, size * size))
        flat_x, flat_cross = flatten(x_pattern), flatten(cross)
        flat_elapsed = measure_ms(mac_flat, flat_x, flat_cross)
        print("[평탄화 비교] 2차원 {:.6f} ms | 1차원 {:.6f} ms (변환 시간 제외)".format(
            elapsed, flat_elapsed))


def generator_mode():
    while True:
        raw = read_line("크기 N (3~1000): ")
        if raw is None:
            return
        try:
            size = int(raw.strip())
            if 3 <= size <= 1000:
                break
        except ValueError:
            pass
        print("3부터 1000까지의 정수를 입력하세요.")
    run_generator(size)


def json_mode():
    path = os.path.join(BASE_DIR, "data.json")
    print("\n[모드 2] JSON 분석\n데이터 파일: {}".format(path))
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("data.json 파일이 없습니다. main.py와 같은 폴더에 넣고 다시 실행하세요.")
        return
    except (OSError, json.JSONDecodeError) as exc:
        print("data.json을 읽지 못했습니다: {}".format(exc))
        return
    if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
        print("전체 FAIL: 최상위 patterns 객체가 필요합니다.")
        return

    print("\n[1] 필터 로드")
    try:
        filters = load_filters(data)
    except ValueError as exc:
        print("필터 스키마 오류: {}".format(exc))
        filters = {}

    print("\n[2] 패턴 분석")
    results, failures, timing_inputs = [], [], {}
    counts = {"pass": 0, "undecided": 0, "mismatch": 0, "schema": 0}
    for case_id, case in data["patterns"].items():
        try:
            result = check_case(case_id, case, filters)
            results.append(result)
            counts[result["category"]] += 1
            timing_inputs.setdefault(result["size"], (result["pattern"], result["filter"]))
            status = "PASS" if result["ok"] else "FAIL"
            print("{} | Cross {} | X {} | 판정 {} | expected {} | {}".format(
                case_id, result["cross"], result["x"], result["result"], result["expected"], status))
            if not result["ok"]:
                failures.append((case_id, result["reason"]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            counts["schema"] += 1
            failures.append((case_id, "데이터/스키마 오류: {}".format(exc)))
            print("{} | FAIL | 데이터/스키마 오류: {}".format(case_id, exc))

    print("\n[3] 성능 분석")
    performance_table(timing_inputs)
    print("\n[4] 최적화 비교")
    optimization_table(timing_inputs)
    total, passed = len(data["patterns"]), counts["pass"]
    print("\n[5] 결과 요약")
    print("총 테스트: {}개 | 통과: {}개 | 실패: {}개".format(total, passed, total - passed))
    print("  - 동점(UNDECIDED): {}개".format(counts["undecided"]))
    print("  - 판정/expected 불일치: {}개".format(counts["mismatch"]))
    print("  - 데이터/스키마 오류: {}개".format(counts["schema"]))
    if failures:
        print("실패 케이스:")
        for case_id, reason in failures:
            print("- {}: {}".format(case_id, reason))


def main():
    print("=== Mini NPU Simulator ===")
    while True:
        print("\n1. 사용자 입력 (3×3)\n2. data.json 분석\n3. 패턴 생성기\n4. 종료")
        choice = read_line("선택: ")
        if choice is None or choice.strip() == "4":
            print("프로그램을 종료합니다.")
            return
        if choice.strip() == "1":
            user_mode()
        elif choice.strip() == "2":
            json_mode()
        elif choice.strip() == "3":
            generator_mode()
        else:
            print("1, 2, 3, 4 중에서 선택해 주세요.")


if __name__ == "__main__":
    main()
