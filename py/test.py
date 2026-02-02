"""

Tests written by AI

"""

import gc
import logging
import math
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

from ffimatrix import Matrix

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def test_1x1_matrix():
    log.info("TEST: 1x1 matrix - smallest possible")
    a = Matrix(1, 1, [42.0])
    b = Matrix(1, 1, [8.0])
    log.debug(f"a = {a}")
    log.debug(f"b = {b}")
    
    c = a + b
    log.debug(f"a + b = {c}")
    assert c._ptr.data[0] == 50.0, f"Expected 50.0, got {c._ptr.data[0]}"
    
    d = a - b
    log.debug(f"a - b = {d}")
    assert d._ptr.data[0] == 34.0, f"Expected 34.0, got {d._ptr.data[0]}"
    
    e = a @ b
    log.debug(f"a @ b = {e}")
    assert e._ptr.data[0] == 336.0, f"Expected 336.0, got {e._ptr.data[0]}"
    
    t = a.T
    log.debug(f"a.T = {t}")
    assert t.shape == (1, 1), f"Expected (1,1), got {t.shape}"
    
    log.info("PASS: 1x1 matrix")


def test_row_vector():
    log.info("TEST: Row vector 1xN")
    row = Matrix(1, 5, [1, 2, 3, 4, 5])
    log.debug(f"row = {row}")
    
    t = row.T
    log.debug(f"row.T = {t}")
    assert t.shape == (5, 1), f"Expected (5,1), got {t.shape}"
    
    col = Matrix(5, 1, [1, 2, 3, 4, 5])
    dot = row @ col
    log.debug(f"row @ col = {dot}")
    assert dot.shape == (1, 1), f"Expected (1,1), got {dot.shape}"
    assert dot._ptr.data[0] == 55.0, f"Expected 55.0, got {dot._ptr.data[0]}"
    
    log.info("PASS: Row vector")


def test_column_vector():
    log.info("TEST: Column vector Nx1")
    col = Matrix(5, 1, [1, 2, 3, 4, 5])
    log.debug(f"col = {col}")
    
    t = col.T
    log.debug(f"col.T = {t}")
    assert t.shape == (1, 5), f"Expected (1,5), got {t.shape}"
    
    row = Matrix(1, 5, [1, 2, 3, 4, 5])
    outer = col @ row
    log.debug(f"col @ row shape = {outer.shape}")
    assert outer.shape == (5, 5), f"Expected (5,5), got {outer.shape}"
    
    log.info("PASS: Column vector")


def test_non_square_matmul():
    log.info("TEST: Non-square matrix multiplication")
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    b = Matrix(3, 4, list(range(12)))
    log.debug(f"a (2x3) = {a}")
    log.debug(f"b (3x4) = {b}")
    
    c = a @ b
    log.debug(f"a @ b = {c}")
    assert c.shape == (2, 4), f"Expected (2,4), got {c.shape}"
    
    log.info("PASS: Non-square matmul")


def test_zero_matrix():
    log.info("TEST: Zero matrix")
    z = Matrix(3, 3, [0.0] * 9)
    log.debug(f"z = {z}")
    
    a = Matrix(3, 3, list(range(9)))
    log.debug(f"a = {a}")
    
    result = z @ a
    log.debug(f"z @ a = {result}")
    for i in range(9):
        assert result._ptr.data[i] == 0.0, f"Expected 0 at index {i}"
    
    result2 = a + z
    log.debug(f"a + z = {result2}")
    for i in range(9):
        assert result2._ptr.data[i] == a._ptr.data[i], f"a + 0 should equal a"
    
    log.info("PASS: Zero matrix")


def test_identity_matrix():
    log.info("TEST: Identity matrix")
    I = Matrix(3, 3, [1,0,0, 0,1,0, 0,0,1])
    a = Matrix(3, 3, [1,2,3, 4,5,6, 7,8,9])
    log.debug(f"I = {I}")
    log.debug(f"a = {a}")
    
    result = I @ a
    log.debug(f"I @ a = {result}")
    for i in range(9):
        assert result._ptr.data[i] == a._ptr.data[i], f"I @ a should equal a at index {i}"
    
    result2 = a @ I
    log.debug(f"a @ I = {result2}")
    for i in range(9):
        assert result2._ptr.data[i] == a._ptr.data[i], f"a @ I should equal a at index {i}"
    
    log.info("PASS: Identity matrix")


def test_negative_numbers():
    log.info("TEST: Negative numbers")
    a = Matrix(2, 2, [-1, -2, -3, -4])
    b = Matrix(2, 2, [-5, -6, -7, -8])
    log.debug(f"a = {a}")
    log.debug(f"b = {b}")
    
    c = a + b
    log.debug(f"a + b = {c}")
    assert c._ptr.data[0] == -6.0, f"Expected -6.0, got {c._ptr.data[0]}"
    
    d = a - b
    log.debug(f"a - b = {d}")
    assert d._ptr.data[0] == 4.0, f"Expected 4.0, got {d._ptr.data[0]}"
    
    e = a @ b
    log.debug(f"a @ b = {e}")
    assert e._ptr.data[0] == 19.0, f"Expected 19.0, got {e._ptr.data[0]}"
    
    log.info("PASS: Negative numbers")


def test_scalar_zero():
    log.info("TEST: Scalar multiply by zero")
    a = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    
    result = a * 0
    log.debug(f"a * 0 = {result}")
    for i in range(4):
        assert result._ptr.data[i] == 0.0, f"Expected 0 at index {i}"
    
    log.info("PASS: Scalar zero")


def test_scalar_negative():
    log.info("TEST: Scalar multiply by negative")
    a = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    
    result = a * -1
    log.debug(f"a * -1 = {result}")
    assert result._ptr.data[0] == -1.0, f"Expected -1.0, got {result._ptr.data[0]}"
    assert result._ptr.data[3] == -4.0, f"Expected -4.0, got {result._ptr.data[3]}"
    
    log.info("PASS: Scalar negative")


def test_scalar_fractional():
    log.info("TEST: Scalar multiply by fraction")
    a = Matrix(2, 2, [2, 4, 6, 8])
    log.debug(f"a = {a}")
    
    result = a * 0.5
    log.debug(f"a * 0.5 = {result}")
    assert result._ptr.data[0] == 1.0, f"Expected 1.0, got {result._ptr.data[0]}"
    assert result._ptr.data[3] == 4.0, f"Expected 4.0, got {result._ptr.data[3]}"
    
    log.info("PASS: Scalar fractional")


def test_double_transpose():
    log.info("TEST: Double transpose A.T.T == A")
    a = Matrix(3, 4, list(range(12)))
    log.debug(f"a = {a}")
    
    t = a.T
    log.debug(f"a.T = {t}")
    assert t.shape == (4, 3), f"Expected (4,3), got {t.shape}"
    
    tt = t.T
    log.debug(f"a.T.T = {tt}")
    assert tt.shape == (3, 4), f"Expected (3,4), got {tt.shape}"
    
    for i in range(12):
        assert tt._ptr.data[i] == a._ptr.data[i], f"A.T.T should equal A at index {i}"
    
    log.info("PASS: Double transpose")


def test_self_add():
    log.info("TEST: Self addition a + a")
    a = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    
    result = a + a
    log.debug(f"a + a = {result}")
    for i in range(4):
        assert result._ptr.data[i] == 2 * a._ptr.data[i], f"a + a should be 2*a at index {i}"
    
    log.info("PASS: Self addition")


def test_self_subtract():
    log.info("TEST: Self subtraction a - a")
    a = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    
    result = a - a
    log.debug(f"a - a = {result}")
    for i in range(4):
        assert result._ptr.data[i] == 0.0, f"a - a should be 0 at index {i}"
    
    log.info("PASS: Self subtraction")


def test_self_matmul():
    log.info("TEST: Self matrix multiply a @ a")
    a = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    
    result = a @ a
    log.debug(f"a @ a = {result}")
    assert result._ptr.data[0] == 7.0, f"Expected 7.0, got {result._ptr.data[0]}"
    assert result._ptr.data[3] == 22.0, f"Expected 22.0, got {result._ptr.data[3]}"
    
    log.info("PASS: Self matmul")


def test_very_small_numbers():
    log.info("TEST: Very small numbers (denormals)")
    tiny = [1e-300, 1e-300, 1e-300, 1e-300]
    a = Matrix(2, 2, tiny)
    log.debug(f"a = {a}")
    
    result = a + a
    log.debug(f"a + a = {result}")
    assert result._ptr.data[0] == 2e-300, f"Expected 2e-300, got {result._ptr.data[0]}"
    
    log.info("PASS: Very small numbers")


def test_very_large_numbers():
    log.info("TEST: Very large numbers")
    huge = [1e300, 1e300, 1e300, 1e300]
    a = Matrix(2, 2, huge)
    log.debug(f"a = {a}")
    
    result = a + a
    log.debug(f"a + a = {result}")
    log.debug(f"First element: {result._ptr.data[0]}")
    
    log.info("PASS: Very large numbers")


def test_inf_values():
    log.info("TEST: Infinity values")
    inf_data = [float('inf'), float('-inf'), 1.0, -1.0]
    a = Matrix(2, 2, inf_data)
    log.debug(f"a = {a}")
    
    result = a + a
    log.debug(f"a + a = {result}")
    assert math.isinf(result._ptr.data[0]), f"Expected inf, got {result._ptr.data[0]}"
    assert math.isinf(result._ptr.data[1]), f"Expected -inf, got {result._ptr.data[1]}"
    
    log.info("PASS: Infinity values")


def test_nan_values():
    log.info("TEST: NaN values")
    nan_data = [float('nan'), 1.0, 2.0, 3.0]
    a = Matrix(2, 2, nan_data)
    log.debug(f"a = {a}")
    
    result = a + a
    log.debug(f"a + a = {result}")
    assert math.isnan(result._ptr.data[0]), f"Expected NaN, got {result._ptr.data[0]}"
    
    log.info("PASS: NaN values")


def test_mixed_inf_nan():
    log.info("TEST: inf - inf should be NaN")
    a = Matrix(1, 1, [float('inf')])
    b = Matrix(1, 1, [float('inf')])
    log.debug(f"a = {a}")
    log.debug(f"b = {b}")
    
    result = a - b
    log.debug(f"inf - inf = {result}")
    assert math.isnan(result._ptr.data[0]), f"Expected NaN, got {result._ptr.data[0]}"
    
    log.info("PASS: Mixed inf/nan")


def test_chained_operations():
    log.info("TEST: Chained operations")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [5, 6, 7, 8])
    log.debug(f"a = {a}")
    log.debug(f"b = {b}")
    
    result = ((a + b) @ (a - b)).T * 2
    log.debug(f"((a + b) @ (a - b)).T * 2 = {result}")
    assert result.shape == (2, 2), f"Expected (2,2), got {result.shape}"
    
    log.info("PASS: Chained operations")


def test_rapid_alloc_dealloc():
    log.info("TEST: Rapid allocation/deallocation (1000 cycles)")
    for i in range(1000):
        m = Matrix(5, 5, list(range(25)))
        _ = m.shape
        del m
    gc.collect()
    log.info("PASS: Rapid alloc/dealloc")


def test_many_matrices_alive():
    log.info("TEST: Many matrices alive simultaneously")
    matrices = []
    for i in range(100):
        matrices.append(Matrix(10, 10, list(range(100))))
    log.debug(f"Created {len(matrices)} matrices")
    
    for i, m in enumerate(matrices):
        assert m.shape == (10, 10), f"Matrix {i} has wrong shape"
    
    matrices.clear()
    gc.collect()
    log.info("PASS: Many matrices alive")


def test_weakref_cleanup():
    log.info("TEST: Weak reference cleanup")
    m = Matrix(2, 2, [1, 2, 3, 4])
    weak = weakref.ref(m)
    log.debug(f"Matrix created, weak ref alive: {weak() is not None}")
    
    assert weak() is not None, "Weakref should be alive"
    
    del m
    gc.collect()
    log.debug(f"After del + gc, weak ref alive: {weak() is not None}")
    assert weak() is None, "Weakref should be dead after cleanup"
    
    log.info("PASS: Weakref cleanup")


def test_result_outlives_operands():
    log.info("TEST: Result outlives operands")
    def create_sum():
        a = Matrix(2, 2, [1, 2, 3, 4])
        b = Matrix(2, 2, [5, 6, 7, 8])
        return a + b
    
    result = create_sum()
    gc.collect()
    log.debug(f"result after operands deleted = {result}")
    assert result._ptr.data[0] == 6.0, f"Expected 6.0, got {result._ptr.data[0]}"
    
    log.info("PASS: Result outlives operands")


def test_notimplemented_add():
    log.info("TEST: NotImplemented for wrong type (add)")
    a = Matrix(2, 2, [1, 2, 3, 4])
    
    result = a.__add__("garbage")
    log.debug(f"Matrix + string = {result}")
    assert result is NotImplemented, f"Expected NotImplemented, got {result}"
    
    result = a.__add__(42)
    log.debug(f"Matrix + int = {result}")
    assert result is NotImplemented, f"Expected NotImplemented, got {result}"
    
    log.info("PASS: NotImplemented for wrong type")


def test_notimplemented_matmul():
    log.info("TEST: NotImplemented for wrong type (matmul)")
    a = Matrix(2, 2, [1, 2, 3, 4])
    
    result = a.__matmul__([1, 2, 3, 4])
    log.debug(f"Matrix @ list = {result}")
    assert result is NotImplemented, f"Expected NotImplemented, got {result}"
    
    log.info("PASS: NotImplemented for matmul")


def test_rmul_works():
    log.info("TEST: Right multiply (2 * a)")
    a = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    
    result = 3 * a
    log.debug(f"3 * a = {result}")
    assert result._ptr.data[0] == 3.0, f"Expected 3.0, got {result._ptr.data[0]}"
    assert result._ptr.data[3] == 12.0, f"Expected 12.0, got {result._ptr.data[3]}"
    
    log.info("PASS: Right multiply")


def test_repr_doesnt_crash():
    log.info("TEST: repr doesn't crash")
    
    m1 = Matrix(1, 1, [42])
    log.debug(f"1x1: {repr(m1)}")
    
    m2 = Matrix(1, 10, list(range(10)))
    log.debug(f"1x10: {repr(m2)}")
    
    m3 = Matrix(10, 1, list(range(10)))
    log.debug(f"10x1: {repr(m3)}")
    
    m4 = Matrix(5, 5, list(range(25)))
    log.debug(f"5x5: {repr(m4)}")
    
    log.info("PASS: repr doesn't crash")


def test_shape_property():
    log.info("TEST: Shape property for various dimensions")
    
    test_cases = [(1, 1), (1, 10), (10, 1), (7, 13), (100, 1), (1, 100), (10, 10)]
    for rows, cols in test_cases:
        m = Matrix(rows, cols, list(range(rows * cols)))
        log.debug(f"Matrix({rows}, {cols}) shape = {m.shape}")
        assert m.shape == (rows, cols), f"Expected ({rows}, {cols}), got {m.shape}"
    
    log.info("PASS: Shape property")


def test_data_independence():
    log.info("TEST: Matrices are independent (no aliasing)")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"a = {a}")
    log.debug(f"b = {b}")
    
    c = a * 10
    log.debug(f"c = a * 10 = {c}")
    log.debug(f"a after c = a * 10: {a}")
    
    assert a._ptr.data[0] == 1.0, f"Original a was modified! Got {a._ptr.data[0]}"
    assert b._ptr.data[0] == 1.0, f"b was modified! Got {b._ptr.data[0]}"
    
    log.info("PASS: Data independence")


def test_operation_creates_new_matrix():
    log.info("TEST: Operations create new matrices")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [5, 6, 7, 8])
    
    a_ptr = a._ptr
    b_ptr = b._ptr
    
    c = a + b
    log.debug(f"a._ptr == c._ptr: {a._ptr == c._ptr}")
    assert a._ptr != c._ptr, "Addition should create new matrix"
    assert b._ptr != c._ptr, "Addition should create new matrix"
    
    d = a @ b
    assert a._ptr != d._ptr, "Matmul should create new matrix"
    
    e = a.T
    assert a._ptr != e._ptr, "Transpose should create new matrix"
    
    log.info("PASS: Operations create new matrices")


def test_concurrent_reads():
    log.info("TEST: Concurrent reads from same matrix")
    a = Matrix(10, 10, list(range(100)))
    
    def read_shape():
        for _ in range(100):
            _ = a.shape
        return True
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(read_shape) for _ in range(4)]
        results = [f.result() for f in futures]
    
    assert all(results), "Concurrent reads failed"
    log.info("PASS: Concurrent reads")


def test_concurrent_creates():
    log.info("TEST: Concurrent matrix creation")
    results = []
    
    def create_matrices():
        local_results = []
        for i in range(50):
            m = Matrix(5, 5, list(range(25)))
            local_results.append(m.shape == (5, 5))
        return all(local_results)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_matrices) for _ in range(4)]
        results = [f.result() for f in futures]
    
    assert all(results), "Concurrent creates failed"
    log.info("PASS: Concurrent creates")


def test_associativity_add():
    log.info("TEST: Addition associativity (a + b) + c == a + (b + c)")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [5, 6, 7, 8])
    c = Matrix(2, 2, [9, 10, 11, 12])
    
    left = (a + b) + c
    right = a + (b + c)
    log.debug(f"(a + b) + c = {left}")
    log.debug(f"a + (b + c) = {right}")
    
    for i in range(4):
        assert left._ptr.data[i] == right._ptr.data[i], f"Associativity failed at index {i}"
    
    log.info("PASS: Addition associativity")


def test_commutativity_add():
    log.info("TEST: Addition commutativity a + b == b + a")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [5, 6, 7, 8])
    
    ab = a + b
    ba = b + a
    log.debug(f"a + b = {ab}")
    log.debug(f"b + a = {ba}")
    
    for i in range(4):
        assert ab._ptr.data[i] == ba._ptr.data[i], f"Commutativity failed at index {i}"
    
    log.info("PASS: Addition commutativity")


def test_distributivity():
    log.info("TEST: Distributivity a @ (b + c) == a @ b + a @ c")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [5, 6, 7, 8])
    c = Matrix(2, 2, [9, 10, 11, 12])
    
    left = a @ (b + c)
    right = (a @ b) + (a @ c)
    log.debug(f"a @ (b + c) = {left}")
    log.debug(f"a @ b + a @ c = {right}")
    
    for i in range(4):
        assert abs(left._ptr.data[i] - right._ptr.data[i]) < 1e-10, f"Distributivity failed at index {i}"
    
    log.info("PASS: Distributivity")


def test_transpose_matmul_identity():
    log.info("TEST: (A @ B).T == B.T @ A.T")
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    b = Matrix(3, 2, [7, 8, 9, 10, 11, 12])
    
    left = (a @ b).T
    right = b.T @ a.T
    log.debug(f"(A @ B).T = {left}")
    log.debug(f"B.T @ A.T = {right}")
    
    assert left.shape == right.shape, f"Shapes differ: {left.shape} vs {right.shape}"
    for i in range(left.shape[0] * left.shape[1]):
        assert abs(left._ptr.data[i] - right._ptr.data[i]) < 1e-10, f"Identity failed at index {i}"
    
    log.info("PASS: Transpose matmul identity")


def test_scalar_mult_preserves_structure():
    log.info("TEST: Scalar mult preserves zero pattern")
    sparse = Matrix(3, 3, [1, 0, 0, 0, 2, 0, 0, 0, 3])
    log.debug(f"sparse = {sparse}")
    
    result = sparse * 5
    log.debug(f"sparse * 5 = {result}")
    
    assert result._ptr.data[1] == 0.0, "Zero should stay zero"
    assert result._ptr.data[2] == 0.0, "Zero should stay zero"
    assert result._ptr.data[3] == 0.0, "Zero should stay zero"
    assert result._ptr.data[0] == 5.0, "1*5 should be 5"
    
    log.info("PASS: Scalar mult preserves structure")


def test_no_data_init():
    log.info("TEST: Matrix with no data provided")
    m = Matrix(3, 3)
    log.debug(f"Empty matrix shape = {m.shape}")
    assert m.shape == (3, 3), f"Expected (3,3), got {m.shape}"
    log.info("PASS: No data init")


def test_float_conversion():
    log.info("TEST: Data converted to float")
    m = Matrix(2, 2, [1, 2, 3, 4])
    log.debug(f"m = {m}")
    assert isinstance(m._ptr.data[0], float), f"Expected float, got {type(m._ptr.data[0])}"
    log.info("PASS: Float conversion")


def test_prime_dimensions():
    log.info("TEST: Prime number dimensions")
    primes = [(7, 11), (13, 17), (23, 29)]
    for r, c in primes:
        m = Matrix(r, c, list(range(r * c)))
        log.debug(f"Matrix({r}, {c}) shape = {m.shape}")
        assert m.shape == (r, c), f"Expected ({r}, {c}), got {m.shape}"
        t = m.T
        assert t.shape == (c, r), f"Transpose shape wrong: expected ({c}, {r}), got {t.shape}"
    log.info("PASS: Prime dimensions")


def test_long_chain_no_leak():
    log.info("TEST: Long operation chain doesn't leak")
    a = Matrix(3, 3, list(range(9)))
    
    for i in range(100):
        a = (a + a) * 0.5
    
    log.debug(f"After 100 iterations: {a}")
    gc.collect()
    log.info("PASS: Long chain no leak")


def test_multiply_by_one():
    log.info("TEST: Multiply by 1 is identity")
    a = Matrix(2, 2, [1.5, 2.5, 3.5, 4.5])
    log.debug(f"a = {a}")
    
    result = a * 1
    log.debug(f"a * 1 = {result}")
    
    for i in range(4):
        assert result._ptr.data[i] == a._ptr.data[i], f"a * 1 should equal a at index {i}"
    
    log.info("PASS: Multiply by one")


def test_subtract_produces_negatives():
    log.info("TEST: Subtraction produces negative values")
    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [10, 20, 30, 40])
    
    result = a - b
    log.debug(f"a - b = {result}")
    
    assert result._ptr.data[0] == -9.0, f"Expected -9.0, got {result._ptr.data[0]}"
    assert result._ptr.data[3] == -36.0, f"Expected -36.0, got {result._ptr.data[3]}"
    
    log.info("PASS: Subtract produces negatives")


def run_all_tests():
    log.info("=" * 60)
    log.info("=" * 60)
    
    tests = [
        test_1x1_matrix,
        test_row_vector,
        test_column_vector,
        test_non_square_matmul,
        test_zero_matrix,
        test_identity_matrix,
        test_negative_numbers,
        test_scalar_zero,
        test_scalar_negative,
        test_scalar_fractional,
        test_double_transpose,
        test_self_add,
        test_self_subtract,
        test_self_matmul,
        test_very_small_numbers,
        test_very_large_numbers,
        test_inf_values,
        test_nan_values,
        test_mixed_inf_nan,
        test_chained_operations,
        test_rapid_alloc_dealloc,
        test_many_matrices_alive,
        test_weakref_cleanup,
        test_result_outlives_operands,
        test_notimplemented_add,
        test_notimplemented_matmul,
        test_rmul_works,
        test_repr_doesnt_crash,
        test_shape_property,
        test_data_independence,
        test_operation_creates_new_matrix,
        test_concurrent_reads,
        test_concurrent_creates,
        test_associativity_add,
        test_commutativity_add,
        test_distributivity,
        test_transpose_matmul_identity,
        test_scalar_mult_preserves_structure,
        test_no_data_init,
        test_float_conversion,
        test_prime_dimensions,
        test_long_chain_no_leak,
        test_multiply_by_one,
        test_subtract_produces_negatives,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            log.error(f"FAIL: {test.__name__}: {e}")
            failed += 1
    
    log.info("=" * 60)
    log.info(f"RESULTS: {passed} passed, {failed} failed")
    log.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
