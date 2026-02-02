import logging

from ffimatrix import Matrix

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("Testing Matrix class")

    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Matrix(2, 2, [5, 6, 7, 8])

    log.debug("Created matrices A and B")
    log.debug(f"A:\n{a}")
    log.debug(f"B:\n{b}")

    c = a + b
    assert c._ptr.data[0] == 6.0, "Addition failed"
    log.info("Addition: PASS")

    d = a - b
    assert d._ptr.data[0] == -4.0, "Subtraction failed"
    log.info("Subtraction: PASS")

    e = a @ b
    log.debug(f"A @ B:\n{e}")
    log.info("Matmul: PASS")

    f = a * 2
    g = 2 * a
    assert f._ptr.data[0] == g._ptr.data[0] == 2.0, "Scalar mul failed"
    log.info("Scalar multiplication: PASS")

    t = a.T
    assert t.shape == (2, 2), "Transpose shape wrong"
    log.info("Transpose: PASS")

    assert a.shape == (2, 2), "Shape failed"
    log.info("Shape: PASS")

    log.info("All tests passed!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
