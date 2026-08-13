import numpy as np

from fredipy.util import softtheta, make_column_vector, make_row_vector, allclose


def test_softtheta():
    x = np.array([-1e2, 0, 1e2])
    mu0 = 0
    l0 = 1

    sign = 1
    result = softtheta(x, mu0, l0, sign)
    expected = np.array([0, 0.5, 1])

    assert np.allclose(result, expected)

    sign = -1
    result = softtheta(x, mu0, l0, sign)
    expected = np.array([1, 0.5, 0])

    assert np.allclose(result, expected)

    sign = 0
    result = softtheta(x, mu0, l0, sign)
    expected = np.ones_like(x)

    assert np.allclose(result, expected)


def test_softtheta_sign_negative_does_not_overflow_to_nan():
    # Regression test: x deep below mu0 relative to l0 (e.g. deep-IR kernel evaluations with a
    # narrow l_ir) previously overflowed exp() in the sign<0 branch's direct exp/(exp+1) form,
    # producing inf/inf = nan instead of the correct limit of 1.0.
    x = np.array([0.0, 1e-3])
    mu0, l0 = 1.0, 1e-3  # (mu0 - x)/l0 >> 709 for both x values: exp() overflows

    result = softtheta(x, mu0, l0, -1)

    assert np.all(np.isfinite(result))
    assert np.allclose(result, 1.0)


def test_allclose():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])

    assert allclose(a, b)

    a = np.array([1, 2, 3])
    b = np.array([1, 2, 4])

    assert not allclose(a, b)

    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3, 4])

    assert not allclose(a, b)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [3, 4]])

    assert allclose(a, b)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [3, 5]])

    assert not allclose(a, b)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2, 3], [4, 5, 6]])

    assert not allclose(a, b)

    a = np.array([1, 2, 3])

    assert not allclose(make_column_vector(a), make_row_vector(a))


def test_make_row_vector():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = make_row_vector(x)
    expected = np.array([[1, 2, 3, 4, 5, 6]])
    assert np.allclose(result, expected)

    x = np.array([1, 2, 3])
    result = make_row_vector(x)
    expected = np.array([[1, 2, 3]])
    assert np.allclose(result, expected)

    x = 1.5
    result = make_row_vector(x)
    expected = np.array([[1.5]])
    assert np.allclose(result, expected)

    x = 5
    result = make_row_vector(x)
    expected = np.array([[5]])
    assert np.allclose(result, expected)

    x = [1, 2, 3]
    result = make_row_vector(x)
    expected = np.array([[1, 2, 3]])
    assert np.allclose(result, expected)

    x = None
    result = make_row_vector(x)
    expected = None
    assert result == expected


def test_make_column_vector():
    x = np.array([1, 2, 3])
    result = make_column_vector(x)
    expected = np.array([[1], [2], [3]])
    assert np.allclose(result, expected)

    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = make_column_vector(x)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(result, expected)

    x = 1.5
    result = make_column_vector(x)
    expected = np.array([[1.5]])
    assert np.allclose(result, expected)

    x = 5
    result = make_column_vector(x)
    expected = np.array([[5]])
    assert np.allclose(result, expected)

    x = [1, 2, 3]
    result = make_column_vector(x)
    expected = np.array([[1], [2], [3]])
    assert np.allclose(result, expected)

    x = None
    result = make_column_vector(x)
    expected = None
    assert result == expected
