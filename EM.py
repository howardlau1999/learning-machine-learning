from functools import reduce

observations = [
    ("B", "HTTTHHTHTH"),
    ("A", "HHHHTHHHHH"),
    ("A", "HTHHHHHTHH"),
    ("B", "HTHTTTHHTT"),
    ("A", "THHHTHHHTH"),
]


def factorial(k): return reduce(lambda a, b: a * b, range(2, k + 1), 1)


def binomial(k, n, p):
    assert k <= n, "k should be less than n"
    return factorial(n) / (factorial(n) * factorial(n - k)) * (p ** k) * ((1 - p) ** (n - k))


def step(pi, p, q):
    A, B, z, n = 0, 0, 0, len(observations)
    for _, observation in observations:
        horizon = len(observation)
        heads = reduce(lambda acc, ch: acc + 1 if ch ==
                       'H' else acc, observation, 0)
        tails = horizon - heads
        _A = binomial(heads, horizon, p) * pi
        _B = binomial(heads, horizon, q) * (1 - pi)
        _z = _A / (_A + _B)

        A += _z * heads
        B += (1 - _z) * heads

        z += _z
    return z / n, A / (10 * z),  B / (10 * (n - z))


THRESHOLD = 1e-6
MAX_ITER = 100

pi, p, q = 0.5, 0.6, 0.5
for i in range(MAX_ITER):
    theta = step(pi, p, q)
    error = 0
    for old, new in zip(theta, (pi, p, q)):
        error += (old - new) ** 2
    if error < THRESHOLD:
        print("no update in params, stop iteration")
        break
    pi, p, q = theta
    print(f"Iteration {i + 1} pi = {pi:.3f} p_A = {p:.3f}, p_B = {q:.3f}")

print(f"Result pi = {pi:.3f} p_A = {p:.3f}, p_B = {q:.3f}")
