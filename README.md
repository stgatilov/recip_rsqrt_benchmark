# recip_rsqrt_benchmark
This benchmark is related to http://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision.

It is known that division and square root are rather slow.
Instructions rcpps and rsqrtps allow to compute approximate results very fast for single precision floats.
Often they are immediately followed by Newton-Raphson iteration or some other numeric procedure with superlinear convergence.
Several approximate implementations of recip(x) = 1 / x and rsqrt(x) = 1 / sqrt(x) are gathered in this repo.
They are benchmarked for precision (maximum relative error is measured), and for throughput (number of CPU cycles per call).

