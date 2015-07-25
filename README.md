# recip_rsqrt_benchmark
This benchmark is related to http://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision.

It is known that division and square root are rather slow.
Instructions `rcpps` and `rsqrtps` allow to compute approximate results very fast for single precision floats.
Often they are immediately followed by Newton-Raphson iteration or some other numeric procedure with superlinear convergence.
Several approximate implementations of `recip(x) = 1 / x` and `rsqrt(x) = 1 / sqrt(x)` are gathered in this repo.
They are benchmarked for precision (maximum relative error is measured), and for throughput (number of CPU cycles per call).

## Sample results

Obtained with MSVC2013 C++ compiler in x64 mode on Intel Core i7-3770 (Ivy Bridge):

```
recip_float4_ieee: maximal error = 0
recip_float4_ieee: cycles per call = 7.30   (1.26497e+012)
recip_float4_fast: maximal error = 0.000300257
recip_float4_fast: cycles per call = 1.13   (1.26486e+012)
recip_float4_nr1: maximal error = 1.47033e-007
recip_float4_nr1: cycles per call = 3.69   (1.26497e+012)
recip_double2_ieee: maximal error = 0
recip_double2_ieee: cycles per call = 14.17   (6.34749e+011)
recip_double2_fast: maximal error = 0.000300175
recip_double2_fast: cycles per call = 3.36   (6.34688e+011)
recip_double2_nr1: maximal error = 9.01048e-008
recip_double2_nr1: cycles per call = 6.00   (6.34749e+011)
recip_double2_nr2: maximal error = 8.1719e-015
recip_double2_nr2: cycles per call = 9.42   (6.34749e+011)
recip_double2_r3: maximal error = 2.70471e-011
recip_double2_r3: cycles per call = 8.94   (6.34749e+011)
recip_double2_r4: maximal error = 8.33416e-015
recip_double2_r4: cycles per call = 10.09   (6.34749e+011)
recip_double2_r5: maximal error = 2.22045e-016
recip_double2_r5: cycles per call = 10.51   (6.34749e+011)
rsqrt_float4_ieee: maximal error = 0
rsqrt_float4_ieee: cycles per call = 14.46   (4.64887e+007)
rsqrt_float4_fast: maximal error = 0.000326109
rsqrt_float4_fast: cycles per call = 1.12   (4.6488e+007)
rsqrt_float4_nr1: maximal error = 3.10286e-007
rsqrt_float4_nr1: cycles per call = 5.52   (4.64887e+007)
rsqrt_double2_ieee: maximal error = 0
rsqrt_double2_ieee: cycles per call = 28.21   (2.73461e+007)
rsqrt_double2_fast: maximal error = 0.00032605
rsqrt_double2_fast: cycles per call = 3.38   (2.73475e+007)
rsqrt_double2_nr1: maximal error = 1.5948e-007
rsqrt_double2_nr1: cycles per call = 7.83   (2.7346e+007)
rsqrt_double2_nr2: maximal error = 3.8152e-014
rsqrt_double2_nr2: cycles per call = 13.18   (2.73461e+007)
rsqrt_double2_r2: maximal error = 1.5948e-007
rsqrt_double2_r2: cycles per call = 8.38   (2.7346e+007)
rsqrt_double2_r3: maximal error = 8.66757e-011
rsqrt_double2_r3: cycles per call = 10.65   (2.73461e+007)
rsqrt_double2_r4: maximal error = 4.95501e-014
rsqrt_double2_r4: cycles per call = 12.38   (2.73461e+007)
rsqrt_double2_r5: maximal error = 3.24557e-016
rsqrt_double2_r5: cycles per call = 13.51   (2.73461e+007)
```
