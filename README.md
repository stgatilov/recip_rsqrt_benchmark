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
recip_float4_ieee: cycles per call = 7.31   (1.26497e+012)
recip_float4_half: maximal error = 0.000300257
recip_float4_half: cycles per call = 1.11   (1.26486e+012)
recip_float4_single: maximal error = 1.47033e-007
recip_float4_single: cycles per call = 3.69   (1.26497e+012)
recip_double2_ieee: maximal error = 0
recip_double2_ieee: cycles per call = 14.20   (6.34749e+011)
recip_double2_half: maximal error = 0.000300175
recip_double2_half: cycles per call = 3.35   (6.34688e+011)
recip_double2_single: maximal error = 9.01048e-008
recip_double2_single: cycles per call = 6.02   (6.34749e+011)
recip_double2_double: maximal error = 8.1719e-015
recip_double2_double: cycles per call = 9.47   (6.34749e+011)
recip_double2_full: maximal error = 2.22045e-016
recip_double2_full: cycles per call = 10.59   (6.34749e+011)
rsqrt_float4_ieee: maximal error = 0
rsqrt_float4_ieee: cycles per call = 14.46   (4.64887e+007)
rsqrt_float4_half: maximal error = 0.000326109
rsqrt_float4_half: cycles per call = 1.11   (4.6488e+007)
rsqrt_float4_single: maximal error = 3.10286e-007
rsqrt_float4_single: cycles per call = 5.50   (4.64887e+007)
rsqrt_double2_ieee: maximal error = 0
rsqrt_double2_ieee: cycles per call = 28.26   (2.73461e+007)
rsqrt_double2_half: maximal error = 0.00032605
rsqrt_double2_half: cycles per call = 3.35   (2.73475e+007)
rsqrt_double2_single: maximal error = 1.5948e-007
rsqrt_double2_single: cycles per call = 7.99   (2.7346e+007)
rsqrt_double2_double: maximal error = 3.8152e-014
rsqrt_double2_double: cycles per call = 13.18   (2.73461e+007)
```
