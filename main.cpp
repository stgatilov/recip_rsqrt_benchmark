#include <xmmintrin.h>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <cfloat>
#ifdef _MSC_VER
  #define noexcept
#endif
#include "aligned_allocator.h"
#include "rdtsc.h"
using namespace std;

#if defined(_MSC_VER)
  #define FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define FORCEINLINE __attribute__((always_inline))
#endif

typedef __m128 (*float4_func)(__m128);
typedef __m128d (*double2_func)(__m128d);

//================ Implementations ===============

#include "routines_sse.h"


//================ Testing for correctness ===============

//taken from http://stackoverflow.com/a/31474088/556899
template<class Vector> void enum_all_floats(Vector &res, float tmin, float tmax) {
  for (float x = tmin; x < tmax; x = nextafterf(x, tmax))
    res.push_back(x);
  res.push_back(tmax);
}
template<class Vector> void enum_all_floats(Vector &res, double tmin, double tmax) {
  for (double x = tmin; x < tmax; x = nextafter(x, tmax))
    res.push_back(x);
  res.push_back(tmax);
}

vector<float, aligned_allocator<float, Alignment::AVX>> test_values_float;
vector<double, aligned_allocator<float, Alignment::AVX>> test_values_double;

float test_precision_float4(float4_func tested_func, float4_func correct_func) {
  float *ptr = test_values_float.data();
  size_t n = test_values_float.size();

  __m128 maxErr = _mm_setzero_ps();
  for (size_t i = 0; i < n; i += 4) {
    __m128 x = _mm_load_ps(&ptr[i]);
    __m128 res = tested_func(x);
    __m128 ans = correct_func(x);
    __m128 diff = _mm_sub_ps(res, ans);
    diff = _mm_max_ps(diff, _mm_sub_ps(_mm_setzero_ps(), diff));
    __m128 relative = _mm_div_ps(diff, ans);
    maxErr = _mm_max_ps(maxErr, relative);
  }

  float tmp[4];
  _mm_storeu_ps(tmp, maxErr);
  return max(max(tmp[0], tmp[1]), max(tmp[2], tmp[3]));
}

double test_precision_double2(double2_func tested_func, double2_func correct_func) {
  double *ptr = test_values_double.data();
  size_t n = test_values_double.size();

  __m128d maxErr = _mm_setzero_pd();
  for (size_t i = 0; i < n; i += 2) {
    __m128d x = _mm_load_pd(&ptr[i]);
    __m128d res = tested_func(x);
    __m128d ans = correct_func(x);
    __m128d diff = _mm_sub_pd(res, ans);
    diff = _mm_max_pd(diff, _mm_sub_pd(_mm_setzero_pd(), diff));
    __m128d relative = _mm_div_pd(diff, ans);
    maxErr = _mm_max_pd(maxErr, relative);
  }

  double tmp[2];
  _mm_storeu_pd(tmp, maxErr);
  return max(tmp[0], tmp[1]);
}

#define TEST_PRECISION(tested_func, correct_func, vec) {\
  auto err = test_precision_##vec(tested_func, correct_func);\
  printf("%s: maximal error = %g\n", #tested_func, err);\
}

//================ Testing for performance ===============

#define TEST_PERFORMANCE_float4(tested_func) {\
  auto time_start = rdtsc();\
  float *ptr = test_values_float.data();\
  size_t n = test_values_float.size();\
  __m128 checkSum = _mm_setzero_ps();\
  for (size_t i = 0; i < n; i += 16) {\
    __m128 x0 = _mm_load_ps(&ptr[i+0]);\
    __m128 x1 = _mm_load_ps(&ptr[i+4]);\
    __m128 x2 = _mm_load_ps(&ptr[i+8]);\
    __m128 x3 = _mm_load_ps(&ptr[i+12]);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    __m128 r = _mm_add_ps(_mm_add_ps(x0, x1), _mm_add_ps(x2, x3));\
    checkSum = _mm_add_ps(checkSum, r);\
  }\
  float tmp[4];\
  _mm_storeu_ps(tmp, checkSum);\
  float checkSumAll = tmp[0] + tmp[1] + tmp[2] + tmp[3];\
  auto time_end = rdtsc();\
  printf("%s: cycles per call = %0.2f   (%g)\n", #tested_func, double(time_end - time_start) / n, checkSumAll);\
}

#define TEST_PERFORMANCE_double2(tested_func) {\
  auto time_start = rdtsc();\
  double *ptr = test_values_double.data();\
  size_t n = test_values_double.size();\
  __m128d checkSum = _mm_setzero_pd();\
  for (size_t i = 0; i < n; i += 8) {\
    __m128d x0 = _mm_load_pd(&ptr[i+0]);\
    __m128d x1 = _mm_load_pd(&ptr[i+2]);\
    __m128d x2 = _mm_load_pd(&ptr[i+4]);\
    __m128d x3 = _mm_load_pd(&ptr[i+6]);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    x0 = tested_func(x0);\
    x1 = tested_func(x1);\
    x2 = tested_func(x2);\
    x3 = tested_func(x3);\
    __m128d r = _mm_add_pd(_mm_add_pd(x0, x1), _mm_add_pd(x2, x3));\
    checkSum = _mm_add_pd(checkSum, r);\
  }\
  double tmp[2];\
  _mm_storeu_pd(tmp, checkSum);\
  double checkSumAll = tmp[0] + tmp[1];\
  auto time_end = rdtsc();\
  printf("%s: cycles per call = %0.2f   (%g)\n", #tested_func, double(time_end - time_start) / (2*n), checkSumAll);\
}

#define TEST_PERFORMANCE(tested_func, vec) TEST_PERFORMANCE_##vec(tested_func)


int main() {
  //generate test values
  mt19937 rnd;
  uniform_real_distribution<float> distrf(-20.0f, 20.0f);
  enum_all_floats(test_values_float, 1.0f, 2.0f);
  for (size_t i = 0; i < (1<<25) || test_values_float.size() % 256 != 0; i++)
    test_values_float.push_back(exp2f(distrf(rnd)));
  uniform_real_distribution<double> distrd(-20.0, 20.0);
  enum_all_floats(test_values_double, 1.0, 1.0 + DBL_EPSILON * (1<<23));
  for (size_t i = 0; i < (1<<24) || test_values_double.size() % 256 != 0; i++)
    test_values_double.push_back(exp2(distrd(rnd)));

  //test reciprocal

  TEST_PRECISION   (recip_float4_ieee, recip_float4_ieee, float4);
  TEST_PERFORMANCE (recip_float4_ieee, float4);
  TEST_PRECISION   (recip_float4_fast, recip_float4_ieee, float4);
  TEST_PERFORMANCE (recip_float4_fast, float4);
  TEST_PRECISION   (recip_float4_nr1, recip_float4_ieee, float4);
  TEST_PERFORMANCE (recip_float4_nr1, float4);

  TEST_PRECISION   (recip_double2_ieee, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_ieee, double2);
  TEST_PRECISION   (recip_double2_fast, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_fast, double2);
  TEST_PRECISION   (recip_double2_nr1, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_nr1, double2);
  TEST_PRECISION   (recip_double2_nr2, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_nr2, double2);

  TEST_PRECISION   (recip_double2_r3, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_r3, double2);
  TEST_PRECISION   (recip_double2_r4, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_r4, double2);
  TEST_PRECISION   (recip_double2_r5, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_r5, double2);

  //test reciprocal square root

  TEST_PRECISION   (rsqrt_float4_ieee, rsqrt_float4_ieee, float4);
  TEST_PERFORMANCE (rsqrt_float4_ieee, float4);
  TEST_PRECISION   (rsqrt_float4_fast, rsqrt_float4_ieee, float4);
  TEST_PERFORMANCE (rsqrt_float4_fast, float4);
  TEST_PRECISION   (rsqrt_float4_nr1, rsqrt_float4_ieee, float4);
  TEST_PERFORMANCE (rsqrt_float4_nr1, float4);
  
  TEST_PRECISION   (rsqrt_double2_ieee, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_ieee, double2);
  TEST_PRECISION   (rsqrt_double2_fast, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_fast, double2);
  TEST_PRECISION   (rsqrt_double2_nr1, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_nr1, double2);
  TEST_PRECISION   (rsqrt_double2_nr2, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_nr2, double2);

  TEST_PRECISION   (rsqrt_double2_r2, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_r2, double2);
  TEST_PRECISION   (rsqrt_double2_r3, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_r3, double2);
  TEST_PRECISION   (rsqrt_double2_r4, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_r4, double2);
  TEST_PRECISION   (rsqrt_double2_r5, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_r5, double2);


  return 0;
}
