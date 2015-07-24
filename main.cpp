#include <xmmintrin.h>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
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

static FORCEINLINE __m128 recip_float4_ieee(__m128 x) {
  return _mm_div_ps(_mm_set1_ps(1.0f), x);
}
static FORCEINLINE __m128 rsqrt_float4_ieee(__m128 x) { 
  return _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(x));
}
static FORCEINLINE __m128 recip_float4_half(__m128 x) {
  return _mm_rcp_ps(x);
}
static FORCEINLINE __m128 rsqrt_float4_half(__m128 x) {
  return _mm_rsqrt_ps(x);
}
static FORCEINLINE __m128 recip_float4_single(__m128 x) {
  //inspired by http://nume.googlecode.com/svn/trunk/fosh/src/sse_approx.h
  __m128 res, muls;
  res = _mm_rcp_ps(x);
  muls = _mm_mul_ps(x, _mm_mul_ps(res, res));
  res = _mm_sub_ps(_mm_add_ps(res, res), muls);
  return res;
}
static FORCEINLINE __m128 rsqrt_float4_single(__m128 x) {
  __m128 three = _mm_set1_ps(3.0f), half = _mm_set1_ps(0.5f);
  //taken from http://stackoverflow.com/q/14752399/556899
  __m128 res, muls;
  res = _mm_rsqrt_ps(x); 
  muls = _mm_mul_ps(_mm_mul_ps(x, res), res); 
  res = _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls)); 
  return res;
}

static FORCEINLINE __m128d recip_double2_ieee(__m128d x) {
  return _mm_div_pd(_mm_set1_pd(1.0), x);
}
static FORCEINLINE __m128d rsqrt_double2_ieee(__m128d x) { 
  return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(x));
}
static FORCEINLINE __m128d recip_double2_half(__m128d x) {
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rcp_ps(f);
  return _mm_cvtps_pd(f);
}
static FORCEINLINE __m128d rsqrt_double2_half(__m128d x) { 
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rsqrt_ps(f);
  return _mm_cvtps_pd(f);
}
static FORCEINLINE __m128d recip_double2_single(__m128d x) {
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rcp_ps(f);
  __m128d res, muls;
  res = _mm_cvtps_pd(f);
  muls = _mm_mul_pd(x, _mm_mul_pd(res, res));
  res = _mm_sub_pd(_mm_add_pd(res, res), muls);
  return res;
}
static FORCEINLINE __m128d rsqrt_double2_single(__m128d x) { 
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rsqrt_ps(f);
  __m128d res, muls;
  __m128d three = _mm_set1_pd(3.0), half = _mm_set1_pd(0.5);
  res = _mm_cvtps_pd(f);
  muls = _mm_mul_pd(_mm_mul_pd(x, res), res);
  res = _mm_mul_pd(_mm_mul_pd(half, res), _mm_sub_pd(three, muls)); 
  return res;
}
static FORCEINLINE __m128d recip_double2_double(__m128d x) {
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rcp_ps(f);
  __m128d res, muls;
  res = _mm_cvtps_pd(f);
  muls = _mm_mul_pd(x, _mm_mul_pd(res, res));
  res = _mm_sub_pd(_mm_add_pd(res, res), muls);
  muls = _mm_mul_pd(x, _mm_mul_pd(res, res));
  res = _mm_sub_pd(_mm_add_pd(res, res), muls);
  return res;
}
static FORCEINLINE __m128d rsqrt_double2_double(__m128d x) { 
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rsqrt_ps(f);
  __m128d res, muls;
  __m128d three = _mm_set1_pd(3.0), half = _mm_set1_pd(0.5);
  res = _mm_cvtps_pd(f);
  muls = _mm_mul_pd(_mm_mul_pd(x, res), res);
  res = _mm_mul_pd(_mm_mul_pd(half, res), _mm_sub_pd(three, muls)); 
  muls = _mm_mul_pd(_mm_mul_pd(x, res), res);
  res = _mm_mul_pd(_mm_mul_pd(half, res), _mm_sub_pd(three, muls)); 
  return res;
}


static FORCEINLINE __m128d recip_double2_full(__m128d x) {
  //taken from http://www.mersenneforum.org/showthread.php?t=11765
/*struct DIVIDE
{
// compute y/x
    const __m128d operator() ( __m128d y, __m128d x ) const
    {
        __m128d temp;
        asm volatile(
        "movaps     %[r0],%[r3]         \n"
        "andps      %[pdnotexp],%[r0]   \n"// mantissa & sign
        "orps       %[pd1],%[r0]        \n"// 1<=x<2
        "andps      %[pdexp],%[r3]      \n"// exponent
        "psubd      %[pdexpbias],%[r3]  \n"// normalize exponent
        "movaps     %[r0],%[r1]         \n"// z
        "cvtpd2ps   %[r0],%[r0]         \n"
        "rcpps      %[r0],%[r0]         \n"// x (12-bit approx)
        "cvtps2pd   %[r0],%[r0]         \n"
        "mulpd      %[r0],%[r1]         \n"// z*x
        "mulpd      %[y],%[r0]          \n"// y*x - more precision if done here!
        "movaps     %[pd1],%[r2]        \n"
        "subpd      %[r1],%[r2]         \n"// h=1-z*x
        "movaps     %[r2],%[r1]         \n"// h
        "mulpd      %[r2],%[r2]         \n"// h^2
        "addpd      %[r2],%[r1]         \n"// h+h^2
        "addpd      %[pd1],%[r2]        \n"// 1+h^2
        "mulpd      %[r0],%[r1]         \n"// y*x*(h+h^2)
        "mulpd      %[r1],%[r2]         \n"// y*x*(1+h^2)*(h+h^2)
        "addpd      %[r2],%[r0]         \n"// y*x+y*x*(1+h^2)*(h+h^2)
        "psubd      %[r3],%[r0]         \n"// -exponent
        :
        [r0]"+x"(x),
        [r1]"=&x"(temp),
        [r2]"=&x"(temp),
        [r3]"=&x"(temp)
        :
        [y]"X"(y),
        [pdexp]"X"(CONST::pdexp),
        [pdnotexp]"X"(CONST::pdnotexp),
        [pdexpbias]"X"(CONST::pdexpbias),
        [pd1]"X"(CONST::pd1)
        :
        );
        return x;
    }
};*/
  __m128d one = _mm_set1_pd(1.0);       // 1
  __m128d t = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(x)));    // t ~= 1 / x
  __m128d tx = _mm_mul_pd(t, x);        // tx
  __m128d h = _mm_sub_pd(one, tx);      // h = (1 - tx)
  __m128d h2 = _mm_mul_pd(h, h);        // h^2
  __m128d h2h = _mm_add_pd(h, h2);      // h^2 + h
  __m128d h21 = _mm_add_pd(h2, one);    // h^2 + 1
  __m128d h2h_t = _mm_mul_pd(h2h, t);   // t (h^2 + h)
  __m128d omg = _mm_mul_pd(h21, h2h_t); // t (h^2 + h) (h^2 + 1)
  __m128d wtf = _mm_add_pd(omg, t);     // t ((h^2 + h) (h^2 + 1) + 1)
  return wtf;
}



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
  TEST_PRECISION   (recip_float4_half, recip_float4_ieee, float4);
  TEST_PERFORMANCE (recip_float4_half, float4);
  TEST_PRECISION   (recip_float4_single, recip_float4_ieee, float4);
  TEST_PERFORMANCE (recip_float4_single, float4);

  TEST_PRECISION   (recip_double2_ieee, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_ieee, double2);
  TEST_PRECISION   (recip_double2_half, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_half, double2);
  TEST_PRECISION   (recip_double2_single, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_single, double2);
  TEST_PRECISION   (recip_double2_double, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_double, double2);

  TEST_PRECISION   (recip_double2_full, recip_double2_ieee, double2);
  TEST_PERFORMANCE (recip_double2_full, double2);

  //test reciprocal square root

  TEST_PRECISION   (rsqrt_float4_ieee, rsqrt_float4_ieee, float4);
  TEST_PERFORMANCE (rsqrt_float4_ieee, float4);
  TEST_PRECISION   (rsqrt_float4_half, rsqrt_float4_ieee, float4);
  TEST_PERFORMANCE (rsqrt_float4_half, float4);
  TEST_PRECISION   (rsqrt_float4_single, rsqrt_float4_ieee, float4);
  TEST_PERFORMANCE (rsqrt_float4_single, float4);
  
  TEST_PRECISION   (rsqrt_double2_ieee, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_ieee, double2);
  TEST_PRECISION   (rsqrt_double2_half, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_half, double2);
  TEST_PRECISION   (rsqrt_double2_single, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_single, double2);
  TEST_PRECISION   (rsqrt_double2_double, rsqrt_double2_ieee, double2);
  TEST_PERFORMANCE (rsqrt_double2_double, double2);


  return 0;
}
