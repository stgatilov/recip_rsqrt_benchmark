//=================== float =================

//canonical
static FORCEINLINE __m128 recip_float4_ieee(__m128 x) {
  return _mm_div_ps(_mm_set1_ps(1.0f), x);
}
static FORCEINLINE __m128 rsqrt_float4_ieee(__m128 x) { 
  return _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(x));
}
//fast approximation
static FORCEINLINE __m128 recip_float4_fast(__m128 x) {
  return _mm_rcp_ps(x);
}
static FORCEINLINE __m128 rsqrt_float4_fast(__m128 x) {
  return _mm_rsqrt_ps(x);
}
//with Newton-Raphson
static FORCEINLINE __m128 recip_float4_nr1(__m128 x) {
  //inspired by http://nume.googlecode.com/svn/trunk/fosh/src/sse_approx.h
  __m128 res, muls;
  res = _mm_rcp_ps(x);
  muls = _mm_mul_ps(x, _mm_mul_ps(res, res));
  res = _mm_sub_ps(_mm_add_ps(res, res), muls);
  return res;
}
static FORCEINLINE __m128 rsqrt_float4_nr1(__m128 x) {
  __m128 three = _mm_set1_ps(3.0f), half = _mm_set1_ps(0.5f);
  //taken from http://stackoverflow.com/q/14752399/556899
  __m128 res, muls;
  res = _mm_rsqrt_ps(x); 
  muls = _mm_mul_ps(_mm_mul_ps(x, res), res); 
  res = _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls)); 
  return res;
}

//=================== double =================

//canonical
static FORCEINLINE __m128d recip_double2_ieee(__m128d x) {
  return _mm_div_pd(_mm_set1_pd(1.0), x);
}
static FORCEINLINE __m128d rsqrt_double2_ieee(__m128d x) { 
  return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(x));
}
//fast approximation (with conversion)
static FORCEINLINE __m128d recip_double2_fast(__m128d x) {
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rcp_ps(f);
  return _mm_cvtps_pd(f);
}
static FORCEINLINE __m128d rsqrt_double2_fast(__m128d x) { 
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rsqrt_ps(f);
  return _mm_cvtps_pd(f);
}
//with Newton-Raphson
static FORCEINLINE __m128d recip_double2_nr1(__m128d x) {
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rcp_ps(f);
  __m128d res, muls;
  res = _mm_cvtps_pd(f);
  muls = _mm_mul_pd(x, _mm_mul_pd(res, res));
  res = _mm_sub_pd(_mm_add_pd(res, res), muls);
  return res;
}
static FORCEINLINE __m128d rsqrt_double2_nr1(__m128d x) { 
  __m128 f = _mm_cvtpd_ps(x);
  f = _mm_rsqrt_ps(f);
  __m128d res, muls;
  __m128d three = _mm_set1_pd(3.0), half = _mm_set1_pd(0.5);
  res = _mm_cvtps_pd(f);
  muls = _mm_mul_pd(_mm_mul_pd(x, res), res);
  res = _mm_mul_pd(_mm_mul_pd(half, res), _mm_sub_pd(three, muls)); 
  return res;
}
static FORCEINLINE __m128d recip_double2_nr2(__m128d x) {
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
static FORCEINLINE __m128d rsqrt_double2_nr2(__m128d x) { 
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

//relative correctors in form:
//   r := 1 - a x | 1 - a x^2;
//   x := x * poly(r);
//method's order is equal to degree of first removed monomial in series poly(r)
//for reciprocal: poly(r) = 1 + r + r^2 + r^3 + r^4 + ...                     //a_k = 1
//for rsqrt: poly(r) = 1 + 1/2 r + 3/8 r^2 + 15/48 r^3 + 105/384 r^4 + ...    //a_k = (2k-1)!! / (2^k k!)
//see http://numbers.computation.free.fr/Constants/Algorithms/inverse.html

static FORCEINLINE __m128d recip_double2_r5(__m128d a) {
  //inspired by http://www.mersenneforum.org/showthread.php?t=11765
  __m128d one = _mm_set1_pd(1.0);
  __m128d x = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(a, x));
  __m128d r2 = _mm_mul_pd(r, r);
  __m128d r2r = _mm_add_pd(r2, r);      // r^2 + r
  __m128d r21 = _mm_add_pd(r2, one);    // r^2 + 1
  __m128d poly = _mm_mul_pd(r2r, r21);
  __m128d res = _mm_add_pd(_mm_mul_pd(poly, x), x);
  return res;
}

static FORCEINLINE __m128d recip_double2_r4(__m128d a) {
  __m128d one = _mm_set1_pd(1.0);
  __m128d x = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(a, x));
  __m128d r2 = _mm_mul_pd(r, r);
  __m128d r1 = _mm_add_pd(r, one);      // r + 1
  __m128d r21 = _mm_add_pd(r2, one);    // r^2 + 1
  __m128d poly = _mm_mul_pd(r1, r21);
  __m128d res = _mm_mul_pd(poly, x);
  return res;
}

static FORCEINLINE __m128d recip_double2_r3(__m128d a) {
  __m128d one = _mm_set1_pd(1.0);
  __m128d x = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(a, x));
  __m128d poly = _mm_add_pd(_mm_mul_pd(r, r), r);  // r^2 + r
  __m128d res = _mm_add_pd(_mm_mul_pd(poly, x), x);
  return res;
}

static FORCEINLINE __m128d rsqrt_double2_r2(__m128d a) {
  __m128d one = _mm_set1_pd(1.0), c1_2 = _mm_set1_pd(1.0/2.0);
  __m128d x = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(_mm_mul_pd(a, x), x));
  __m128d res = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(c1_2, x), r), x);
  return res;
}

static FORCEINLINE __m128d rsqrt_double2_r3(__m128d a) {
  __m128d one = _mm_set1_pd(1.0), c1 = _mm_set1_pd(1.0/2.0), c2 = _mm_set1_pd(3.0/8.0);
  __m128d x = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(_mm_mul_pd(a, x), x));
  __m128d t1 = _mm_add_pd(_mm_mul_pd(c2, r), c1);
  __m128d res = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(r, x), t1), x);
  return res;
}

static FORCEINLINE __m128d rsqrt_double2_r4(__m128d a) {
  __m128d one = _mm_set1_pd(1.0), c1 = _mm_set1_pd(1.0/2.0), c2 = _mm_set1_pd(3.0/8.0), c3 = _mm_set1_pd(15.0/48.0);
  __m128d x = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(_mm_mul_pd(a, x), x));
  __m128d r2 = _mm_mul_pd(r, r);
  __m128d t1 = _mm_add_pd(_mm_mul_pd(c2, r), c1);
  __m128d poly = _mm_add_pd(_mm_mul_pd(r2, c3), t1);
  __m128d res = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, r), poly), x);
  return res;
}

static FORCEINLINE __m128d rsqrt_double2_r5(__m128d a) {
  __m128d one = _mm_set1_pd(1.0), c1 = _mm_set1_pd(1.0/2.0), c2 = _mm_set1_pd(3.0/8.0), c3 = _mm_set1_pd(15.0/48.0), c4 = _mm_set1_pd(105.0/384.0);
  __m128d x = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(_mm_mul_pd(a, x), x));
  __m128d r2 = _mm_mul_pd(r, r);
  __m128d t1 = _mm_add_pd(_mm_mul_pd(c2, r), c1);
  __m128d t3 = _mm_add_pd(_mm_mul_pd(c4, r), c3);
  __m128d poly = _mm_add_pd(_mm_mul_pd(r2, t3), t1);
  __m128d res = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, r), poly), x);
  return res;
}
