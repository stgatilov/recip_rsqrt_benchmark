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
//   h := 1 - a*x | 1 - a*x*x;
//   x := x * poly(h);
//see http://numbers.computation.free.fr/Constants/Algorithms/inverse.html


static FORCEINLINE __m128d recip_double2_r5(__m128d x) {
  //taken from http://www.mersenneforum.org/showthread.php?t=11765
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
