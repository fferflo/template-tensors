#pragma once

namespace template_tensors {

template <metal::int_ TMaxLen = 64, typename TIntegral>
__host__ __device__
auto itoa(TIntegral i)
{
  static_assert(std::is_integral<TIntegral>::value, "Must be integer");
  template_tensors::AllocVectorT<char, mem::alloc::heap, template_tensors::ColMajor> result(TMaxLen);
  size_t index = TMaxLen - 1;
  if (i == 0)
  {
    result(index--) = '0';
  }
  else
  {
    bool negative = std::is_signed<TIntegral>::value ? i < 0 : false;
    if (negative)
    {
      i = -i;
    }
    while (i > 0)
    {
      result(index--) = '0' + (i % 10);
      i /= 10;
    }
    if (negative)
    {
      result(index--) = '-';
    }
  }
  return template_tensors::tail(util::move(result), TMaxLen - 1 - index);
}

template <metal::int_ TMaxLen = 64, typename TFloat>
__host__ __device__
auto ftoa(TFloat f, size_t max_significant_digits = 8)
{
  static_assert(std::is_floating_point<TFloat>::value, "Must be floating point");
  template_tensors::AllocVectorT<char, mem::alloc::heap, template_tensors::ColMajor> result(TMaxLen);

  size_t index = 0;
  if (math::isnan(f))
  {
    result(index++) = 'n';
    result(index++) = 'a';
    result(index++) = 'n';
  }
  else if (math::isinf(f))
  {
    if (f > 0)
    {
      result(index++) = 'i';
      result(index++) = 'n';
      result(index++) = 'f';
    }
    else
    {
      result(index++) = '-';
      result(index++) = 'i';
      result(index++) = 'n';
      result(index++) = 'f';
    }
  }
  else
  {
    if (f < 0)
    {
      result(index++) = '-';
      f = -f;
    }

    int32_t magnitude = math::log10(f);
    int32_t scientific_e = 0;
    if (math::abs(magnitude) > 5)
    {
      if (magnitude < 0)
      {
        magnitude -= 1.0;
      }
      scientific_e = magnitude;
      f = f / math::pow(10.0, magnitude);
      magnitude = 0;
    }

    int32_t digit_index = math::max(magnitude, 0);
    size_t significant_digits = 0;
    while (((significant_digits < max_significant_digits && f > 0) || digit_index >= -1) && index < TMaxLen)
    {
      TFloat weight = math::pow(10.0, digit_index);
      char digit = math::floor(f / weight);
      if (significant_digits > 0 || digit != 0)
      {
        significant_digits++;
      }
      f -= digit * weight;
      result(index++) = '0' + digit;
      if (digit_index == 0)
      {
        result(index++) = '.';
      }
      digit_index--;
    }

    if (scientific_e != 0)
    {
      result(index++) = 'e';
      result(index++) = scientific_e > 0 ? '+' : '-';
      auto e_str = itoa(math::abs(scientific_e));
      for (size_t i = 0; i < e_str.rows(); i++)
      {
        result(index++) = e_str(i);
      }
    }
  }
  return template_tensors::head(util::move(result), index);
}



template <typename TChar = char, typename T, ENABLE_IF(std::is_floating_point<typename std::decay<T>::type>::value)>
__host__ __device__
auto to_string(T&& t)
RETURN_AUTO(ftoa(t))
// TODO: different char/ encoding types
template <typename TChar = char, typename T, ENABLE_IF(std::is_integral<typename std::decay<T>::type>::value)>
__host__ __device__
auto to_string(T&& t)
RETURN_AUTO(itoa(t))

template <typename TChar = char, typename T, ENABLE_IF(is_string_v<T, TChar>::value)>
__host__ __device__
auto to_string(T&& t)
RETURN_AUTO(util::forward<T>(t))

template <typename TChar>
__host__ __device__
size_t count_null_terminated_length(const TChar* ptr)
{
  const TChar* start = ptr;
  while (*ptr != 0)
  {
    ptr++;
  }
  return ptr - start;
}

template <typename TChar = char, typename TCharIn, ENABLE_IF(std::is_same<typename std::decay<TChar>::type, typename std::decay<TCharIn>::type>::value)>
__host__ __device__
auto to_string(TCharIn* ptr)
RETURN_AUTO(template_tensors::ref<template_tensors::ColMajor, mem::LOCAL>(ptr, count_null_terminated_length(ptr)))

} // end of ns tensor
