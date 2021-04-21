#pragma once

namespace atomic {

template <typename T>
struct is_atomic
{
  static const bool value = false;
};

// TODO: this depends on platform, and might differ between cuda and non-cuda code
template <>
struct is_atomic<bool>
{
  static const bool value = true;
};

template <>
struct is_atomic<float>
{
  static const bool value = true;
};

template <>
struct is_atomic<double>
{
  static const bool value = true;
};

template <>
struct is_atomic<char>
{
  static const bool value = true;
};

template <>
struct is_atomic<int8_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<int16_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<int32_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<int64_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<uint8_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<uint16_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<uint32_t>
{
  static const bool value = true;
};

template <>
struct is_atomic<uint64_t>
{
  static const bool value = true;
};

} // end of ns atomic
