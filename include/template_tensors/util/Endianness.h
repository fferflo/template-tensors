#pragma once

#include <template_tensors/TemplateTensors.h>

namespace endian {

#define IS_LITTLE_ENDIAN (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define IS_BIG_ENDIAN (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
static_assert(IS_LITTLE_ENDIAN || IS_BIG_ENDIAN, "Endianness is not supported");

#define IS_NETWORK_ENDIAN IS_BIG_ENDIAN

template <size_t TSize>
struct EndianSwap;

template <>
struct EndianSwap<1>
{
  template <typename T>
  static T get(T element)
  {
    return element;
  }
};

template <>
struct EndianSwap<2>
{
  template <typename T>
  static T get(T element)
  {
    uint16_t swapped = __builtin_bswap16(*reinterpret_cast<uint32_t*>(&element));
    return *reinterpret_cast<T*>(&swapped);
  }
};

template <>
struct EndianSwap<4>
{
  template <typename T>
  static T get(T element)
  {
    uint32_t swapped = __builtin_bswap32(*reinterpret_cast<uint32_t*>(&element));
    return *reinterpret_cast<T*>(&swapped);
  }
};

template <>
struct EndianSwap<8>
{
  template <typename T>
  static T get(T element)
  {
    uint64_t swapped = __builtin_bswap64(*reinterpret_cast<uint64_t*>(&element));
    return *reinterpret_cast<T*>(&swapped);
  }
};

template <typename T>
T swap(T element)
{
  return EndianSwap<sizeof(T)>::get(element);
}

template <typename T>
T hton(T element)
{
#if IS_NETWORK_ENDIAN
  return element;
#else
  return swap(element);
#endif
}

template <typename T>
T ntoh(T element)
{
#if IS_NETWORK_ENDIAN
  return element;
#else
  return swap(element);
#endif
}

template <typename T>
void swap_array(T* array, size_t num)
{
  template_tensors::for_each([](T& el){el = swap(el);}, template_tensors::ref<template_tensors::ColMajor, mem::HOST>(array, num));
}

template <typename T>
void hton_array(T* array, size_t num)
{
#if !IS_NETWORK_ENDIAN
  return swap_array(array, num);
#endif
}

template <typename T>
void ntoh_array(T* array, size_t num)
{
#if !IS_NETWORK_ENDIAN
  return swap_array(array, num);
#endif
}

template <size_t TSize>
void swap(uint8_t* data)
{
  using type = template_tensors::VectorXT<uint8_t, TSize>;
  *reinterpret_cast<type*>(data) = EndianSwap<TSize>::get(*reinterpret_cast<type*>(data));
}

template <size_t TNum>
void hton(uint8_t* data)
{
#if !IS_NETWORK_ENDIAN
  swap<TNum>(data);
#endif
}

template <size_t TNum>
void ntoh(uint8_t* data)
{
#if !IS_NETWORK_ENDIAN
  swap<TNum>(data);
#endif
}

} // end of ns endian