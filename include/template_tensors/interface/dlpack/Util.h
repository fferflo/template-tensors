#pragma once

#ifdef DLPACK_INCLUDED

#include <dlpack/dlpack.h>

namespace template_tensors {

class InvalidDlPackShapeException : public std::exception
{
public:
  InvalidDlPackShapeException(size_t got, size_t expected)
    : m_message(std::string("Invalid DlPack shape. Got rank ") + util::to_string(got) + " expected rank " + util::to_string(expected))
  {
  }

  template <typename TVector1, typename TVector2>
  InvalidDlPackShapeException(TVector1&& got, TVector2&& expected)
    : m_message(std::string("Invalid DlPack shape. Got dimensions ") + util::to_string(got) + " expected dimensions " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidDlPackElementTypeException : public std::exception
{
public:
  InvalidDlPackElementTypeException(
      DLDataTypeCode expected_code, uint8_t expected_bits, uint16_t expected_lanes,
      DLDataTypeCode got_code, uint8_t got_bits, uint16_t got_lanes)
    : m_message(
      std::string("Invalid DlPack elementtype. Got {code=") + util::to_string(got_code) + ", bits=" + util::to_string(got_bits) + ", lanes=" + util::to_string(got_lanes) + "}"
      " Expected {code=" + util::to_string(expected_code) + ", bits=" + util::to_string(expected_bits) + ", lanes=" + util::to_string(expected_lanes) + "}"
    )
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidDlPackMemoryType : public std::exception
{
public:
  InvalidDlPackMemoryType(DLDeviceType got, DLDeviceType expected)
    : m_message(std::string("Invalid DlPack memory type. Got type ") + util::to_string(got) + " expected type " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

template <typename TElementType>
struct dlpack_elementtype;

#define DLPACK_ELEMENTTYPE(TYPENAME, CODE, BITS, LANES) \
  template <> \
  struct dlpack_elementtype<TYPENAME> \
  { \
    static DLDataTypeCode getCode() \
    { \
      return CODE; \
    } \
    static uint8_t getBits() \
    { \
      return BITS; \
    } \
    static uint16_t getLanes() \
    { \
      return LANES; \
    } \
  }

DLPACK_ELEMENTTYPE(float, kDLFloat, 32, 1);
DLPACK_ELEMENTTYPE(double, kDLFloat, 64, 1);
DLPACK_ELEMENTTYPE(uint8_t, kDLUInt, 8, 1);
DLPACK_ELEMENTTYPE(uint16_t, kDLUInt, 16, 1);
DLPACK_ELEMENTTYPE(uint32_t, kDLUInt, 32, 1);
DLPACK_ELEMENTTYPE(uint64_t, kDLUInt, 64, 1);
DLPACK_ELEMENTTYPE(int8_t, kDLInt, 8, 1);
DLPACK_ELEMENTTYPE(int16_t, kDLInt, 16, 1);
DLPACK_ELEMENTTYPE(int32_t, kDLInt, 32, 1);
DLPACK_ELEMENTTYPE(int64_t, kDLInt, 64, 1);

#undef DLPACK_ELEMENTTYPE

template <mem::MemoryType TMemoryType>
struct dlpack_devicetype;

template <>
struct dlpack_devicetype<mem::HOST>
{
  static const DLDeviceType value = kDLCPU;
};

template <>
struct dlpack_devicetype<mem::DEVICE>
{
  static const DLDeviceType value = kDLCUDA;
};

} // end of ns template_tensors

#endif