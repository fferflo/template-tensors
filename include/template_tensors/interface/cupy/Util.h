#pragma once

namespace template_tensors::python {

class InvalidCupyShapeException : public std::exception
{
public:
  InvalidCupyShapeException(size_t got, size_t expected)
    : m_message(std::string("Invalid cupy shape. Got rank ") + std::to_string(got) + " expected rank " + std::to_string(expected))
  {
  }

  template <typename TVector1, typename TVector2>
  InvalidCupyShapeException(TVector1&& got, TVector2&& expected)
    : m_message(std::string("Invalid cupy shape. Got dimensions ") + util::to_string(got) + " expected dimensions " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidCupyElementTypeException : public std::exception
{
public:
  InvalidCupyElementTypeException(std::string got, std::string expected)
    : m_message(std::string("Invalid cupy elementtype. Got size ") + got + " expected size " + expected)
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
struct cupy_elementtype_name;

template <>
struct cupy_elementtype_name<float>
{
  static std::string get()
  {
    return "float32";
  }
};

template <>
struct cupy_elementtype_name<double>
{
  static std::string get()
  {
    return "float64";
  }
};

template <>
struct cupy_elementtype_name<uint8_t>
{
  static std::string get()
  {
    return "uint8";
  }
};

template <>
struct cupy_elementtype_name<uint16_t>
{
  static std::string get()
  {
    return "uint16";
  }
};

template <>
struct cupy_elementtype_name<uint32_t>
{
 static std::string get()
  {
    return "uint32";
  }
};

template <>
struct cupy_elementtype_name<uint64_t>
{
  static std::string get()
  {
    return "uint64";
  }
};

template <>
struct cupy_elementtype_name<int8_t>
{
  static std::string get()
  {
    return "int8";
  }
};

template <>
struct cupy_elementtype_name<int16_t>
{
  static std::string get()
  {
    return "int16";
  }
};

template <>
struct cupy_elementtype_name<int32_t>
{
  static std::string get()
  {
    return "int32";
  }
};

template <>
struct cupy_elementtype_name<int64_t>
{
  static std::string get()
  {
    return "int64";
  }
};

} // end of ns template_tensors::python