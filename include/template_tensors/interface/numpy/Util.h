#pragma once

namespace template_tensors::python {

class InvalidNumpyShapeException : public std::exception
{
public:
  InvalidNumpyShapeException(size_t got, size_t expected)
    : m_message(std::string("Invalid numpy shape. Got rank ") + std::to_string(got) + " expected rank " + std::to_string(expected))
  {
  }

  template <typename TVector1, typename TVector2>
  InvalidNumpyShapeException(TVector1&& got, TVector2&& expected)
    : m_message(std::string("Invalid numpy shape. Got dimensions ") + util::to_string(got) + " expected dimensions " + util::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidNumpyElementTypeException : public std::exception
{
public:
  InvalidNumpyElementTypeException(size_t got, size_t expected)
    : m_message(std::string("Invalid numpy elementtype. Got size ") + std::to_string(got) + " expected size " + std::to_string(expected))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

} // end of ns template_tensors::python