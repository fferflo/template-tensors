#pragma once

namespace template_tensors {

template <typename TDummy>
class PrintStream
{
public:
  __host__ __device__
  ~PrintStream()
  {
    auto message = m_stream.str();
    printf("%s", message.data());
  }

private:
  template <typename TDummy2, typename T>
  __host__ __device__
  friend PrintStream<TDummy2>& operator<<(PrintStream<TDummy2>& stream, T&& object);

  template <typename TDummy2, typename T>
  __host__ __device__
  friend PrintStream<TDummy2>&& operator<<(PrintStream<TDummy2>&& stream, T&& object);

  stringstream<char> m_stream;
};

template <typename TDummy, typename T>
__host__ __device__
PrintStream<TDummy>& operator<<(PrintStream<TDummy>& stream, T&& object)
{
  stream.m_stream << std::forward<T>(object);
  return stream;
}

template <typename TDummy, typename T>
__host__ __device__
PrintStream<TDummy>&& operator<<(PrintStream<TDummy>&& stream, T&& object)
{
  stream.m_stream << std::forward<T>(object);
  return std::move(stream);
}

} // end of ns template_tensors
