#pragma once

namespace atomic {

namespace op {

class Void
{
public:
  template <typename TData1, typename TData2>
  __host__ __device__
  bool load(TData1& data, TData2& out)
  {
    out = data;
    return true;
  }

  template <typename TData1, typename TData2>
  __host__ __device__
  bool store(TData1& data, const TData2& value)
  {
    data = value;
    return true;
  }

  template <typename TData1, typename TData2>
  __host__ __device__
  bool add(TData1& data, const TData2& value)
  {
    data += value;
    return true;
  }

  template <typename TData1, typename TData2>
  __host__ __device__
  bool subtract(TData1& data, const TData2& value)
  {
    data -= value;
    return true;
  }

  template <typename TData>
  __host__ __device__
  bool inc(TData& data, const TData& max)
  {
    data = (data >= max) ? 0 : (data + 1);
    return true;
  }

  template <typename TData>
  __host__ __device__
  bool dec(TData& data, const TData& max)
  {
    data = (data == 0 || data > max) ? max : (data - 1);
    return true;
  }

  template <typename TData1, typename TData2, typename TData3>
  __host__ __device__
  bool cas(TData1& data, const TData2& compare, const TData3& val, bool& swapped)
  {
    swapped = template_tensors::eq(data, compare);
    if (swapped)
    {
      data = val;
    }
    return true;
  }
};

} // end of ns op

} // end of ns atomic
