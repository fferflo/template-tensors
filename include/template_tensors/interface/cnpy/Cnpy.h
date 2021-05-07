#ifdef CNPY_INCLUDED

#include <boost/filesystem.hpp>
#include <cnpy.h>

namespace template_tensors {

namespace cnpy {

#define ThisType FromCnpyWrapperMatrix<TElementType, TRank>
#define SuperType IndexedPointerTensor< \
                                        ThisType, \
                                        TElementType, \
                                        template_tensors::RowMajor, \
                                        mem::HOST, \
                                        dyn_dimseq_t<TRank> \
                              >
template <typename TElementType, metal::int_ TRank>
class FromCnpyWrapperMatrix : public SuperType
{
public:
  FromCnpyWrapperMatrix(::cnpy::NpyArray npy_array)
    : SuperType(VectorXs<TRank>(template_tensors::fromStdVector(npy_array.shape)))
    , m_npy_array(npy_array)
  {
    ASSERT(TRank == npy_array.shape, "Invalid shape");
  }

  TT_ARRAY_SUBCLASS_ASSIGN(ThisType)

  HD_WARNING_DISABLE
  template <typename TThisType2>
  __host__ __device__
  static auto data2(TThisType2&& self)
  RETURN_AUTO(self.m_npy_array.template data<TElementType>())
  FORWARD_ALL_QUALIFIERS(data, data2)

  template <metal::int_ TIndex>
  dim_t getDynDim() const
  {
    return TIndex < TRank ? m_npy_array.shape[TIndex] : 1;
  }

  dim_t getDynDim(size_t index) const
  {
    return index < TRank ? m_npy_array.shape[index] : 1;
  }

private:
  ::cnpy::NpyArray m_npy_array;
};
#undef SuperType
#undef ThisType

class InvalidWordsizeException : public std::exception
{
public:
  InvalidWordsizeException(size_t got, size_t excepted)
    : m_message(std::string("Invalid word size. Got ") + std::to_string(got) + " expected " + std::to_string(excepted))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

template <typename TElementType, metal::int_ TRank>
FromCnpyWrapperMatrix<TElementType, TRank> load_npz(const boost::filesystem::path& path, std::string name = "arr_0")
{
  ::cnpy::NpyArray arr;
  try
  {
    arr = ::cnpy::npz_load(path.string(), name);
  }
  catch (const std::runtime_error& e)
  {
    throw std::runtime_error("Failed to load npz file " + path.string() + "\n" + std::string(e.what()));
  }
  if (arr.shape.size() != TRank)
  {
    throw boost::filesystem::filesystem_error("Invalid npz shape: " + util::to_string(template_tensors::fromStdVector(arr.shape)),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
  if (arr.word_size != sizeof(TElementType))
  {
    throw InvalidWordsizeException(arr.word_size, sizeof(TElementType));
  }

  return FromCnpyWrapperMatrix<TElementType, TRank>(arr);
}

template <typename TElementType, metal::int_ TRank>
FromCnpyWrapperMatrix<TElementType, TRank> load_npz(const boost::filesystem::path& path, VectorXs<TRank> dims, std::string name = "arr_0")
{
  FromCnpyWrapperMatrix<TElementType, TRank> result = load_npz<TElementType, TRank>(path, name);
  if (!template_tensors::all(result.template dims<TRank>() == dims))
  {
    throw boost::filesystem::filesystem_error("Invalid dimensions. Got " + util::to_string(result.dims()) + ", expected " + util::to_string(dims),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
  return result;
}

template <typename TElementType, metal::int_ TRank>
FromCnpyWrapperMatrix<TElementType, TRank> load_npy(const boost::filesystem::path& path)
{
  ::cnpy::NpyArray arr;
  try
  {
    arr = ::cnpy::npy_load(path.string());
  }
  catch (const std::runtime_error& e)
  {
    throw std::runtime_error("Failed to load npy file " + path.string() + "\n" + std::string(e.what()));
  }

  if (arr.shape.size() != TRank)
  {
    throw boost::filesystem::filesystem_error("Invalid npy shape: " + util::to_string(template_tensors::fromStdVector(arr.shape)),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
  if (arr.word_size != sizeof(TElementType))
  {
    throw InvalidWordsizeException(arr.word_size, sizeof(TElementType));
  }

  return FromCnpyWrapperMatrix<TElementType, TRank>(arr);
}

template <typename TElementType, metal::int_ TRank>
FromCnpyWrapperMatrix<TElementType, TRank> load_npy(const boost::filesystem::path& path, VectorXs<TRank> dims)
{
  FromCnpyWrapperMatrix<TElementType, TRank> result = load_npy<TElementType, TRank>(path);
  if (!template_tensors::all(result.template dims<TRank>() == dims))
  {
    throw boost::filesystem::filesystem_error("Invalid dimensions. Got " + util::to_string(result.dims()) + ", expected " + util::to_string(dims),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
  return result;
}

template <typename TElementType, metal::int_ TRank>
FromCnpyWrapperMatrix<TElementType, TRank> load(const boost::filesystem::path& path)
{
  if (boost::filesystem::extension(path) == ".npy")
  {
    return load_npy<TElementType, TRank>(path);
  }
  else if (boost::filesystem::extension(path) == ".npz")
  {
    return load_npz<TElementType, TRank>(path);
  }
  else
  {
    throw boost::filesystem::filesystem_error("Unsupported file format. Got " + boost::filesystem::extension(path) + ", expected .npy or .npz",
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
}

template <typename TElementType, metal::int_ TRank>
FromCnpyWrapperMatrix<TElementType, TRank> load(const boost::filesystem::path& path, VectorXs<TRank> dims)
{
  if (boost::filesystem::extension(path) == ".npy")
  {
    return load_npy<TElementType, TRank>(path, dims);
  }
  else if (boost::filesystem::extension(path) == ".npz")
  {
    return load_npz<TElementType, TRank>(path, dims);
  }
  else
  {
    throw boost::filesystem::filesystem_error("Unsupported file format. Got " + boost::filesystem::extension(path) + ", expected .npy or .npz",
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
}

template <metal::int_ TRank2 = DYN, typename TTensorType, metal::int_ TRank = TRank2 == DYN ? non_trivial_dimensions_num_v<TTensorType>::value : TRank2>
void save_npz(const boost::filesystem::path& path, TTensorType&& tensor, std::string name = "arr_0")
{
  auto data = template_tensors::eval<template_tensors::RowMajor>(tensor);
  ::cnpy::npz_save(path.string(), name, data.data(), template_tensors::toStdVector(data.dims()), "w");
}

template <metal::int_ TRank2 = DYN, typename TTensorType, metal::int_ TRank = TRank2 == DYN ? non_trivial_dimensions_num_v<TTensorType>::value : TRank2>
void save_npy(const boost::filesystem::path& path, TTensorType&& tensor)
{
  auto data = template_tensors::eval<template_tensors::RowMajor>(tensor);
  ::cnpy::npy_save(path.string(), data.data(), template_tensors::toStdVector(data.dims()), "w");
}

template <metal::int_ TRank = DYN, typename TTensorType>
void save(const boost::filesystem::path& path, TTensorType&& tensor)
{
  if (boost::filesystem::extension(path) == ".npy")
  {
    return save_npy<TRank>(path, util::forward<TTensorType>(tensor));
  }
  else if (boost::filesystem::extension(path) == ".npz")
  {
    return save_npz<TRank>(path, util::forward<TTensorType>(tensor));
  }
  else
  { // TODO: replace with correct filesystem exception everywhere
    throw boost::filesystem::filesystem_error("Unsupported file format. Got " + boost::filesystem::extension(path) + ", expected .npy or .npz",
        boost::system::errc::make_error_code(boost::system::errc::io_error));
  }
}

} // end of ns cnpy

} // end of ns tensor

#endif
