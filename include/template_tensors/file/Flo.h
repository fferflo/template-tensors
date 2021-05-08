#include <boost/filesystem.hpp>

namespace template_tensors {

#define TT_FLO_MAGIC_BYTES 202021.250
#define TT_FLO_UNKNOWN_THRESH 1e9
#define TT_FLO_UNKNOWN 1e10

inline template_tensors::AllocMatrixT<template_tensors::Vector2f, mem::alloc::host_heap, template_tensors::RowMajor> readFlo(boost::filesystem::path path)
{
  std::ifstream stream(path.string(), std::ios::binary);
  if (!stream)
  {
    throw boost::filesystem::filesystem_error("File " + path.string() + " could not be opened", boost::system::errc::make_error_code(boost::system::errc::io_error));
  }

  float magic_bytes;
  stream.read(reinterpret_cast<char*>(&magic_bytes), sizeof(float));
  if (magic_bytes != TT_FLO_MAGIC_BYTES)
  {
    stream.close();
    throw boost::filesystem::filesystem_error("Flo file does not start with magic bytes", boost::system::errc::make_error_code(boost::system::errc::io_error));
  }

  int32_t width, height;
  stream.read(reinterpret_cast<char*>(&width), sizeof(int32_t));
  stream.read(reinterpret_cast<char*>(&height), sizeof(int32_t));

  template_tensors::AllocMatrixT<template_tensors::Vector2f, mem::alloc::host_heap, template_tensors::RowMajor> result(height, width);
  stream.read(reinterpret_cast<char*>(result.data()), width * height * sizeof(template_tensors::Vector2f));

  stream.close();

  return result;
}

} // end of ns template_tensors
