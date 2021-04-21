#ifdef TINYPLY_INCLUDED

#include <tinyply.h>
#include <boost/filesystem.hpp>
#include <sstream>

namespace template_tensors {

namespace tinyply {

namespace detail {

template <typename TScalar>
struct ElementType;

template <>
struct ElementType<int8_t>
{
  static const ::tinyply::Type value = ::tinyply::Type::INT8;
};

template <>
struct ElementType<uint8_t>
{
  static const ::tinyply::Type value = ::tinyply::Type::UINT8;
};

template <>
struct ElementType<int16_t>
{
  static const ::tinyply::Type value = ::tinyply::Type::INT16;
};

template <>
struct ElementType<uint16_t>
{
  static const ::tinyply::Type value = ::tinyply::Type::UINT16;
};

template <>
struct ElementType<int32_t>
{
  static const ::tinyply::Type value = ::tinyply::Type::INT32;
};

template <>
struct ElementType<uint32_t>
{
  static const ::tinyply::Type value = ::tinyply::Type::UINT32;
};

template <>
struct ElementType<float>
{
  static const ::tinyply::Type value = ::tinyply::Type::FLOAT32;
};

template <>
struct ElementType<double>
{
  static const ::tinyply::Type value = ::tinyply::Type::FLOAT64;
};

} // end of ns detail

template <typename TScalar>
TVALUE(::tinyply::Type, scalar_v, template_tensors::tinyply::detail::ElementType<TScalar>::value);



namespace detail {

struct ElementRequester
{
  ElementRequester(::tinyply::PlyFile& ply_file)
    : ply_file(ply_file)
  {
  }

  template <typename TElement>
  void operator()(TElement&& element)
  {
    try
    {
      element.request(ply_file);
    }
    catch (const std::exception & e)
    {
      throw boost::filesystem::filesystem_error("Failed to request element " + element.getName() + ":\n" + e.what(),
        boost::system::errc::make_error_code(boost::system::errc::io_error));
    }

    if (element.m_data->t != scalar_v<typename std::decay<TElement>::type::PropertyType>::value)
    {
      throw boost::filesystem::filesystem_error("Invalid scalar type " + ::tinyply::PropertyTable[element.m_data->t].str,
        boost::system::errc::make_error_code(boost::system::errc::io_error));
    }
  }

  ::tinyply::PlyFile& ply_file;
};

struct ElementAdder
{
  ElementAdder(::tinyply::PlyFile& ply_file)
    : ply_file(ply_file)
  {
  }

  template <typename TElement>
  void operator()(TElement&& element)
  {
    element.add(ply_file);
  }

  ::tinyply::PlyFile& ply_file;
};

} // end of ns detail



template <typename TScalar, size_t TNum, typename TIndexType = uint32_t>
class ReadProperty
{
private:
  std::shared_ptr<::tinyply::PlyData> m_data;
  std::string m_element_key;
  std::vector<std::string> m_property_keys;
  bool m_is_list;

public:
  using PropertyType = TScalar;

  ReadProperty(std::string element_key, std::vector<std::string> property_keys)
    : m_element_key(element_key)
    , m_property_keys(property_keys)
    , m_is_list(false)
  {
    ASSERT(TNum == m_property_keys.size(), "Invalid number of property keys");
  }

  ReadProperty(std::string element_key, std::string list_property_key)
    : m_element_key(element_key)
    , m_property_keys{list_property_key}
    , m_is_list(true)
  {
  }

  ReadProperty(std::string element_key)
    : m_element_key(element_key)
    , m_property_keys(0)
    //, m_list is overriden below
  {
  }

  ReadProperty()
    : m_element_key("")
    , m_property_keys(0)
  {
  }

  void setPropertyKey(std::string val)
  {
    m_property_keys.resize(1);
    m_property_keys[0] = val;
  }

  void setPropertyKeys(std::vector<std::string> val)
  {
    m_property_keys = val;
  }

  auto data()
  RETURN_AUTO(
    template_tensors::ref<template_tensors::RowMajor, mem::HOST>(reinterpret_cast<template_tensors::VectorXT<TScalar, TNum>*>(m_data->buffer.get()), m_data->count)
  )

  size_t size() const
  {
    return m_data->count;
  }

  template_tensors::VectorXT<TScalar, TNum>& operator[](size_t index)
  {
    ASSERT(index < size(), "Index out of bounds");
    return reinterpret_cast<template_tensors::VectorXT<TScalar, TNum>*>(m_data->buffer.get())[index];
  }

  const template_tensors::VectorXT<TScalar, TNum>& operator[](size_t index) const
  {
    ASSERT(index < size(), "Index out of bounds");
    return reinterpret_cast<template_tensors::VectorXT<TScalar, TNum>*>(m_data->buffer.get())[index];
  }

  friend class detail::ElementRequester;
  friend class detail::ElementAdder;
  // TODO: iterators
private:
  void request(::tinyply::PlyFile& ply_file)
  {
    if (m_property_keys.size() == 0)
    {
      auto elements = ply_file.get_elements();
      ::tinyply::PlyElement* element_it = nullptr;
      for (auto& el : elements)
      {
        if (el.name == m_element_key)
        {
          element_it = &el;
          break;
        }
      }
      if (element_it == nullptr)
      {
        throw boost::filesystem::filesystem_error("Element with key '" + m_element_key + "' not found in ply file",
          boost::system::errc::make_error_code(boost::system::errc::io_error));
      }
      ::tinyply::PlyElement& element = *element_it;
      for (auto& property : element.properties)
      {
        m_property_keys.push_back(property.name);
      }
      if (element.properties.size() == 1)
      {
        m_is_list = element.properties[0].isList;
      }
    }

    if (!m_is_list)
    {
      m_data = ply_file.request_properties_from_element(m_element_key, m_property_keys);
    }
    else
    {
      m_data = ply_file.request_properties_from_element(m_element_key, m_property_keys, TNum);
    }
  }

  void add(::tinyply::PlyFile& ply_file)
  {
    if (!m_is_list)
    {
      ply_file.add_properties_to_element(m_element_key, m_property_keys, scalar_v<TScalar>::value, m_data->count, m_data->buffer.get(), ::tinyply::Type::INVALID, 0);
    }
    else
    {
      ply_file.add_properties_to_element(m_element_key, m_property_keys, scalar_v<TScalar>::value, m_data->count, m_data->buffer.get(), scalar_v<TIndexType>::value, TNum);
    }
  }

  std::string getName() const
  {
    std::stringstream str;
    str << m_element_key << "(";
    for (size_t i = 0; i < m_property_keys.size(); i++)
    {
      if (i > 0)
      {
        str << ", ";
      }
      str << m_property_keys[i];
    }
    str << ")";
    return str.str();
  }
};

template <typename... TElements>
void read(const boost::filesystem::path& path, TElements&&... elements)
{
  std::ifstream file_stream(path.string(), std::ios::binary);
  if (!file_stream)
  {
    throw boost::filesystem::filesystem_error("File " + path.string() + " could not be opened", boost::system::errc::make_error_code(boost::system::errc::io_error));
  }

  ::tinyply::PlyFile ply_file;
  ply_file.parse_header(file_stream);

  util::for_each(detail::ElementRequester(ply_file), util::forward<TElements>(elements)...);

  ply_file.read(file_stream);
}



template <typename TScalar, size_t TNum, typename TIndexType = uint32_t>
class WriteProperty
{
private:
  std::string m_element_key;
  std::vector<std::string> m_property_keys;
  bool m_is_list;
  uint8_t* m_data;
  size_t m_count;

public:
  using PropertyType = TScalar;

  template <typename TTensorType>
  WriteProperty(std::string element_key, std::vector<std::string> property_keys, TTensorType&& tensor)
    : m_element_key(element_key)
    , m_property_keys(property_keys)
    , m_is_list(false)
    , m_count(template_tensors::prod(tensor.dims()))
  {
    m_data = new uint8_t[m_count * sizeof(template_tensors::VectorXT<TScalar, TNum>)];
    template_tensors::ref<template_tensors::RowMajor, mem::HOST>(reinterpret_cast<template_tensors::VectorXT<TScalar, TNum>*>(m_data), m_count) = tensor;
  }

  template <typename TTensorType>
  WriteProperty(std::string element_key, std::string list_property_key, TTensorType&& tensor)
    : m_element_key(element_key)
    , m_property_keys{list_property_key}
    , m_is_list(true)
    , m_count(template_tensors::prod(tensor.dims()))
  {
    m_data = new uint8_t[m_count * sizeof(template_tensors::VectorXT<TScalar, TNum>)];
    template_tensors::ref<template_tensors::RowMajor, mem::HOST>(reinterpret_cast<template_tensors::VectorXT<TScalar, TNum>*>(m_data), m_count) = tensor;
  }

  ~WriteProperty()
  {
    delete[] m_data;
  }

  friend class detail::ElementAdder;

private:
  void add(::tinyply::PlyFile& ply_file)
  {
    if (!m_is_list)
    {
      ply_file.add_properties_to_element(m_element_key, m_property_keys, scalar_v<TScalar>::value, m_count, m_data, ::tinyply::Type::INVALID, 0);
    }
    else
    {
      ply_file.add_properties_to_element(m_element_key, m_property_keys, scalar_v<TScalar>::value, m_count, m_data, scalar_v<TIndexType>::value, TNum);
    }
  }

  std::string getName() const
  {
    std::stringstream str;
    str << m_element_key << "(";
    for (size_t i = 0; i < TNum; i++)
    {
      if (i > 0)
      {
        str << ", ";
      }
      str << m_property_keys[i];
    }
    str << ")";
    return str.str();
  }
};

template <typename... TElements>
void write(const boost::filesystem::path& path, bool binary, TElements&&... elements)
{
  std::ofstream file_stream(path.string(), std::ios::out);
  if (!file_stream)
  {
    throw boost::filesystem::filesystem_error("File " + path.string() + " could not be opened", boost::system::errc::make_error_code(boost::system::errc::io_error));
  }

  ::tinyply::PlyFile ply_file;

  util::for_each(detail::ElementAdder(ply_file), util::forward<TElements>(elements)...);

  ply_file.write(file_stream, binary);
}

} // end of ns tinyply

} // end of ns tensor

#endif