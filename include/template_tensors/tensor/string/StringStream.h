#pragma once

namespace template_tensors {

template <typename TChar>
class stringstream
{
public:
  __host__ __device__
  stringstream()
    : m_root(nullptr)
    , m_length(0)
  {
  }

  __host__ __device__
  ~stringstream()
  {
    node* n = m_root;
    while (n != nullptr)
    {
      node* next_node = n->next;
      delete n;
      n = next_node;
    }
  }

  __host__ __device__
  stringstream(const stringstream<TChar>&) = delete;
  __host__ __device__
  stringstream(stringstream<TChar>&&) = delete;
  __host__ __device__
  stringstream<TChar>& operator=(const stringstream<TChar>&) = delete;
  __host__ __device__
  stringstream<TChar>& operator=(stringstream<TChar>&&) = delete;

  template <typename T>
  __host__ __device__
  stringstream<TChar>& append(T&& object)
  {
    m_length += object.rows();

    node* new_node = new node(object.rows());
    new_node->string = util::forward<T>(object);
    if (m_root == nullptr)
    {
      m_root = new_node;
    }
    else
    {
      m_tail->next = new_node;
    }
    m_tail = new_node;

    return *this;
  }

  __host__ __device__
  template_tensors::AllocVectorT<TChar, mem::alloc::heap, template_tensors::ColMajor> str() const
  {
    template_tensors::AllocVectorT<TChar, mem::alloc::heap, template_tensors::ColMajor> result(m_length);
    node* n = m_root;
    size_t index = 0;
    while (n != nullptr)
    {
      for (size_t i = 0; i < n->string.rows(); i++)
      {
        result(index++) = n->string(i);
      }
      n = n->next;
    }
    return result;
  }

private:
  struct node
  {
    template_tensors::AllocVectorT<TChar, mem::alloc::heap, template_tensors::ColMajor> string;
    node* next;

    __host__ __device__
    node(size_t len)
      : string(len)
      , next(nullptr)
    {
    }
  };

  node* m_root;
  node* m_tail;
  size_t m_length;
};

template <typename TChar, typename T>
__host__ __device__
auto operator<<(stringstream<TChar>& stream, T&& object)
RETURN_AUTO(stream.append(to_string<TChar>(util::forward<T>(object))))

} // end of ns template_tensors