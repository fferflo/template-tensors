#ifdef DLPACK_INCLUDED

#include <dlpack/dlpack.h>

namespace template_tensors {

class SafeDLManagedTensor
{
public:
  SafeDLManagedTensor(DLManagedTensor* dl)
    : m_dl(dl)
  {
  }

  SafeDLManagedTensor()
    : SafeDLManagedTensor(nullptr)
  {
  }

  SafeDLManagedTensor(SafeDLManagedTensor&& other)
  {
    m_dl = other.m_dl;
    other.m_dl = nullptr;
  }

  SafeDLManagedTensor(const SafeDLManagedTensor&) = delete;

  ~SafeDLManagedTensor()
  {
    if (m_dl != nullptr && m_dl->deleter != nullptr)
    {
      m_dl->deleter(m_dl);
      m_dl = nullptr;
    }
  }

  SafeDLManagedTensor& operator=(SafeDLManagedTensor&& other)
  {
    this->~SafeDLManagedTensor();
    m_dl = other.m_dl;
    other.m_dl = nullptr;
    return *this;
  }

  SafeDLManagedTensor& operator=(const SafeDLManagedTensor&) = delete;

  DLManagedTensor* operator->()
  {
    return m_dl;
  }

  const DLManagedTensor* operator->() const
  {
    return m_dl;
  }

  volatile DLManagedTensor* operator->() volatile
  {
    return m_dl;
  }

  const volatile DLManagedTensor* operator->() const volatile
  {
    return m_dl;
  }

  DLManagedTensor* use()
  {
    DLManagedTensor* result = m_dl;
    m_dl = nullptr;
    return result;
  }

private:
  DLManagedTensor* m_dl;
};

} // end of ns tensor

#endif
