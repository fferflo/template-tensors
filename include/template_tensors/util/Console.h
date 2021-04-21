#pragma once

#include <string>
#include <iomanip>

namespace console {

class ProgressBar
{
public:
  ProgressBar(std::string prefix, size_t num_elements, size_t width = 60)
    : m_prefix(prefix)
    , m_num_elements(num_elements)
    , m_width(width - prefix.length() - 3)
    , m_last_width(static_cast<size_t>(-1))
    , m_current(0)
  {
    std::cout << m_prefix << "\r" << std::flush;
  }

  void operator()()
  {
    size_t next_width = (m_last_width == static_cast<size_t>(-1)) ? 0 : m_width * m_current / m_num_elements;
    if (next_width != m_last_width)
    {
      std::cout << m_prefix << " [";
      for (size_t i = 0; i < m_width; i++)
      {
        if (i < next_width)
        {
          std::cout << "=";
        }
        else if (i == next_width)
        {
          std::cout << ">";
        }
        else
        {
          std::cout << " ";
        }
      }
      std::cout << "]\r" << std::flush;
      m_last_width = next_width;
    }
    m_current++;
    if (m_current == m_num_elements)
    {
      std::cout << std::left << std::setw(m_width + m_prefix.length() + 3) << m_prefix + " done." << std::endl << std::flush;
    }
    else if (m_current > m_num_elements)
    {
      std::cout << std::left << std::setw(m_width + m_prefix.length() + 3) << "Invalid progress" << std::endl << std::flush;
      exit(-1);
    }
  }

private:
  std::string m_prefix;
  size_t m_num_elements;
  size_t m_width;
  size_t m_last_width;
  size_t m_current;
};

} // end of ns console
