#include <iomanip>  // for operator<<, setw, setprecision
#include <iostream> // for char_traits, basic_ostream, oper...
#include <sstream>  // for basic_ostringstream
#include <string>   // for allocator, operator<<

#include "highmap/array.hpp" // for Array

namespace hmap
{

void print_array(const Array &arr,
                 bool         show_coords = false,
                 int          width = 6,
                 int          precision = 2)
{
  const int nx = arr.shape.x;
  const int ny = arr.shape.y;

  // formatter helper
  auto format_value = [&](float v)
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
  };

  std::cout << "Array: \n";

  // print from top (ny-1) to bottom (0)
  for (int j = ny - 1; j >= 0; --j)
  {
    for (int i = 0; i < nx; ++i)
    {
      if (show_coords)
      {
        std::ostringstream oss;
        oss << "(" << i << "," << j << ")=" << format_value(arr(i, j));
        std::cout << std::setw(width * 2) << oss.str();
      }
      else
      {
        std::cout << std::setw(width) << format_value(arr(i, j));
      }
    }
    std::cout << '\n';
  }
}

} // namespace hmap
