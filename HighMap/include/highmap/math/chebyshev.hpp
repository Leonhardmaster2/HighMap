/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */

/**
 * @file chebyshev.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @copyright Copyright (c) 2026
 */
#pragma once

namespace hmap
{

#pragma once

#include <cassert>
#include <cmath>
#include <vector>

/**
 * @brief Evaluates a Chebyshev polynomial series using Clenshaw recurrence.
 *
 * Evaluates:
 *
 *     f(x) = sum_k c_k T_k(x)
 *
 * where T_k are Chebyshev polynomials of the first kind.
 *
 * Designed for very large numbers of evaluations with minimal overhead.
 */
class ChebyshevEvaluator
{
public:
  using value_type = float;

  /**
   * @brief Default constructor.
   */
  ChebyshevEvaluator() = default;

  /**
   * @brief Construct from Chebyshev coefficients.
   * @param coefficients Series coefficients ordered by polynomial degree.
   */
  explicit ChebyshevEvaluator(std::vector<value_type> coefficients)
      : m_coefficients(std::move(coefficients))
  {
  }

  /**
   * @brief Set Chebyshev coefficients.
   * @param coefficients Series coefficients ordered by polynomial degree.
   */
  void set_coefficients(std::vector<value_type> coefficients)
  {
    m_coefficients = std::move(coefficients);
  }

  /**
   * @brief Evaluate the polynomial series at x.
   *
   * Uses the Clenshaw recurrence algorithm.
   *
   * @param  x Evaluation coordinate.
   * @return   Evaluated value.
   */
  [[nodiscard]]
  inline value_type evaluate(value_type x) const noexcept
  {
    const int n = static_cast<int>(m_coefficients.size());

    if (n == 0) return value_type(0);

    if (n == 1) return m_coefficients[0];

    value_type b_kplus1 = 0.0;
    value_type b_kplus2 = 0.0;

    const value_type two_x = 2.0 * x;

    for (int k = n - 1; k >= 1; --k)
    {
      const value_type b_k = m_coefficients[k] + two_x * b_kplus1 - b_kplus2;

      b_kplus2 = b_kplus1;
      b_kplus1 = b_k;
    }

    return m_coefficients[0] + x * b_kplus1 - b_kplus2;
  }

  /**
   * @brief Evaluate the series on a range of input values.
   *
   * @tparam InputIt Input iterator type.
   * @tparam OutputIt Output iterator type.
   *
   * @param begin Input begin iterator.
   * @param end   Input end iterator.
   * @param out   Output iterator.
   */
  template <typename InputIt, typename OutputIt>
  inline void evaluate_batch(InputIt  begin,
                             InputIt  end,
                             OutputIt out) const noexcept
  {
    for (; begin != end; ++begin, ++out)
    {
      *out = evaluate(*begin);
    }
  }

  /**
   * @brief Get polynomial degree.
   * @return Highest polynomial degree.
   */
  [[nodiscard]]
  inline int degree() const noexcept
  {
    return static_cast<int>(m_coefficients.size()) - 1;
  }

  /**
   * @brief Get series coefficients.
   * @return Coefficient array.
   */
  [[nodiscard]]
  inline const std::vector<value_type> &coefficients() const noexcept
  {
    return m_coefficients;
  }

private:
  std::vector<value_type> m_coefficients;
};

} // namespace hmap