/*
*  This file is part of nunnlib
*
*  nunnlib is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  nunnlib is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with nunnlib; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
*
*  Author: <antonino.calderone@ericsson.com>, <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

#ifndef __NU_VECTOR_H__
#define __NU_VECTOR_H__


/* -------------------------------------------------------------------------- */

#include "nu_noexcept.h"

#include <vector>
#include <ostream>
#include <sstream>
#include <functional>
#include <cmath>


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

template< class T = double >
class vector_t
{
public:
   using item_t = T;
   using vr_t = std::vector < item_t > ;

   using iterator = typename vr_t::iterator;
   using const_iterator = typename vr_t::const_iterator;

   enum class exception_t
   {
      size_mismatch
   };

   //Ctors
   vector_t() = default;

   vector_t(const vector_t&) = default;

   vector_t(const double* v, size_t v_len) : _v(v_len)
   {
      memcpy(_v.data(), v, v_len);
   }

   vector_t& operator=(const vector_t&) = default;

   vector_t& operator=( const T& value )
   {
      if ( !_v.empty() )
         std::fill(_v.begin(), _v.end(), value);

      return *this;
   }

   vector_t(const std::initializer_list<T> l) NU_NOEXCEPT :
      _v(l) {}

   vector_t(vector_t&& other) NU_NOEXCEPT :
      _v(std::move(other._v)) {}

   vector_t& operator=(vector_t&& other) NU_NOEXCEPT
   {
      if (this != &other)
         _v = std::move(other._v);

      return *this;
   }

   vector_t(const size_t & size, item_t v = 0.0) NU_NOEXCEPT
      : _v(size, v) {}

   vector_t(const vr_t& v) NU_NOEXCEPT
      : _v(v){}


   size_t size() const NU_NOEXCEPT
   {
      return _v.size();
   }


   bool empty() const NU_NOEXCEPT
   {
      return _v.empty();
   }

   void resize(const size_t & size, item_t v = 0.0) NU_NOEXCEPT
   {
      _v.resize(size, v);
   }


   iterator begin() NU_NOEXCEPT
   {
      return _v.begin();
   }


   const_iterator cbegin() const NU_NOEXCEPT
   {
      return _v.cbegin();
   }


   iterator end() NU_NOEXCEPT
   {
      return _v.end();
   }


   const_iterator cend() const NU_NOEXCEPT
   {
      return _v.cend();
   }


   item_t operator[](size_t idx) const NU_NOEXCEPT
   {
      return _v[idx];
   }


   item_t& operator[](size_t idx) NU_NOEXCEPT
   {
      return _v[idx];
   }


   void push_back(const item_t& item)
   {
      _v.push_back(item);
   }


   size_t max_item_index() NU_NOEXCEPT
   {
      if ( empty() )
         return size_t(-1);
      
      item_t max = _v[0];

      size_t idx = 1;
      size_t max_idx = 0;

      for (; idx<size(); ++idx )
         if ( max < _v[idx] )
         {
            max_idx = idx;
            max = _v[idx];
         }

      return max_idx;
   }


   //dot product
   item_t dot(const vector_t& other)
   {
      if (other.size() != size())
         throw exception_t::size_mismatch;

      item_t sum = 0;
      size_t idx = 0;

      for (auto & i : _v)
         sum += i*other[idx++];

      return sum;
   }


   const vector_t& apply(const std::function<T(T)> & f)
   {
      for ( auto & i : _v )
         i = f(i);

      return *this;
   }


   const vector_t& abs() NU_NOEXCEPT
   {
      return apply(::abs);
   }


   const vector_t& log() NU_NOEXCEPT
   {
      return apply([](double x) { return std::log(x); });
   }


   const vector_t& negate() NU_NOEXCEPT
   {
      return apply([](double x) { return -x; });
   }


   T sum() const NU_NOEXCEPT
   {
      T res = T(0);
      for ( auto & i : _v )
         res += i;

      return res;
   }


   bool operator==(const vector_t& other) const NU_NOEXCEPT
   {
      return (this == &other) || _v == other._v;
   }


   bool operator!=(const vector_t& other) const NU_NOEXCEPT
   {
      return (this != &other) && _v != other._v;
   }


   bool operator<(const vector_t& other) const NU_NOEXCEPT
   {
      return _v < other._v;
   }


   bool operator<=(const vector_t& other) const NU_NOEXCEPT
   {
      return _v <= other._v;
   }



   bool operator>=(const vector_t& other) const NU_NOEXCEPT
   {
      return _v >= other._v;
   }


   bool operator>(const vector_t& other) const NU_NOEXCEPT
   {
      return _v > other._v;
   }



   vector_t& operator+=(const vector_t& other)
   {
      return _op(other, [](item_t& d, const item_t& s) { d += s; });
   }


   vector_t& operator*=(const vector_t& other)
   {
      return _op(other, [](item_t& d, const item_t& s) { d *= s; });
   }


   vector_t& operator-=(const vector_t& other)
   {
      return _op(other, [](item_t& d, const item_t& s) { d -= s; });
   }


   vector_t& operator/=(const vector_t& other)
   {
      return _op(other, [](item_t& d, const item_t& s) { d /= s; });
   }


   vector_t& operator*=(const item_t& s)
   {
      vector_t other(size(), s);
      return this->operator*=(other);
   }


   vector_t& operator+=(const item_t& s)
   {
      vector_t other(size(), s);
      return this->operator+=(other);
   }


   vector_t& operator-=(const item_t& s)
   {
      vector_t other(size(), s);
      return this->operator-=(other);
   }


   vector_t& operator/=(const item_t& s)
   {
      vector_t other(size(), s);
      return this->operator/=(other);
   }


   void get_vector(vr_t & d) NU_NOEXCEPT
   {
      d = _v;
   }


   friend std::stringstream& operator<<(std::stringstream& ss, const vector_t& v)
   {
      ss << v.size() << std::endl;

      for ( auto i = v.cbegin(); i != v.cend(); ++i )
         ss << *i << std::endl;

      return ss;
   }


   friend std::stringstream& operator>>(std::stringstream& ss, vector_t& v)
   {
      size_t size = 0;
      ss >> size;

      v.resize(size);

      for (size_t i = 0; i < size; ++i)
         ss >> v[i];

      return ss;
   }


   friend std::ostream& operator<<(std::ostream& os, const vector_t& v)
   {
      os << "[ ";

      for (auto i = v.cbegin(); i!=v.cend();++i)
         os << *i << " ";

      os << " ]";

      return os;
   }


   friend vector_t operator +( const vector_t& v1, const vector_t& v2 )
   {
      auto vr = v1;
      vr += v2;
      return vr;
   }


   friend vector_t operator -( const vector_t& v1, const vector_t& v2 )
   {
      auto vr = v1;
      vr -= v2;
      return vr;
   }


   item_t euclidean_norm2() const NU_NOEXCEPT
   {
      item_t res = 0.0;

      for ( size_t i = 0; i < _v.size(); ++i )
         res += _v[i] * _v[i];

      return res;
   }


   item_t euclidean_norm() const NU_NOEXCEPT
   {
      return std::sqrt(euclidean_norm2());
   }


   const std::vector<T>& to_stdvec() const NU_NOEXCEPT
   {
      return _v;
   }

   std::vector<T>& to_stdvec() NU_NOEXCEPT
   {
      return _v;
   }

private:
   vr_t _v;

   vector_t& _op(
      const vector_t& other,
      std::function<void(item_t&, const item_t&)> f)
   {
      if (other.size() != size())
         throw exception_t::size_mismatch;

      size_t idx = 0;
      for (auto & i : _v)
         f(i, other[idx++]);

      return *this;
   }

};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_VECTOR_H__
