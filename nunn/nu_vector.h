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

   vector_t& operator=(const vector_t&) = default;

   vector_t& operator=( const T& value )
   {
      if ( !_v.empty() )
         std::fill(_v.begin(), _v.end(), value);

      return *this;
   }

   vector_t(const std::initializer_list<T> l) throw() :
      _v(l) {}

   vector_t(vector_t&& other) throw() :
      _v(std::move(other._v)) {}

   vector_t& operator=(vector_t&& other) throw()
   {
      if (this != &other)
         _v = std::move(other._v);

      return *this;
   }

   vector_t(const size_t & size, item_t v = 0.0) throw()
      : _v(size, v) {}

   vector_t(const vr_t& v) throw()
      : _v(v){}


   size_t size() const throw()
   {
      return _v.size();
   }


   bool empty() const throw( )
   {
      return _v.empty();
   }

   void resize(const size_t & size, item_t v = 0.0) throw()
   {
      _v.resize(size, v);
   }


   iterator begin() throw()
   {
      return _v.begin();
   }


   const_iterator cbegin() const throw()
   {
      return _v.cbegin();
   }


   iterator end() throw()
   {
      return _v.end();
   }


   const_iterator cend() const throw()
   {
      return _v.cend();
   }


   item_t operator[](size_t idx) const throw()
   {
      return _v[idx];
   }


   item_t& operator[](size_t idx) throw()
   {
      return _v[idx];
   }


   void push_back(const item_t& item)
   {
      _v.push_back(item);
   }


   size_t max_item_index() throw( )
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


   const vector_t& abs() throw()
   {
      for (auto & i : _v)
         i = ::abs(i);

      return *this;
   }


   const vector_t& log() throw( )
   {
      for ( auto & i : _v )
         i = std::log(i);

      return *this;
   }


   const vector_t& negate() throw( )
   {
      for ( auto & i : _v )
         i = -i;

      return *this;
   }


   T sum() const throw( )
   {
      T res = T(0);
      for ( auto & i : _v )
         res += i;

      return res;
   }


   bool operator==(const vector_t& other) const throw()
   {
      return (this == &other) || _v == other._v;
   }


   bool operator!=(const vector_t& other) const throw()
   {
      return (this != &other) && _v != other._v;
   }


   bool operator<(const vector_t& other) const throw()
   {
      return _v < other._v;
   }


   bool operator<=(const vector_t& other) const throw()
   {
      return _v <= other._v;
   }



   bool operator>=(const vector_t& other) const throw()
   {
      return _v >= other._v;
   }


   bool operator>(const vector_t& other) const throw()
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


   void get_vector(vr_t & d) throw()
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


   item_t euclidean_norm2() const throw( )
   {
      item_t res = 0.0;

      for ( size_t i = 0; i < _v.size(); ++i )
         res += _v[i] * _v[i];

      return res;
   }


   item_t euclidean_norm() const throw( )
   {
      return std::sqrt(euclidean_norm2());
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
