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

#ifndef __MNIST_H__
#define __MNIST_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"

#include <list>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#endif


/* -------------------------------------------------------------------------- */

//! This class represents a single handwritten digit and its classification 
//! (label)
class digit_data_t
{
public:
   using data_t = std::vector < char > ;

private:
   size_t _dx;
   size_t _dy;
   int _label;
   data_t _data;

public:

   //! ctor
   digit_data_t(size_t dx, size_t dy, int label, const data_t& data) throw( ) :
      _dx(dx),
      _dy(dy),
      _label(label),
      _data(data)
   {
   }


   //! copy ctor
   digit_data_t(const digit_data_t&) = default;

   //! copy assign operator
   digit_data_t& operator=( const digit_data_t& other ) = default;


   //! move ctor
   digit_data_t(digit_data_t&& other) throw( ) :
      _dx(std::move(_dx)),
      _dy(std::move(_dy)),
      _label(std::move(_label)),
      _data(std::move(_data))
   {
   }


   //! move assign operator
   digit_data_t& operator=( digit_data_t&& other ) throw( )
   {
      if ( this != &other )
      {
         _dx = std::move(_dx);
         _dy = std::move(_dy);
         _label = std::move(_label);
         _data = std::move(_data);
      }

      return *this;
   }
   

   //! Returns the digit width in pixels
   size_t get_dx() const throw( )
   {
      return _dx;
   }


   //! Returns the digit height in pixels
   size_t get_dy() const throw( )
   {
      return _dy;
   }


   //! Returns the digit classification
   int get_label() const throw( )
   {
      return _label;
   }


   //! Returns a reference to internal data
   const data_t& data() const throw( )
   {
      return _data;
   }


   //! Converts the image data into a vector normalizing each item
   //! within the range [0.0, 1.0]
   void to_vect(nu::vector_t < double > & v) const throw( );


   //! Converts a label into a vector where the items are all zeros
   //! except for the item with index corrisponding to the label value
   //! its self (which is within range [0, 9]
   void label_to_target(nu::vector_t < double > & v) const throw( );


#ifdef _WIN32
   //! Draw the digit image on the window
   void paint(int xoff, int yoff, HWND hwnd = nullptr) const throw( );
#endif

};


/* -------------------------------------------------------------------------- */

//! This class provides a method to load MNIST pair of images and labels files 
//! The data can be retrieved as a list of digit_data_t objects
class training_data_t
{
public:

   using data_t = std::list< std::unique_ptr<digit_data_t> >;

   //! Return a reference to a list of digit_data_t objects
   const data_t & data() const throw( )
   {
      return _data;
   }


   //! reshuffle objects
   void reshuffle()
   {
      std::vector< std::unique_ptr< digit_data_t> > data;

      for ( auto & e : _data )
         data.push_back(std::move(e));

      std::random_shuffle(data.begin(), data.end());
      _data.clear();

      for ( auto & e : data )
         _data.push_back(std::move(e));
   }


   enum class exception_t
   {
      lbls_file_not_found,
      imgs_file_not_found,
      lbls_file_read_error,
      imgs_file_read_error,
      lbls_file_wrong_magic,
      imgs_file_wrong_magic,
      n_of_items_mismatch,
   };


   training_data_t() = delete;

   
   training_data_t(
      const std::string& lbls_file,
      const std::string& imgs_file) throw ( ) :
      _lbls_file(lbls_file),
      _imgs_file(imgs_file)
   {}


   //! Load data.
   //! @return number of loaded items or -1 in case of error
   int load();


private:
   std::string _lbls_file;
   std::string _imgs_file;

   data_t _data;
};


/* -------------------------------------------------------------------------- */

#endif // __MNIST_DB_UTILS_H__
