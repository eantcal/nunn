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
*  Author: Antonino Calderone <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

/*

FILE FORMATS FOR THE MNIST DATABASE

All the integers in the files are stored in the MSB first (high endian).

LABEL FILE FORMAT:

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  ??               number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

IMAGE FILE FORMAT:

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  ??               number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise.
Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

*/


/* -------------------------------------------------------------------------- */

#include "mnist.h"


/* -------------------------------------------------------------------------- */

void digit_data_t::to_vect(nu::vector_t < double > & v) const NU_NOEXCEPT
{
   size_t vsize = data().size();
   v.resize(vsize);

   for ( size_t i = 0; i < vsize; ++i )
      v[i] = double(( unsigned char ) data()[i]) / 255.0;
}


/* -------------------------------------------------------------------------- */

void digit_data_t::label_to_target(nu::vector_t < double > & v) const NU_NOEXCEPT
{
   v.resize(10);
   std::fill(v.begin(), v.end(), 0.0);
   v[get_label() % 10] = 1.0;
}


/* -------------------------------------------------------------------------- */


#ifdef _WIN32
void digit_data_t::paint(int xoff, int yoff, HWND hwnd) const NU_NOEXCEPT
{
   size_t dx = get_dx();
   size_t dy = get_dy();

   if (! hwnd )
      hwnd = GetConsoleWindow();

   HDC hdc = GetDC(hwnd);

   const auto & data_bits = data();

   int idx = 0;

   for ( size_t y = 0; y < dy; ++y )
   {
      for ( size_t x = 0; x < dx; ++x )
      {
         int c = data_bits[idx++];

         SetPixel(
            hdc, 
            int(x) + xoff, 
            int(y) + yoff, 
            RGB(255 - c, 255 - c, 255 - c));
      }
   }

   ReleaseDC(GetConsoleWindow(), hdc);
}
#endif


/* -------------------------------------------------------------------------- */

int training_data_t::load()
{
   const std::vector<char> magic_lbls = { 0, 0, 8, 1 };
   const std::vector<char> magic_imgs = { 0, 0, 8, 3 };

   int ret = -1;

   FILE *flbls = nullptr, *fimgs = nullptr;

   auto to_int32 = [&](const std::vector<char>& buf)
   {
      assert(buf.size() >= 4);

      return
         ( ( buf[0] << 24 ) & 0xff000000 ) |
         ( ( buf[1] << 16 ) & 0x00ff0000 ) |
         ( ( buf[2] << 8 ) & 0x0000ff00 ) |
         ( ( buf[3] << 0 ) & 0x000000ff );
   };

   try {
      flbls = fopen(_lbls_file.c_str(), "rb");

      if ( !flbls )
         throw exception_t::lbls_file_not_found;

      fimgs = fopen(_imgs_file.c_str(), "rb");

      if ( !fimgs )
         throw exception_t::imgs_file_not_found;

      std::vector<char> buf(4);

      if ( fread(buf.data(), 1, 4, flbls) != 4 )
         throw exception_t::lbls_file_read_error;

      if ( buf[0] != magic_lbls[0] || buf[1] != magic_lbls[1] ||
         buf[2] != magic_lbls[2] || buf[3] != magic_lbls[3] )
         throw exception_t::lbls_file_wrong_magic;

      if ( fread(buf.data(), 1, 4, fimgs) != 4 )
         throw exception_t::imgs_file_read_error;


      if ( buf[0] != magic_imgs[0] || buf[1] != magic_imgs[1] ||
         buf[2] != magic_imgs[2] || buf[3] != magic_imgs[3] )
         throw exception_t::imgs_file_wrong_magic;


      if ( fread(buf.data(), 1, 4, flbls) != 4 )
         throw exception_t::lbls_file_read_error;

      int32_t n_of_lbls = to_int32(buf);

      if ( fread(buf.data(), 1, 4, fimgs) != 4 )
         throw exception_t::imgs_file_read_error;

      int32_t n_of_imgs = to_int32(buf);

      if ( n_of_lbls != n_of_imgs )
         throw exception_t::n_of_items_mismatch;

      ret = int(n_of_imgs);

      if ( fread(buf.data(), 1, 4, fimgs) != 4 )
         throw exception_t::imgs_file_read_error;

      int32_t n_rows = to_int32(buf);

      if ( fread(buf.data(), 1, 4, fimgs) != 4 )
         throw exception_t::imgs_file_read_error;

      int32_t n_cols = to_int32(buf);

      size_t img_size = size_t(n_rows * n_cols);

      for ( int32_t i = 0; i < n_of_lbls; ++i )
      {
         if ( fread(buf.data(), 1, 1, flbls) != 1 )
            break;

         digit_data_t::data_t data(img_size);

         int label = int(buf[0]) & 0xff;

         if ( fread(data.data(), 1, data.size(), fimgs) != data.size() )
            break;

         std::unique_ptr< digit_data_t > digit_info(
            new digit_data_t(size_t(n_cols), size_t(n_rows), label, data));

         _data.push_back(std::move(digit_info));
      }

   }
   catch ( exception_t )
   {
      if ( flbls )
         fclose(flbls);

      if ( fimgs )
         fclose(fimgs);

      throw;
   }

   fclose(flbls);
   fclose(fimgs);

   return ret;
}


/* -------------------------------------------------------------------------- */



