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

#ifndef __NU_TRAINER_H__
#define __NU_TRAINER_H__


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

//! The trainer class is a helper class for neural networks training
template< class Net, class Input, class Target >
class nn_trainer_t
{
   friend class iterator;

public:
   using type_t = nn_trainer_t<Net, Input, Target>;
   using cost_func_t = std::function<double(Net&, const Target&)>;
   using progress_cbk_t = std::function<void(Net&, const Input&, const Target&, size_t)>;

   struct iterator
   {
      friend class nn_trainer_t;
      nn_trainer_t * _trainer = nullptr;
      size_t _epoch = 0;

   private:
      iterator(type_t & trainer, size_t epoch) NU_NOEXCEPT
         : _trainer(&trainer),
         _epoch(epoch)
      {
      }

   public:
      iterator(iterator & it) NU_NOEXCEPT :
         _trainer(it._trainer),
         _epoch(it._epoch)
      {
      }

      iterator& operator=( iterator & it ) NU_NOEXCEPT
      {
         if ( &it != this )
         {
            _trainer = it._trainer;
            _epoch = it._epoch;
         }

         return *this;
      }

      iterator(iterator && it) NU_NOEXCEPT :
         _trainer(std::move(it._trainer)),
         _epoch(std::move(it._epoch))
      {
      }

      iterator& operator=( iterator && it ) NU_NOEXCEPT
      {
         if ( &it != this )
         {
            _trainer = std::move(it._trainer);
            _epoch = std::move(it._epoch);
         }

         return *this;
      }

      size_t get_epoch() const NU_NOEXCEPT
      {
         return _epoch;
      }

      type_t& operator*( ) const NU_NOEXCEPT
      {
         return *_trainer;
      }

      type_t* operator->( ) const NU_NOEXCEPT
      {
         return _trainer;
      }

      iterator operator++( ) NU_NOEXCEPT
      {
         ++_epoch;
         return *this;
      }

      iterator operator++( int ) NU_NOEXCEPT // post
      {
         iterator ret = *this;
         ++_epoch;
         return ret;
      }

      bool operator==( iterator & other ) const NU_NOEXCEPT
      {
         return ( 
            _trainer == other._trainer && 
            _epoch == other._epoch );
      }

      bool operator!=( iterator & other ) const NU_NOEXCEPT
      {
         return !this->operator==( other );
      }
   };


   iterator begin()
   {
      return iterator(*this, 0);
   }


   iterator end()
   {
      return iterator(*this, this->_epochs + 1);
   }


   nn_trainer_t(
      Net & nn,
      size_t epochs,
      double min_err
      ) :
      _nn(nn),
      _epochs(epochs),
      _min_err(min_err),
      _err(0.0)
   {}


   size_t get_epochs() const NU_NOEXCEPT
   {
      return _epochs;
   }


   double get_min_err() const NU_NOEXCEPT
   {
      return _min_err;
   }


   double get_error() const NU_NOEXCEPT
   {
      return _err;
   }


   //! Trains the net using a single sample
   bool train(
      const Input& input, 
      const Target& target, 
      cost_func_t err_cost_f)
   {
      _nn.set_inputs(input);
      _nn.back_propagate(target);

      // Compute error for this sample
      _err = err_cost_f(_nn, target);

      return _err < _min_err;
   }


   //! Trains the net using a training set of samples
   template <class TSet>
   size_t train(
      const TSet& training_set, 
      cost_func_t err_cost_f,
      progress_cbk_t * progress_cbk = nullptr)
   {
      size_t epoch = 0;

      for (; epoch < _epochs; ++epoch )
      {
         for ( const auto & sample : training_set )
         {
            if ( progress_cbk )
               ( *progress_cbk )( _nn, sample.first, sample.second, epoch );

            if ( train(sample.first, sample.second, err_cost_f) == true )
               return epoch;
         }
      }

      return epoch;
   }

   protected:
      Net & _nn;
      size_t _epochs;
      double _min_err;
      double _err;

};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_TRAINER_H__
