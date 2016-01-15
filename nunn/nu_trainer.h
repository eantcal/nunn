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

   //! Cost function pointer
   using cost_fptr_t = double(Net&, const Target&);

   //! Cost function object wrapper
   using cost_func_t = std::function<cost_fptr_t>;

   //! Progress call back function pointer
   using progress_fptr_t = 
      void(
         Net&, 
         const Input&, 
         const Target&, 
         size_t /*epoch*/, 
         size_t /* sample_idx */,
         double /* err */);

   //! Progress call back function object wrapper
   using progress_cbk_t = std::function<progress_fptr_t>;

   //! Trainer iterator
   struct iterator
   {
      friend class nn_trainer_t;

   private:
      nn_trainer_t * _trainer = nullptr;
      size_t _epoch = 0;

      iterator(type_t & trainer, size_t epoch) NU_NOEXCEPT
         : 
         _trainer(&trainer),
         _epoch(epoch)
      {
      }

   public:
      //! Copy constructor
      iterator(const iterator & it) = default;

      //! Assignment constructor
      iterator& operator=(const iterator & it) = default;

      //! Move constructor
      iterator(iterator && it) NU_NOEXCEPT :
      _trainer(std::move(it._trainer)),
         _epoch(std::move(it._epoch))
      {
      }

      //! Move assignment operator
      iterator& operator=(iterator && it) NU_NOEXCEPT
      {
         if (&it != this)
         {
            _trainer = std::move(it._trainer);
            _epoch = std::move(it._epoch);
         }

         return *this;
      }

      //! Return epoch number
      size_t get_epoch() const NU_NOEXCEPT
      {
         return _epoch;
      }

      //! Pointer Dereference operator
      type_t& operator*() const NU_NOEXCEPT
      {
         return *_trainer;
      }

      //! Arrow operator
      type_t* operator->() const NU_NOEXCEPT
      {
         return _trainer;
      }

      //! Increment operator
      iterator operator++() NU_NOEXCEPT
      {
         ++_epoch;
         return *this;
      }

      //! Post increment operator
      iterator operator++(int) NU_NOEXCEPT // post
      {
         iterator ret = *this;
         ++_epoch;
         return ret;
      }

      //! Equal-To operator
      bool operator==(iterator & other) const NU_NOEXCEPT
      {
         return (
            _trainer == other._trainer &&
            _epoch == other._epoch);
      }

      //! Not-Equal-To operator
      bool operator!=(iterator & other) const NU_NOEXCEPT
      {
         return !this->operator==(other);
      }
   };


   //! Return an iterator to the first epoch
   iterator begin() NU_NOEXCEPT
   {
      return iterator(*this, 0);
   }


   //! Return an iterator to the last epoch
   iterator end() NU_NOEXCEPT
   {
      return iterator(*this, this->_epochs + 1);
   }


   //! Constructor
   //! @nn:      the network to train
   //! @epochs:  max epoch count at which to stop training
   //! @min_err: min error value at which to stop training
   nn_trainer_t(
      Net & nn,
      size_t epochs,
      double min_err
      )  NU_NOEXCEPT :
      _nn(nn),
      _epochs(epochs),
      _min_err(min_err),
      _err(0.0)
   {}


   //! Return the max number of epochs
   size_t get_epochs() const NU_NOEXCEPT
   {
      return _epochs;
   }

   
   //! Return the min error value at which to stop training
   double get_min_err() const NU_NOEXCEPT
   {
      return _min_err;
   }


   //! Return current epoch error
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
   size_t run_training(
      const TSet& training_set,
      cost_func_t err_cost_f,
      progress_cbk_t progress_cbk = nullptr)
   {
      size_t epoch = 0;

      for (; epoch < _epochs; ++epoch)
      {
         size_t sample_idx = 0;

         for (const auto & sample : training_set)
         {
            if (progress_cbk)
               progress_cbk(
                  _nn,
                  sample.first,
                  sample.second,
                  epoch,
                  sample_idx++,
                  _err);

            if (train(sample.first, sample.second, err_cost_f) == true)
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
