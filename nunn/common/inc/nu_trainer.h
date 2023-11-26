//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//

#pragma once
namespace nu {

//! The trainer class is a helper class for neural networks training
template <class Net, class Input, class Target>
class NNTrainer
{
    friend class iterator;
public:
    using type_t = NNTrainer<Net, Input, Target>;

    //! Cost function pointer
    using cost_fptr_t = double(Net&, const Target&);

    //! Cost function object wrapper
    using costFunction_t = std::function<cost_fptr_t>;

    //! Progress call back function prototype
    //! It can break training session returning true
    using progressCallbackPrototype_t =
        bool(Net&, 
             const Input&, 
             const Target&,
             size_t /*epoch*/, 
             size_t /* sampleIdx */,
             double /* err */);

    //! Progress call back function object wrapper
    using progressCallback_t = std::function<progressCallbackPrototype_t>;

    //! Trainer iterator
    struct iterator {
        friend class NNTrainer;

    private:
        NNTrainer* _trainer = nullptr;
        size_t _epoch = 0;

        iterator(type_t& trainer, size_t epoch) noexcept
          : _trainer(&trainer),
            _epoch(epoch)
        {
        }

    public:
        //! Copy constructor
        iterator(const iterator& it) = default;

        //! Assignment constructor
        iterator& operator=(const iterator& it) = default;

        //! Move constructor
        iterator(iterator&& it) noexcept : 
            _trainer(std::move(it._trainer)),
            _epoch(std::move(it._epoch))
        {
        }

        //! Move assignment operator
        iterator& operator=(iterator&& it) noexcept {
            if (&it != this) {
                _trainer = std::move(it._trainer);
                _epoch = std::move(it._epoch);
            }

            return *this;
        }

        //! Return epoch number
        size_t get_epoch() const noexcept { 
            return _epoch; 
        }

        //! Pointer Dereference operator
        type_t& operator*() const noexcept { 
            return *_trainer; 
        }

        //! Arrow operator
        type_t* operator->() const noexcept { 
            return _trainer; 
        }

        //! Increment operator
        iterator operator++() noexcept {
            ++_epoch;
            return *this;
        }

        //! Post increment operator
        iterator operator++(int)noexcept {
            iterator ret = *this;
            ++_epoch;
            return ret;
        }

        //! Equal-To operator
        bool operator==(iterator& other) const noexcept {
            return (_trainer == other._trainer && _epoch == other._epoch);
        }

        //! Not-Equal-To operator
        bool operator!=(iterator& other) const noexcept {
            return !this->operator==(other);
        }
    };


    //! Return an iterator to the first epoch
    iterator begin() noexcept { 
        return iterator(*this, 0); 
    }

    //! Return an iterator to the last epoch
    iterator end() noexcept { 
        return iterator(*this, this->_epochs + 1); 
    }

    //! Constructor
    //! @nn:      the network to train
    //! @epochs:  max epoch count at which to stop training
    //! @minErr: min error value at which to stop training
    NNTrainer(Net& nn, size_t epochs, double minErr = -1.0) noexcept
      : _nn(nn),
        _epochs(epochs),
        _minError(minErr),
        _err(0.0)
    {
    }

    //! Return the max number of epochs
    size_t getEpochs() const noexcept { 
        return _epochs; 
    }

    //! Return the expected min error value at which to stop training 
    //! on sample basis.
    //! If negative it will be ignored
    double getMinErr() const noexcept { 
        return _minError; 
    }

    //! Return current epoch error
    double getError() const noexcept { 
        return _err; 
    }

    //! Trains the net using a single sample
    bool train(const Input& input, const Target& target, costFunction_t errCost) {
        _nn.setInputVector(input);
        _nn.backPropagate(target);

        // Compute error for this sample
        _err = errCost(_nn, target);

        return _err < getMinErr();
    }


    //! Trains the net using a training set of samples
    template <class TSet>
    size_t runTraining(
        const TSet& trainingSet, 
        costFunction_t errCost,
        progressCallback_t progressCbk = nullptr,
        double p2use = 1.0)
    {
        size_t epoch = 0;
        size_t end_idx = size_t(double(trainingSet.size()) * p2use);

        for (bool bContinue = true; 
             bContinue && epoch < _epochs; 
             ++epoch) 
        {
            size_t sampleIdx = 0;

            for (const auto& [input, target] : trainingSet) {
                if (progressCbk)
                    bContinue = !progressCbk(
                        _nn, input, target, epoch, sampleIdx++, _err);

                if (train(input, target, errCost) == true)
                    return epoch;

                if (!bContinue || (p2use < 1.0 && sampleIdx >= end_idx)) 
                    return epoch;
            }
        }

        return epoch;
    }

  protected:
    Net& _nn;
    size_t _epochs;
    double _minError;
    double _err;
};

} // namespace nu

