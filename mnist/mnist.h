//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_vector.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif


//! This class represents a single handwritten digit and its classification
//! (label)
class DigitData {
public:
    using data_t = std::vector<char>;

private:
    size_t _dx {};
    size_t _dy {};
    int _label {};
    data_t _data;

public:
    //! ctor
    DigitData(size_t dx, size_t dy, int label, const data_t& data) noexcept
        : _dx(dx)
        , _dy(dy)
        , _label(label)
        , _data(data)
    {
    }

    //! copy ctor
    DigitData(const DigitData&) = default;

    //! copy assign operator
    DigitData& operator=(const DigitData& other) = default;

    //! move ctor
    DigitData(DigitData&& other) noexcept = default;

    //! move assign operator
    DigitData& operator=(DigitData&& other) noexcept = default;

    //! Returns the digit width in pixels
    size_t get_dx() const noexcept { return _dx; }

    //! Returns the digit height in pixels
    size_t get_dy() const noexcept { return _dy; }

    //! Returns the digit classification
    int getLabel() const noexcept { return _label; }

    //! Returns a reference to internal data
    const data_t& data() const noexcept { return _data; }

    //! Converts the image data into a vector normalizing each item
    //! within the range [0.0, 1.0]
    void toVect(nu::Vector& v) const noexcept;

    //! Converts a label into a vector where the items are all zeros
    //! except for the item with index corrisponding to the label value
    //! its self (which is within range [0, 9]
    void labelToTarget(nu::Vector& v) const noexcept;

#ifdef _WIN32
    //! Draw the digit image on the window
    void paint(int xoff, int yoff, HWND hwnd = nullptr) const noexcept;
#endif
};

//! This class provides a method to load MNIST pair of images and labels files
//! The data can be retrieved as a list of DigitData objects
class TrainingData {
public:
    using data_t = std::list<std::unique_ptr<DigitData>>;

    //! Return a reference to a list of DigitData objects
    const data_t& data() const noexcept { return _data; }

    //! reshuffle objects
    void reshuffle()
    {
        std::vector<std::unique_ptr<DigitData>> data;

        for (auto& e : _data) {
            data.emplace_back(std::move(e));
        }

        std::random_device rng;
        std::mt19937 urng(rng());
        std::shuffle(data.begin(), data.end(), urng);

        _data.clear();

        for (auto& e : data) {
            _data.push_back(std::move(e));
        }
    }

    class LabelsFileNotFoundException : public std::runtime_error {
    public:
        LabelsFileNotFoundException()
            : std::runtime_error("Labels file not found")
        {
        }
    };

    class ImagesFileNotFoundException : public std::runtime_error {
    public:
        ImagesFileNotFoundException()
            : std::runtime_error("Images file not found")
        {
        }
    };

    class LabelsFileReadErrorException : public std::runtime_error {
    public:
        LabelsFileReadErrorException()
            : std::runtime_error("Error reading labels file")
        {
        }
    };

    class ImagesFileReadErrorException : public std::runtime_error {
    public:
        ImagesFileReadErrorException()
            : std::runtime_error("Error reading images file")
        {
        }
    };

    class LabelsFileWrongMagicException : public std::runtime_error {
    public:
        LabelsFileWrongMagicException()
            : std::runtime_error("Wrong magic number in labels file")
        {
        }
    };

    class ImagesFileWrongMagicException : public std::runtime_error {
    public:
        ImagesFileWrongMagicException()
            : std::runtime_error("Wrong magic number in images file")
        {
        }
    };

    class NumberOfItemsMismatchException : public std::runtime_error {
    public:
        NumberOfItemsMismatchException()
            : std::runtime_error("Number of items mismatch")
        {
        }
    };

    TrainingData() = delete;

    TrainingData(const std::string& lbls_file,
        const std::string& imgs_file) throw()
        : _lblsFile(lbls_file)
        , _imgsFile(imgs_file)
    {
    }

    //! Load data.
    //! @return number of loaded items or -1 in case of error
    int load();

private:
    std::string _lblsFile;
    std::string _imgsFile;

    data_t _data;
};
