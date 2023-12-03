//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

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
Pixel values are 0 to 255. 0 means background (white), 255 means foreground
(black).
*/

#include "mnist.h"

#include <fstream>
#include <vector>

void DigitData::toVect(nu::Vector<double>& v) const noexcept
{
    size_t vsize = data().size();
    v.resize(vsize);

    for (size_t i = 0; i < vsize; ++i)
        v[i] = double((unsigned char)data()[i]) / 255.0;
}

void DigitData::labelToTarget(nu::Vector<double>& v) const noexcept
{
    v.resize(10);
    std::fill(v.begin(), v.end(), 0.0);
    v[getLabel() % 10] = 1.0;
}

#ifdef _WIN32
void DigitData::paint(int xoff, int yoff, HWND hwnd) const noexcept
{
    size_t dx = get_dx();
    size_t dy = get_dy();

    if (!hwnd)
        hwnd = GetConsoleWindow();

    HDC hdc = GetDC(hwnd);

    const auto& dataBits = data();

    int idx = 0;

    for (size_t y = 0; y < dy; ++y) {
        for (size_t x = 0; x < dx; ++x) {
            int c = dataBits[idx++];

            SetPixel(hdc,
                int(x) + xoff,
                int(y) + yoff,
                RGB(255 - c, 255 - c, 255 - c));
        }
    }

    ReleaseDC(GetConsoleWindow(), hdc);
}
#endif

int TrainingData::load()
{
    const std::vector<char> magicLbls = { 0, 0, 8, 1 };
    const std::vector<char> magicImgs = { 0, 0, 8, 3 };

    int ret = -1;

    auto to_int32 = [&](const std::vector<char>& buf) {
        assert(buf.size() >= 4);
        return ((buf[0] << 24) & 0xff000000) | ((buf[1] << 16) & 0x00ff0000) | ((buf[2] << 8) & 0x0000ff00) | (buf[3] & 0x000000ff);
    };

    std::ifstream flbls(_lblsFile, std::ios::binary);
    std::ifstream fimgs(_imgsFile, std::ios::binary);

    if (!flbls) {
        throw Exception::lbls_file_not_found;
    }

    if (!fimgs) {
        throw Exception::imgs_file_not_found;
    }

    std::vector<char> buf(4);

    if (!flbls.read(buf.data(), 4)) {
        throw Exception::lbls_file_read_error;
    }

    if (buf != magicLbls) {
        throw Exception::lbls_file_wrong_magic;
    }

    if (!fimgs.read(buf.data(), 4)) {
        throw Exception::imgs_file_read_error;
    }

    if (buf != magicImgs) {
        throw Exception::imgs_file_wrong_magic;
    }

    if (!flbls.read(buf.data(), 4)) {
        throw Exception::lbls_file_read_error;
    }

    const int32_t n_of_lbls = to_int32(buf);

    if (!fimgs.read(buf.data(), 4)) {
        throw Exception::imgs_file_read_error;
    }

    const int32_t n_of_imgs = to_int32(buf);

    if (n_of_lbls != n_of_imgs) {
        throw Exception::n_of_items_mismatch;
    }

    ret = n_of_imgs;

    if (!fimgs.read(buf.data(), 4)) {
        throw Exception::imgs_file_read_error;
    }

    const int32_t n_rows = to_int32(buf);

    if (!fimgs.read(buf.data(), 4)) {
        throw Exception::imgs_file_read_error;
    }

    const int32_t n_cols = to_int32(buf);

    const size_t imgSize = size_t(n_rows * n_cols);

    for (int32_t i = 0; i < n_of_lbls; ++i) {
        if (!flbls.read(buf.data(), 1)) {
            break;
        }

        DigitData::data_t data(imgSize);

        const int label = int(buf[0]) & 0xff;

        if (!fimgs.read(reinterpret_cast<char*>(data.data()), data.size())) {
            break;
        }

        std::unique_ptr<DigitData> digit_info(new DigitData(size_t(n_cols), size_t(n_rows), label, data));
        _data.push_back(std::move(digit_info));
    }

    return ret;
}
