#pragma once 
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using half_placeholder = uint16_t ; 

template<typename T>
void save_to_npy(const std::string& filename, const T* data, const std::vector<int>& shape) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("");

    std::string descr = "<";
    
    if constexpr (std::is_same_v<T, float>) {
        descr += "f4";
    } else if constexpr (std::is_same_v<T, double>) {
        descr += "f8";
    } else if constexpr (sizeof(T) == 2) { 
        descr += "f2"; 
    } else if constexpr (std::is_same_v<T, int32_t>) {
        descr += "i4";
    } else {
        descr += "u1";
    }

    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]) + ( (shape.size()==1 || i < shape.size()-1) ? "," : "" );
    }
    dict += "), }";

    int current_len = 6 + 1 + 1 + 2 + dict.length() + 1;
    int padding = 64 - (current_len % 64);
    dict.append(padding, ' ');
    dict += '\n';

    file.write("\x93NUMPY\x01\x00", 8);
    uint16_t header_len = static_cast<uint16_t>(dict.length());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    file.write(dict.c_str(), dict.length());

    size_t total_elements = 1;
    for (size_t dim : shape) total_elements *= dim;
    file.write(reinterpret_cast<const char*>(data), total_elements * sizeof(T));
}

