#pragma once
//
// Created by moinshaikh on 1/27/26.
//

#ifndef LUNARALIGHTINGRL_SPACE_HPP
#define LUNARALIGHTINGRL_SPACE_HPP

#include<string>
#include<torch/torch.h>

struct ActionSpace
{
    std::string type;
    std::vector<int64_t> shape;
};



#endif //LUNARALIGHTINGRL_SPACE_HPP