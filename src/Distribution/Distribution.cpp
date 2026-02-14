//
// Created by moinshaikh on 1/30/26.
//

#include"../../include/Distribution/Distribution.hpp"
#include<vector>
#include<ctype.h>

namespace LunarAlighting
{
    std::vector<int64_t> Distribution::extendedShape(c10::ArrayRef<int64_t> &sampleShapes)
    {
        std::vector<int64_t> outputShape;
        outputShape.insert(outputShape.end(), sampleShapes.begin(), sampleShapes.end());
        outputShape.insert(outputShape.end(),batch_shape.begin(), batch_shape.end());
        outputShape.insert(outputShape.end(),event_shape.begin(), event_shape.end());
        return outputShape;

    }
}
