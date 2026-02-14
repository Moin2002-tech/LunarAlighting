//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_MODELUTILS_HPP
#define LUNARALIGHTINGRL_MODELUTILS_HPP

#include<torch/nn.h>

#include<vector>

namespace LunarAlighting
{
    /**
     * @brief Flattens multi-dimensional tensors into 1D vectors
     *
     * `FlattenImpl` is a PyTorch neural network module that reshapes input tensors
     * by flattening all dimensions except the batch dimension (dimension 0).
     * This is commonly used to convert feature maps from convolutional layers
     * into feature vectors for fully-connected layers.
     *
     * @note Inherits from `nn::Module` to integrate seamlessly with PyTorch's
     *       module system and training pipeline.
    */
    struct FlattenImpl : torch::nn::Module
    {
        /**
         * @brief Flattens the input tensor
         *
         * Reshapes the input tensor such that the batch dimension is preserved
         * and all other dimensions are merged into a single feature dimension.
         *
         * @param x Input tensor with shape [batch_size, dim1, dim2, ...].
         *          Can have arbitrary number of dimensions >= 2.
         * @return Flattened tensor with shape [batch_size, dim1 * dim2 * ...].
         *         The batch dimension remains at index 0.
         *
         * @note This operation does not change the order of elements in memory,
         *       only the shape metadata.
         */
        torch::Tensor forward(torch::Tensor &x);

    };
    TORCH_MODULE(Flatten);

    /**
     * @brief Initializes weights and biases for neural network parameters
     *
     * Applies Xavier/Glorot initialization to weight tensors and zero-initialization
     * to bias tensors. Weight gains allow custom scaling of the standard initialization,
     * useful for different activation functions or architecture requirements.
     *
     * @param parameters Ordered dictionary of named tensors representing model
     *                   parameters. Names typically follow PyTorch convention
     *                   (e.g., "layer.weight", "layer.bias").
     * @param weightGain Multiplicative scaling factor for weight initialization.
     *                    A gain of 1.0 applies standard Xavier initialization.
     *                    Gains > 1.0 increase variance (steeper initialization),
     *                    gains < 1.0 decrease variance (flatter initialization).
     *                    Common values: 1.0 for sigmoid/tanh, sqrt(2) for ReLU.
     * @param biasGain Multiplicative scaling factor for bias initialization.
     *                  Most commonly 0.0 to initialize biases to zero.
     *                  Non-zero values can be used for specialized initialization
     *                  strategies (e.g., biasing towards certain activation regions).
     *
     * @note Automatically identifies weight vs. bias tensors based on parameter names.
     *       Parameters with "weight" or "weight_*" in the name receive weight
     *       initialization; parameters with "bias" in the name receive bias
     *       initialization. Others are left unchanged.
     *
     * @note This function directly modifies the tensors in the provided dictionary.
     *       No new tensors are created; weights are initialized in-place.
     */
    void  initWeights(torch::OrderedDict<std::string,torch::Tensor> parameters, double weightGain, double biasGain);

}


#endif //LUNARALIGHTINGRL_MODELUTILS_HPP