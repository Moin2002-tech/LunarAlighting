//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_NNBASE_HPP
#define LUNARALIGHTINGRL_NNBASE_HPP
#include<torch/torch.h>
#include<torch/nn.h>
#include<vector>

namespace LunarAlighting
{
    /**
     * @brief Base class for neural network policies
     *
     * `NNBase` provides a common interface for neural network modules used in
     * reinforcement learning policies. It supports both feed-forward and recurrent
     * (GRU-based) architectures, with virtual methods that must be overridden by
     * derived classes to implement specific policy architectures.
     *
     */

    class NNBase  : public torch::nn::Module
    {
    private:
        torch::nn::GRU gatedRecurrentUnit; // GRU Module for Recurrent processing
        unsigned int  hiddenSize;
        bool recurrent;
    public:
        /**
         * @brief Constructs a NNBase neural network module
         *
         * Initializes the base module with recurrent configuration options.
         * If `recurrent` is true, a GRU module is initialized with the provided
         * input and hidden sizes.
         *
         * @param recurrent Whether to use a recurrent (GRU) architecture
         * @param recurrentInputSize Input size for the GRU (if recurrent)
         * @param hiddenSize Size of the hidden state dimension
        */
        NNBase(bool recurrent,
            unsigned int recurrentInputSize,
            unsigned int hiddenSize
            );


        /**
         * @brief Forward pass through the neural network
         *
         * Pure virtual method that must be implemented by derived classes.
         * Processes inputs and hidden states, returning transformed representations
         * and potentially updated hidden states.
         *
         * @param inputs Input tensor for the forward pass
         * @param hxs Hidden state tensor (for recurrent architectures)
         * @param masks Mask tensor indicating valid timesteps
         * @return Vector of output tensors containing processed representations
         */

        virtual std::vector<torch::Tensor> forward(
            torch::Tensor inputs,
            torch::Tensor hxs,
            torch::Tensor masks)= 0;
        /**
         * @brief Forward pass through a GRU module
         *
         * Helper method that applies GRU processing to inputs with the provided
         * hidden states and masks. Useful for recurrent policy architectures.
         *
         * @param x Input tensor for GRU processing
         * @param hxs Hidden state tensor for the GRU
         * @param masks Mask tensor to handle variable-length sequences
         * @return Vector of tensors containing GRU outputs and updated hidden states
        */
        std::vector<torch::Tensor> forwardGatedRecurrentUnits(torch::Tensor inputs,
            torch::Tensor hxs,
            torch::Tensor masks);

        unsigned int getHiddenSize() const;

        inline int getOutputSize() const
        {
            return hiddenSize;
        }

        inline bool isRecurrent() const
        {
            return recurrent;
        }

    };
 }

#endif //LUNARALIGHTINGRL_NNBASE_HPP