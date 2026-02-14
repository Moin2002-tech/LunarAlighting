//
// Created by moinshaikh on 1/28/26.
//

#ifndef LUNARALIGHTINGRL_CNNBASE_HPP
#define LUNARALIGHTINGRL_CNNBASE_HPP

#include<torch/torch.h>

#include"nnBase.hpp"

namespace LunarAlighting
{
    /**
     * @brief Convolutional neural network (CNN) base policy network
     *
     * `CnnBase` is a convolutional architecture designed for processing image or
     * spatial observation inputs in reinforcement learning. It extracts visual
     * features through convolutional layers and produces both policy and value
     * outputs. Supports both feed-forward and recurrent (GRU-based) temporal
     * processing through inheritance from `NNBase`.
     *
     * The architecture consists of:
     * - Main CNN feature extraction pipeline (convolutional layers with pooling)
     * - Flattened feature representation from main network
     * - Critic linear layers for value function estimation
     *
     * Typical use cases:
     * - Atari game environments with visual observations
     * - Robotic vision tasks with image inputs
     * - Any environment with multi-channel spatial data
     *
     * @note Inherits from `NNBase` to leverage recurrent infrastructure and maintain
     *       consistent interface across policy types.
     */
    class CnnBase : public NNBase
    {
    private:
        torch::nn::Sequential main;          /**< Sequential CNN module for feature extraction.
                                       Contains convolutional layers, activation functions,
                                       and pooling operations that progressively extract
                                       spatial features from image inputs. Output is flattened
                                       to feed into critic networks. */
        torch::nn::Sequential criticLinear; /**< Sequential module for critic (value) network.
                                           Processes flattened features from main CNN through
                                           fully-connected layers to produce scalar value estimate. */

    public:
        /**
         * @brief Constructs a CNN-based policy network
         *
         * Initializes convolutional and critic networks optimized for processing
         * image/spatial observations. The CNN stack automatically adapts to input
         * channels provided while building stable feature representations.
         *
         * @param num_inputs Number of input channels in observation images.
         *                   Typical values:
         *                   - 1 for grayscale observations
         *                   - 3 for RGB observations
         *                   - N for multi-frame stacking (e.g., 4 stacked frames)
         *                   Determines the first convolutional layer input channels.
         * @param recurrent  If true, uses GRU recurrence for temporal processing;
         *                   if false, processes each frame independently (feed-forward).
         *                   Recurrent mode useful for temporal dependencies in video.
         *                   Default: false.
         * @param hidden_size Dimension of internal representations and GRU state.
         *                    Larger values increase model capacity to learn complex features.
         *                    Default: 512 (larger than MlpBase due to visual complexity).
         *                    Typical range: 256-1024 for image inputs.
         *
         * @note The constructor initializes convolutional layers but does not perform
         *       weight initialization. Call init_weights() separately for custom schemes.
         */
        CnnBase(unsigned int numInput,
            bool recurrect = false,
            unsigned int hiddenSize = 512);

        /**
        * @brief Forward pass through the CNN policy network
        *
        * Processes image observations through convolutional feature extraction,
        * then computes policy and value outputs. Overrides the virtual `forward()`
        * method from `NNBase`.
        *
        * @param inputs Image tensor with shape [sequence_length, batch_size, num_inputs, height, width]
        *               or [batch_size, num_inputs, height, width] for single frames.
        *               Typically:
        *               - For Atari: [T, B, 1, 84, 84] with stacked grayscale frames
        *               - For RGB: [T, B, 3, height, width]
        *               - For multi-frame: [T, B, num_frames*channels, height, width]
        * @param hxs Hidden state tensor for recurrent processing.
        *            Shape: [num_layers, batch_size, hidden_size] for GRU, or
        *            empty tensor for feed-forward architecture.
        *            Carries temporal context across frames in recurrent mode.
        * @param masks Mask tensor indicating valid frames and episode boundaries.
        *              Used to reset recurrent state at episode ends and handle
        *              variable-length sequences. Shape: [sequence_length, batch_size].
        *
        * @return Vector of output tensors containing:
        *         - [0] Value estimate from critic network (scalar per sample)
        *         - [1] Policy output (action distribution parameters)
        *         - [2] Updated hidden states (for recurrent architectures)
        */
        std::vector<torch::Tensor> forward(torch::Tensor inputs,
            torch::Tensor hxs,
            torch::Tensor masks) override;
    };
}
#endif //LUNARALIGHTINGRL_CNNBASE_HPP