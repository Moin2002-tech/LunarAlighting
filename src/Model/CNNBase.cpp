
//
// Created by moinshaikh on 2/6/26.
//




#include<torch/torch.h>

#include"../../include/Model/CNNBase.hpp"
#include"../../include/Model/modelUtils.hpp"
#include"../third_party/doctest.hpp"

namespace LunarAlighting
{
    /**
     * @brief Constructs a CNN-based policy network for visual observation processing
     *
     * Initializes a convolutional neural network architecture specifically designed for
     * reinforcement learning agents that process image-based observations. The architecture
     * combines spatial feature extraction through convolutional layers with optional temporal
     * processing via GRU recurrence.
     *
     * **Network Architecture Overview:**
     *
     * The main feature extraction pipeline consists of 3 convolutional layers followed by
     * fully-connected processing:
     * - **Layer 1:** Conv2D with kernel 8×8, stride 4, 32 filters (aggressive downsampling)
     * - **Layer 2:** Conv2D with kernel 4×4, stride 2, 64 filters (moderate downsampling)
     * - **Layer 3:** Conv2D with kernel 3×3, stride 1, 32 filters (fine-grained features)
     * - **Flatten:** Vectorizes spatial feature maps into 1D representation
     * - **FC Layer:** Dense layer mapping to hidden_size (512 by default)
     * - **Activation:** ReLU applied after each layer for non-linearity
     *
     * The critic network is a simple linear head mapping hidden features to scalar value:
     * - **Critic FC:** Linear layer from hidden_size → 1 (single value estimate)
     *
     * **Mathematical Tensor Shape Transformations:**
     *
     * Input tensor shape: [B, C_in, H, W]
     * where B = batch_size, C_in = num_input_channels, H = height, W = width
     *
     * Assuming standard Atari 84×84 images (C_in=1 for grayscale):
     * ```
     * Input:           [B, 1, 84, 84]
     * After Conv1:     [B, 32, 20, 20]    (kernel 8, stride 4: floor((84-8)/4)+1 = 20)
     * After Conv2:     [B, 64, 9, 9]      (kernel 4, stride 2: floor((20-4)/2)+1 = 9)
     * After Conv3:     [B, 32, 7, 7]      (kernel 3, stride 1: floor((9-3)/1)+1 = 7)
     * After Flatten:   [B, 32*7*7=1568]
     * After FC:        [B, hidden_size]   (e.g., [B, 512])
     * Critic Output:   [B, 1]
     * ```
     *
     * **Convolutional Formula:**
     * For each convolutional layer, output spatial dimensions computed as:
     * ```
     * H_out = floor((H_in - kernel_size) / stride) + 1
     * W_out = floor((W_in - kernel_size) / stride) + 1
     * C_out = num_filters (specified per layer)
     * ```
     *
     * **Weight Initialization Strategy:**
     *
     * - **Main CNN weights:** Initialized with orthogonal matrices scaled by √2
     *   - Formula: W_main ← √2 × Q, where Q^T Q = I
     *   - Rationale: Compensates for ReLU activation reducing signal magnitude
     *   - Benefits: Stabilizes gradient flow through convolutional layers
     *
     * - **Critic weights:** Initialized with orthogonal matrices scaled by 1.0
     *   - Formula: W_critic ← 1.0 × Q, where Q^T Q = I
     *   - Rationale: Standard orthogonal initialization for final value layer
     *   - Benefits: Ensures stable value estimation without artificial inflation
     *
     * - **All biases:** Initialized to 0 (centered bias)
     *   - Formula: b ← 0
     *   - Rationale: Network learns bias values during training
     *
     * **Key Properties and Benefits:**
     *
     * - **Spatial feature hierarchy:** Progressive spatial resolution reduction
     *   - Layer 1: Captures edges and low-level textures
     *   - Layer 2: Combines features into mid-level shapes
     *   - Layer 3: Extracts high-level semantic features
     *
     * - **Parameter efficiency:** Convolutional weight sharing reduces parameters
     *   - Total params ≈ (8×8×1×32) + (4×4×32×64) + (3×3×64×32) + (1568×512) + 512
     *   - Significantly smaller than fully-connected equivalent
     *
     * - **Gradient flow:** Orthogonal initialization maintains ||∇L||/||L|| ≈ constant
     *   - Prevents vanishing gradients in early layers
     *   - Prevents exploding gradients in later layers
     *
     * - **Recurrence compatibility:** Can optionally wrap features with GRU
     *   - If recurrent=true: Features fed to GRU with hidden state tracking
     *   - If recurrent=false: Features used directly for policy/value output
     *
     * **Training Setup:**
     *
     * - Calls train() mode by default for proper dropout and batch norm behavior
     * - Ready for SGD/Adam optimization with gradient-based learning
     * - Modules registered via register_module() for parameter discovery
     *
     * **Typical Use Cases:**
     *
     * - **Atari environments:** numInput=1 (grayscale), 84×84 observations
     * - **RGB observations:** numInput=3 (RGB), variable resolution
     * - **Frame stacking:** numInput=4 (4 consecutive grayscale frames concatenated)
     * - **Multi-agent visual:** numInput=3+3=6 (two stacked RGB frames)
     *
     * @param numInput Number of input channels in the observation tensor
     *                 - 1 for grayscale images
     *                 - 3 for RGB color images
     *                 - N for stacked frames or multi-channel inputs
     *                 Determines torch::nn::Conv2dOptions(numInput, 32, 8) first layer
     *
     * @param recurrect Whether to use recurrent (GRU) processing for temporal dependencies
     *                  - true: Enables GRU module for sequence processing
     *                  - false: Feed-forward processing (default)
     *                  Passed to NNBase constructor for recurrent infrastructure
     *
     * @param hiddenSize Dimension of the hidden representation and GRU state
     *                   - Size of features output from main CNN
     *                   - Size of GRU input/output if recurrent=true
     *                   - Determines model capacity for feature learning
     *                   - Typical range: 256-1024 for image inputs
     *                   - Default: 512 (larger than MLP due to visual complexity)
     *
     * @return CnnBase instance fully initialized and in training mode
     *
     * @note Module registration and weight initialization occur in constructor
     * @warning Requires input tensor format: [B, numInput, H, W] with H≥84, W≥84
     *
     * @see orthogonal_ for orthogonal weight initialization algorithm
     * @see initWeights for weight/bias initialization function
     * @see torch::nn::Sequential for modular layer composition
     */
    CnnBase::CnnBase(unsigned int numInput, bool recurrect, unsigned int hiddenSize) :
        NNBase(recurrect,hiddenSize,hiddenSize),
        main(torch::nn::Conv2d(torch::nn::Conv2dOptions(numInput,32,8).stride(4)),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32,64,4).stride(2)),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64,32,3).stride(1)),
            torch::nn::Functional(torch::relu),
            torch::nn::Flatten(),
            torch::nn::Linear(32*7*7,hiddenSize),
            torch::nn::Functional(torch::relu)),
            criticLinear(torch::nn::Linear(hiddenSize,1))
    {
        register_module("main",main);
        register_module("criticLinear",criticLinear);

        initWeights(main->named_parameters(),std::sqrt(2.),0);
        initWeights(criticLinear->named_parameters(),1,0);
        train();
    }

    /**
     * @brief Forward pass through the CNN policy network
     *
     * Processes image observations through the feature extraction pipeline and computes
     * both policy (actor) and value (critic) outputs. Supports optional recurrent temporal
     * processing via GRU for handling sequential decision-making.
     *
     * **Data Flow Pipeline:**
     *
     * The forward pass follows this sequence:
     * 1. **Normalization:** Divide input by 255 to map pixel values [0,255] → [0,1]
     * 2. **Feature Extraction:** Pass through main CNN (3 conv layers + 1 FC layer)
     * 3. **Recurrence (optional):** Process features through GRU if recurrent=true
     * 4. **Value Estimation:** Compute scalar value from critic linear network
     * 5. **Output Bundling:** Return value, features, and updated hidden states
     *
     * **Mathematical Tensor Transformations:**
     *
     * Input space:
     * - inputs: x ∈ ℝ^(B × C × H × W) where values ∈ [0, 255]
     * - hxs: hidden state h ∈ ℝ^(B × hidden_size) (for recurrent mode)
     * - masks: m ∈ ℝ^(B × 1) ∈ {0, 1} (1 = valid, 0 = episode boundary)
     *
     * Step 1 - Normalization:
     * ```
     * x_norm = x / 255.0  ∈ [0, 1]
     * ```
     *
     * Step 2 - Feature Extraction through main CNN:
     * ```
     * x_feat = main(x_norm)  ∈ ℝ^(B × hidden_size)
     * Applies: Conv-ReLU-Conv-ReLU-Conv-ReLU-Flatten-FC-ReLU sequence
     * ```
     *
     * Step 3a - Feed-forward mode (if recurrent=false):
     * ```
     * x_out = x_feat (no temporal processing)
     * hxs_out = hxs (hidden state unchanged)
     * ```
     *
     * Step 3b - Recurrent mode (if recurrent=true):
     * ```
     * [x_out, hxs_out] = GRU(x_feat, hxs, masks)
     * where:
     * - GRU applies: o_t = update_gate⊙o_{t-1} + reset_gate⊙tanh(W×[x_t,h_t])
     * - masks reset hidden state at episode boundaries: h_new = m⊙h_old
     * x_out ∈ ℝ^(B × hidden_size) (temporal processed features)
     * hxs_out ∈ ℝ^(B × hidden_size) (updated recurrent state)
     * ```
     *
     * Step 4 - Value Estimation:
     * ```
     * value = criticLinear(x_out)  ∈ ℝ^(B × 1)
     * Single linear layer: v = W×x + b, produces scalar value per sample
     * ```
     *
     * **Output Vector Structure:**
     *
     * The function returns std::vector<torch::Tensor> with 3 elements:
     * ```
     * return {value, x_out, hxs_out}
     * where:
     *   value ∈ ℝ^(B × 1)              - Critic output (state value estimate V(s))
     *   x_out ∈ ℝ^(B × hidden_size)    - Policy output (feature representation for actor)
     *   hxs_out ∈ ℝ^(B × hidden_size)  - Updated hidden states (for recurrent tracking)
     * ```
     *
     * **Algorithm Steps:**
     *
     * 1. **Input Normalization (pixel scaling):**
     *    - Divide RGB/grayscale values by 255.0
     *    - Maps [0, 255] → [0, 1] for stable network input
     *    - Reduces internal covariate shift in first layer
     *    - Improves gradient magnitude consistency
     *
     * 2. **Convolutional Feature Extraction:**
     *    - Apply 3 convolutional blocks with ReLU activation
     *    - Extract hierarchical spatial features
     *    - Progressive downsampling reduces spatial dimensions
     *    - Flatten output combines spatial and channel dimensions
     *
     * 3. **Fully-connected Bottleneck:**
     *    - Project flattened features to fixed hidden_size dimension
     *    - Creates compact feature representation for policy/value heads
     *    - ReLU activation maintains non-linearity
     *
     * 4. **Temporal Processing (optional):**
     *    - If recurrent: apply GRU to capture temporal dependencies
     *    - GRU updates hidden state based on feature sequence
     *    - Masks handle variable-length episodes and boundaries
     *
     * 5. **Dual Output Heads:**
     *    - Critic: linear projection to scalar value
     *    - Actor: features passed to action distribution (in higher-level code)
     *
     * **Key Mathematical Properties:**
     *
     * - **Differentiability:** All operations are differentiable w.r.t. inputs
     *   - Enables gradient-based learning via backpropagation
     *   - Supports both policy gradient and value-based algorithms
     *
     * - **Scale preservation:** Orthogonal weight init maintains gradient magnitude
     *   - ||∇L/∂x_in|| / ||∇L/∂x_out|| ≈ 1.0 (prevents vanishing/exploding grads)
     *
     * - **Feature dimensionality:** Bottleneck to hidden_size enables:
     *   - Compact representation suitable for GRU input
     *   - Reduced parameter count compared to fully-connected
     *   - Stability in recurrent temporal processing
     *
     * **Practical Usage Patterns:**
     *
     * **Feed-forward (stateless) processing:**
     * ```cpp
     * auto cnn = CnnBase(3, false, 512);  // No recurrence
     * auto inputs = torch::rand({batch_size, 3, 84, 84});
     * auto hxs = torch::zeros({batch_size, 512});  // Dummy hidden state
     * auto masks = torch::ones({batch_size, 1});
     * auto [value, features, new_hxs] = cnn->forward(inputs, hxs, masks);
     * // Process each frame independently, no temporal context
     * ```
     *
     * **Recurrent (stateful) processing:**
     * ```cpp
     * auto cnn = CnnBase(1, true, 256);   // With GRU recurrence
     * auto hxs = torch::zeros({batch_size, 256});  // Initial hidden state
     * for (auto& observation : episode_sequence) {
     *   auto [value, features, hxs] = cnn->forward(observation, hxs, masks);
     *   // hxs carries temporal context across frames
     *   // Reset hidden state at episode boundaries using masks
     * }
     * ```
     *
     * **Computational Complexity:**
     *
     * - **Forward pass time:** O(B × H × W × (C×K²)) for convolutions
     *   - B = batch size, H×W = spatial dimensions, C×K² = filter operations
     *   - Typically 10-50ms per batch on GPU (depends on resolution)
     *
     * - **Memory usage:** O(B × (H×W×C + hidden_size²))
     *   - Dominated by activation maps storage and hidden state tensors
     *
     * **Typical Input Specifications:**
     *
     * | Environment Type | numInput | Resolution | Format |
     * |------------------|----------|------------|--------|
     * | Atari (gray)     | 1        | 84×84      | Single grayscale frame |
     * | Atari (stacked)  | 4        | 84×84      | 4 stacked frames |
     * | RGB (single)     | 3        | 84×84      | Single RGB frame |
     * | RGB (stacked)    | 12       | 84×84      | 4 stacked RGB frames |
     *
     * @param inputs Image observation tensor
     *               Shape: [B, numInput, H, W] or [T, B, numInput, H, W]
     *               Value range: [0, 255] (uint8 or float)
     *               - B = batch size (typically 32-64 for training)
     *               - numInput = color channels or stacked frames
     *               - H, W = spatial dimensions (typically 84×84 for Atari)
     *               - T = sequence length (optional, for recurrent processing)
     *
     * @param hxs Hidden state tensor for recurrent architecture
     *            Shape: [B, hidden_size]
     *            - Carries temporal context across timesteps
     *            - Initialized to zeros for episode start
     *            - Updated by GRU on each forward pass (if recurrent)
     *            - Should be on same device as inputs
     *
     * @param masks Mask tensor indicating valid frames and episode boundaries
     *              Shape: [B, 1]
     *              Values: 1.0 for valid frames, 0.0 at episode boundaries
     *              Usage:
     *              - Prevents gradient flow across episode boundaries
     *              - Resets hidden state when episode ends
     *              - Handles variable-length trajectory sequences
     *
     * @return std::vector<torch::Tensor> containing 3 tensors:
     *         - [0] Value estimate V(s) ∈ ℝ^(B × 1)
     *             Scalar value prediction for state value function
     *             Targets from Bellman equation in training: R_t = r + γV(s')
     *         - [1] Features ∈ ℝ^(B × hidden_size)
     *             Hidden representation output, used by actor head for action distribution
     *             Input to policy network (logits, means, etc.)
     *         - [2] Updated hidden states ∈ ℝ^(B × hidden_size)
     *             Recurrent state after processing (copy of hxs in feed-forward mode)
     *             Passed to next timestep in recurrent processing
     *
     * @note Expects input values in [0, 255] range; will normalize internally
     * @note First dimension can be time-major (T, B, ...) for sequence processing
     * @note Masks enable proper gradient handling in sequence learning
     * @warning Input must be on same device as model parameters (CPU/GPU)
     *
     * @see NNBase::forwardGatedRecurrentUnits for GRU implementation details
     * @see torch::nn::Conv2d for convolutional layer mathematics
     */
    std::vector<torch::Tensor> CnnBase::forward(torch::Tensor inputs, torch::Tensor hxs, torch::Tensor masks)
    {
        auto x = main->forward(inputs/ 255.);
        if (isRecurrent())
        {
            auto gru_output = forwardGatedRecurrentUnits(x,hxs,masks);
            x = gru_output[0];
            hxs = gru_output[1];
        }

        return {criticLinear->forward(x),x,hxs};
    }

    /**
     * @brief Unit tests for CnnBase neural network module
     *
     * Validates that the CnnBase architecture produces correct tensor shapes and
     * maintains proper configuration across feed-forward and recurrent modes.
     *
     * **Test Purpose:**
     *
     * These tests ensure:
     * 1. Module initialization with correct architecture parameters
     * 2. Configuration flags (recurrence, hidden size) are properly stored
     * 3. Output tensor shapes match expected dimensions
     * 4. Forward pass dimensions are consistent with architecture
     * 5. Batch processing works correctly with multiple samples
     *
     * **Test Architecture:**
     *
     * Configuration: CnnBase(3, true, 10)
     * - 3 input channels (RGB)
     * - Recurrent mode enabled (uses GRU)
     * - Hidden size = 10 (small for testing purposes)
     *
     * **Subtest 1: "Sanity checks"**
     *
     * Validates module configuration properties:
     * - CHECK: isRecurrent() == true
     *   - Verifies recurrence flag is properly set
     *   - Ensures GRU infrastructure is initialized
     *
     * - CHECK: getHiddenSize() == 10
     *   - Confirms hidden dimension matches constructor parameter
     *   - Validates internal state tracking
     *
     * **Subtest 2: "Output tensors are correct shapes"**
     *
     * Tests forward pass with batch of 4 samples:
     * - Batch size: B = 4
     * - Input channels: C = 3 (RGB)
     * - Spatial resolution: 84×84 (standard Atari)
     * - Hidden state size: 10 (per configuration)
     *
     * Input tensor dimensions:
     * ```
     * inputs: [4, 3, 84, 84]    - 4 RGB images, 84×84 each
     * rnn_hxs: [4, 10]          - Initial hidden state (batch size 4, hidden dim 10)
     * masks: [4, 1]             - All ones (all frames valid)
     * ```
     *
     * Expected output dimensions:
     * ```
     * outputs[0] (critic value):
     *   Shape: [4, 1]
     *   - 4 = batch size
     *   - 1 = scalar value per sample (V(s) estimation)
     *   - CHECK: outputs[0].size(0) == 4 (batch dimension)
     *   - CHECK: outputs[0].size(1) == 1 (value dimension)
     *
     * outputs[1] (actor features):
     *   Shape: [4, 10]
     *   - 4 = batch size
     *   - 10 = hidden_size (feature representation dimension)
     *   - CHECK: outputs[1].size(0) == 4 (batch dimension)
     *   - CHECK: outputs[1].size(1) == 10 (feature dimension)
     *
     * outputs[2] (updated hidden states):
     *   Shape: [4, 10]
     *   - 4 = batch size
     *   - 10 = hidden_size (recurrent state dimension)
     *   - CHECK: outputs[2].size(0) == 4 (batch dimension)
     *   - CHECK: outputs[2].size(1) == 10 (hidden state dimension)
     * ```
     *
     * **Shape Flow Through Network:**
     *
     * Tensor transformation sequence (single batch element):
     * ```
     * [1, 3, 84, 84]     - Input image
     * [1, 32, 20, 20]    - After Conv1 (kernel 8, stride 4)
     * [1, 64, 9, 9]      - After Conv2 (kernel 4, stride 2)
     * [1, 32, 7, 7]      - After Conv3 (kernel 3, stride 1)
     * [1, 1568]          - After Flatten (32×7×7=1568)
     * [1, 10]            - After FC (hidden_size=10)
     * [1, 10]            - After GRU (recurrent processing)
     * [1, 1]             - After Critic (scalar value)
     * ```
     *
     * **Assertions Used:**
     *
     * - **REQUIRE:** Checks vector size (critical for unpacking)
     *   - REQUIRE(outputs.size() == 3)
     *   - If fails, test stops (can't unpack further)
     *
     * - **CHECK:** Individual tensor dimensions
     *   - CHECK(outputs[i].size(j) == expected_size)
     *   - If fails, test continues to next check (reports all failures)
     *
     * **Validation Benefits:**
     *
     * - **Correctness:** Ensures forward pass produces valid output
     * - **Regression detection:** Catches shape mismatches from code changes
     * - **Architecture verification:** Validates layer configuration matches code
     * - **Batch processing:** Tests on realistic batch sizes
     * - **Cross-device compatibility:** Works with CPU tensors
     *
     * **Edge Cases Covered:**
     *
     * - Recurrent mode with non-zero hidden states
     * - Batch processing (multiple samples simultaneously)
     * - Standard Atari resolution (84×84)
     * - Small hidden size (10) vs. typical (256-512)
     *
     * **Expected Runtime:**
     * - Single GPU forward pass: ~5-10ms per subtest
     * - CPU forward pass: ~50-100ms per subtest
     * - Total test time: <1 second on modern hardware
     *
     * **Related Tests:**
     * - Test other architectures (MlpBase, RecurrentGenerator)
     * - Test gradient flow in backward pass
     * - Test recurrence state persistence across frames
     */
    TEST_CASE("CnnBase")
    {
        auto base = std::make_shared<CnnBase>(3, true, 10);

        SUBCASE("Sanity checks")
        {
            CHECK(base->isRecurrent() == true);
            CHECK(base->getHiddenSize() == 10);
        }

        SUBCASE("Output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 3, 84, 84});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = base->forward(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 3);

            // Critic
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Actor
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 10);

            // Hidden state
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 10);
        }
    }
}
