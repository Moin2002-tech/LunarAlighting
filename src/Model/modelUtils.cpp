//
// Created by moinshaikh on 2/4/26.
//

#include<torch/torch.h>


#include"../../include/Model/modelUtils.hpp"
#include"../third_party/doctest.hpp"


using torch::nn::Linear;

namespace LunarAlighting
{
    /**
     * @brief Fills the input `tensor` with a (semi) orthogonal matrix using QR decomposition
     *
     * Implements orthogonal weight initialization as described in "Exact solutions to the
     * nonlinear dynamics of learning in deep linear neural networks". This technique maintains
     * well-conditioned weight matrices and improves gradient flow during backpropagation.
     *
     * **Algorithm Overview:**
     * - Step 1: Generate random matrix from standard normal distribution
     * - Step 2: Compute QR decomposition to extract orthogonal component
     * - Step 3: Apply phase correction based on diagonal signs
     * - Step 4: Scale result by gain parameter
     *
     * **Mathematical Representation:**
     *
     * Given tensor T ∈ ℝ^(m×n), the initialization proceeds as:
     *
     * 1. Generate: A ∈ ℝ^(rows × columns) from 𝒩(0, 1)
     *
     * 2. QR Decomposition: A = Q × R
     *    - Q ∈ ℝ^(rows × columns) is semi-orthogonal (Q^T Q = I)
     *    - R ∈ ℝ^(columns × columns) is upper triangular
     *
     * 3. Phase Correction: Q' = Q × diag(sign(diag(R)))
     *    - Ensures consistent sign patterns
     *    - Improves numerical stability
     *
     * 4. Scaling: W = gains × Q'
     *    - Final weight matrix maintains orthogonal structure
     *    - Property: W^T W ≈ I (near-unitary)
     *
     * **Key Properties:**
     * - Orthogonality condition: ||W^T W - I||_F ≈ 0 (Frobenius norm)
     * - Prevents vanishing/exploding gradients
     * - Maintains singular values ≈ 1.0
     * - Useful for initializing recurrent and deep networks
     *
     * @param tensor an n-dimensional tensor, where n >= 2
     *               Will be reshaped to 2D if necessary
     * @param gains the multiplier scalar for the weights (typically 1.0 or √2 ≈ 1.414)
     * @return torch::Tensor the orthogonally initialized tensor with shape preserved
     *
     * @note Returns tensor unmodified if dimensions < 2
     * @warning Computation is done without gradient tracking (torch::NoGradGuard enabled)
     *
     * @see torch::qr QR decomposition function from LibTorch
     */
    torch::Tensor orthogonal_(torch::Tensor tensor,double gains)
    {
        torch::NoGradGuard gaurd;
        if (tensor.dim() < 2)
        {
            return tensor;
        }

        const auto rows = tensor.size(0);
        const auto columns = tensor.numel() / rows;
        auto flattened = torch::randn({rows,columns});
        if (rows<columns)
        {
            flattened.t_();
        }
        torch::Tensor q,r;
        std::tie(q,r )= torch::qr(flattened);
        auto d = torch::diag(r, 0);
        auto ph = d.sign();
        q *= ph;

        if (rows < columns)
        {
            q.t_();
        }

        tensor.view_as(q).copy_(q);
        tensor.mul_(gains);

        return tensor;
    }

    /**
     * @brief Flattens a multi-dimensional tensor into a 2D tensor while preserving batch dimension
     *
     * Reshapes input tensor by collapsing all feature dimensions into a single column vector.
     * This is commonly used for transitioning between convolutional/recurrent layers and
     * fully-connected layers in neural networks.
     *
     * **Algorithm Overview:**
     * - Preserve the first dimension (batch size)
     * - Collapse all remaining dimensions into a single dimension
     * - Maintain memory layout (row-major order)
     *
     * **Mathematical Representation:**
     *
     * Input tensor: x ∈ ℝ^(B × d₁ × d₂ × ... × dₙ)
     * where:
     *   - B = batch size (first dimension)
     *   - d₁, d₂, ..., dₙ = feature dimensions
     *
     * Output tensor: y ∈ ℝ^(B × D)
     * where:
     *   - D = d₁ × d₂ × ... × dₙ (total flattened features)
     *
     * Transformation: y[i, :] = vec(x[i, :, :, ..., :])
     *   - vec() denotes vectorization operator
     *   - Preserves element ordering (C-contiguous)
     *
     * **Shape Transformation Examples:**
     * - (5, 32, 32, 3) → (5, 3072)  [5 images, 32×32 RGB pixels]
     * - (10, 64, 64) → (10, 4096)   [10 samples, 64×64 features]
     * - (N, 256) → (N, 256)         [no change, already 2D]
     *
     * **Key Properties:**
     * - Bijective mapping: no information loss
     * - Total element count preserved: N_in = N_out
     * - Time complexity: O(1) for view operation (no data copy)
     * - Memory complexity: O(B × D)
     * - Enables feature map vectorization after convolutional layers
     *
     * @param x The input tensor to flatten (dimension must be ≥ 1)
     *          Shape: (batch_size, dim₁, dim₂, ..., dimₙ)
     * @return torch::Tensor the flattened 2D tensor
     *         Shape: (batch_size, dim₁ × dim₂ × ... × dimₙ)
     *
     * @note Uses torch::Tensor::view which creates a view without copying data
     * @note Batch dimension (index 0) is always preserved
     * @warning Input must be contiguous; use .contiguous() if needed
     *
     * @example
     * ```cpp
     * auto flatten = Flatten();
     *
     * // Flatten CNN output
     * auto cnn_output = torch::rand({5, 64, 8, 8});  // batch_size=5, 64 channels, 8×8 spatial
     * auto flattened = flatten->forward(cnn_output);  // shape: (5, 4096)
     *
     * // Can then feed to fully-connected layer
     * auto fc_input = torch::nn::Linear(4096, 128);
     * auto logits = fc_input(flattened);  // shape: (5, 128)
     * ```
     */
    torch::Tensor FlattenImpl::forward(torch::Tensor& x)
    {
        return x.view({x.size(0), -1});
    }

    /**
     * @brief Initializes network weights and biases using orthogonal and constant initialization
     *
     * Iterates through all named parameters in a neural network module and applies
     * appropriate initialization strategies based on parameter type. This is crucial for
     * improving convergence speed and preventing gradient flow issues in deep networks.
     *
     * **Algorithm Overview:**
     * - Step 1: Iterate through all network parameters
     * - Step 2: Filter non-empty parameters (size(0) ≠ 0)
     * - Step 3: Check parameter name for type identification
     * - Step 4: Apply bias or weight initialization accordingly
     *
     * **Initialization Strategies:**
     *
     * **For Bias Parameters:**
     * - Condition: parameter name contains substring "bias"
     * - Method: Constant initialization
     * - Formula: b ← bias_gain × 1
     * - Property: All bias values set to single scalar
     * - Typical value: bias_gain = 0.0 (centered initialization)
     *
     * **For Weight Parameters:**
     * - Condition: parameter name contains substring "weight"
     * - Method: Orthogonal initialization (via orthogonal_ function)
     * - Formula: W ← weight_gain × Q, where Q is semi-orthogonal
     * - Property: Q^T Q ≈ I (preserves gradient magnitudes)
     * - Typical value: weight_gain = 1.0 or √2 ≈ 1.414
     *
     * **Mathematical Representation:**
     *
     * For each parameter p ∈ P = {W₁, b₁, W₂, b₂, ..., Wₙ, bₙ}:
     *
     * IF (name(p) contains "bias"):
     *   p ← bias_gain (scalar ∈ ℝ)
     * ELSE IF (name(p) contains "weight"):
     *   p ← weight_gain × Q, where Q ∈ ℝ^(m×n) with Q^T Q = I
     *
     * **Benefits of Orthogonal Weight Initialization:**
     * - Prevents vanishing gradient problem: ||∇L/∂p|| remains ≈ 1.0
     * - Prevents exploding gradient problem: ||∂y/∂x|| ≈ 1.0
     * - Improves convergence: reduces training iterations needed
     * - Maintains singular value spectrum: σ₁ ≈ σ₂ ≈ ... ≈ σₙ ≈ 1.0
     * - Reduces internal covariate shift
     * - Particularly effective for RNNs and deep networks
     *
     * **Benefits of Zero Bias Initialization:**
     * - Symmetric initialization: no bias toward any direction
     * - Allows network to learn bias naturally
     * - Reduces redundancy with weight initialization
     * - Faster convergence in early training phase
     *
     * **Network Architecture Compatibility:**
     * - Linear/Dense layers: ✓ (primary use case)
     * - Convolutional layers: ✓ (initializes conv kernels)
     * - Recurrent layers: ✓ (recommended for stability)
     * - Batch normalization: ✓ (biases are still useful)
     * - Layer normalization: ✓ (typically beneficial)
     *
     * @param parameters Ordered dictionary of network parameters
     *                   Format: {layer_name.weight, layer_name.bias, ...}
     *                   Type: torch::OrderedDict<std::string, torch::Tensor>
     * @param weight_gain Scaling factor for weight initialization
     *                    Range: typically [0.5, 2.0]
     *                    Default: 1.0 (standard orthogonal)
     *                    Alternative: √2 ≈ 1.414 (ReLU networks)
     * @param bias_gain Scaling factor for bias initialization
     *                  Range: typically [−0.1, 0.1]
     *                  Default: 0.0 (centered)
     *                  Alternative: 0.01 (small positive bias)
     *
     * @return void (modifies parameters in-place)
     *
     * @note Parameters with size(0) == 0 are skipped
     * @note Parameter identification uses string matching, must contain exact keywords
     * @warning Modifies tensor values in-place; existing values are overwritten
     *
     * @example
     * ```cpp
     * // Create a simple neural network
     * auto model = torch::nn::Sequential(
     *     torch::nn::Linear(28 * 28, 128),
     *     torch::nn::Functional(torch::relu),
     *     torch::nn::Linear(128, 64),
     *     torch::nn::Functional(torch::relu),
     *     torch::nn::Linear(64, 10));
     *
     * // Initialize with orthogonal weights and zero biases
     * initWeights(model->named_parameters(), 1.0, 0.0);
     *
     * // Alternative: ReLU network with √2 gain
     * initWeights(model->named_parameters(), std::sqrt(2.0), 0.0);
     * ```
     *
     * @see orthogonal_ for details on weight initialization algorithm
     * @see torch::nn::init::constant_ for bias initialization implementation
     */
    void initWeights(torch::OrderedDict<std::string, torch::Tensor> parameters,
                  double weight_gain,
                  double bias_gain)
    {
        for (const auto &parameter : parameters)
        {
            if (parameter.value().size(0) != 0)
            {
                if (parameter.key().find("bias") != std::string::npos)
                {
                    torch::nn::init::constant_(parameter.value(), bias_gain);
                }
                else if (parameter.key().find("weight") != std::string::npos)
                {
                    orthogonal_(parameter.value(), weight_gain);
                }
            }
        }
    }

    TEST_CASE("Flatten")
    {
        auto flatten = Flatten();

        SUBCASE("Flatten converts 3 dimensional vector to 2 dimensional")
        {
            auto input = torch::rand({5, 5, 5});
            auto output = flatten->forward(input);

            CHECK(output.size(0) == 5);
            CHECK(output.size(1) == 25);
        }

        SUBCASE("Flatten converts 5 dimensional vector to 2 dimensional")
        {
            auto input = torch::rand({2, 2, 2, 2, 2});
            auto output = flatten->forward(input);

            CHECK(output.size(0) == 2);
            CHECK(output.size(1) == 16);
        }

        SUBCASE("Flatten converts 1 dimensional vector to 2 dimensional")
        {
            auto input = torch::rand({10});
            auto output = flatten->forward(input);

            CHECK(output.size(0) == 10);
            CHECK(output.size(1) == 1);
        }
    }

    TEST_CASE("init_weights()")
    {
        auto module = torch::nn::Sequential(
            torch::nn::Linear(5, 10),
            torch::nn::Functional(torch::relu),
            torch::nn::Linear(10, 8));

        initWeights(module->named_parameters(), 1, 0);

        SUBCASE("Bias weights are initialized to 0")
        {
            for (const auto &parameter : module->named_parameters())
            {
                if (parameter.key().find("bias") != std::string::npos)
                {
                    CHECK(parameter.value()[0].item().toDouble() == doctest::Approx(0));
                }
            }
        }
    }
}