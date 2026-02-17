# Doxygen Documentation - modelUtils.cpp

## Overview
Enhanced Doxygen documentation for `src/Model/modelUtils.cpp` with comprehensive mathematical representations, algorithm explanations, and practical examples.

---

## 1. `orthogonal_()` Function

### Purpose
Fills the input tensor with a (semi) orthogonal matrix using QR decomposition for optimal neural network weight initialization.

### Algorithm Overview
- **Step 1:** Generate random matrix from standard normal distribution (𝒩(0, 1))
- **Step 2:** Compute QR decomposition to extract orthogonal component
- **Step 3:** Apply phase correction based on diagonal signs
- **Step 4:** Scale result by gain parameter

### Mathematical Representation

Given tensor T ∈ ℝ^(m×n), the initialization proceeds as:

```
1. Generate: A ∈ ℝ^(rows × columns) from 𝒩(0, 1)

2. QR Decomposition: A = Q × R
   - Q ∈ ℝ^(rows × columns) is semi-orthogonal (Q^T Q = I)
   - R ∈ ℝ^(columns × columns) is upper triangular

3. Phase Correction: Q' = Q × diag(sign(diag(R)))
   - Ensures consistent sign patterns
   - Improves numerical stability

4. Scaling: W = gains × Q'
   - Final weight matrix maintains orthogonal structure
   - Property: W^T W ≈ I (near-unitary)
```

### Key Properties
- **Orthogonality condition:** ||W^T W - I||_F ≈ 0 (Frobenius norm)
- **Gradient flow:** Prevents vanishing/exploding gradients
- **Singular values:** Maintains σ₁ ≈ σ₂ ≈ ... ≈ σₙ ≈ 1.0
- **Use cases:** Optimal for initializing recurrent and deep networks

### Parameters
- `tensor`: n-dimensional tensor (n ≥ 2), will be reshaped to 2D if necessary
- `gains`: Multiplier scalar for weights (typically 1.0 or √2 ≈ 1.414)

### Returns
- `torch::Tensor`: Orthogonally initialized tensor with shape preserved

---

## 2. `FlattenImpl::forward()` Function

### Purpose
Flattens a multi-dimensional tensor into a 2D tensor while preserving the batch dimension.

### Algorithm Overview
- Preserve the first dimension (batch size)
- Collapse all remaining dimensions into a single dimension
- Maintain memory layout (row-major order)

### Mathematical Representation

**Input tensor:** x ∈ ℝ^(B × d₁ × d₂ × ... × dₙ)
- B = batch size (first dimension)
- d₁, d₂, ..., dₙ = feature dimensions

**Output tensor:** y ∈ ℝ^(B × D)
- D = d₁ × d₂ × ... × dₙ (total flattened features)

**Transformation:** y[i, :] = vec(x[i, :, :, ..., :])
- vec() denotes vectorization operator
- Preserves element ordering (C-contiguous)

### Shape Transformation Examples
- (5, 32, 32, 3) → (5, 3072)  [5 images, 32×32 RGB pixels]
- (10, 64, 64) → (10, 4096)   [10 samples, 64×64 features]
- (N, 256) → (N, 256)         [no change, already 2D]

### Key Properties
- **Bijective mapping:** No information loss
- **Element count:** Total count preserved (N_in = N_out)
- **Time complexity:** O(1) for view operation (no data copy)
- **Memory complexity:** O(B × D)
- **Use case:** Feature map vectorization after convolutional layers

### Parameters
- `x`: Input tensor (dimension ≥ 1), Shape: (batch_size, dim₁, dim₂, ..., dimₙ)

### Returns
- `torch::Tensor`: Flattened 2D tensor, Shape: (batch_size, dim₁ × dim₂ × ... × dimₙ)

### Example Usage
```cpp
auto flatten = Flatten();

// Flatten CNN output
auto cnn_output = torch::rand({5, 64, 8, 8});  // batch_size=5, 64 channels, 8×8 spatial
auto flattened = flatten->forward(cnn_output);  // shape: (5, 4096)

// Can then feed to fully-connected layer
auto fc_input = torch::nn::Linear(4096, 128);
auto logits = fc_input(flattened);  // shape: (5, 128)
```

---

## 3. `initWeights()` Function

### Purpose
Initializes network weights and biases using orthogonal and constant initialization strategies for improved convergence and gradient flow.

### Algorithm Overview
- **Step 1:** Iterate through all network parameters
- **Step 2:** Filter non-empty parameters (size(0) ≠ 0)
- **Step 3:** Check parameter name for type identification
- **Step 4:** Apply bias or weight initialization accordingly

### Initialization Strategies

#### For Bias Parameters
- **Condition:** Parameter name contains substring "bias"
- **Method:** Constant initialization
- **Formula:** b ← bias_gain × 1
- **Property:** All bias values set to single scalar
- **Typical value:** bias_gain = 0.0 (centered initialization)

#### For Weight Parameters
- **Condition:** Parameter name contains substring "weight"
- **Method:** Orthogonal initialization (via `orthogonal_()` function)
- **Formula:** W ← weight_gain × Q, where Q is semi-orthogonal
- **Property:** Q^T Q ≈ I (preserves gradient magnitudes)
- **Typical value:** weight_gain = 1.0 or √2 ≈ 1.414

### Mathematical Representation

For each parameter p ∈ P = {W₁, b₁, W₂, b₂, ..., Wₙ, bₙ}:

```
IF (name(p) contains "bias"):
  p ← bias_gain (scalar ∈ ℝ)
  
ELSE IF (name(p) contains "weight"):
  p ← weight_gain × Q, where Q ∈ ℝ^(m×n) with Q^T Q = I
```

### Benefits of Orthogonal Weight Initialization
- **Gradient stability:** ||∇L/∂p|| remains ≈ 1.0 (prevents vanishing gradients)
- **Gradient magnitude:** ||∂y/∂x|| ≈ 1.0 (prevents exploding gradients)
- **Convergence speed:** Reduces training iterations needed
- **Singular values:** Maintains σ₁ ≈ σ₂ ≈ ... ≈ σₙ ≈ 1.0
- **Regularization:** Reduces internal covariate shift
- **Deep networks:** Particularly effective for RNNs and deep architectures

### Benefits of Zero Bias Initialization
- **Symmetry:** No bias toward any direction
- **Learning:** Allows network to learn bias naturally
- **Efficiency:** Reduces redundancy with weight initialization
- **Convergence:** Faster convergence in early training phase

### Network Architecture Compatibility
- **Linear/Dense layers:** ✓ (primary use case)
- **Convolutional layers:** ✓ (initializes conv kernels)
- **Recurrent layers:** ✓ (recommended for stability)
- **Batch normalization:** ✓ (biases are still useful)
- **Layer normalization:** ✓ (typically beneficial)

### Parameters
- `parameters`: Ordered dictionary of network parameters
  - Format: {layer_name.weight, layer_name.bias, ...}
  - Type: `torch::OrderedDict<std::string, torch::Tensor>`

- `weight_gain`: Scaling factor for weight initialization
  - Range: typically [0.5, 2.0]
  - Default: 1.0 (standard orthogonal)
  - Alternative: √2 ≈ 1.414 (ReLU networks)

- `bias_gain`: Scaling factor for bias initialization
  - Range: typically [−0.1, 0.1]
  - Default: 0.0 (centered)
  - Alternative: 0.01 (small positive bias)

### Returns
- `void` (modifies parameters in-place)

### Example Usage
```cpp
// Create a simple neural network
auto model = torch::nn::Sequential(
    torch::nn::Linear(28 * 28, 128),
    torch::nn::Functional(torch::relu),
    torch::nn::Linear(128, 64),
    torch::nn::Functional(torch::relu),
    torch::nn::Linear(64, 10));

// Initialize with orthogonal weights and zero biases
initWeights(model->named_parameters(), 1.0, 0.0);

// Alternative: ReLU network with √2 gain
initWeights(model->named_parameters(), std::sqrt(2.0), 0.0);
```

---

## Documentation Standards

### Doxygen Features Used
✓ **@brief** - Function description  
✓ **Algorithm Overview** - Bullet points explaining steps  
✓ **Mathematical Representation** - Formal mathematical notation  
✓ **Key Properties** - Relevant properties and benefits  
✓ **@param** - Parameter descriptions with types and ranges  
✓ **@return** - Return value description  
✓ **@note** - Important implementation notes  
✓ **@warning** - Critical warnings  
✓ **@example** - Code examples with usage  
✓ **@see** - Cross-references to related functions  

### Mathematical Notation
- **Vectors/Matrices:** Bold or Greek letters (e.g., W, Q, R)
- **Set notation:** Standard mathematical set symbols (e.g., ∈, ℝ)
- **Operations:** Matrix operations (e.g., A = Q × R, Q^T Q = I)
- **Dimensions:** Subscripts for tensor dimensions (e.g., d₁, d₂)
- **Properties:** Big-O notation and mathematical properties

---

## Implementation Status
✅ **Complete** - All three functions fully documented with:
- Detailed algorithm explanations
- Mathematical formulas and representations
- Practical examples and use cases
- Parameter ranges and typical values
- Benefits and compatibility notes

