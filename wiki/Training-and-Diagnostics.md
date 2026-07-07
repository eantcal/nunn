# Training and Diagnostics

This page is a practical companion to the theory pages. It focuses on the questions that appear when a model does not learn, learns too slowly, or behaves differently after being saved and loaded.

## The Basic Training Loop

Most supervised examples in nuNN follow the same loop:

1. Convert raw data into numeric vectors.
2. Run a forward pass.
3. Compute a loss against the target.
4. Backpropagate gradients.
5. Update weights and biases.
6. Periodically evaluate on data not used for the update.

![Training loop](assets/training-loop.png)

The exact classes differ, but the diagnostic logic is shared.

## Reading a Cost Curve

A cost-convergence chart should answer three questions:

- is the loss decreasing?
- is it flattening?
- is it noisy, divergent, or stuck?

Typical patterns:

| Curve shape | Likely meaning | First checks |
| --- | --- | --- |
| Smooth decrease then plateau | normal convergence | try more epochs or a smaller learning rate |
| Loss jumps upward or explodes | learning rate too high, bad scaling, unstable activation | reduce learning rate, normalize input |
| Flat from the beginning | no useful gradient or wrong target/input path | check labels, topology, activation, loss |
| Training loss falls but test accuracy is poor | overfitting or train/runtime mismatch | compare preprocessing and test set |
| Good before save, bad after reload | serialization or wrong model file | check topology, input size, selected file |

For OCR, a correct model should produce one dominant output neuron for a familiar digit. If all outputs are similar, the network is uncertain. If the dominant output is stable but wrong, the input preprocessing may differ from MNIST.

## Learning Rate

The learning rate controls update size:

```text
theta <- theta - eta * grad
```

Too small: convergence is slow.

Too large: the optimizer overshoots and may oscillate or diverge.

MNIST defaults use `0.025` because it is a conservative value for the didactic MLP setup. For deeper topologies or different activations, it is worth trying lower values.

## Momentum

Momentum keeps a fraction of the previous update:

```text
v_t = alpha * v_(t-1) - eta * grad
theta <- theta + v_t
```

When gradients point in the same direction for several steps, momentum accelerates movement. When gradients alternate direction, it damps oscillations.

In nuNN, momentum is exposed in MLP training so the user can observe its effect on convergence rather than treating it as a hidden optimizer detail.

## Batch Size

Batch size changes the noise and cost of updates:

| Batch style | Behavior |
| --- | --- |
| `1` | online SGD, noisy but very direct |
| small batch | useful noise, efficient enough |
| larger batch | smoother gradients, better matrix acceleration |

For MNIST, the current default is `100`, paired with `MlpMatrixNN`. This makes training more stable and lets Eigen/OpenCL acceleration matter.

## Activation and Loss Pairing

For classification, cross-entropy often converges faster than MSE because it keeps the output gradient informative even when predictions are confidently wrong.

MSE remains useful in didactic examples because it is easy to inspect and works well for reconstruction/regression-style tasks.

A useful rule of thumb:

- regression or reconstruction: start with MSE;
- classification: prefer cross-entropy when available;
- hidden layers: ReLU or Leaky ReLU often train faster than saturated sigmoid/tanh in deeper networks;
- small didactic networks: sigmoid/tanh are easier to visualize and compare with textbook formulas.

## Input Scaling

Training is sensitive to input scale. MNIST pixels are normalized from bytes to `[0, 1]`. Linear regression with gradient descent often needs normalized features. RBF, K-Means, SOM, and PCA are also sensitive to scale because they depend on distances or variance.

If learning is unstable, check input scale before changing the model.

## Save, Load, and Reproducibility

When validating persistence, test the full path:

1. Train a model.
2. Save it.
3. Load it into a fresh object.
4. Run the same input through both models.
5. Compare outputs within a small tolerance.

This is especially important for JSON model files used by `ocr_test`, because recognition depends on the loaded topology, activation choices, weights, biases, and preprocessing convention.

## Runtime Backend

`MlpMatrixNN` can use:

- CPU through Eigen;
- OpenCL through ArrayFire;
- `Auto`, which prefers OpenCL when available and falls back to CPU.

The backend changes execution, not the mathematical model. For a fixed model and input, CPU and OpenCL outputs should be numerically close, allowing for floating-point differences.

If an installed GUI fails because OpenCL runtime DLLs are missing, use the CPU/non-OpenCL executable or the launcher fallback. The GUI should diagnose the missing runtime clearly instead of failing silently.

## MNIST/OCR Checklist

For a healthy MNIST/OCR workflow:

- train with `MlpMatrixNN`, backend `Auto`, batch size `100`;
- verify the cost curve decreases;
- save as JSON;
- reload the JSON in the same process or in `ocr_test`;
- check that model input size is `784`;
- draw centered digits with similar stroke thickness to MNIST;
- compare the ten output activations, not only the final digit.

