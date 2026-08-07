# Convolutional Networks and Transformer

Convolution and self-attention solve different structure problems. A convolution reuses a local detector at every position; self-attention lets each token combine information from selected tokens in its context. nuNN keeps both implementations deliberately small: a 1D convolutional stack and a decoder-only mini transformer.

## At a glance

| Model | Structural bias | Public entry point | Runnable example |
| --- | --- | --- | --- |
| 1D CNN | locality and shared filters | [`nu_convnet.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_convnet.h) | [`cnn_seq`](https://github.com/eantcal/nunn/blob/main/examples/cnn_seq/cnn_seq.cc) |
| Decoder-only transformer | content-dependent token interaction with a causal mask | [`nu_transformer.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_transformer.h) | [`transformer_char`](https://github.com/eantcal/nunn/blob/main/examples/transformer_char/transformer_char.cc) |

## 1D convolution: local features with shared weights

A filter of width `K` slides across a channel-major input:

```text
y_o(t) = activation(
    b_o + sum_c sum_k W(o,c,k) * x_c(t+k)
)
```

![Convolution filter](assets/conv-filter.png)

`Conv1DLayer` uses valid padding and stride 1:

```text
output_length = input_length - kernel_size + 1
```

Its public contract and tensor layout are explicit in [`nu_conv.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_conv.h):

```cpp
nu::Conv1DLayer conv(
    1,                         // input channels
    16,                        // input length
    8,                         // output channels / filters
    5,                         // kernel width
    nu::Activation::Tanh,
    0.005
);

const auto& features = conv.forward(input);
```

Flat inputs and outputs are channel-major: all positions for channel 0, followed by all positions for channel 1, and so on. The implementation uses an `im2col` matrix so all windows become columns and the filter bank becomes one Eigen product. Read the construction and reverse `col2im` path in [`nu_conv.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_conv.cc).

## Max pooling: reduce and route

`MaxPool1DLayer` uses non-overlapping windows:

```text
output_length = floor(input_length / pool_size)
```

![Max pooling](assets/maxpool.png)

During `forward()` it stores the winning input index for every output. During `backward()` only that index receives the upstream gradient. Any remainder shorter than a complete pool window is discarded.

`MaxPool1DLayer` has no trainable parameters; the learning-rate argument on the common `backward()` interface is ignored.

## `ConvNet`: end-to-end builder

`ConvNet` owns the convolution/pooling stack and an `MlpMatrixNN` head. This is the exact architecture in [`cnn_seq.cc`](https://github.com/eantcal/nunn/blob/main/examples/cnn_seq/cnn_seq.cc):

```cpp
using LC = nu::MlpMatrixNN::LayerConfig;

nu::ConvNet cnn(1, 16);
cnn.addConv1D(8, 5, nu::Activation::Tanh, 0.005);
cnn.addMaxPool1D(4);

const size_t flatSize = cnn.flatFeatureSize(); // 8 * 3 = 24
cnn.setFCHead({
    LC(flatSize),
    LC(16, nu::Activation::Tanh),
    LC(2, nu::Activation::Sigmoid)
}, 0.005);

double loss = cnn.train(sample, oneHotTarget);
auto output = cnn.predict(sample);
```

The size calculation is worth checking manually:

```text
input:             1 x 16
valid conv K=5:    8 x 12
pool size 4:       8 x 3
flattened head:       24
```

`setFCHead()` rejects a first layer size that differs from `flatFeatureSize()`. During training, the head exposes `dLoss/dInput`; [`nu_convnet.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_convnet.cc) passes it backward through the stack in reverse order.

Tests isolate each responsibility:

- [`test_cnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_cnn.cc) checks convolution dimensions, pooling, gradients, and end-to-end learning;
- [`test_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpmatrixnn.cc) checks the fully connected head and its input gradient.

Run `cnn_seq [epochs] [learning_rate]` to classify noisy one-cycle versus two-cycle signals.

## Self-attention: content-dependent mixing

For token representations `X`, learned projections form queries, keys, and values:

```text
Q = X W_Q
K = X W_K
V = X W_V

Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
```

Each attention head uses its own projections. The head results are concatenated and projected back to the model dimension.

Unlike convolution, the mixing weights are recomputed from the current content. Unlike a recurrent network, every allowed token pair can interact within one attention layer.

## Causality and the decoder-only model

Next-token training must not leak future tokens. A causal mask sets attention scores above the diagonal to negative infinity before softmax:

```text
token t can attend to positions 0 ... t
token t cannot attend to positions t+1 ... T-1
```

The [`SelfAttentionLayer::forward` implementation](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_transformer.cc) accepts the causal flag. `MiniTransformer` uses it for decoder-only language modeling.

## The implemented transformer stack

`MiniTransformer` is Pre-LN:

```text
token ids
  -> token embedding + fixed sinusoidal position
  -> N * (
       LayerNorm -> causal multi-head attention -> residual
       LayerNorm -> feed-forward ReLU          -> residual
     )
  -> output projection
  -> logits [sequence_length x vocabulary_size]
```

The source is divided into four inspectable types:

| Type | Responsibility |
| --- | --- |
| `LayerNorm` | row-wise normalization plus learned scale and shift |
| `SelfAttentionLayer` | per-head Q/K/V projections, causal softmax, output projection |
| `TransformerBlock` | Pre-LN attention and feed-forward residual paths |
| `MiniTransformer` | embeddings, positions, blocks, logits, loss, generation |

All declarations are together in [`nu_transformer.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_transformer.h), their forward and backward paths are in [`nu_transformer.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_transformer.cc), and numerical behavior is covered by [`test_transformer.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_transformer.cc).

## Source-backed character model

[`transformer_char.cc`](https://github.com/eantcal/nunn/blob/main/examples/transformer_char/transformer_char.cc) uses:

```cpp
constexpr size_t sequenceLength = 32;
constexpr size_t modelDimension = 64;
constexpr size_t heads = 4;
constexpr size_t feedForwardDimension = 128;
constexpr size_t layers = 2;

nu::MiniTransformer model(
    vocabulary.size(),
    sequenceLength,
    modelDimension,
    heads,
    feedForwardDimension,
    layers,
    0.005
);

double loss = model.train(inputTokens, nextTokens);

std::mt19937 rng(42);
auto continuation = model.generate(
    prompt,
    80,
    0.8,                       // temperature
    &rng
);
```

Constraints follow directly from the implementation:

- `modelDimension` must be divisible by `heads`;
- `forward()` and `train()` use the fixed context length passed to the constructor;
- token IDs must be in `[0, vocabularySize)`;
- `train()` returns mean cross-entropy over the sequence;
- `generate()` is autoregressive and repeatedly uses the most recent context window.

Run:

```sh
transformer_char
transformer_char 1500 0.003 120
```

Positional arguments are `epochs learning_rate generated_length`.

## Temperature during generation

Given logits `z`, generation samples from `softmax(z / temperature)`:

- below 1 sharpens the distribution and favors high-probability characters;
- above 1 flattens it and increases variety;
- extremely small values approach greedy selection;
- high values expose weakly learned alternatives and noise.

Always compare generation with training loss. A plausible short sample is not a substitute for held-out evaluation.

## Choosing between the two

Use `ConvNet` when local patterns and translation matter, especially for fixed-size 1D signals. Use `MiniTransformer` when relationships depend on token content and direct long-range interaction is useful. For sequential state carried one step at a time, compare with [Recurrent Networks](Recurrent-Networks).

## Keep reading

Use [Theory Notes](Theory-Notes) for the mathematical bridge and [Training and Diagnostics](Training-and-Diagnostics) for gradient, shape, and evaluation checks.
