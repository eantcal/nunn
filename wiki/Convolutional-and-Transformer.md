# Convolutional Networks and Transformer

This page covers two families that process structured inputs in different ways. Convolution uses local filters and weight sharing. Transformers use attention so each token can directly combine information from other tokens.

## Conv1DLayer / MaxPool1DLayer / ConvNet

nuNN includes a compact 1D convolutional pipeline:

- `Conv1DLayer`
- `MaxPool1DLayer`
- `ConvNet`

The 1D implementation keeps the mechanics visible while preserving the essential CNN ideas: local windows, shared filters, feature maps, pooling, and end-to-end backpropagation.

## Local Filters

A convolutional filter is a small learned template. It slides over the input and computes a dot product with each local window.

![Convolution filter](assets/conv-filter.png)

For a 1D input, a filter response can be read as:

```text
y_i(t) = f(b_i + sum_c sum_k W_i(c,k) * x_c(t+k))
```

`Conv1DLayer` uses valid padding and stride 1. Internally it uses an `im2col`-style transformation so the convolution can be expressed as a matrix product. This keeps the implementation close to the math and lets Eigen do the heavy numeric work.

## Pooling

Max pooling reduces the sequence length by keeping the strongest response in each local window.

![Max pooling](assets/maxpool.png)

This has two effects:

- fewer activations reach later layers;
- small local shifts often produce the same pooled response.

`MaxPool1DLayer` stores the position of the maximum during the forward pass, then routes the gradient back to that position during the backward pass.

## ConvNet Builder

`ConvNet` chains convolution and pooling layers, then attaches an `MlpMatrixNN` head. The head computes the final prediction, and its input gradient is propagated backward through pooling and convolution.

Demo:

- `cnn_seq`

## MiniTransformer

The transformer implementation is intentionally compact and educational. It includes:

- `LayerNorm`
- `SelfAttentionLayer`
- `TransformerBlock`
- `MiniTransformer`

The central mechanism is self-attention. Each token creates a query, key, and value. Queries are compared with keys; the resulting weights combine values:

```text
Attention(Q,K,V) = softmax((Q * K^T) / sqrt(d_k)) * V
```

Multi-head attention repeats this process in several projection spaces so different heads can specialize in different relationships.

## Decoder-Only Language Modeling

`MiniTransformer` is decoder-only and autoregressive. It predicts the next token from previous context, so it uses a causal mask: token `t` may attend to tokens `0..t`, but not to future tokens.

The implementation uses:

- token embeddings;
- sinusoidal positional encoding;
- Pre-LN transformer blocks;
- residual connections;
- position-wise feed-forward layers;
- softmax cross-entropy for training;
- temperature scaling for generation.

Demo:

- `transformer_char`

