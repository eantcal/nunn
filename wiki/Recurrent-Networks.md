# Recurrent Networks

Recurrent networks process sequences by carrying state from one time step to the next. They are useful when order matters: signals, time series, text, characters, and other streams where the current element depends on context.

nuNN includes:

- `VanillaRnn`
- `Gru`
- `Lstm`

![RNN unrolled](assets/rnn-unrolled.png)

## Why State Matters

A feedforward network sees one input vector and produces one output. A recurrent network also receives its previous hidden state:

```text
h_t = f(W_x * x_t + W_h * h_(t-1) + b)
y_t = g(W_y * h_t + c)
```

The hidden state acts as memory. In a character model, it can carry information about earlier letters. In a sine-wave predictor, it can carry phase information. In an adding task, it can retain selected values until they are needed.

## `VanillaRnn`

`VanillaRnn` is an Elman-style recurrent network. It is the simplest recurrent baseline and is useful for seeing how the same weights are reused at every time step.

Its limitation is gradient stability. During backpropagation through time, gradients are repeatedly multiplied through the recurrent connection. They can shrink until learning long-range dependencies becomes difficult, or grow until training becomes unstable.

Demo programs:

- `rnn_sine`
- `rnn_char`
- `rnn_adding`

## GRU

`Gru` adds gates that decide how much previous state to keep and how much new information to write. It has fewer parameters than an LSTM and is often a strong practical default for short-to-medium dependencies.

The key idea is selective memory: the model learns when to preserve state and when to replace it.

## LSTM

`Lstm` separates hidden state from cell state. The cell state is a more direct memory path, and the gates regulate input, forgetting, and output.

![LSTM cell](assets/lstm-cell.png)

This structure helps the model preserve information across longer spans than a vanilla RNN.

## Truncated BPTT

Backpropagation through time unfolds the recurrent computation across sequence steps. Unfolding indefinitely would be expensive, so nuNN uses truncated BPTT: it trains over a bounded window.

This keeps computation predictable while still assigning credit across recent history.

## Reading the Examples

The recurrent demos are intentionally small:

- `rnn_sine` checks whether the model can learn a smooth periodic signal.
- `rnn_char` shows character-level next-symbol prediction.
- `rnn_adding` tests whether the model can retain sparse information over a sequence.

When comparing RNN, GRU, and LSTM, watch both the final loss and the speed of convergence. A gated model may use more computation per step, but it often learns dependencies that a vanilla RNN struggles to keep.

