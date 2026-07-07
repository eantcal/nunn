# Recurrent Networks

Recurrent networks process sequences by carrying state from one time step to the next.

nuNN includes:

- `VanillaRnn`
- `Gru`
- `Lstm`

![RNN unrolled](assets/rnn-unrolled.png)

## VanillaRnn

`VanillaRnn` is an Elman-style recurrent network. It is useful as the simplest recurrent baseline, but it can struggle on long dependencies because gradients may vanish or explode through time.

Demo programs:

- `rnn_sine`
- `rnn_char`
- `rnn_adding`

## GRU

`Gru` introduces gates that control how much previous state is retained and how much new information is written. It has fewer parameters than LSTM and is often a strong practical default for short-to-medium sequences.

## LSTM

`Lstm` separates hidden state and cell state. Its gates regulate input, forgetting, and output, making it better suited than vanilla RNNs for longer dependencies.

![LSTM cell](assets/lstm-cell.png)

## BPTT

All recurrent classes use truncated backpropagation through time. Instead of unfolding the sequence indefinitely, training backpropagates through a limited window. This keeps computation bounded while still allowing temporal credit assignment.

