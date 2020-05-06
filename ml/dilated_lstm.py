import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

# Used to develop and test the dilated LSTM
def main():
    n_input = 3
    n_hidden = 6
    n_layers = 4
    batch_size = 4
    seq_size = 8
    cell_type = 'LSTM'

    model = DRNN(n_input, n_hidden, n_layers, cell_type=cell_type)

    x1 = torch.randn(seq_size, batch_size, n_input)
    x2 = torch.randn(seq_size, batch_size, n_input)

    out, hidden = model(x1)
    # out, hidden = model(x2, hidden)


# Original implementation of the dLSTM
class DRNN(nn.Module):

    def __init__(self, num_features, n_hidden, n_layers, dropout=0, cell_type='GRU',
                 batch_first=False):
        super(DRNN, self).__init__()  # Init module

        self.dilations = [2 ** i for i in range(n_layers)]  # Exponential
        # dilations as per the paper [2, 4, 8, 16, ...]
        self.cell_type = cell_type
        self.batch_first = batch_first

        # Select the RNN type to use
        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        # For each layer, create the hidden cell
        for i in range(n_layers):
            if i == 0:
                c = cell(num_features, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)

        # Combine the layers into one RNN block (one output feeding into the
        # next?)
        # self.cells = nn.Sequential(*layers)
        self.cells = layers

    # Initially, no hidden layer. After every output we pass the output
    # hidden layer to the next input (if stateful, if not we reset).
    # hidden is a list of hidden states for each layer I think.
    # hidden[0] is hidden state of layer 0, hidden[1] is hidden state of
    # layer 1, etc. For the very first (batch of) inputs, we don't have an
    # initial hidden state to feed in. After our first bat
    def forward(self, inputs, hidden=None):

        # Handle the case where the batch axis is first
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        outputs = []

        # Zip the cells and dilations into tuples, then enumerate the tuples
        # (loop through each layer)
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            # If we don't input a hidden state (i.e it must be the various
            # first batch), then we don't care about the final hidden state
            # value. The hidden state for each layer will be all 0s initially.
            if hidden is None:
                print("Dilation:", dilation)
                inputs, outs = self.drnn_layer(cell, inputs, dilation)
                print("Output Size", inputs.size())
                print("h_n Size", outs[0].size())
                print("c_n Size", outs[1].size())
                print()

                # inputs = (seq_len, batch, hidden_size) = (8, 4, 6)
                # outs[0] = outs[1] = (num_layers, batch, hidden_size) = (1,
                # 4, 6)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation,
                                                    hidden[i])

            # inputs contains outputs which is of dimension:
            # [seq_len, batch_size, hidden_size]
            # we then append final 1, 2, 4, 8, 16, etc outputs
            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    # Cell = the cell type for this layer (LSTM, GRU, etc)
    # Inputs = the inp
    # Rate = the dilation of this layer
    # hidden = ??
    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)  # Corresponds to window_size in our case
        batch_size = inputs[0].size(0)  # Size of batch
        hidden_size = cell.hidden_size  # Size of hidden state

        # Pad the inputs to ensure that the sequence length is a multiple of
        # the dilation
        inputs, _ = self._pad_inputs(inputs, n_steps, rate)

        # Convert the inputs into the correct dilated sequences. e.g
        # [0, 1, 2, 3, 4, 5, 6, 7] to:
        # [0, 4], [1, 5], [2, 6], [3, 7] for dilation = 4
        dilated_inputs = self._prepare_inputs(inputs, rate)

        # Input the values into the layer. If the layer is the first layer
        # we must initialise the hidden layer, if not then we don't need to.
        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        # Split the outputs up back into the correct shape, and remove any
        # padding.
        print("Dilated Output Size", dilated_outputs.size())
        print("Dilated h_n Size", hidden[0].size())
        print("Dilated c_n Size", hidden[1].size())
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        # splitted_hn = self._split_outputs(hidden[0], rate)
        # hidden = (splitted_hn, hidden[1])

        # Return the outputs (usual outputs, including all hidden layers),
        # and the hidden layer tuple (h_n, c_n).
        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        # If we are inputting into the initial layer
        if hidden is None:
            # If we are using LSTM as the blocks
            if self.cell_type == 'LSTM':
                # Initialise the c, h layers with zeros. We have gone from
                # seq_len * batch size to (seq_len/rate) * (batch_size * rate)
                # for our input, so we need to initialise our hidden state
                # accordingly
                c, m = self.init_hidden(batch_size * rate, hidden_size)


                # Combined c, m into a single variable, and add an extra
                # dimension along the 0th axis (required because we could
                # have an extra dimension if we are using a Bi-LSTM)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
                print("h_0 Input Size:", hidden[0].size())
                print("c_0 Input Size:", hidden[1].size())
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)
        else:
            pass
        # print(hidden.size())
        # Input the values, and the hidden state (either 0s or previous
        # hidden state), and get the outputs
        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    # Simply remove the padding by taking the first seq_len values from the
    # output.
    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    # Split the outputs of our dilated RNN back into the correct size
    # of [seq_len, batch_size, hidden_size]. Currrently:
    # dilated outputs.size = (new_seq_len, new_batch_size, hidden_size)
    # = [seq_len/rate, batch_size * rate, hidden_size]
    def _split_outputs(self, dilated_outputs, rate):
        # Get the original batch_size = the current batch size // rate
        # = batch_size * rate // rate
        batchsize = dilated_outputs.size(1) // rate

        # Divide our new bigger batch size into old batch size chunks
        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        # stack() without an axis specified concatenates by adding a new axis
        # transpose(1, 0) swaps the dimensions of axis 1 and axis 0
        # A tensor of the same shape as torch.stack((blocks).transpose(1,
        # 0) would have a different memory layout to what it has here,
        # hence we need to call contiguous to create a copy with the correct
        # underlying memory arrangement. spooky

        # Stack and tranpose them to re-interleave
        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()

        # Now we convert back to [seq_len/rate * rate,  batch_size *
        # rate/rate, hidden_size)
        # = [seq_len, batch_size, hidden_size]
        interleaved = interleaved.view(
            dilated_outputs.size(0) * rate,
            batchsize,
            dilated_outputs.size(2)
        )
        return interleaved

    # This function extends each sequence so that its length is a multiple
    # of the dilation, by padding it with 0s - possibly try padding with the
    # mean values?
    def _pad_inputs(self, inputs, n_steps, rate):
        # If window_size (seq_len) divides dilation exactly
        # 48 / 8 = 6 e.g,
        is_even = (n_steps % rate) == 0

        # Pad if necessary
        if not is_even:
            # eg :
            # rate (dilation) = 32
            # dilated_steps = 48 // 32 + 1 = 1 + 1 = 2
            dilated_steps = n_steps // rate + 1

            # zeros.shape = [2 * 32 - 48, batch_size, num_features]
            # = [16, batch_size, num_features]
            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            # Concatenate sequence with the zeros
            inputs = torch.cat((inputs, zeros_))

        # No need to pad
        else:
            dilated_steps = n_steps // rate

        # Return possibly padded input, and the number of dilation steps
        return inputs, dilated_steps

    # inputs = inputs in a batch
    # rate = dilation
    def _prepare_inputs(self, inputs, rate):
        # inputs.shape = (seq_len, batch_size, num_features]
        # Let dilation (rate) = 4
        # Ignoring the batch size and extra inputs for the moment,
        # and suppose we have a sequence of [0, 1, 2, 3, 4, 5, 6, 7],
        # and a rate of 4. Then inputs[j::rate] for j in range(rate)] will
        # create [[0, 4], [1, 5], [2, 6], [3, 7]]. i.e it starts at 0, 1,
        # ..., rate, and then for each creates the sequence of every
        # 'rate'th value. These are the inputs which are dilated together.
        # This is a nice property of the dilated LSTM - we don't actually
        # have to modify the LSTM blocks, as the RNN inputs can just be
        # restructured - see paper. We will then have, instead of a batch of
        # size 1 of an 8 length sequence, a batch of 4 length-2 sequences.
        # The only issue with this is that it changes the order of our
        # inputs, so we need to change the order of our outputs again to
        # match the label ordering.

        # For a hidden vector coming in which is of shape
        # [num_layers = 1, batch_size, hidden_size], the hidden size won't
        # change, the

        # Suppose instead that hidden vector does not include c_n,
        # and instead should be just h_n, and what is passed in is
        # [num_dilations, batch_size, hidden_size]
        # [
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    # Zero the hidden layers. If using LSTM, we have both a hidden state and
    # a cell state, defined as 'memory' here. For RNN/GRU, we have only the
    # hidden layer
    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden

# main()