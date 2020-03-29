import torch
import torch.nn as nn


# def main():
#     seq_len = 32
#     hidden_size = 10
#     num_layers = 4
#     num_features = 2
#     batch_size = 4
#     cell_type = 'LSTM'
#
#     model = DRNN(2, hidden_size, num_layers, cell_type=cell_type)
#
#     x1 = torch.randn(seq_len, batch_size, num_features)
#     x2 = torch.randn(seq_len, batch_size, num_features)
#
#     out, (h_n, c_n) = model(x1)
#     print(out.shape)
#     print(len(h_n), [h.shape for h in h_n])
#     print(len(c_n), [c.shape for c in c_n])
#
#     out2, (h_n2, c_n2) = model(x2, (h_n, c_n))
#     print(out2.shape)
#     print(len(h_n2), [h.shape for h in h_n2])
#     print(len(c_n2), [c.shape for c in c_n2])


class DRNN(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout=0,
                 cell_type='LSTM', batch_first=False, dilations=None,
                 residuals=tuple([])):

        super().__init__()

        # Exponentially increasing dilations by default
        if dilations is None:
            self.dilations = [2 ** i for i in range(num_layers)]
        else:
            self.dilations = dilations

        self.batch_first=batch_first
        self.cell_type=cell_type
        self.residuals=residuals

        # LSTM by default
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        # Create the layers of the model
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                c = cell(num_features, hidden_size, dropout=dropout)
                c.double()  # TODO FOR SOME REASON THIS FIXED IT. Think it's
                # because the model itself needs to be using doubles for the
                # weight matrices, whereas prior to this it uses floats by
                # default. possibly consider changing all to floats and
                # removing this to save time and memory
            else:
                c = cell(hidden_size, hidden_size, dropout=dropout)
                c.double()
            self.layers.append(c)

        for i, c in enumerate(self.layers):
            self.add_module("rnn_layer_" + str(i), c)

    # Apply one forward pass. Expects batched inputs. Allow for stateful
    # LSTM by specifying initial hidden values
    # Hidden (for non-LSTM, no c_n): (h_n, c_n), where both h_n, c_n are
    # a list of num_layers elements, where each of elements is of
    # shape (1, original_batch_size * dilation, hidden_size)
    def forward(self, inputs, hidden=None):

        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        if hidden is not None and self.cell_type == "LSTM":
            hidden = list(zip(*hidden))

        hidden_out = []
        cell_state_out = []
        outputs = []  # Just to stop the linter complaining

        connections = {}
        saved_outputs = {}

        for start, finish in self.residuals:
            connections[finish] = start
            saved_outputs[start] = None


        # Input initial inputs to first layer, then iteratively take layer
        # outputs and feed as inputs into the next layer
        for i, (cell, dilation) in enumerate(zip(self.layers, self.dilations)):
            if hidden is None:
                outputs, h = self.apply_layer(cell, inputs, dilation)

                # Apply residual connection
                if i in saved_outputs.keys():
                    saved_outputs[i] = outputs

                if i in connections:
                    outputs += saved_outputs[connections[i]]

                inputs = outputs

            else:
                outputs, h = self.apply_layer(
                    cell, inputs, dilation, hidden[i]
                )
                inputs = outputs

            if self.cell_type == "LSTM":
                h_n, c_n = h
                hidden_out.append(h_n)
                cell_state_out.append(c_n)
            else:
                hidden_out.append(h)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        if self.cell_type == "LSTM":
            return outputs, (hidden_out, cell_state_out)
        else:
            return outputs, hidden_out

    # Takes previous outputs as inputs, pads the sequences to the correct
    # length, transforms the sequence to model the dilated connections,
    # and feeds in the batch
    def apply_layer(self, cell, inputs, dilation, hidden=None):
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        hidden_size = cell.hidden_size

        padded_inputs, _ = self.pad_inputs(inputs, seq_len, dilation)
        dilated_inputs = self.prepare_inputs(padded_inputs, dilation)

        # print(all((inputs == dilated_inputs).view(-1)))

        if hidden is None:
            dilated_outputs, hidden = self.apply_cell(
                dilated_inputs, cell, batch_size, dilation, hidden_size
            )
        else:
            dilated_outputs, hidden = self.apply_cell(
                dilated_inputs, cell, batch_size,
                dilation, hidden_size, hidden
            )

        # Split the outputs back into correct, non-dilated sequences,
        # and then removing any padding that was added
        split_outputs = self.split_outputs(dilated_outputs, dilation)
        outputs = self.unpad_outputs(split_outputs, seq_len)

        return outputs, hidden

    # Generate the default hidden values if necessary, the input the hidden
    # states and the inputs into the LSTM
    def apply_cell(self, dilated_inputs, cell, batch_size, dilation, hidden_size, hidden=None):
        new_bs = batch_size * dilation

        # Generate initial hidden states for the batch if not specified
        if hidden is None:
            if self.cell_type == 'LSTM':
                h, c = self.init_hidden(new_bs, hidden_size)
                hidden = (h.unsqueeze(0), c.unsqueeze(0))

            else:
                hidden = self.init_hidden(new_bs, hidden_size).unsqueeze(0)

        # Feed the batch into the cell, get the outputs
        dilated_outputs, hidden_out = cell(dilated_inputs.double(), hidden)

        return dilated_outputs, hidden_out

    # Simply remove the padding by taking the first seq_len values from the
    # output (thus removing any extra zeros that were appended as padding)
    @staticmethod
    def unpad_outputs(split_outputs, n_steps):
        return split_outputs[:n_steps]

    # Split the outputs of our dilated RNN back into the correct size
    @staticmethod
    def split_outputs(dilated_outputs, dilation):
        dilated_bs = dilated_outputs.size(1)
        original_bs = dilated_bs // dilation

        # Split outputs into batches of the correct size. However,
        # the sequences are still dilated at this point.
        original_bs_chunks = [
            dilated_outputs[:, i * original_bs: (i + 1) * original_bs, :]
            for i in range(dilation)
        ]

        # Stack, without a specified dimension, concatenates the chunks into
        # one tensor by adding a new dimension (0). This just combines the
        # list above into a tensor and so stacked will have four dimensions -
        # the sequences are still dilated at this point
        stacked = torch.stack(original_bs_chunks)

        # The following lines cleverly ensure that the outputs are now back
        # to being correctly interleaved. Not exactly sure how they work.
        interleaved = stacked.transpose(1, 0).contiguous()
        interleaved = interleaved.view(
            dilated_outputs.size(0) * dilation,
            original_bs,
            dilated_outputs.size(2)
        )
        return interleaved

    # Pad each sequence in the batch so that its length is a multiple of the
    # specified dilation
    @staticmethod
    def pad_inputs(inputs, seq_len, dilation):
        is_multiple = (seq_len % dilation) == 0

        if not is_multiple:
            dilated_steps = seq_len // dilation + 1
            zeros = torch.zeros(
                dilated_steps * dilation - inputs.size(0),
                inputs.size(1),
                inputs.size(2),
                dtype=torch.double
            )
            inputs = torch.cat((inputs, zeros))

        else:
            dilated_steps = seq_len // dilation

        return inputs, dilated_steps

    # Transform each sequence in the batch into a dilated sequence. Sequence
    # of length seq_len are converted into sequences of length seq_len /
    # dilation, and each sequence is made up of the values connected by a
    # dilated connection. The number of sequences in the batch increases by
    # a factor of dilation, so the new batch_size is batch_size * dilation
    @staticmethod
    def prepare_inputs(inputs, dilation):
        dilated_seqs = [inputs[i::dilation, :, :] for i in range(dilation)]
        dilated_inputs = torch.cat(dilated_seqs, 1)
        return dilated_inputs

    # Zero the hidden layers. If using LSTM, we have both a hidden state and
    # a cell state, defined as 'memory' here. For RNN/GRU, we have only the
    # hidden layer
    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim, dtype=torch.double)

        if self.cell_type == "LSTM":
            cell_state = torch.zeros(
                batch_size, hidden_dim, dtype=torch.double
            )

            return hidden.double(), cell_state.double()
        else:
            return hidden.double()


# main()
