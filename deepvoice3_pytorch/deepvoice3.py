# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import numpy as np

from fairseq.modules import GradMultiply
from fairseq.modules.conv_tbc import ConvTBC as _ConvTBC

from .modules import Conv1d, LinearizedConv1d, ConvTBC, Embedding, Linear


def has_dilation(convolutions):
    return np.any(np.array(list(map(lambda x: x[2], convolutions))) > 1)


class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_speakers, speaker_embed_dim,
                 padding_idx=None, convolutions=((64, 5, .1),) * 7,
                 max_positions=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None

        # Text input embeddings
        self.embed_tokens = Embedding(n_vocab, embed_dim, padding_idx)

        # Text position embedding
        self.embed_text_positions = Embedding(
            max_positions, embed_dim, padding_idx)
        self.embed_text_positions.weight.data = position_encoding_init(
            max_positions, embed_dim)

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_fc1 = Linear(speaker_embed_dim, embed_dim)
            self.speaker_fc2 = Linear(speaker_embed_dim, embed_dim)
        self.n_speakers = n_speakers

        # Non-causual convolutions
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.speaker_projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()

        Conv1dLayer = Conv1d if has_dilation(convolutions) else ConvTBC

        for (out_channels, kernel_size, dilation) in convolutions:
            pad = (kernel_size - 1) // 2 * dilation
            dilation = (dilation,)
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.speaker_projections.append(
                Linear(speaker_embed_dim, out_channels) if n_speakers > 1 else None)
            self.convolutions.append(
                Conv1dLayer(in_channels, out_channels * 2, kernel_size, padding=pad,
                            dilation=dilation, dropout=dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # embed text_sequences
        x = self.embed_tokens(text_sequences)
        if text_positions is not None:
            x += self.embed_text_positions(text_positions)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # embed speakers
        if speaker_embed is not None:
            # expand speaker embedding for all time steps
            # (B, N) -> (B, T, N)
            ss = speaker_embed.size()
            speaker_embed = speaker_embed.unsqueeze(1).expand(
                ss[0], x.size(1), ss[-1])
            speaker_embed_btc = speaker_embed
            speaker_embed_tbc = speaker_embed.transpose(0, 1)
            x += F.softsign(self.speaker_fc1(speaker_embed_btc))

        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        use_convtbc = isinstance(self.convolutions[0], _ConvTBC)
        # TBC case: B x T x C -> T x B x C
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        # １D conv blocks
        for proj, speaker_proj, conv in zip(
                self.projections, self.speaker_projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            splitdim = -1 if use_convtbc else 1
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            if speaker_proj is not None:
                softsign = F.softsign(speaker_proj(
                    speaker_embed_tbc if use_convtbc else speaker_embed_btc))
                softsign = softsign if use_convtbc else softsign.transpose(1, 2)
                a = a + softsign
            x = a * F.sigmoid(b)
            x = (x + residual) * math.sqrt(0.5)

        # Back to batch first
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        # project back to size of embedding
        keys = self.fc2(x)
        if speaker_embed is not None:
            keys += F.softsign(self.speaker_fc2(speaker_embed_btc))

        # scale gradients (this only affects backward, not forward)
        if self.num_attention_layers is not None:
            keys = GradMultiply.apply(keys, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, dropout=0.1):
        super(AttentionLayer, self).__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)
        self.dropout = dropout

    def forward(self, query, encoder_out, mask=None, last_attended=None,
                window_size=3):
        keys, values = encoder_out
        residual = query

        # attention
        x = self.in_projection(query)
        x = torch.bmm(x, keys)

        mask_value = -float("inf")
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)

        if last_attended is not None:
            if last_attended > 0:
                x[:, :, :last_attended] = mask_value
            ahead = last_attended + window_size
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores


def position_encoding_init(n_position, d_pos_vec, position_rate=1.0):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [position_rate * pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class Decoder(nn.Module):
    def __init__(self, embed_dim, n_speakers, speaker_embed_dim,
                 in_dim=80, r=5,
                 max_positions=512, padding_idx=None,
                 convolutions=((128, 5, 1),) * 4,
                 attention=True, dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=True,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r

        in_channels = in_dim * r
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = Embedding(
            max_positions, convolutions[0][0], padding_idx)
        self.embed_query_positions.weight.data = position_encoding_init(
            max_positions, convolutions[0][0], position_rate=query_position_rate)
        self.embed_keys_positions = Embedding(
            max_positions, embed_dim, padding_idx)
        self.embed_keys_positions.weight.data = position_encoding_init(
            max_positions, embed_dim, position_rate=key_position_rate)

        self.fc1 = Linear(in_channels, convolutions[0][0], dropout=dropout)
        in_channels = convolutions[0][0]

        # Causual convolutions
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        Conv1dLayer = Conv1d if has_dilation(convolutions) else LinearizedConv1d

        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            pad = (kernel_size - 1) * dilation
            dilation = (dilation,)
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1dLayer(in_channels, out_channels * 2, kernel_size,
                            padding=pad, dilation=dilation, dropout=dropout))
            self.attention.append(AttentionLayer(out_channels, embed_dim,
                                                 dropout=dropout)
                                  if attention[i] else None)
            in_channels = out_channels
        self.fc2 = Linear(in_channels, in_dim * r)

        # decoder states -> Done binary flag
        self.fc3 = Linear(in_channels, 1)

        self._is_inference_incremental = False
        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = \
                [force_monotonic_attention] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None):

        if inputs is None:
            assert text_positions is not None
            self._start_incremental_inference()
            outputs = self._incremental_forward(encoder_out, text_positions)
            self._stop_incremental_inference()
            return outputs

        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        keys, values = encoder_out

        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None

        # position encodings
        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions)
            keys += text_pos_embed
        if frame_positions is not None:
            frame_pos_embed = self.embed_query_positions(frame_positions)

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        x = inputs
        x = F.dropout(x, p=self.dropout, training=self.training)

        # project to size of convolution
        x = F.relu(self.fc1(x), inplace=False)

        use_convtbc = isinstance(self.convolutions[0], _ConvTBC)
        # TBC case: B x T x C -> T x B x C
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        # temporal convolutions
        alignments = []
        for idx, (proj, conv, attention) in enumerate(zip(
                self.projections, self.convolutions, self.attention)):
            residual = x if proj is None else proj(x)
            if idx > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            splitdim = -1 if use_convtbc else 1
            if use_convtbc:
                x = conv.remove_future_timesteps(x)
            else:
                x = x[:, :, :residual.size(-1)]
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            x = a * F.sigmoid(b)

            # Feed conv output to attention layer as query
            if attention is not None:
                # (B x T x C)
                x = x.transpose(1, 0) if use_convtbc else x.transpose(1, 2)
                x = x if frame_positions is None else x + frame_pos_embed
                x, alignment = attention(x, (keys, values), mask=mask)
                # (T x B x C)
                x = x.transpose(1, 0) if use_convtbc else x.transpose(1, 2)
                alignments += [alignment]

            # residual
            x = (x + residual) * math.sqrt(0.5)

        # Back to batch first
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        # well, I'm not sure this is really necesasary
        decoder_states = x

        # project to mel-spectorgram
        x = F.sigmoid(self.fc2(decoder_states))

        # Done flag
        done = F.sigmoid(self.fc3(decoder_states))

        return x, torch.stack(alignments), done

    def incremental_inference(self, beam_size=None):
        """Context manager for incremental inference.
        This provides an optimized forward pass for incremental inference
        (i.e., it predicts one time step at a time). If the input order changes
        between time steps, call model.decoder.reorder_incremental_state to
        update the relevant buffers. To generate a fresh sequence, first call
        model.decoder.start_fresh_sequence.
        Usage:
        ```
        with model.decoder.incremental_inference():
            for step in range(maxlen):
                out = model.decoder(tokens[:, :step], positions[:, :step],
                                    encoder_out)
                probs = F.log_softmax(out[:, -1, :])
        ```
        """
        class IncrementalInference(object):

            def __init__(self, decoder, beam_size):
                self.decoder = decoder
                self.beam_size = beam_size

            def __enter__(self):
                self.decoder._start_incremental_inference(self.beam_size)

            def __exit__(self, *args):
                self.decoder._stop_incremental_inference()

        return IncrementalInference(self, beam_size)

    def _start_incremental_inference(self):
        assert not self._is_inference_incremental, \
            'already performing incremental inference'
        self._is_inference_incremental = True

        # save original forward
        self._orig_forward = self.forward

        # switch to incremental forward
        self.forward = self._incremental_forward

        # start a fresh sequence
        self.start_fresh_sequence()

    def _stop_incremental_inference(self):
        # restore original forward
        self.forward = self._orig_forward

        self._is_inference_incremental = False

    def _incremental_forward(self, encoder_out, text_positions,
                             initial_input=None, test_inputs=None):
        assert self._is_inference_incremental

        keys, values = encoder_out
        B = keys.size(0)

        # position encodings
        text_pos_embed = self.embed_keys_positions(text_positions)
        keys += text_pos_embed

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        outputs = []
        alignments = []
        dones = []
        # intially set to zeros
        last_attended = [None] * len(self.attention)
        for idx, v in enumerate(self.force_monotonic_attention):
            last_attended[idx] = 0 if v else None

        num_attention_layers = sum([layer is not None for layer in self.attention])
        t = 0
        if initial_input is None:
            initial_input = Variable(
                keys.data.new(B, 1, self.in_dim * self.r).zero_())
        current_input = initial_input
        while True:
            frame_pos = Variable(keys.data.new(B, 1).zero_().add_(t + 1)).long()
            frame_pos_embed = self.embed_query_positions(frame_pos)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
            x = current_input
            x = F.dropout(x, p=self.dropout, training=self.training)

            # project to size of convolution
            x = F.relu(self.fc1(x), inplace=False)

            # temporal convolutions
            ave_alignment = None
            for idx, (proj, conv, attention) in enumerate(zip(
                    self.projections, self.convolutions, self.attention)):
                residual = x if proj is None else proj(x)
                if idx > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv.incremental_forward(x)
                a, b = x.split(x.size(-1) // 2, dim=-1)
                x = a * F.sigmoid(b)

                # attention
                if attention is not None:
                    x = x + frame_pos_embed
                    x, alignment = attention(x, (keys, values),
                                             last_attended=last_attended[idx])
                    if self.force_monotonic_attention:
                        last_attended[idx] = alignment.max(-1)[1].view(-1).data[0]
                    if ave_alignment is None:
                        ave_alignment = alignment
                    else:
                        ave_alignment = ave_alignment + ave_alignment

                # residual
                x = (x + residual) * math.sqrt(0.5)

            ave_alignment = ave_alignment.div_(num_attention_layers)

            # Ooutput & done flag predictions
            output = F.sigmoid(self.fc2(x))
            done = F.sigmoid(self.fc3(x))

            outputs += [output]
            alignments += [ave_alignment]
            dones += [done]

            t += 1
            if (done > 0.5).all() and t > self.min_decoder_steps:
                break
            elif t > self.max_decoder_steps:
                print("Warning! doesn't seems to be converged")
                break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments, dones

    def start_fresh_sequence(self):
        """Clear all state used for incremental generation.
        **For incremental inference only**
        This should be called before generating a fresh sequence.
        beam_size is required if using BeamableMM.
        """
        if self._is_inference_incremental:
            self.prev_state = None
            for conv in self.convolutions:
                conv.clear_buffer()


class Converter(nn.Module):
    def __init__(self, in_dim, out_dim, convolutions=((256, 5, 1),) * 4, dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Non-causual convolutions
        in_channels = convolutions[0][0]
        self.fc1 = Linear(in_dim, in_channels)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()

        Conv1dLayer = Conv1d if has_dilation(convolutions) else ConvTBC
        for (out_channels, kernel_size, dilation) in convolutions:
            pad = (kernel_size - 1) // 2 * dilation
            dilation = (dilation,)
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1dLayer(in_channels, out_channels * 2, kernel_size,
                            padding=pad, dilation=dilation, dropout=dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, out_dim)

    def forward(self, x):
        # project to size of convolution
        x = F.relu(self.fc1(x), inplace=False)

        use_convtbc = isinstance(self.convolutions[0], _ConvTBC)
        # TBC case: B x T x C -> T x B x C
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        # １D conv blocks
        for idx, (proj, conv) in enumerate(zip(self.projections, self.convolutions)):
            residual = x if proj is None else proj(x)
            if idx > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            splitdim = -1 if use_convtbc else 1
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            x = a * F.sigmoid(b)
            x = (x + residual) * math.sqrt(0.5)

        # Back to batch first
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        return F.sigmoid(self.fc2(x))
