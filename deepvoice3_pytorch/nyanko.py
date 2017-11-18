# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import numpy as np

from .modules import Embedding, Linear, Conv1d, ConvTranspose1d
from .modules import get_mask_from_lengths
from .deepvoice3 import position_encoding_init, AttentionLayer


class HighwayConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=None,
                 dilation=1, causual=False, dropout=0, std_mul=4.0, glu=False):
        super(HighwayConv1d, self).__init__()
        if padding is None:
            # no future time stamps available
            if causual:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causual = causual
        self.glu = glu

        self.conv = Conv1d(in_channels, 2 * out_channels,
                           kernel_size=kernel_size, padding=padding,
                           dilation=dilation, dropout=dropout,
                           std_mul=std_mul)
        self.dropout = dropout

    def _forward(self, x, is_incremental):
        """Forward

        Args:
            x: (B, in_channels, T)
        returns:
            (B, out_channels, T)
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causual else x

        if self.glu:
            x = F.glu(x, dim=splitdim)
            return (x + residual) * math.sqrt(0.5)
        else:
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            T = F.sigmoid(b)
            return (T * a + (1 - T) * residual) * math.sqrt(0.5)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def clear_buffer(self):
        self.conv.clear_buffer()


class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, channels,
                 n_speakers=1, speaker_embed_dim=16,
                 padding_idx=None,
                 max_positions=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.dropout = dropout

        # Text input embeddings
        self.embed_tokens = Embedding(n_vocab, embed_dim, padding_idx)

        E = embed_dim
        D = channels
        self.convnet = nn.Sequential(
            Conv1d(E, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(),
            Conv1d(2 * D, 2 * D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=1, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=3, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=9, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=27, std_mul=4.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=1, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=3, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=9, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=27, std_mul=4.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=1, std_mul=4.0, dropout=dropout),
            HighwayConv1d(2 * D, 2 * D, kernel_size=3, padding=None,
                          dilation=1, std_mul=4.0, dropout=dropout),

            HighwayConv1d(2 * D, 2 * D, kernel_size=1, padding=0,
                          dilation=1, std_mul=4.0, dropout=dropout),
        )

        # self.fc = Linear(E, D) if E != D else None

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        # embed text_sequences
        # (B, T, E)
        x = self.embed_tokens(text_sequences)
        input_embed = x

        # (B, E, T)
        x = x.transpose(1, 2)
        # (B, 2*D, T)
        x = self.convnet(x)
        # (B, T, 2*D)
        x = x.transpose(1, 2)
        # (B, T, D) and (B, T, D)

        keys, values = x.split(x.size(-1) // 2, dim=-1)

        # residual connection
        # TODO
        # input_embed = input_embed if self.fc is None else self.fc(input_embed)
        # values = (values + input_embed) * math.sqrt(0.5)

        return keys, values


class Decoder(nn.Module):
    def __init__(self, embed_dim, in_dim=80, r=5, channels=256,
                 n_speakers=1, speaker_embed_dim=16,
                 max_positions=512, padding_idx=None,
                 dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r

        D = channels
        F = in_dim * r  # should be r = 1 to replicate
        self.audio_encoder_modules = nn.ModuleList([
            Conv1d(F, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),

            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=1, causual=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=3, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=9, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=27, causual=True, std_mul=4.0, dropout=dropout),

            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=1, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=3, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=9, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=27, causual=True, std_mul=4.0, dropout=dropout),

            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=3, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=3, causual=True, std_mul=4.0, dropout=dropout),
        ])

        self.attention = AttentionLayer(D, D, dropout=dropout)

        self.audio_decoder_modules = nn.ModuleList([
            Conv1d(2 * D, D, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=1, causual=True, std_mul=1.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=3, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=9, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=27, causual=True, std_mul=4.0, dropout=dropout),

            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=1, causual=True, std_mul=4.0, dropout=dropout),
            HighwayConv1d(D, D, kernel_size=3, padding=None,
                          dilation=1, causual=True, std_mul=4.0, dropout=dropout),

            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=4.0),
            nn.ReLU(),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(),
            Conv1d(D, D, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(),

            Conv1d(D, F, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            # nn.Sigmoid()
        ])

        # Done prediction
        self.fc = Linear(F, 1)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = Embedding(
            max_positions, D, padding_idx)
        self.embed_query_positions.weight.data = position_encoding_init(
            max_positions, D, position_rate=query_position_rate)
        self.embed_keys_positions = Embedding(
            max_positions, D, padding_idx)
        self.embed_keys_positions.weight.data = position_encoding_init(
            max_positions, D, position_rate=key_position_rate)

        # options
        self._is_inference_incremental = False
        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
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
            keys = keys + text_pos_embed
        if frame_positions is not None:
            frame_pos_embed = self.embed_query_positions(frame_positions)

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        # (B, T, C)
        x = inputs

        # (B, C, T)
        x = x.transpose(1, 2)

        # Apply audio encoder
        for f in self.audio_encoder_modules:
            x = f(x)
        Q = x

        # Attention modules assume query as (B, T, C)
        x = x.transpose(1, 2)
        R, alignments = self.attention(
            x + frame_pos_embed, (keys, values), mask=mask)
        R = R.transpose(1, 2)

        # (B, C*2, T)
        Rd = torch.cat((R, Q), dim=1)
        x = Rd

        # Apply audio decoder
        for f in self.audio_decoder_modules:
            x = f(x)

        # (B, T, C)
        x = x.transpose(1, 2)

        # Mel
        outputs = F.sigmoid(x)

        # Done prediction
        done = F.sigmoid(self.fc(x))

        # Adding extra dim for convenient
        alignments = alignments.unsqueeze(0)

        return outputs, alignments, done

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
        if text_positions is not None:
            text_pos_embed = self.embed_keys_positions(text_positions)
            keys = keys + text_pos_embed

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        outputs = []
        alignments = []
        dones = []
        # intially set to zeros
        last_attended = 0 if self.force_monotonic_attention else None

        t = 0
        if initial_input is None:
            initial_input = Variable(
                keys.data.new(B, 1, self.in_dim * self.r).zero_())
        current_input = initial_input
        while True:
            # frame pos start with 1.
            frame_pos = Variable(keys.data.new(B, 1).fill_(t + 1)).long()
            frame_pos_embed = self.embed_query_positions(frame_pos)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]

            # (B, 1, C)
            x = current_input

            for f in self.audio_encoder_modules:
                try:
                    x = f.incremental_forward(x)
                except AttributeError as e:
                    x = f(x)
            Q = x

            R, alignment = self.attention(
                x + frame_pos_embed, (keys, values), last_attended=last_attended)

            Rd = torch.cat((R, Q), dim=-1)
            x = Rd
            for f in self.audio_decoder_modules:
                try:
                    x = f.incremental_forward(x)
                except AttributeError as e:
                    x = f(x)

            # Ooutput & done flag predictions
            output = F.sigmoid(x)
            done = F.sigmoid(self.fc(x))

            outputs += [output]
            alignments += [alignment]
            dones += [done]

            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments, dones

    def start_fresh_sequence(self):
        if self._is_inference_incremental:
            _clear_modules(self.audio_encoder_modules)
            _clear_modules(self.audio_decoder_modules)


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    def __init__(self, in_dim, out_dim, channels=512, dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

        F = in_dim
        Fd = out_dim
        C = channels
        self.convnet = nn.Sequential(
            Conv1d(F, C, kernel_size=1, padding=0, dilation=1, std_mul=1.0),

            HighwayConv1d(C, C, kernel_size=3, padding=None, dilation=1, std_mul=1.0),
            HighwayConv1d(C, C, kernel_size=3, padding=None, dilation=3, std_mul=4.0),

            ConvTranspose1d(C, C, kernel_size=2, padding=0, stride=2, std_mul=4.0),
            HighwayConv1d(C, C, kernel_size=3, padding=None, dilation=1, std_mul=1.0),
            HighwayConv1d(C, C, kernel_size=3, padding=None, dilation=3, std_mul=4.0),
            ConvTranspose1d(C, C, kernel_size=2, padding=0, stride=2, std_mul=4.0),
            HighwayConv1d(C, C, kernel_size=3, padding=None, dilation=1, std_mul=1.0),
            HighwayConv1d(C, C, kernel_size=3, padding=None, dilation=3, std_mul=4.0),

            Conv1d(C, 2 * C, kernel_size=1, padding=0, dilation=1, std_mul=4.0),

            HighwayConv1d(2 * C, 2 * C, kernel_size=3, padding=None, dilation=1, std_mul=1.0),
            HighwayConv1d(2 * C, 2 * C, kernel_size=3, padding=None, dilation=1, std_mul=4.0),

            Conv1d(2 * C, Fd, kernel_size=1, padding=0, dilation=1, std_mul=4.0),

            Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=1.0),
            nn.ReLU(),
            Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.ReLU(),

            Conv1d(Fd, Fd, kernel_size=1, padding=0, dilation=1, std_mul=2.0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.convnet(x.transpose(1, 2)).transpose(1, 2)
