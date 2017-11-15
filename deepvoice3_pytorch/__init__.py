# coding: utf-8

from .version import __version__

import torch
from torch import nn

from .modules import Embedding


class MultiSpeakerTTSModel(nn.Module):
    """Attention seq2seq model + post processing network
    """

    def __init__(self, seq2seq, postnet,
                 mel_dim=80, linear_dim=513,
                 n_speakers=1, speaker_embed_dim=16, padding_idx=None,
                 trainable_positional_encodings=False):
        super(MultiSpeakerTTSModel, self).__init__()
        self.seq2seq = seq2seq
        self.postnet = postnet  # referred as "Converter" in DeepVoice3
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.trainable_positional_encodings = trainable_positional_encodings

        # Speaker embedding
        if n_speakers > 1:
            self.embed_speakers = Embedding(
                n_speakers, speaker_embed_dim, padding_idx)
        self.n_speakers = n_speakers
        self.speaker_embed_dim = speaker_embed_dim

    def get_trainable_parameters(self):
        if self.trainable_positional_encodings:
            return self.parameters()

        decoder = self.seq2seq.decoder

        # Avoid updating the position encoding
        pe_query_param_ids = set(map(id, decoder.embed_query_positions.parameters()))
        pe_keys_param_ids = set(map(id, decoder.embed_keys_positions.parameters()))
        freezed_param_ids = pe_query_param_ids | pe_keys_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, text_sequences, mel_targets=None, speaker_ids=None,
                text_positions=None, frame_positions=None, input_lengths=None):
        B = text_sequences.size(0)

        if speaker_ids is not None:
            speaker_embed = self.embed_speakers(speaker_ids)
        else:
            speaker_embed = None

        # Apply seq2seq
        # (B, T//r, mel_dim*r)
        mel_outputs, alignments, done = self.seq2seq(
            text_sequences, mel_targets, speaker_embed,
            text_positions, frame_positions, input_lengths)

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # (B, T, linear_dim)
        # Convert coarse mel-spectrogram to high resolution spectrogram
        linear_outputs = self.postnet(mel_outputs)
        assert linear_outputs.size(-1) == self.linear_dim

        return mel_outputs, linear_outputs, alignments, done


class AttentionSeq2Seq(nn.Module):
    """Encoder + Decoder with attention
    """

    def __init__(self, encoder, decoder):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.num_attention_layers = sum(
            [layer is not None for layer in decoder.attention])

    def forward(self, text_sequences, mel_targets=None, speaker_embed=None,
                text_positions=None, frame_positions=None, input_lengths=None):
        # (B, T, text_embed_dim)
        encoder_outputs = self.encoder(
            text_sequences, lengths=input_lengths, speaker_embed=speaker_embed)

        # Mel: (B, T//r, mel_dim*r)
        # Alignments: (N, B, T_target, T_input)
        # Done: (B, T//r, 1)
        mel_outputs, alignments, done = self.decoder(
            encoder_outputs, mel_targets,
            text_positions=text_positions, frame_positions=frame_positions,
            speaker_embed=speaker_embed, lengths=input_lengths)

        return mel_outputs, alignments, done


def build_deepvoice3(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
                     n_speakers=1, speaker_embed_dim=16, padding_idx=0,
                     dropout=(1 - 0.95), kernel_size=5,
                     encoder_channels=128,
                     decoder_channels=256,
                     converter_channels=256,
                     query_position_rate=1.0,
                     key_position_rate=1.29,
                     use_memory_mask=False,
                     trainable_positional_encodings=False,
                     ):
    from deepvoice3_pytorch.deepvoice3 import Encoder, Decoder, Converter

    # Seq2seq
    h = encoder_channels  # hidden dim (channels)
    k = kernel_size   # kernel size
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1), (h, k, 1), (h, k, 1), (h, k, 1),
                      (h, k, 2), (h, k, 4), (h, k, 8)],
    )

    h = decoder_channels
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout,
        convolutions=[(h, k, 1), (h, k, 1), (h, k, 2), (h, k, 4), (h, k, 8)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=[True, False, False, False, True],
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask)

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    # Post net
    # NOTE: In deepvoice3, they use decoder states as inputs of converter, but
    # for simplicity I use decoder outoputs (i.e., mel spectrogram) for inputs.
    # This makes it possible to train seq2seq and postnet separately, as
    # described in https://arxiv.org/abs/1710.08969
    in_dim = mel_dim
    h = converter_channels
    converter = Converter(
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        convolutions=[(h, k, 1), (h, k, 1), (h, k, 2), (h, k, 4), (h, k, 8)])

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings)

    return model
