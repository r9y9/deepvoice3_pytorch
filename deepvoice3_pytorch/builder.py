import torch
from torch import nn

from deepvoice3_pytorch import MultiSpeakerTTSModel, AttentionSeq2Seq


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
                     force_monotonic_attention=True,
                     ):
    """Build deepvoice3
    """
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
        force_monotonic_attention=force_monotonic_attention,
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


def build_nyanko(n_vocab, embed_dim=128, mel_dim=80, linear_dim=513, r=4,
                 n_speakers=1, speaker_embed_dim=16, padding_idx=0,
                 dropout=(1 - 0.95), kernel_size=5,
                 encoder_channels=256,
                 decoder_channels=256,
                 converter_channels=512,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 use_memory_mask=False,
                 trainable_positional_encodings=False,
                 force_monotonic_attention=True,):
    from deepvoice3_pytorch.nyanko import Encoder, Decoder, Converter

    test = False

    if test:
        from deepvoice3_pytorch.deepvoice3 import Encoder as _Encoder
        h = encoder_channels
        k = 3
        encoder = _Encoder(
            n_vocab, embed_dim, padding_idx=padding_idx,
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            dropout=dropout,
            # (channels, kernel_size, dilation)
            convolutions=[(h, k, 1), (h, k, 1), (h, k, 1), (h, k, 1),
                          (h, k, 2), (h, k, 4), (h, k, 8)],
        )
    else:
        # Seq2seq
        encoder = Encoder(
            n_vocab, embed_dim, channels=encoder_channels, padding_idx=padding_idx,
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            dropout=dropout,
        )

    test = True

    if test:
        from deepvoice3_pytorch.deepvoice3 import Decoder as _Decoder
        h = decoder_channels
        k = 3
        decoder = _Decoder(
            embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            dropout=dropout,
            convolutions=[(h, k, 1), (h, k, 1), (h, k, 2), (h, k, 4), (h, k, 8)],
            attention=[True, False, False, False, True],
            force_monotonic_attention=force_monotonic_attention,
            query_position_rate=query_position_rate,
            key_position_rate=key_position_rate,
            use_memory_mask=use_memory_mask)
    else:
        decoder = Decoder(
            embed_dim, in_dim=mel_dim, r=r, channels=decoder_channels,
            padding_idx=padding_idx,
            n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
            dropout=dropout,
            force_monotonic_attention=force_monotonic_attention,
            query_position_rate=query_position_rate,
            key_position_rate=key_position_rate,
            use_memory_mask=use_memory_mask)

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    converter = Converter(
        in_dim=mel_dim, out_dim=linear_dim, channels=converter_channels,
        dropout=dropout)

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings)

    return model
