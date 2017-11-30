import torch
from torch import nn

from deepvoice3_pytorch import MultiSpeakerTTSModel, AttentionSeq2Seq


def deepvoice3(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
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
               use_decoder_state_for_postnet_input=True,
               max_positions=512,
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
        dropout=dropout, max_positions=max_positions,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1), (h, k, 1), (h, k, 1), (h, k, 1),
                      (h, k, 2), (h, k, 4), (h, k, 8)],
        use_glu=True,
    )

    h = decoder_channels
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        preattention=[(h, k, 1), (h, k, 2), (h, k, 4), (h, k, 8)],
        convolutions=[(h, k, 1), (h, k, 1), (h, k, 2), (h, k, 4), (h, k, 8)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask,
        use_glu=True,
    )

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    # Post net
    if use_decoder_state_for_postnet_input:
        in_dim = h // r
    else:
        in_dim = mel_dim
    h = converter_channels
    converter = Converter(
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        convolutions=[(h, k, 1), (h, k, 1), (h, k, 2), (h, k, 4), (h, k, 8)],
        use_glu=True,
    )

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input)

    return model


def nyanko(n_vocab, embed_dim=128, mel_dim=80, linear_dim=513, r=1,
           n_speakers=1, speaker_embed_dim=16, padding_idx=0,
           dropout=(1 - 0.95), kernel_size=3,
           encoder_channels=256,
           decoder_channels=256,
           converter_channels=512,
           query_position_rate=1.0,
           key_position_rate=1.29,
           use_memory_mask=False,
           trainable_positional_encodings=False,
           force_monotonic_attention=True,
           use_decoder_state_for_postnet_input=False,
           max_positions=512):
    from deepvoice3_pytorch.nyanko import Encoder, Decoder, Converter
    assert encoder_channels == decoder_channels

    # Seq2seq
    encoder = Encoder(
        n_vocab, embed_dim, channels=encoder_channels, kernel_size=kernel_size,
        padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout,
    )

    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, channels=decoder_channels,
        kernel_size=kernel_size, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask)

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    if use_decoder_state_for_postnet_input:
        in_dim = decoder_channels // r
    else:
        in_dim = mel_dim

    converter = Converter(
        in_dim=in_dim, out_dim=linear_dim, channels=converter_channels,
        kernel_size=kernel_size, dropout=dropout)

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input)

    return model


# TODO:
latest = nyanko
