import tensorflow as tf

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="deepvoice3",

    # Text:
    # [en, jp]
    frontend='en',

    # Replace words to its pronunciation with fixed probability.
    # e.g., 'hello' to 'HH AH0 L OW1'
    # [en, jp]
    # en: Word -> pronunciation using CMUDict
    # jp: Word -> pronounciation usnig MeCab
    # [0 ~ 1.0]: 0 means no replacement happens.
    replace_pronunciation_prob=0.5,

    # Convenient model builder
    # [deepvoice3, nyanko, latest]
    # Definitions can be found at deepvoice3_pytorch/builder.py
    # deepvoice3: build DeepVoice3　https://arxiv.org/abs/1710.07654
    # nyanko: https://arxiv.org/abs/1710.08969
    # latest: Latest model I (@r9y9) have been working on.
    builder="latest",

    # Must be configured depends on the dataset and model you use
    n_speakers=1,
    speaker_embed_dim=16,

    # Presets known to work good.
    # NOTE: If True, this will overwride params with presets[builder]
    use_preset=False,
    presets={
        "deepvoice3": {
            "n_speakers": 1,
            "downsample_step": 4,
            "outputs_per_step": 1,
            "embedding_weight_std": 0.1,
            "dropout": 1 - 0.95,
            "kernel_size": 3,
            "text_embed_dim": 256,
            "encoder_channels": 512,
            "decoder_channels": 256,
            "converter_channels": 256,
            "use_guided_attention": True,
            "guided_attention_sigma": 0.2,
            "binary_divergence_weight": 0.1,
            "use_decoder_state_for_postnet_input": True,

            "clip_thresh": 0.1,
            "initial_learning_rate": 5e-4,
        },
        "deepvoice3_vctk": {
            "n_speakers": 108,
            "speaker_embed_dim": 16,
            "downsample_step": 4,
            "outputs_per_step": 1,
            "embedding_weight_std": 0.1,
            "speaker_embedding_weight_std": 0.01,
            "dropout": 1 - 0.95,
            "kernel_size": 3,
            "text_embed_dim": 256,
            "encoder_channels": 512,
            "decoder_channels": 256,
            "converter_channels": 256,
            "use_guided_attention": True,
            "guided_attention_sigma": 0.2,
            "binary_divergence_weight": 0.1,
            "use_decoder_state_for_postnet_input": True,

            "query_position_rate": 1.0,
            "key_position_rate": 1.385,

            "clip_thresh": 0.1,
            "initial_learning_rate": 5e-4,
        },
        "nyanko": {
            "downsample_step": 4,
            "outputs_per_step": 1,
            "embedding_weight_std": 0.01,
            "dropout": 1 - 0.95,
            "kernel_size": 3,
            "text_embed_dim": 128,
            "encoder_channels": 256,
            "decoder_channels": 256,
            "converter_channels": 256,
            "use_guided_attention": True,
            "guided_attention_sigma": 0.2,
            "binary_divergence_weight": 0.1,
            "use_decoder_state_for_postnet_input": True,

            "clip_thresh": 0.1,
            "initial_learning_rate": 5e-4,
        },
        "latest": {},
    },

    # Audio:
    num_mels=80,
    fft_size=1024,
    hop_size=256,
    sample_rate=22050,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    downsample_step=4,  # must be 4 when builder="nyanko"
    outputs_per_step=1,  # must be 1 when builder="nyanko"
    embedding_weight_std=0.1,
    speaker_embedding_weight_std=0.01,
    padding_idx=0,
    max_positions=512,
    dropout=1 - 0.95,
    kernel_size=3,
    text_embed_dim=128,
    encoder_channels=256,
    decoder_channels=256,
    # Note: large converter channels requires significant computational cost
    converter_channels=256,
    query_position_rate=1.0,
    key_position_rate=1.385,  # 2.37 for jsut
    use_memory_mask=True,
    trainable_positional_encodings=False,
    freeze_embedding=False,
    # If True, use decoder's internal representation for postnet inputs,
    # otherwise use mel-spectrogram.
    use_decoder_state_for_postnet_input=True,

    # Data loader
    pin_memory=True,
    num_workers=2,

    # Loss
    masked_loss_weight=0.0,  # (1-w)*loss + w * masked_loss
    priority_freq=3000,  # heuristic: priotrize [0 ~ priotiry_freq] for linear loss
    priority_freq_weight=0.0,  # (1-w)*linear_loss + w*priority_linear_loss
    # https://arxiv.org/pdf/1710.08969.pdf
    # Adding the divergence to the loss stabilizes training, expecially for
    # very deep (> 10 layers) networks.
    # Binary div loss seems has approx 10x scale compared to L1 loss, so I choose 0.1.
    binary_divergence_weight=0.1,  # set 0 to disable
    use_guided_attention=True,
    guided_attention_sigma=0.2,

    # Training:
    batch_size=16,
    adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    initial_learning_rate=5e-4,  # 0.001,
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=0.1,

    # Save
    checkpoint_interval=10000,
    eval_interval=10000,
    save_optimizer_state=True,

    # Eval:
    # this can be list for multple layers of attention
    # e.g., [True, False, False, False, True]
    force_monotonic_attention=True,
    power=1.4,  # Power to raise magnitudes to prior to phase retrieval
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
