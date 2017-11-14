import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="deepvoice3",

    # Text:
    # [en, jp]
    frontend='en',
    # en: Word -> pronunciation using CMUDict
    # jp: Word -> pronounciation usnig MeCab
    # [0 ~ 1.0]: 0 means no replacement happens.
    replace_pronunciation_prob=0.5,

    # Audio:
    num_mels=80,
    fft_size=1024,
    hop_size=256,
    sample_rate=22050,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    downsample_step=1,
    outputs_per_step=4,
    padding_idx=0,
    dropout=1 - 0.95,
    kernel_size=5,
    encoder_channels=128,
    decoder_channels=256,
    converter_channels=256,
    query_position_rate=1.0,
    key_position_rate=1.29,  # 2.37 for jsut
    use_memory_mask=True,

    # Data loader
    pin_memory=True,
    num_workers=2,

    # Loss
    priority_freq=3000,  # heuristic: priotrize [0 ~ priotiry_freq] for linear loss
    priority_freq_weight=0.5,  # (1-w)*linear_loss + w*priority_linear_loss
    # https://arxiv.org/pdf/1710.08969.pdf
    binary_divergence_weight=0.0,  # set 0 to disable it
    use_guided_attention=False,
    guided_attention_sigma=0.2,

    # Training:
    batch_size=16,
    adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    initial_learning_rate=0.001,
    decay_learning_rate=True,
    lr_schedule=None,
    lr_schedule_kwargs={
        "anneal_rate": 0.98,
        "anneal_interval": 30000,
    },
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=5.0,

    # Save
    checkpoint_interval=5000,

    # Eval:
    max_iters=200,
    griffin_lim_iters=60,
    power=1.4,              # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
