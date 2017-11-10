"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>         Directory contains preprocessed features.
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    --log-event-path=<name>     Log event path.
    -h, --help                Show this help message and exit
"""
from docopt import docopt

import sys
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

# The deepvoice3 model
from deepvoice3_pytorch import frontend, build_deepvoice3
import audio
import lrschedule

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from numba import jit

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser

import librosa.display
from matplotlib import pyplot as plt
import sys
import os
import tensorboard_logger
from tensorboard_logger import log_value
from hparams import hparams, hparams_debug_string

fs = hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = True

_frontend = None  # to be set later


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')


class TextDataSource(FileDataSource):
    def __init__(self, data_root):
        self.data_root = data_root

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[-1], lines))
        return lines

    def collect_features(self, text):
        seq = _frontend.text_to_sequence(text, p=hparams.replace_pronunciation_prob)
        return np.asarray(seq, dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col):
        self.data_root = data_root
        self.col = col

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), lines))
        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root):
        super(MelSpecDataSource, self).__init__(data_root, 1)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root):
        super(LinearSpecDataSource, self).__init__(data_root, 0)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    # Add single zeros frame at least, so plus 1
    max_target_len = int(np.max([len(x[1]) for x in batch]) + 1)
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    max_target_len += b_pad

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)

    # text positions
    text_positions = np.array([_pad(np.arange(1, len(x[0]) + 1), max_input_len)
                               for x in batch], dtype=np.int)
    text_positions = torch.LongTensor(text_positions)

    # frame positions
    s, e = 1, max_target_len // r + 1
    # if b_pad > 0:
    #    s, e = s - 1, e - 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
        len(batch), max_target_len // r)

    # done flags
    done = np.array([_pad(np.zeros(len(x[1]) // r - 1), max_target_len // r, constant_values=1)
                     for x in batch])
    done = torch.FloatTensor(done)

    return x_batch, input_lengths, mel_batch, y_batch, (text_positions, frame_positions), done


def save_alignment(path, attn):
    plot_alignment(attn.T, path, info="deepvoice3, step={}".format(global_step))


def save_spectrogram(path, linear_output):
    spectrogram = audio._denormalize(linear_output)
    plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    # Multi-hop attention
    if attn.dim() == 4:
        for i, alignment in enumerate(attn):
            alignment_dir = join(checkpoint_dir, "alignment_layer{}".format(i + 1))
            os.makedirs(alignment_dir, exist_ok=True)
            path = join(alignment_dir, "step{:09d}_layer_{}_alignment.png".format(
                global_step, i + 1))
            alignment = alignment[idx].cpu().data.numpy()
            save_alignment(path, alignment)

        # Save averaged alignment
        alignment_dir = join(checkpoint_dir, "alignment_ave")
        os.makedirs(alignment_dir, exist_ok=True)
        path = join(alignment_dir, "step{:09d}_alignment.png".format(global_step))
        alignment = attn.mean(0)[idx].cpu().data.numpy()
        save_alignment(path, alignment)
    else:
        assert False

    # Predicted spectrogram
    path = join(checkpoint_dir, "step{:09d}_predicted_spectrogram.png".format(
        global_step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = audio.inv_spectrogram(linear_output.T)
    path = join(checkpoint_dir, "step{:09d}_predicted.wav".format(
        global_step))
    audio.save_wav(signal, path)

    # Target spectrogram
    path = join(checkpoint_dir, "step{:09d}_target_spectrogram.png".format(
        global_step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)


def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def spec_loss(y_hat, y):
    l1_loss = nn.L1Loss()(y_hat, y)
    y_hat_logits = logit(y_hat)
    binary_div = torch.mean(-y * y_hat_logits + torch.log(1 + torch.exp(y_hat_logits)))
    return l1_loss, binary_div


@jit(nopython=True)
def guided_attention(N, T, g=0.2):
    W = np.empty((N, T), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T)**2 / 2 / g / g)
    return W


def train(model, data_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim
    r = hparams.outputs_per_step
    current_lr = init_lr

    binary_criterion = nn.BCELoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, input_lengths, mel, y, positions, done) in tqdm(enumerate(data_loader)):
            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            x, mel, y = x[indices], mel[indices], y[indices]
            text_positions, frame_positions = text_positions[indices], frame_positions[indices]
            done = done[indices]
            done = done.unsqueeze(-1) if done.dim() == 2 else done

            # Feed data
            x, mel, y = Variable(x), Variable(mel), Variable(y)
            text_positions = Variable(text_positions)
            frame_positions = Variable(frame_positions)
            done = Variable(done)
            if use_cuda:
                x, mel, y = x.cuda(), mel.cuda(), y.cuda()
                text_positions = text_positions.cuda()
                frame_positions = frame_positions.cuda()
                done = done.cuda()

            mel_outputs, linear_outputs, attn, done_hat = model(
                x, mel,
                text_positions=text_positions, frame_positions=frame_positions,
                input_lengths=sorted_lengths)

            # Losses
            w = hparams.binary_divergence_weight

            # mel:
            mel_l1_loss, mel_binary_div = spec_loss(mel_outputs[:, :-r, :], mel[:, r:, :])
            mel_loss = (1 - w) * mel_l1_loss + w * mel_binary_div

            # done:
            done_loss = binary_criterion(done_hat, done)

            # linear:
            linear_l1_loss, linear_binary_div = spec_loss(linear_outputs[:, :-r, :], y[:, r:, :])
            linear_loss = (1 - w) * linear_l1_loss + w * linear_binary_div

            loss = mel_loss + linear_loss + done_loss

            # attention
            if hparams.use_guided_attention:
                soft_mask = guided_attention(attn.size(-2), attn.size(-1),
                                             g=hparams.guided_attention_sigma)
                soft_mask = Variable(torch.from_numpy(soft_mask))
                soft_mask = soft_mask.cuda() if use_cuda else soft_mask
                attn_loss = (attn * soft_mask).mean()
                loss += attn_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, mel_outputs, linear_outputs, attn, y,
                    sorted_lengths, checkpoint_dir)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.get_trainable_parameters(), clip_thresh)
            optimizer.step()

            # Logs
            log_value("loss", float(loss.data[0]), global_step)
            log_value("done_loss", float(done_loss.data[0]), global_step)
            log_value("mel loss", float(mel_loss.data[0]), global_step)
            log_value("mel_l1_loss", float(mel_l1_loss.data[0]), global_step)
            log_value("mel_binary_div", float(mel_binary_div.data[0]), global_step)
            log_value("linear_loss", float(linear_loss.data[0]), global_step)
            log_value("linear_l1_loss", float(linear_l1_loss.data[0]), global_step)
            log_value("linear_binary_div", float(linear_binary_div.data[0]), global_step)
            if hparams.use_guided_attention:
                log_value("attn_loss", float(attn_loss.data[0]), global_step)

            if clip_thresh > 0:
                log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.data[0]

        averaged_loss = running_loss / (len(data_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint-path"]
    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(expanduser("~"), "data", "ljspeech")

    log_event_path = args["--log-event-path"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource(data_root))
    Mel = FileSourceDataset(MelSpecDataSource(data_root))
    Y = FileSourceDataset(LinearSpecDataSource(data_root))

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = build_deepvoice3(n_vocab=_frontend.n_vocab,
                             embed_dim=256,
                             mel_dim=hparams.num_mels,
                             linear_dim=hparams.num_freq,
                             r=hparams.outputs_per_step,
                             padding_idx=hparams.padding_idx,
                             dropout=hparams.dropout,
                             kernel_size=hparams.kernel_size,
                             encoder_channels=hparams.encoder_channels,
                             decoder_channels=hparams.decoder_channels,
                             converter_channels=hparams.converter_channels,
                             )

    optimizer = optim.Adam(model.get_trainable_parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    # Setup tensorboard logger
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("Los event path: {}".format(log_event_path))
    tensorboard_logger.configure(log_event_path)

    print(hparams_debug_string())

    # Train!
    try:
        train(model, data_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)
