import numpy as np
import os
from grad_tts.model import GradTTS, params
import argparse
import torch
from grad_tts.text.symbols import symbols
from grad_tts.text import text_to_sequence, cmudict
from grad_tts.utils import intersperse
from scipy.io.wavfile import write
from grad_tts.model.hifi_gan.models import Generator as HiFiGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10,
                        help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None,
                        help='speaker id for multispeaker model')

    parser.add_argument('--hifigan_config', type=str, required=False, default=None,
                        help='path to a config of HiFi-GAN')
    parser.add_argument('--hifigan_checkpoint', type=str, required=True,
                        help='path to a checkpoint of HiFi-GAN')
    parser.add_argument('-o', '--outdir', type=str, required=True, help='dir path to output wav files')
    args = parser.parse_args()

    generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.eval()
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
        texts = [text for text in texts if text != '']
    cmu = cmudict.CMUDict()

    if args.hifigan_config is None:
        vocoder = HiFiGAN()
    else:
        vocoder = HiFiGAN(args.hifigan_config)
    vocoder.load_state_dict(torch.load(args.hifigan_checkpoint, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.eval()
    vocoder.remove_weight_norm()

    with torch.no_grad():
        for i, text in enumerate(texts):
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols)))[None]
            x_lengths = torch.LongTensor([x.shape[-1]])
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=None, length_scale=0.91)
            print(y_dec)
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32767).astype(np.int16)
            write(os.path.join(args.outdir, f'sample_{i}.wav'), 22050, audio)
