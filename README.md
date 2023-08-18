# Grad-TTS

A fork of the [official implementation](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) of the Grad-TTS model based on Diffusion Probabilistic Modelling. This fork cleans up the code to focus on easy installation and inference.



## Installation


## Inference
See `grad_tts_cli.py` for how to use the model for inference.
```bash
python grad_tts_cli.py \
    --file /PATH/TO/TEXT_FILE \
    --checkpoint /PATH/TO/GRAD_TTS_CHECKPOINT \
    --hifigan_checkpoint /PATH/TO/HIFIGAN_CHECKPOINT \
    --outdir /PATH/TO/OUTPUT_DIR
```

You can download [Grad-TTS](https://drive.google.com/file/d/1YrlswCD2Q_IUlvFtQQ-gnfkG7FEvRoPJ/view?usp=drive_link) and [HiFi-GAN](https://drive.google.com/file/d/15AeZO2Zo4NBl7PG8oGgfQk0J1PpjaOgI/view?usp=drive_link) checkpoints trained on LJSpeech. 

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).
