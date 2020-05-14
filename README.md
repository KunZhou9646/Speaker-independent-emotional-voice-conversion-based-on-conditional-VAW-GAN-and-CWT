# Speaker-independent-emotional-voice-conversion-based-on-conditional-VAW-GAN-and-CWT

This is the implementation of the paper "[Converting anyone's emotion: steps towards speaker-independent emotional voice conversion][paper link]". Please kindly cite our paper if you are using our codes.
[paper link]: www.

## Getting Started

### Prerequisites

- Ubuntu 16.04  
- Python 3.6 
  - Tensorflow-gpu 1.5.0
  - PyWorld
  - librosa
  - soundfile
  - numpy 1.14.0
  - sklearn
  - glob
  - sprocket-vc
  - pycwt
  - scipy
<br/>

## Usage
1. **Activate your virtual enviroment.**
```bash
source activate [your env]
```
2. **Train VAW-GAN for prosody.**
```bash
./train_f0.sh
```
3. **Train VAW-GAN for spectrum.**
```bash
./train_sp.sh
```
4. **Generate the converted emotional speech.**
```bash
./convert.sh
```
**Note:** 
The codes are based on VAW-GAN Voice Conversion: https://github.com/JeremyCCHsu/vae-npvc/tree/vawgan
