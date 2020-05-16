# Speaker-independent-emotional-voice-conversion-based-on-conditional-VAW-GAN-and-CWT

This is the implementation of the paper "[Converting anyone's emotion: steps towards speaker-independent emotional voice conversion](https://www.researchgate.net/publication/341388058_Converting_Anyone's_Emotion_Towards_Speaker-Independent_Emotional_Voice_Conversion)". Please kindly cite our paper if you are using our codes.


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
