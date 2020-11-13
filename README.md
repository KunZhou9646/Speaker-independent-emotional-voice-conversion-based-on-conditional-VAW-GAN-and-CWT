# Speaker-independent-emotional-voice-conversion-based-on-conditional-VAW-GAN-and-CWT

This is the implementation of the Interspeech 2020 paper "[Converting anyone's emotion: towards speaker-independent emotional voice conversion](https://www.researchgate.net/publication/341388058_Converting_Anyone's_Emotion_Towards_Speaker-Independent_Emotional_Voice_Conversion)". Please kindly cite our paper if you are using the codes.


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
1. **Prepare your dataset.**
```
Please follow the file structure:

training_dir: ./data/wav/training_set/*/*.wav

evaluation_dir ./data/wav/evaluation_set/*/*.wav

For example: "./data/wav/training_set/Angry/0001.wav"
```
2. **Activate your virtual enviroment.**
```bash
source activate [your env]
```
3. **Train VAW-GAN for prosody.**
```bash
./train_f0.sh
# Remember to change the source and target dir in "architecture-vawgan-vcc2016.json"
```
4. **Train VAW-GAN for spectrum.**
```bash
./train_sp.sh
# Remember to change the source and target dir in "architecture-vawgan-vcc2016.json"
```
5. **Generate the converted emotional speech.**
```bash
./convert.sh
```
**Note:** 
The codes are based on VAW-GAN Voice Conversion: https://github.com/JeremyCCHsu/vae-npvc/tree/vawgan
