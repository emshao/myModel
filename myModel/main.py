import torch
import autoencoder
import train
import test


# first focus on compression

def main(num, t=False):
    SAVE_PATH = 'C:\\Users\\Emily Shao\\Desktop\myModel\\myModel\\'
    
    if t:
        autoEncode = autoencoder.AutoEncoder()
        model = autoEncode.createModel()
        model.load_state_dict(torch.load(SAVE_PATH + 'model{num}.pt'))
        
        test.testModel(model)

    else:
        autoEncode = autoencoder.AutoEncoder() 
        model = autoEncode.createModel()
        finishedModel = train.trainModel(model) # default train fashionmnist

        torch.save(finishedModel.state_dict(), SAVE_PATH + 'model{num}.pt')

# calculate PSNR


if __name__ == "__main__":
    main(0)
    main(0, t=True)










"""
Aug 11, 2023

original transformer:
used in machine translation
- 1D input
- used in NLP (text)

therefore, when handling images, you must flatten into 1D sequence


Audio = 1D (but the sampling rate is super high, aka a super longggggggg sequence)
- another possible route: don't STFT data into image, simply put in 1D audio
- spectrogram --> visualization of STFT data

Convolution is able to handle 2D things


patchify: 600*192 sequence * 600*192 (this is like original transformer, too computationally heavy)

attention in transformers
sequence with length t, hidden size (dimension) --> input in self attention
query, key, value

scale factor = ( query * key^T ) / sqrt(dim(key))
^ creates softmax

softmax distribution * value ---> attention 


for images (h*w), first flatten into a sequence (of length h*w)

still really big ---> h*w * h*w

thus, images typically use patchify
- kind of like a kernal, take into pieces (like a small window, called a patch)
- the image is cut into these "patches"
- so the h*w length is smaller, and reduces complexity

linear map to hidden dimension

image: h*w*3
patchify: if patch size was 2x2 ---> (h/2) * (w/2) * 3 (smaller!)

using transformer --> next step is linear mapping
map the 3 to a hidden dimension (3 is a very small parameter)


after attention, there are two linear networks
- multilinear layers put together
- positional feedfoward network





VIT (vision transformer)*****************
- patchify, then flatten
- for images

patchify goal: flattened sequence is too long
- there is some information loss, like a compression
- images can handle this loss, later you can recover this
- but for audio, there is some loss of unrecoverable information





SWIN transformer (shifted window transformer) ********************
- window attention (sliding)
    - image is cut into smaller pieces
- within the window, flatten it there


-- shifted windown attention
    - you can't cut the image into smaller pieces and expect to do attention on the smaller pieces
    - would only give you local trends
    - slide the window to create overlap, helps with capturing global trends

    


Longformer transformer
- to help with the 1D sequence implementation
    - help with having sequences that are too long
    - still a type of window transformer
- cannot just swap transformers easily (modular?)
    - no need for STFT

    

    

5 encoder layers, 5 decoder layers, same structure as the cross scale paper
changed convolution to transformer
with patchify



next steps:
- compression
    - calculate PSNR
        - what is the performance on fashionmnist
        - ex. rgb dataset
    - then the model may not be good enough
        - group quantizer
        - compare performance again
    - try cross scale structure - need larger images
    - residual VQ

- classification
    - transformer
        - ViT --> used for image classification
    - classification head
        - long sequence in the beginning that tells you
            - when patchified, flatten to sequence
            - have additional token on sequence (pad)
            - only use the pad to map to the labels (classify)
    - Vision transformer article





----------------------------------------------------------------------------------------------------------------------------------


8/21
dataset = https://github.com/microsoft/DNS-Challenge
https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-2.sh 


within this folder: /hpc/group/tarokhlab/eys9/data/DNS_CHALLENGE/DNS2021 (has some folders, but not all)

put composed data into the processed folder (like this one): /hpc/group/tarokhlab/eys9/data/DNS_CHALLENGE/processed 
make another folder

recompose some datasets:
compose two: train and test

train:
3 sec audio clips
180,000 data samples

test:
10 sec audio clips
1158 data samples (without overlapping with training data)
(make sure the speaker is also different from the training data)
(note that if some test data isn't 10 seconds long, put them together)

for both:
have (given) proportional subset of each language
don't cut audio clips (aka if you already make one clip from the first three seconds, discard the rest, avoid audio cuts)



when loading data:
simply load them (1d), and compose them 
need to iterate through all the original files

output: train.pt, test.pt (saving tensor)

shape should be: 
train -- 180000, 16000*3
test --- 1158, 16000*10


also save these audio clip as actual audio clips that we can hear: .wav files
start by doing this with the test clips





source audio:
some audio is 48k sampling rate, discard these (don't use!)
make sure that the data/audio i am using is 16k sampling rate

this is how to get source audio ====> torchaudio.load function --> will return wav form (1d tensor) and sampling rate
then check sampling rate
then cut and compose wav form***
then torchaudio.save ===> saves composed wav tensor as a .wav file (save this, add index in name)

store wav form into output tensor


note: train and test data and speaker should be different!!



"""