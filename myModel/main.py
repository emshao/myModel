import torch
import torch.nn as nn
import autoencoder
import train
import test

def main(test=False):
    autoEncode = autoencoder.AutoEncoder()
    model = autoEncode.createModel()
    finalized = train.trainModel(model)

    if (test):
        test.testModel(finalized)


# calculate PSNR


if __name__ == "__main__":
    main()





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
- calculate PSNR
- try cross scale structure




"""