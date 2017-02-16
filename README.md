# Pytorch Poetry Generation
More info about project: http://bdp.glia.ca/

Adapted from: Pytorch http://pytorch.org/

# Specs
PyTorch (early-release Beta)

Anaconda3 4.3

Python 3.6

Cuda 8.0

# to TRAIN
python main_pf.py --cuda --model=LSTM --emsize=512 --nhid=320 --nlayers=8 --batch-size=32

# to GENERATE
Github/pytorch-poetry-generation/word_language_model$ python generate_pf-INFINITE.py --cuda --checkpoint='/media/jhave/429C7AC09C7AADD3/Github/PyTorch/models/2017-02-09T18-07-03/model-LSTM-emsize-2048-nhid_200-nlayers_8-batch_size_64-epoch_20-loss_6.68-ppl_792.65.pt' 

# VIDEOS of GENERATED Poetry
https://vimeo.com/channels/1201489

# Acknowlegdement
The GTX TitanX GPU used in this research was generously donated by the NVIDIA Corporation.
