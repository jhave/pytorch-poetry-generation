# Pytorch Poetry Generation
More info about project: http://bdp.glia.ca/
Adaptated from: Pytorch http://pytorch.org/

PyTorch is an early release beta software (developed by a consortium led by Facebook and NIVIDIA), a “deep learning software that puts Python first.”

So since I luckily received an NVIDIA GTX TitanX (Maxwell) before leaving Hong Kong under the generous NVIDIA academic GPU Grant program, and having last week finally bought a custom-build to house it, and 2 days ago finally got Ubuntu installed with CUDA and CUDNN drivers, and having found that the Tensorflow 0.11 version no longer runs under Python 3.6 Anaconda, I decided to give a PyTorch example a try, specifically Word-level language modeling RNN

I posted results after 10 minutes here:
http://bdp.glia.ca/pytorch-prelim/

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
https://vimeo.com/203369645


#FROM PYTORCH docs: Word-level language modeling RNN
https://github.com/pytorch/examples/tree/master/word_language_model

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda  # Train an LSTM on ptb with cuda (cuDNN). Should reach perplexity of 113
python generate.py     # Generate samples from the trained LSTM model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        humber of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```
