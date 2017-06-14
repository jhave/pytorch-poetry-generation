# PyTorch-for-Poetry-Generation 
is a repurposing of [http://pytorch.org/](Pytorch): an early release beta software (developed by a consortium led by Facebook and NIVIDIA), a “deep learning software that puts Python first.”

So since I luckily received an NVIDIA GTX TitanX (Maxwell) before leaving Hong Kong under the generous NVIDIA academic GPU Grant program, and having in late January 2017 finally bought a custom-build to house it, and on 06-04-2017 finally got Ubuntu installed with CUDA and CUDNN drivers, and having found that the Tensorflow 0.11 version no longer runs under Python 3.6 Anaconda, I decided to give a PyTorch example a try, specifically Word-level language modeling RNN


# Preliminary results
[More here](http://bdp.glia.ca/pytorch-prelim/)
This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task…The trained model can then be used by the generate script to generate new text.

And after only an hour of training on an 11k poem corpus, using the default settings, the results announced “End of training | test loss  5.99 | test ppl   398.41” — Which means that the loss is bad and perplexity is now at the seemingly terrible level of 398….

Then I ran the generate script and the 1000 word text below got generated in less than 30 seconds. I find it stunning. If this is what PyTorch is capable of with a tiny corpus, default settings and a minimal run, language generation is entering a renaissance.  Ok, so it’s veering toward the incomprehensible and has little lived evocative phenomenological resonance, but its grasp on idiomatic cadence is creepily accurate. It’s as if it absorbed several semesters of graduate seminars on romantic and post-modern verse:

	`the embankment
	and your face sad like a nest, grew sorry
	when your cold work made of snow
	broken
	and left a thousand magnifies.

	a little cold, you plant but hold it
	and seems
	the slight arts? face, and ends
	with such prayer as the fingers do,
	this reorganizing contest is how
	to be murdered
	throwing it
	into the arteries obscurity goes disc whispering whole
	affairs, now your instinct
	does a case,
	defense. on her eye, you do not know that every homelands
	is didn’t at the
	risk very psychiatrists, just under bay.

	by the living of life’s melancholy grate.
	i have found a
	wild orange in eden, eight hazy years guzzles
	her neck at the grave turn into every mythological orbit of
	distances,
	person’s there–see then are we told what we understand
	won’t take the slightest danger
	or the
	size of what it means to take up if you can,
	tongue. only your eye exultant whitens again will
	happen.
	i think that the four-oared clouded of one stick in flowerpot
	is part of an antique little
	register on a hiatus
	till i try for you.
	i wash up the door my knee will be
	high.
	if i refuse a limits as i can lift my hand rubicon.

	i can see her
	above the stove tide
	hip. orange as a breaking sty.`

# Pytorch Poetry Generation
More info about project: http://bdp.glia.ca/

Adapted from: Pytorch http://pytorch.org/

# Specs
PyTorch (early-release Beta)
Anaconda3 4.3
Python 3.6
Cuda 8.0

# to TRAIN
`python main_pf.py --cuda --model=LSTM --emsize=512 --nhid=320 --nlayers=8 --batch-size=32`

On April 2nd Pytorch updated their source to include the parameter

`—tied             tie the word embedding and softmax weights`

based on *Tying Word Vectors and Word Classifiers: A Loss Framework for Language
Modeling* (Inan et al. 2016) ... Number of hidden layers and embedding size must now match. Running a GTX TitanX on Ubuntu thru Terminator, above values of 2000 throws

`cuda runtime error(2): out of memory`

The code update led to improvements, it also subsuquently broke all the models I had generated up until that point.


# to GENERATE
`python generate_pf-INFINITE.py --cuda --checkpoint='/media/jhave/429C7AC09C7AADD3/Github/PyTorch/models/2017-02-09T18-07-03/model-LSTM-emsize-2048-nhid_200-nlayers_8-batch_size_64-epoch_20-loss_6.68-ppl_792.65.pt'` 

# VIDEOS of GENERATED Poetry
[https://vimeo.com/channels/1201489](https://vimeo.com/channels/1201489)

# Acknowlegdement
The GTX TitanX GPU used in this research was generously donated by the NVIDIA Corporation.
