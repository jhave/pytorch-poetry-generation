###############################################################################
# Language Modeling on Poetry Foundation Corpus
#
# This file generates new sentences sampled from the language model
#
###############################################################################
from textwrap import TextWrapper


import argparse
import time
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import os

import re

from random import randint
from datetime import datetime
started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# create folder in which to put txt files of generated poems
directory = "GENERATED_SL/"+started_datestring+'/'
if not os.path.exists(directory):
	os.makedirs(directory)


parser = argparse.ArgumentParser(description='PyTorch PF Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/SENSELAB/pytorch_BE',
					help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/2017-08-22T12-35-49/model-GRU-emsize-2500-nhid_2500-nlayers_2-batch_size_20-epoch_69-loss_0.71-ppl_2.03.pt',
					help='model checkpoint to use')
parser.add_argument('--outf', type=str, default=started_datestring+'.txt',
					help='output file for generated text')
parser.add_argument('--words', type=int, default='70',
					help='number of words to generate')
parser.add_argument('--seed', type=int, default=randint(0,99999999),
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
					help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
					help='reporting interval')
args = parser.parse_args()


# GET TECH DETAILS
md=args.checkpoint.split("/")[-1]
style = md.split("-")[1]
emsize= md.split("-")[3]

nhid= md.split("-")[4].split("_")[1]
nlay= md.split("-")[5].split("_")[1]
bs = md.split("-")[6].split("_")[2]
ep= md.split("-")[7].split("_")[1]
loss= md.split("-")[8].split("_")[1]
ppl= md.split("-")[9].split("_")[1]

det = "BRERIN \n\nA Philosobot:\nTrained on the collected book-length works of Erin Manning and Brian Massumi\n\n+~+Library: PyTorch+~+\n\nMode: "+style+"\nEmbedding size: "+str(emsize)+"\nHidden Layers: "+str(nhid)+"\nBatch size: "+bs+"\nEpoch: "+ep+"\nLoss: "+loss+"\nPerplexity: "+ppl

print("\nSystem will generate "+str(args.words)+" word bursts, perpetually, until stopped.")

print("\n"+det)
print ("\nInitializing.\nPlease be patient.\n\n")




##############################
#  Formatting stuff for laptop

def insertNewlines(text, lineLength):
	if len(text) <= lineLength:
		return text
	elif text[lineLength] != ' ':
		return insertNewlines(text[:], lineLength+1)
	else:
		return text[:lineLength] + '\n' + insertNewlines(text[lineLength + 1:], lineLength)

def uppercase(matchobj):
	return matchobj.group(0).upper()

def capitalize(s):
	return re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)', uppercase, s)






while(True):

	# print("****************SLEEPING***************")
	print ("\n\n\n\t\t~ + ~\n")
	# time.sleep(3)

	# Manual sez: Set the random seed manually for reproducibility.
	# Forget reproducibility: this is philosophy.
	torch.manual_seed(randint(0,9999999999))

	#time.sleep(0.25)

	if torch.cuda.is_available():
		if not args.cuda:
			print("")#("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			torch.cuda.manual_seed(args.seed)

	if args.temperature < 1e-3:
		parser.error("--temperature has to be greater or equal 1e-3")

	with open(args.checkpoint, 'rb') as f:
		#
		# GPU
		#model = torch.load(f)
		#
		# changed TO MAKE WORK ON CPU
		model = torch.load(f, map_location=lambda storage, loc: storage)

	if args.cuda:
		model.cuda()
	else:
		model.cpu()

	corpus = data.Corpus(args.data)
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(1)
	input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
	if args.cuda:
		input.data = input.data.cuda()

	
	words=''

	now ="{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
	tfn =directory+"/"+now+"_"+args.checkpoint.split("model-")[1]+".txt"

	with open(tfn, 'w') as outf:

		for i in range(args.words):
			output, hidden = model(input, hidden)
			word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
			word_idx = torch.multinomial(word_weights, 1)[0]
			input.data.fill_(word_idx)

			if word_idx<=len(corpus.dictionary.idx2word)-1:
				word = corpus.dictionary.idx2word[word_idx]

				if word == '<eos>':
					word = '\n'

				if word == '&amp;':
					word = '\n'


					
				words+=word+" "
			
			# Output how many created so far
			print('                                            {}/{}'.format(i+1, args.words), end='\r')


		#titl = words.split('\n', 1)[0].title()

		#erase the output '88/88 words' line
		print('                                                                                                                       ', end='\r')


		# if not titl:
		#     words = "\n"+"\n".join(words.splitlines()[1:])
		# else:
		#     words = "\n"+titl+"\n\n"+"\n".join(words.splitlines()[1:])

		#words = "\n"+"\n".join(words.splitlines()[1:])
		#words = insertNewlines(words, 48)

		# for display in Cathode
		tw = TextWrapper()
		tw.width = 64
		words = "\n".join(tw.wrap(words))

		words = ".".join(words.split(".")[:-1])+ "."

		# Tidy up 
		words = words.replace('“', '')
		words = words.replace('”', '')
		words = words.replace('"', '')
		words = words.replace('(', '')
		words = words.replace(')', '')



		words = capitalize(words)

		# SCREEN OUTPUT
		for char in words:
			time.sleep(0.001)
			sys.stdout.write(char)

		words+="\n\n\n\n\n--------------------------------------------------------------------------------------------\nGenerated on : "+str(started_datestring)+"\n--------------------------------------------------------------------------------------------\n\nTech details\n-------------  \n\nInfo: http://bdp.glia.ca/\nCode: https://github.com/jhave/pytorch-poetry-generation\n\n"+det+"\n\n--------------------------------------------------------------------------------------------\n"+args.checkpoint
		outf.write(words)
		outf.close()
		#print("\n\nsaved to: "+ tfn)


		


