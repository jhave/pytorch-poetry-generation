###############################################################################
# Language Modeling on collected 
# book-length works of Erin Manning and Brian Massumi.
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
import random

from datetime import datetime
started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# create folder in which to put txt files of generated poems
directory = "GENERATED_SL/"+started_datestring+'/'
if not os.path.exists(directory):
	os.makedirs(directory)


#############
# TEMPERATURE
#### RANGE ##

MIN_TEMP=0.25
MAX_TEMP=1.25


parser = argparse.ArgumentParser(description='PyTorch PF Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/SENSELAB/pytorch_BE',
					help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/2017-08-22T12-35-49/model-GRU-emsize-2500-nhid_2500-nlayers_2-batch_size_20-epoch_69-loss_0.71-ppl_2.03.pt',
					help='model checkpoint to use')
parser.add_argument('--outf', type=str, default=started_datestring+'.txt',
					help='output file for generated text')
parser.add_argument('--words', type=int, default='88',
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

det = "BRERIN \n\nA Philosobot:\n\nTrained on collected book-length works of Erin Manning and Brian Massumi.\n\n+~+Library: PyTorch+~+\n\nMode: "+style+"\nEmbedding size: "+str(emsize)+"\nHidden Layers: "+str(nhid)+"\nBatch size: "+bs+"\nEpoch: "+ep+"\nLoss: "+loss+"\nPerplexity: "+ppl+"\n\nTemperature range: "+str(MIN_TEMP)+" to "+str(MAX_TEMP)

#print("\nSystem will generate "+str(args.words)+" word bursts, perpetually, until stopped.")

print("\n"+det)
print ("\n\tBRERIN     PhilosoBot\n\t   Initializing\n\tPlease be patient.\n")




while(True):

	print ("\n\n\t\t~", str(math.ceil(args.temperature*100)/100) , "~\n\n")

	# Manual sez: Set the random seed manually for reproducibility.
	# Forget reproducibility: this is philosophy.
	torch.manual_seed(randint(0,9999999999))

	if torch.cuda.is_available():
		if not args.cuda:
			print("")#("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			torch.cuda.manual_seed(args.seed)




	######### RANDOM TEMPERATURE ##############
	args.temperature = random.uniform(MIN_TEMP, MAX_TEMP)



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
			
		
		# FORMAT FOR DISPLAY
		words.strip()

		#get longest string 
		sent=words.split(".")	

		for snt in words:
			if len(snt)<240 and len(snt)>48:
				words = snt+"."
			else:
				words =max(sent, key=len)+"."


		# for display in Cathode
		words.strip()
		tw = TextWrapper()
		tw.width = 42
		words = "\t"+"\n\t".join(tw.wrap(words))


		# Tidy up 
		words = words.replace('“', '')
		words = words.replace('”', '')
		words = words.replace('"', '')
		words = words.replace('(', '')
		words = words.replace(')', '')


		# SCREEN OUTPUT
		for char in words:
			time.sleep(0.001)
			sys.stdout.write(char)

		#print("\n\n\nComplexity:", math.ceil(args.temperature*100)/100)

		words+="\n\n\n\n\n--------------------------------------------------------------------------------------------\nGenerated on : "+str(started_datestring)+"\n\nTemperature: "+str(math.ceil(args.temperature*100)/100)+"\n--------------------------------------------------------------------------------------------\n\nTech details\n-------------  \n\nInfo: http://bdp.glia.ca/\nCode: https://github.com/jhave/pytorch-poetry-generation\n\n"+det+"\n\n--------------------------------------------------------------------------------------------\n"+args.checkpoint
		outf.write(words)
		outf.close()
		#print("\n\nsaved to: "+ tfn)


		


