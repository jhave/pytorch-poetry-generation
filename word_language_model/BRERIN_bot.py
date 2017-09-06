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
MAX_TEMP=1.5


parser = argparse.ArgumentParser(description='PyTorch PF Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/SENSELAB/pytorch_BE',
					help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/2017-08-22T12-35-49/model-GRU-emsize-2500-nhid_2500-nlayers_2-batch_size_20-epoch_69-loss_0.71-ppl_2.03.pt',
					help='model checkpoint to use')
parser.add_argument('--outf', type=str, default=started_datestring+'.txt',
					help='output file for generated text')
parser.add_argument('--words', type=int, default='66',
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

# det = "BRERIN \n\nA Philosobot:\n\nTrained on the collected book-length works of Erin Manning and Brian Massumi.\n\n+~+Library: PyTorch+~+\n\nMode: "+style+"\nEmbedding size: "+str(emsize)+"\nHidden Layers: "+str(nhid)+"\nBatch size: "+bs+"\nEpoch: "+ep+"\nLoss: "+loss+"\nPerplexity: "#+ppl+"\n\nTemperature range: "+str(MIN_TEMP)+" to "+str(MAX_TEMP)

# #print("\nSystem will generate "+str(args.words)+" word bursts, perpetually, until stopped.")

# print("\n"+det)
# print ("\nAsk a question. \nBegin a discussion. \nType something.\n\n")




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




################## INIT MODEL & CORPUS #####################

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
print("hidden",hidden)
inputs = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
print("inputs",inputs)
if args.cuda:	
	inputs.data = inputs.data.cuda()





#######  BOT RESPONSE ###################
def getResponse(it,model,args,inputs,hidden,corpus):

	if len(it)>20:
		args.words = (len(it))
	else:
		args.words = 20

	args.temperature = random.uniform(MIN_TEMP, MAX_TEMP)
	
	words=''

	for i in range(args.words):
		output, hidden = model(inputs, hidden)
		word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
		word_idx = torch.multinomial(word_weights, 1)[0]
		inputs.data.fill_(word_idx)

		if word_idx<=len(corpus.dictionary.idx2word)-1:
			word = corpus.dictionary.idx2word[word_idx]

			if word == '<eos>':
				word = '\n'

			if word == '&amp;':
				word = '\n'


				
			words+=word+" "
		
		# GENERATING output how many created so far
		print('                                            {}/{}'.format(i+1, args.words), end='\r')


	print('                                                                                                                       ', end='\r')


	# for display in Cathode
	tw = TextWrapper()
	tw.width = 64
	words = "\t\n".join(tw.wrap(words))

	words = ".".join(words.split(".")[:-1])+ "."

	# Tidy up 
	words = words.replace('“', '')
	words = words.replace('”', '')
	words = words.replace('"', '')
	words = words.replace('(', '')
	words = words.replace(')', '')



	words = capitalize(words)


	#print("\n\n\nComplexity:", math.ceil(args.temperature*100)/100)
	# with open(tfn, 'a') as outf:
	# 		outf.write(words,a)

	return (words)

	



#######  MAIN LOOP ###################

while(True):
	it = input("YOU:")
	print("BRERIN: ",getResponse(it,model,args,inputs,hidden,corpus))



