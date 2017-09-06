# coding: utf-8

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

import os

import editdistance

###### Parameters
parser = argparse.ArgumentParser(description='PyTorch PF Language Model')
parser.add_argument('--data', type=str, default='./data/SENSELAB/pytorch_BE/train.txt',
					help='location of the data corpus')
args = parser.parse_args()




################## INIT CORPUS #####################


corpus=open(args.data, encoding="utf-8")

lines=corpus.readlines()
print("# of sentences: ",len(lines))

sentences=[]
for l in lines:
	sentences.append(l.split("."))





#######  BOT RESPONSE ###################
def getResponse(it,sentences):


	words=''
	dist=1000

	for i in range(10):#(len(sentences)):

		# find min levenshstein
		print (it,sentences[i])
		print( editdistance.eval(it,sentences[i]))

		
	# Tidy up 
	words = words.replace('“', '')
	words = words.replace('”', '')
	words = words.replace('"', '')
	words = words.replace('(', '')
	words = words.replace(')', '')

	return (words)

	



#######  MAIN LOOP ###################

while(True):
	it = input("YOU:")
	print("BRERIN: ",getResponse(it,sentences))



