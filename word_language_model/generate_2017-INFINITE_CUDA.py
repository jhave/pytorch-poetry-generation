###############################################################################
# Language Modeling on Poetry Foundation Corpus
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import time
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import os

from random import randint
from datetime import datetime
started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# create folder in which to put txt files of generated poems
directory = "GENERATED/"+started_datestring+'/'
if not os.path.exists(directory):
    os.makedirs(directory)


parser = argparse.ArgumentParser(description='PyTorch PF Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/2017',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
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

det = "PyTorch Poetry Language Model.\nTrained on over 600,000 lines of poetry\n\nCORPUS derived from:\nPoetry Foundation\nJacket2\nCapa\nEvergreen Review\nShampoo\n\nMode: "+style+"\nEmbedding size: "+str(emsize)+"\nHidden Layers: "+str(nhid)+"\nBatch size: "+bs+"\nEpoch: "+ep+"\nLoss: "+loss+"\nPerplexity: "+ppl

print("\nSystem will generate poems of "+str(args.words)+" words each, perpetually, until stopped.")
#print("Using model: "+str(args.checkpoint))

print("\n"+det)
print ("\nInitializing.\nPlease be patient.\n\n")







while(True):

    # print("****************SLEEPING***************")
    print ("\n\n\n\t\t~ + ~")
    # time.sleep(3)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(randint(0,9999999999))
    if torch.cuda.is_available():
        if not args.cuda:
            print("")#("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        #model = torch.load(f)
        #print("YO!")
        #model=torch.load(f, map_location=lambda storage, location: 'cpu')
        model=torch.load(f, map_location={'cuda:0': 'cpu'})

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
            word = corpus.dictionary.idx2word[word_idx]

            if word == '<eos>':
                word = '\n'

            if word == '&amp;':
                word = '\n'

            print("\t\t\t\t\t\t\tGenerating:",i,"/",args.words, end="\r")
                
            words+=word+" "
                
            #outf.write(word + ('\n' if i % 20 == 19 else ' '))

            #if i % args.log_interval == 0:
            #print('Generated {}/{} words'.format(i+1, args.words), end='\r')
        titl = "\n"+words.split('\n', 1)[0].upper()

        if not titl:
            words = "\n"+"\n".join(words.splitlines()[1:])
        else:
            words = titl.title()+"\n\n"+"\n".join(words.splitlines()[1:])
        
        # SCREEN OUTPUT
        print("\t\t\t\t\t\t\t                                              ", end="\r")
        for char in words:
            time.sleep(0.001)
            sys.stdout.write(char)

        words+="\n\n\n\n\n--------------------------------------------------------------------------------------------\nGenerated on : "+str(started_datestring)+"\n--------------------------------------------------------------------------------------------\n\nTech details\n-------------  \n\nInfo: http://bdp.glia.ca/\nCode: https://github.com/jhave/pytorch-poetry-generation\n\n"+det+"\n\n--------------------------------------------------------------------------------------------\n"+args.checkpoint
        outf.write(words)
        outf.close()
        #print("\n\nsaved to: "+ tfn)



        


