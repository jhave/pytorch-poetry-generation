###############################################################################
# Language Modeling on Poetry Foundation Corpus
#
# This file generates new sentences sampled from the language model
#
###############################################################################
from textwrap import TextWrapper

import random
import math

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

det = "PyTorch Poetry Language Model.\nTrained on approx 600,000 lines of poetry\n\n+~+\n\nCORPUS derived from:\n\nPoetry Foundation\nPoets.org\nJacket2\nCapa\nEvergreen Review\nShampoo\n\n+~+\n\nMode: "+style+"\nEmbedding size: "+str(emsize)+"\nHidden Layers: "+str(nhid)+"\nBatch size: "+bs+"\nEpoch: "+ep+"\nLoss: "+loss+"\nPerplexity: "+ppl

print("\nSystem will generate poems of "+str(args.words)+" words each, perpetually, until stopped.")

print("\n"+det)
print ("\nInitializing INCREMENTAL TEMPERATURE RANGE.\nPlease be patient.\n\n")



MIN_temp=0.25
MAX_temp=1.25
temp=0.3
CNT=0



while(True):

    # print("****************SLEEPING***************")
    print ("\n\n\n\t\t~ + ~")
    # time.sleep(3)

    # Manual sez: Set the random seed manually for reproducibility.
    # Forget reproducibility: this is poetry.
    torch.manual_seed(randint(0,99999999))



    if torch.cuda.is_available():
        if not args.cuda:
            print("")#("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


    if temp>MAX_temp:
        temp=MIN_temp

    temp=temp+0.05
    args.temperature = temp#random.uniform(0.3, 1.5)
    

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)

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
            print('                       {}/{} words'.format(i+1, args.words), end='\r')


        titl = words.split('\n', 1)[0].title()

        # #erase the output '88/88 words' line
        print('                                                                                                                       ', end='\r')


        if not titl:
            words = "\n\t"+"\n\t".join(words.splitlines()[1:])
        else:
            words = "\n\t"+titl+"\n\n"+"\n".join(words.splitlines()[1:])

        # tw = TextWrapper()
        # tw.width = 70
        # words = "\n\t".join(tw.wrap(words))

        maxl=55
        words2=""
        for li in words.splitlines():
            
            if len(li)>maxl:
                words = "\n".join(words.splitlines()[1:])
                break
        #         nl=""
        #         nlb=True
        #         # find word then insert newline
        #         for w in li.split(" "):
        #             if len(nl+w)<maxl:
        #                 nl =nl+" "+w     
        #             elif nlb:
        #                 nl = nl+"\n\t\t"+w
        #                 nlb=False
        #             else:
        #                 nl =nl+w 

        #         words2=words2+"\n\t\t"+nl
        #     else:
        #         words2=words2+"\n\t"+li

        # words=words2

                # for display in Cathode
        # tw = TextWrapper()
        # tw.width = 55
        # words = "\n\t".join(tw.wrap(words))

        # words = " ".join(words.split(".")[:-1])+ "."

        #NO title                  words = "\n\t"+"\n\t".join(words.splitlines()[1:])
        
        # SCREEN OUTPUT
        CNT=CNT+1
        print("\n\n\n\n# "+str(CNT),"[", math.ceil(args.temperature*100)/100,"\n\n")

        for char in words:
            time.sleep(0.001)
            sys.stdout.write(char)



        words+="\n\n\n\n\n--------------------------------------------------------------------------------------------\nGenerated on : "+str(started_datestring)+"\n--------------------------------------------------------------------------------------------\n\nTech details\n-------------  \n\nInfo: http://bdp.glia.ca/\nCode: https://github.com/jhave/pytorch-poetry-generation\n\n"+det+"\n\n--------------------------------------------------------------------------------------------\n"+args.checkpoint
        outf.write(words)
        outf.close()
        #print("\n\nsaved to: "+ tfn)



        


