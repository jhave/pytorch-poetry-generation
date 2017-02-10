#!/bin/bash

for i in ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-49523 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-50601 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-50603 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-58102 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-58111 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-59252 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-67777 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-67778 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-84951 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-93541 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-93542 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-102127 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-102129 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-102616 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-103431 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-107721 ./logdir/train/demos/dilations1024_skipChannels4096_qc1024_dc32/model.ckpt-107723


do
	for model in $i
	do
		python generate_pf.py --cuda --checkpoint=$model
	done
done




