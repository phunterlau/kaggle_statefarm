for batch in 1 2 3 4 5
do
    python train_inception_bn.py --train-dataset sf${batch}_train.rec --val-dataset sf${batch}_val.rec --model-prefix ./model/ckpt-shuffle${batch} --gpus 0
    #save the mean image
    cp mean.bin mean-rand-shuffle${batch}.bin
done
