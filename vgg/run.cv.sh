for batch in 1 2 3 4 5
do
    python train.py --train-dataset sf${batch}_train.rec --val-dataset sf${batch}_val.rec --model-prefix ./model/ckpt-${batch} --finetune-lr-scale 5
    #save the mean image
    cp mean.bin mean${batch}.bin
done
