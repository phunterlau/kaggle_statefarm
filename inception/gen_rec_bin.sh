python make_list.py --recursive 1 ./train sf1 --train_ratio 0.9
python make_list.py --recursive 1 ./train sf2 --train_ratio 0.9
python make_list.py --recursive 1 ./train sf3 --train_ratio 0.9
python make_list.py --recursive 1 ./train sf4 --train_ratio 0.9
python make_list.py --recursive 1 ./train sf5 --train_ratio 0.9
#make sure conda install opencv
#im2rec_path=/home/tairuic/Downloads/mxnet/bin/im2rec
im2rec_path=/udir/liuh/github/mxnet/bin/im2rec

$im2rec_path sf1_train.lst ./train/ sf1_train.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf1_val.lst ./train/ sf1_val.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf2_train.lst ./train/ sf2_train.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf2_val.lst ./train/ sf2_val.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf3_train.lst ./train/ sf3_train.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf3_val.lst ./train/ sf3_val.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf4_train.lst ./train/ sf4_train.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf4_val.lst ./train/ sf4_val.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf5_train.lst ./train/ sf5_train.rec resize=224 color=1 encoding='.jpg'
$im2rec_path sf5_val.lst ./train/ sf5_val.rec resize=224 color=1 encoding='.jpg'
