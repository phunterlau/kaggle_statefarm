from collections import defaultdict
from operator import add
cv_list = [1,2,3,4,5]
#directory, label and epoch list
label_list = [
              ('vgg',"test",(5,10,15)),
              ('inception','test',(20,25,30))
              ]
pred_dict = dict()
denominator = 0
for cv in cv_list:
    for path, label, epoch_list in label_list:
        for epoch in epoch_list:
            denominator +=1
            fi = open('./%s/submission_%s_batch_%d_epoch_%d.csv'%(path, label, cv, epoch),'r')
            header = fi.readline()
            for f in fi:
                ll = f.strip().split(',')
                img = ll[0]
                pred = map(float, ll[1:])
                if not img in pred_dict:
                    pred_dict[img]=pred
                else:
                    old_pred = pred_dict[img]
                    pred_dict[img] = map(add, old_pred, pred)
            fi.close()

#expore the mean value of predicted probability
fw_out = open('submission_vgg_5-10-15_inception_20-25-30.csv','w')
fw_out.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
for img, pred in sorted(pred_dict.iteritems()):
    #pred = ['%.6f'%(p/(len(cv_list)*len(epoch_list)*len(label_list))) for p in pred]
    pred = ['%.6f'%(p/(denominator)) for p in pred]
    fw_out.write('%s,%s\n'%(img,','.join(pred)))

fw_out.close()
