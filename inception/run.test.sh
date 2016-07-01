printf %s\\n {1..5} | xargs -n 1 -P 5 python creat_sub-cv.py 30 $1
printf %s\\n {1..5} | xargs -n 1 -P 5 python creat_sub-cv.py 25 $1
printf %s\\n {1..5} | xargs -n 1 -P 5 python creat_sub-cv.py 20 $1
