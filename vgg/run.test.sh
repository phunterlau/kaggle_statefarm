printf %s\\n {1..5} | xargs -n 1 -P 5 python creat_sub-cv.py 5 $1
printf %s\\n {1..5} | xargs -n 1 -P 5 python creat_sub-cv.py 10 $1
printf %s\\n {1..5} | xargs -n 1 -P 5 python creat_sub-cv.py 15 $1
