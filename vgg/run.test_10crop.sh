printf %s\\n {1..5} | xargs -n 1 -P 5 python make_sub_10crop.py 5 $1
printf %s\\n {1..5} | xargs -n 1 -P 5 python make_sub_10crop.py 10 $1
printf %s\\n {1..5} | xargs -n 1 -P 5 python make_sub_10crop.py 15 $1
