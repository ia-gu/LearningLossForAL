SEED=(42 1000 2000 3000)
for i in ${SEED[@]}; do
    echo $i
    python main.py --seed $i
done