# log GPU health in real time
clear
while true
do
    esc="\e[1A\e[K"
    clesc=""
    for i in {1..32}
    do
        clesc+=$esc
    done
    echo -e -n $clesc
    nvidia-smi
    sleep 2
done