# python3 ./eval.py --model llama2-7b-chat-4k

# list directories in pred/
for dir in $(find pred_e/*use_snapFalse*kbits2*_pool* -type d); do
    echo "Processing $dir"
    dir=${dir#pred_e/}
    python3 ./eval.py --model $dir --e 1>/dev/null 2>&1 & 
done

# also directories under pred_e/pyramid/
# for dir in $(ls pred_e/pyramid/); do
#     dir="pyramid/${dir}"
#     echo "Processing $dir"
#     python3 ./eval.py --model $dir --e 1>/dev/null 2>&1 & 
# done
