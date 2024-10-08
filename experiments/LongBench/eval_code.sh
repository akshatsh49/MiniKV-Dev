# python3 ./eval.py --model llama2-7b-chat-4k

# list directories in pred/
for dir in $(ls pred_e/); do
    echo "Processing $dir"
    python3 ./eval.py --model $dir --e 1>/dev/null 2>&1 & 
done
