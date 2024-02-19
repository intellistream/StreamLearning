# the instruction and training config version
version="llm-embedder"
# the output folder
output="llm-embedder"
# the data root where you untar the data
data_root="/home/tongjun/FlagEmbedding/FlagEmbedding/llm_embedder/data/llm-embedder"

load_index=1  # set to be true
if [ "$load_index" -eq 1 ]; then
    save_index=0
else
    save_index=1
fi

torchrun --nproc_per_node=1 run_dense.py --train_data \
    llm-embedder:chat/msc/train.json \
    llm-embedder:convsearch/qrecc/train.concat.json \
    llm-embedder:lrlm/arxiv/train.json \
    llm-embedder:lrlm/books3/train.json \
    llm-embedder:lrlm/codeparrot/train.json \
    llm-embedder:qa/msmarco/train.json \
    llm-embedder:qa/nq/train.json \
    llm-embedder:tool/toolbench/train.json \
    llm-embedder:tool/toolbench/train.json \
    llm-embedder:icl/icl/train.json \
    --output_dir data/outputs/$output \
    --save_steps 1000000 \
    --num_train_epochs 1 \
    --logging_steps 100 \
    --inbatch_same_dataset epoch \
    --use_train_config \
    --gradient_checkpointing \
    --per_device_train_batch_size 100 \
    --deepspeed /home/tongjun/FlagEmbedding/FlagEmbedding/llm_embedder/data/deepspeed/stage0.json \
    --version $version \
    --learning_rate 5e-6 \
    --data_root $data_root

for model in "stream-end"
do
    torchrun --nproc_per_node 1 -m evaluation.eval_mmlu --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_popqa --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_msc --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_tool --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root --save_index $save_index --load_index $load_index
    torchrun --nproc_per_node 1 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/books3/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/arxiv/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/codeparrot/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/pg19/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_icl --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 1 -m evaluation.eval_qrecc --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
done
