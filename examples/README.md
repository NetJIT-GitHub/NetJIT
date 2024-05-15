## NetJIT examples

Four runnable examples with NetJIT are provided: BERT, GPT-1/GPT-2, ResNet-152.

To run the examples, one has to install all the related dependencies and add NetJIT to Python Path, then launch the examples through torchrun. Some parameters of NetJIT may need to be adjusted according to the actual hardware. By default, NetJIT will report communications estimated to happen in [1,3] seconds, and the report content will be written to file `report.txt` under the corresponding directory.

Reference commands for the four examples (replace NetJIT-Path to the corresponding path):

### ResNet-152:
```bash
env PYTHONPATH=$PYTHONPATH:{NetJIT-Path} torchrun --master_addr={master_addr} --master_port={master_port} --nproc_per_node=1 --nnodes={nodes} --node_rank={rank} resnet152/main.py --backend=nccl --use_syn --batch_size=32 --num_epochs={num_epochs} --arch=resnet152
```

### GPT-1:
```bash
env PYTHONPATH=$PYTHONPATH:{NetJIT-Path} torchrun --master_addr={master_addr} --master_port={master_port} --nproc_per_node=1 --nnodes={nodes} --node_rank={rank} mingpt/gpt1.py
```

### GPT-2:
```bash
env PYTHONPATH=$PYTHONPATH:{NetJIT-Path} torchrun --master_addr={master_addr} --master_port={master_port} --nproc_per_node=1 --nnodes={nodes} --node_rank={rank} mingpt/gpt2.py
```

### BERT:

Use `BERT/scripts/data_download.sh` to download the required data first, then use `BERT/scripts/run_squad.sh` to launch the example