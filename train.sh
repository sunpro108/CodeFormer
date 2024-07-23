# python -m torch.distributed.launch --nproc_per_node=4 basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml --launcher pytorch
torchrun \
--standalone \
--nnodes=1 \
--nproc-per-node=4 \
basicsr/train.py \
--launcher pytorch \
-opt options/VQGAN_512_ds32_nearest_stage1_harmer.yml 
# -opt options/CodeFormer_stage2_harmer.yml
