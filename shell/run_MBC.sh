dataset=$1

[ -z "${dataset}" ] && dataset="MBC_s1"
# [ -z "${dataset}" ] && dataset="MBC_s2"

python MBC.py \
	--device 0 \
	--dataset $dataset \
	--mask_method "random" \
	--remask_method "random" \
	--mask_rate 0.3 \
	--in_drop 0.1 \
	--attn_drop 0.1 \
	--num_layers 1 \
	--num_dec_layers 1 \
	--num_hidden 1024 \
	--num_heads 8 \
	--num_out_heads 1 \
	--encoder "gat" \
	--decoder "gat" \
	--max_epoch 1000 \
	--lr 0.001 \
	--weight_decay 0.04 \
	--activation "prelu" \
	--loss_fn "sce" \
	--alpha_l 3 \
	--scheduler \
	--seeds 15 \
	--lam 0.5 \
	--use_cfg
