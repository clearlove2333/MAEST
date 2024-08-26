dataset=$1

[ -z "${dataset}" ] && dataset="DLPFC"
# [ -z "${dataset}" ] && dataset="MBA"
# [ -z "${dataset}" ] && dataset="HBC"

#	--sample "151507" "151508" "151509" "151510" "151669" "151670" "151671" "151672" "151673" "151674" "151675" "151676" \

	# --sample "151507" "151671" "151673" \
	# --sample "151671" \

# python DLPFC.py \

python DLPFC.py \
	--device 0 \
	--dataset $dataset \
	--sample "151671" \
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
	--seeds 41 \
	--lam 0.5 \
	--use_cfg
