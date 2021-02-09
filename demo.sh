#!/bin/bash
data_dir="./EURLex-4K"
train_X_path="$data_dir/trn_fea.txt"
train_lbl_path="$data_dir/trn_lbl.txt"
test_lbl_path="$data_dir/tst_lbl.txt"
save_trn_lbl="$data_dir/trn_lbl_d.txt"
save_tst_lbl="$data_dir/tst_lbl_d.txt"
save_cut_idx="$data_dir/cut_idx.txt"
tail_bound="10"
cut_rate="0.1"

python3 NNPP_cut.py $train_X_path $train_lbl_path $test_lbl_path $save_trn_lbl $save_tst_lbl $save_cut_idx $tail_bound $cut_rate