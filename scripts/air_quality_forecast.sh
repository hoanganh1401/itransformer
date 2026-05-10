#!/bin/bash
# Script chạy iTransformer trên tập dữ liệu air_quality.csv
# 
# Cấu hình mặc định:
#   - features=M : dự báo multivariate (tất cả feature × location)
#   - enc_in=391  : 23 location × 17 features
#   - seq_len=96  : nhìn lại 96 time steps
#   - pred_len=96 : dự báo 96 time steps tiếp theo
#
# Chạy:
#   bash ./scripts/air_quality_forecast.sh

model_name=iTransformer

# --- Multivariate forecasting (M): dự báo tất cả features ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --model_id air_quality_M \
  --model $model_name \
  --data air_quality \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 391 \
  --dec_in 391 \
  --c_out 391 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --itr 1

echo "Done: M forecasting"

# --- Univariate per location (S): chỉ dự báo AQI ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --model_id air_quality_AQI \
  --model $model_name \
  --data air_quality \
  --features S \
  --target aqi \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 23 \
  --dec_in 23 \
  --c_out 23 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --itr 1

echo "Done: S (AQI) forecasting"
