# iTransformer — Air Quality Forecasting (Vietnam)

iTransformer applied to multivariate air quality time series across Vietnamese provinces.

> Paper: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://openreview.net/forum?id=JePfAI8fah) (ICLR 2024 Spotlight)

## Dataset

`dataset/air_quality/air_quality.csv` — hourly air quality measurements from 23 provincial monitoring stations across Vietnam.

**Columns:** `ts_utc`, `pm25`, `pm10`, `no2`, `o3`, `so2`, `co`, `aod`, `dust`, `uv_index`, `co2`, `aqi`, `aqi_pm25`, `aqi_pm10`, `aqi_no2`, `aqi_o3`, `aqi_so2`, `aqi_co`, `location_key`

**Variates (features='M):** 23 locations × 17 features = **391 variates**
**Variates (features='S', target='aqi'):** 23 variates (one per location)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Multivariate forecasting (tất cả features × locations)
bash ./scripts/air_quality_forecast.sh

# Hoặc chạy trực tiếp:
python run.py \
  --is_training 1 \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --model_id air_quality_M \
  --model iTransformer \
  --data air_quality \
  --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 3 \
  --enc_in 391 --dec_in 391 --c_out 391 \
  --d_model 512 --d_ff 512 \
  --batch_size 16 --learning_rate 0.0005 \
  --train_epochs 10 --itr 1

# Chỉ dự báo AQI (features=S)
python run.py \
  --is_training 1 \
  --root_path ./dataset/air_quality/ \
  --data_path air_quality.csv \
  --model_id air_quality_AQI \
  --model iTransformer \
  --data air_quality \
  --features S --target aqi \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --e_layers 3 \
  --enc_in 23 --dec_in 23 --c_out 23 \
  --d_model 512 --d_ff 512 \
  --batch_size 16 --learning_rate 0.0005 \
  --train_epochs 10 --itr 1
```

## Project Structure

```
iTransformer/
├── run.py                          # Entry point
├── requirements.txt
├── dataset/
│   └── air_quality/
│       └── air_quality.csv         # Dữ liệu chất lượng không khí
├── data_provider/
│   ├── data_factory.py             # DataLoader factory (chỉ air_quality)
│   └── data_loader.py              # Dataset_AirQuality class
├── model/
│   └── iTransformer.py             # iTransformer architecture
├── layers/
│   ├── Embed.py
│   ├── SelfAttention_Family.py
│   └── Transformer_EncDec.py
├── experiments/
│   ├── exp_basic.py
│   └── exp_long_term_forecasting.py
├── utils/
│   ├── metrics.py
│   ├── tools.py
│   └── timefeatures.py
└── scripts/
    └── air_quality_forecast.sh     # Script chạy thực nghiệm
```
