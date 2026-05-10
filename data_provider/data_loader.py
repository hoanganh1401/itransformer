"""
Dataset_AirQuality
------------------
Đọc file air_quality.csv (tab-separated) có cấu trúc:
    ts_utc | pm25 | pm10 | ... | aqi_co | location_key

Mỗi dòng là một quan sát tại một trạm (location_key) tại một thời điểm.
Dataset pivot theo location_key để tạo ma trận multivariate time series:
  - index  = timestamp (ts_utc)
  - columns= (feature, location) hoặc chỉ (target, location) tùy chế độ

Ghi chú về kích thước dữ liệu:
  Nếu dữ liệu thực tế của bạn có nhiều timestamp (ví dụ dữ liệu hourly nhiều tháng),
  dataset sẽ tự động dùng toàn bộ. Script chạy mặc định dùng seq_len=96, pred_len=96
  → cần ít nhất ~200 time steps để train/val/test có ý nghĩa.
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

AIR_QUALITY_FEATURE_COLS = [
    'pm25', 'pm10', 'no2', 'o3', 'so2', 'co',
    'aod', 'dust', 'uv_index', 'co2',
    'aqi', 'aqi_pm25', 'aqi_pm10', 'aqi_no2', 'aqi_o3', 'aqi_so2', 'aqi_co'
]


class Dataset_AirQuality(Dataset):
    """
    Dataset cho air_quality.csv.

    features='M'  → input và output là tất cả (feature × location)
    features='S'  → input và output chỉ là `target` × location
    features='MS' → input là tất cả features, output chỉ là `target`

    Split: 70% train / 10% val / 20% test (theo thứ tự thời gian)
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='air_quality.csv',
                 target='aqi', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    # ------------------------------------------------------------------
    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path),
            sep=','
        )
        df_raw = df_raw.rename(columns={'ts_utc': 'date'})
        df_raw['date'] = pd.to_datetime(df_raw['date'], utc=True).dt.tz_localize(None)

        feature_cols = [c for c in AIR_QUALITY_FEATURE_COLS if c in df_raw.columns]
        locations = sorted(df_raw['location_key'].dropna().unique())

        # --- Pivot ---
        if self.features == 'S':
            pivot = df_raw.pivot_table(
                index='date', columns='location_key',
                values=self.target, aggfunc='mean'
            )
            pivot.columns = [f"{self.target}__{loc}" for loc in pivot.columns]
        else:
            frames = []
            for loc in locations:
                sub = (df_raw[df_raw['location_key'] == loc][['date'] + feature_cols]
                       .copy().set_index('date'))
                sub.columns = [f"{col}__{loc}" for col in sub.columns]
                frames.append(sub)
            pivot = pd.concat(frames, axis=1)

        pivot = pivot.sort_index().ffill().bfill()
        self.variate_names = list(pivot.columns)
        self.n_variates = len(self.variate_names)

        n = len(pivot)
        if n < self.seq_len + self.pred_len:
            raise ValueError(
                f"Dữ liệu chỉ có {n} time step sau khi pivot, "
                f"nhưng cần ít nhất {self.seq_len + self.pred_len} "
                f"(seq_len={self.seq_len} + pred_len={self.pred_len}). "
                "Hãy cung cấp thêm dữ liệu hoặc giảm seq_len/pred_len."
            )

        num_train = int(n * 0.7)
        num_val = int(n * 0.1)

        border1s = [0,
                    max(0, num_train - self.seq_len),
                    max(0, num_train + num_val - self.seq_len)]
        border2s = [num_train, num_train + num_val, n]

        b1 = border1s[self.set_type]
        b2 = border2s[self.set_type]

        df_values = pivot.values.astype(np.float32)

        train_data = df_values[border1s[0]:border2s[0]]
        if len(train_data) == 0:
            raise ValueError(
                f"Không có dữ liệu train (dữ liệu chỉ có {n} time steps). "
                "Hãy thêm nhiều timestamp hơn vào air_quality.csv."
            )
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_values)
        else:
            data = df_values

        # --- Time stamp ---
        ts_slice = pivot.index[b1:b2]
        df_stamp = pd.DataFrame({'date': ts_slice})
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop('date', axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq
            ).transpose(1, 0)
        else:
            data_stamp = np.zeros((b2 - b1, 1))

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
