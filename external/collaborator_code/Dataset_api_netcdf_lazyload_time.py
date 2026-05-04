import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch
import xarray as xr
import pandas as pd


STATS_CONFIG = {
    'Pred': {
        'input_variables': [
            {"name": "windspeed_lowatmmodlay", "mean": 5.512597, "std": 2.9025302, "include": False},
            {"name": "precipitation_total_surface_flux", "mean": 1.256179e-05, "std": 0.00017290862, "include": False},
            {"name": "temp_lowatmmodlay", "mean": 285, "std": 10, "include": False},
            {"name": "specific_humidity_lowatmmodlay", "mean": 0.004552208, "std": 0.0028845386, "include": False},
            {"name": "surface_pressure", "mean": 79716.62, "std": 4729.2383, "include": False},
            {"name": "radiation_shortwave_downward_flux", "mean": 235.67114, "std": 303.05682, "include": False},
            {"name": "radiation_longwave_absorbed_flux", "mean": 264.5978, "std": 51.637115, "include": False},
            {"name": "sm_surface_forecast", "mean": 0.13334474, "std": 0.06109628, "include": True},
            {"name": "sm_rootzone_forecast", "mean": 0.18498646, "std": 0.056409765, "include": True},
            {"name": "soil_temp_layer1_forecast", "mean": 284.4739, "std": 9.821837, "include": True},
            {"name": "surface_temp_forecast", "mean": 284.2508, "std": 12.2772665, "include": True},
            {"name": "mask", "mean": 0, "std": 1, "include": False}
        ],
        'target_variables': [
            {"name": "sm_surface_forecast", "mean": 0.13334474, "std": 0.06109628, "include": True},
            {"name": "sm_rootzone_forecast", "mean": 0.18498646, "std": 0.056409765, "include": True},
            {"name": "soil_temp_layer1_forecast", "mean": 285, "std": 10, "include": True},
            {"name": "surface_temp_forecast", "mean": 285, "std": 10, "include": True},
            {"name": "heat_flux_latent", "mean": 31.187826, "std": 41.93185, "include": False},
            {"name": "heat_flux_sensible", "mean": 66.21313, "std": 132.90985, "include": False}
        ]
    },
    'DA': {
        'input_variables': [
            {"name": "sm_surface_forecast", "mean": 0.13334474, "std": 0.06109628, "include": True},
            {"name": "sm_rootzone_forecast", "mean": 0.18498646, "std": 0.056409765, "include": True},
            {"name": "soil_temp_layer1_forecast", "mean": 285, "std": 10, "include": True},
            {"name": "surface_temp_forecast", "mean": 284.2508, "std": 12.2772665, "include": True},
            {"name": "mwrtm_vegopacity", "mean": 0.1, "std": 0.1, "include": False},
            {"name": "tb_h_obs", "mean": 253.6875, "std": 15.8172, "include": True},
            {"name": "tb_v_obs", "mean": 276.8860, "std": 11.6384, "include": True},
            {"name": "tb_h_obs_errstd", "mean": 0.0, "std": 0.0, "include": False},
            {"name": "tb_v_obs_errstd", "mean": 0.0, "std": 0.0, "include": False},
            {"name": "tb_h_obs_assim", "mean": 0.0, "std": 0.0, "include": False},
            {"name": "tb_v_obs_assim", "mean": 0.0, "std": 0.0, "include": False},
            {"name": "mask", "mean": 0, "std": 1, "include": False}
        ],
        'target_variables': [
            {"name": "sm_surface_analysis", "mean": 0.13334474, "std": 0.06109628, "include": True},
            {"name": "sm_rootzone_analysis", "mean": 0.18498646, "std": 0.056409765, "include": True},
            {"name": "soil_temp_layer1_analysis", "mean": 285, "std": 10, "include": True},
            {"name": "surface_temp_analysis", "mean": 285, "std": 10, "include": True}
        ]
    }
}

class WeatherDataset(Dataset):
    def __init__(self, input_dir, start_time, end_time, stats_json_path):
        """
        自定义数据集，用于加载 Input_data 和 Target_data 文件对，根据 JSON 文件选择变量并进行归一化。

        Args:
            input_dir (str): Input_data 文件夹路径
            target_dir (str): Target_data 文件夹路径
            start_time (str): 起始时间，格式为 'YYYYMMDDTHHMMSS'（如 '20150401T000000'）
            end_time (str): 终止时间，格式为 'YYYYMMDDTHHMMSS'（如 '20150430T235959'）
            stats_json_path (str): 包含变量名、mean、std 和 include 的 JSON 文件路径
            transform (callable, optional): 额外的变换
        """
        self.input_dir = input_dir

        # 加载 JSON 文件中的变量名、均值、标准差和 include 字段
        with open(stats_json_path, 'r') as f:
            stats = json.load(f)

        # 验证 JSON 数据
        # if not all(key in stats for key in ['input_variables', 'target_variables']):
        #     raise ValueError("JSON 文件必须包含 'input_variables' 和 'target_variables' 键")
        # if len(stats['input_variables']) != 12:
        #     raise ValueError("input_variables 必须包含 12 个变量")
        # if len(stats['target_variables']) != 6:
        #     raise ValueError("target_variables 必须包含 6 个变量")
        # for var in stats['input_variables'] + stats['target_variables']:
        #     if not all(key in var for key in ['name', 'mean', 'std', 'include']):
        #         raise ValueError("每个变量必须包含 'name', 'mean', 'std', 'include' 键")

        # 提取选中的变量（include: true）
        self.input_vars = [var for var in stats['input_variables'] if var['include']]
        self.target_vars = [var for var in stats['target_variables'] if var['include']]

        # 验证选中的变量数量（可选，根据你的需求）
        if len(self.input_vars) == 0 or len(self.target_vars) == 0:
            raise ValueError("至少需要选择一个输入变量和一个输出变量")

        # 提取均值和标准差（仅针对选中的变量）
        self.input_mean = torch.tensor([var['mean'] for var in self.input_vars]).float().view(-1, 1, 1)  # 形状 [n, 1, 1]
        self.input_std = torch.tensor([var['std'] for var in self.input_vars]).float().view(-1, 1, 1)  # 形状 [n, 1, 1]
        self.target_mean = torch.tensor([var['mean'] for var in self.target_vars]).float().view(-1, 1, 1)  # 形状 [m, 1, 1]
        self.target_std = torch.tensor([var['std'] for var in self.target_vars]).float().view(-1, 1, 1)  # 形状 [m, 1, 1]

        # 保存变量名和索引（用于调试或验证）
        self.input_var_names = [var['name'] for var in self.input_vars]
        self.target_var_names = [var['name'] for var in self.target_vars]
        self.input_indices = [i for i, var in enumerate(stats['input_variables']) if var['include']]
        self.target_indices = [i for i, var in enumerate(stats['target_variables']) if var['include']]

        # 将时间字符串转换为 datetime 对象
        self.start_time = datetime.strptime(start_time, '%Y%m%dT%H%M%S')
        self.end_time = datetime.strptime(end_time, '%Y%m%dT%H%M%S')

        # 获取所有符合时间范围的文件对
        # self.input_data, self.target_data, self.year, self.month, self.day, self.times = self._load_netcdf()
        self.ds = self._lazy_load()

    def _lazy_load(self):
        df = xr.open_dataset(self.input_dir)
        df = df.sel(time = slice(self.start_time, self.end_time))
        return df
    def _load_netcdf(self):

        df = xr.open_dataset(self.input_dir)
        df = df.sel(time = slice(self.start_time, self.end_time))
        input = df['input'].values  # 形状 [time, V_input, H, W]
        target = df['target'].values
        time = df['time'].values  # 形状 [time]
        year = [t.astype('datetime64[ms]').astype(np.datetime64).item().year for t in time]
        month = [t.astype('datetime64[ms]').astype(np.datetime64).item().month for t in time]
        day = [t.astype('datetime64[ms]').astype(np.datetime64).item().day for t in time]
        hour = [t.astype('datetime64[ms]').astype(np.datetime64).item().hour for t in time]

        input = torch.from_numpy(input).float()  # 转换为 PyTorch 张量
        target = torch.from_numpy(target).float()

        input = input[:, self.input_indices, :, :]  # 选择指定的输入变量
        target = target[:, self.target_indices, :, :]

        return input, target, year, month, day, hour


    def __len__(self):
        return self.ds.time.shape[0]
    def __getitem__(self, idx):
        """
        获取索引为 idx 的数据对，选择指定变量并进行归一化。
        """
        input_data = self.ds['input'].isel(time=idx).values[self.input_indices]
        target_data = self.ds['target'].isel(time=idx).values[self.target_indices]

        # ---- 提取时间 ----
        t = self.ds['time'].values[idx].astype('datetime64[ms]').astype(np.datetime64).item()
        year = t.year
        month = t.month
        day = t.day
        hour = t.hour

        # 时间编码 sin/cos 加入月时间编码
        angle = 2 * np.pi * month / 12.
        time_code = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)  # shape=(2,)

        # ---- 转为 Tensor ----
        input_data = torch.from_numpy(input_data).float()
        target_data = torch.from_numpy(target_data).float()
        time_code = torch.from_numpy(time_code).float()

        # ---- 归一化 ----
        input_data = (input_data - self.input_mean) / self.input_std
        target_data = (target_data - self.target_mean) / self.target_std

        return {
            'input': input_data,         # [C, H, W]
            'target': target_data,       # [C_out, H, W]
            'time_code': time_code,      # [2]   ★★★ 新增的季节编码
            'year': year,
            'month': month,
            'day': day,
            'time': hour
        }

if __name__ == '__main__':

    input_dir = '/home/liuzixuan/Data2/Train_data_America/all/Pred.nc'
    stats_json_path = '/home/liuzixuan/PythonProjects/SS4-Phy-git/Configs/Config_Vars_sm.json'
    start_time_str = f'20200101T000000'
    start_time = pd.to_datetime(start_time_str) - pd.DateOffset(hours=3)
    end_time = start_time + pd.DateOffset(days=90)- pd.DateOffset(hours=3)

    dataset = WeatherDataset(input_dir, start_time = start_time.strftime('%Y%m%dT%H%M%S'), end_time = end_time.strftime('%Y%m%dT%H%M%S'), stats_json_path = stats_json_path)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = False, num_workers = 1, pin_memory = True)
    for batch in dataloader:
        print(batch['time'])
        print(batch['input'].shape)
        print(batch['target'].shape)
        print(batch['time_code'].shape)
        print(batch['time_code'][0,0])
        print(batch['time_code'][0,1])
        
        print(batch['time_code'][10,0])
        print(batch['time_code'][10,1])
        print(batch['month'][10])

        break  # 仅打印第一批次
