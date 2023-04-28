from DDPM import DiffusionModel, beta_schedule_cos
from Networks import very_simple_unet
from CandleDataSet import CandleDataSet
from functools import partial

ds = CandleDataSet(32)
train_dataset = ds.get_data()

diffusion_steps = 400
max_beta = 0.02
beta_schedule = partial(beta_schedule_cos, max_beta=max_beta)

model = DiffusionModel(very_simple_unet, diffusion_steps=diffusion_steps, beta_schedule=beta_schedule)
model.train(train_dataset, 100)
