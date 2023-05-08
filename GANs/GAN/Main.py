from GAN import GAN
from CandleDataSet import CandleDataSet

ds = CandleDataSet(batch_size=32)
train_dataset = ds.get_data()

model = GAN()
model.train(train_dataset, 50)
