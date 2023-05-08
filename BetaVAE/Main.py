import tensorflow as tf
from BetaVAE import BetaVAE
from MnistDataSet import get_data_set

tf.keras.backend.clear_session()

model = BetaVAE(latent_size=4, beta=1)
train_dataset, test_dataset = get_data_set()
model.train(train_dataset, test_dataset)
