import tensorflow as tf
from tensorflow import math as tfm
from tensorflow import random as tfr
import matplotlib.pyplot as plt
import math


def beta_schedule_linear(time_steps, beta_from=0.001, beta_to=0.02):
    return tf.linspace(beta_from, beta_to, time_steps)


def beta_schedule_constant(time_steps, beta):
    return beta_schedule_linear(time_steps, beta, beta)


def beta_schedule_cos(time_steps, max_beta=0.999):
    # from https://arxiv.org/abs/2105.05233
    alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(time_steps):
        t1 = i / time_steps
        t2 = (i + 1) / time_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return tf.convert_to_tensor(betas)


class DiffusionModel:

    def __init__(self, network_generator, input_shape=(28, 28, 1), diffusion_steps=1000, learning_rate=0.0001,
                 beta_schedule=beta_schedule_linear):
        self.input_shape = input_shape
        self.time_steps = diffusion_steps - 1
        self.model = network_generator(float(self.time_steps), learning_rate)
        self.beta = beta_schedule(diffusion_steps)
        self.alpha = 1. - self.beta
        self.alpha_hat = tfm.cumprod(self.alpha)  # in other papers also called beta_hat

    def load_parameters(self, model_path):
        self.model.load_weights(model_path)

    def save_parameters(self, model_path):
        self.model.save_weights(model_path)

    def generate_standard_noise(self, shape):
        return tfr.normal(shape=shape, mean=0.0, stddev=1.0)

    def sample2(self):
        x = self.generate_standard_noise((1,) + self.input_shape)
        for t in range(self.time_steps, -1, -1):
            z = self.generate_standard_noise((1,) + self.input_shape) if t > 0 else 0.0
            epsilon = self.model((x, tf.convert_to_tensor([t])))
            x = 1. / tfm.sqrt(self.alpha[t]) * \
                (x - (1 - self.alpha[t]) / tf.sqrt(1 - self.alpha_hat[t]) * epsilon) + \
                self.beta[t] * z
        return (x[0] + 1) * 128.  # denormalize

    @tf.function
    def sample(self):
        t = tf.constant(self.time_steps)
        x = self.generate_standard_noise((1,) + self.input_shape)
        c = lambda t, _: t >= 0

        def b(t, x):
            z = self.generate_standard_noise((1,) + self.input_shape) if t > 0 else 0.0
            epsilon = self.model((x, tf.convert_to_tensor([t])))
            alpha_t = tf.reshape(self.alpha[t], (1, 1, 1, 1))
            alpha_hat_t = tf.reshape(self.alpha_hat[t], (1, 1, 1, 1))
            beta_t = tf.reshape(self.beta[t], (1, 1, 1, 1))

            x = 1. / tfm.sqrt(alpha_t) * (x - (1 - alpha_t) / tf.sqrt(1 - alpha_hat_t) * epsilon) \
                + beta_t * z

            return (t - 1, x)

        res = tf.while_loop(c, b, [t, x], shape_invariants=[t.get_shape(), tf.TensorShape(None)])[1]
        return (res[0] + 1) * 128.  # denormalize

    @tf.function
    def train_step(self, data_set, loss_fn):
        acc_loss = 0.0
        for data in data_set:
            batch_size = tf.shape(data)[0]
            t = tfr.uniform(shape=(batch_size, 1), minval=0, maxval=self.time_steps, dtype=tf.int32)
            noise = self.generate_standard_noise((batch_size,) + self.input_shape)
            alpha_hat_t = tf.reshape(tf.gather_nd(self.alpha_hat, t), (batch_size, 1, 1, 1))
            noisy_data = tfm.sqrt(alpha_hat_t) * data + tfm.sqrt(1 - alpha_hat_t) * noise
            with tf.GradientTape() as tape:
                predicted_noise = self.model((noisy_data, tf.cast(t, tf.float32)))
                loss = loss_fn(noise, predicted_noise)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            acc_loss += loss
        return acc_loss

    def train(self, data_set, epochs):
        tf.keras.backend.clear_session()
        loss_fn = tf.keras.losses.MeanSquaredError()
        for epoch in range(epochs):
            loss = self.train_step(data_set, loss_fn)
            print(f"epoch {epoch}, loss={loss / float(data_set.__len__())}")
            img = self.sample()
            plt.imshow(img.numpy().astype("uint8")[:, :, 0], cmap='gray')
            plt.show()
