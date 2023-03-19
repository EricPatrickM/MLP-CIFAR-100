import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

categorias=["aquatic mammals","fish","flowers","food containers","fruit and vegetables","household electrical devices","household furniture","insects","large carnivores","large man-made outdoor things",
            "large natural outdoor scenes", "large omnivores and herbivores","medium-sized mammals","non-insect invertebrates","people","reptiles","small mammals","trees", "vehicles 1", "vehicles 2"]

# Dividir o conjunto de treinamento em conjuntos de treinamento e validação
x_train, x_test = x_train / 255.0, x_test / 255.0


model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(3072, activation='relu'),
    Dense(1536, activation='relu'),
    Dense(768, activation='relu'),
    Dense(384, activation='relu'),
    Dense(192, activation='softmax')
])



model.compile(loss="sparse_categorical_crossentropy",
    optimizer="Adam",
    metrics=["accuracy"])



model.fit(x_train, y_train, epochs=50)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)