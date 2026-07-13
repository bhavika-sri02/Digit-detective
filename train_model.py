import tensorflow
from tensorflow import keras # type: ignore
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense,Flatten # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def main():
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # to scale the values between 0 and 1
    X_train = X_train/255.0
    X_test = X_test/255.0

    print("\nBuilding model...")
    # to train the model
    model = Sequential()

    # to convert 2D array into 1D input using Flatten
    model.add(Flatten(input_shape=(28,28)))
    model.add( Dense(128,activation = 'relu'))
    model.add( Dense(32,activation = 'relu'))
    model.add(Dense(10,activation = 'softmax'))

    # to get detailed summary of layers and parameters
    model.summary()

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

    print("\nTraining model...")
    history = model.fit(X_train,y_train,epochs=25,validation_split= 0.2)

    print("\nEvaluating on test set...")
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)
    acc_score=accuracy_score(y_test , y_pred)
    print(f"Test accuracy: {acc_score:.4f}")
   
    # to save the trained model
    print("\nSaving model as mnist_model.h5 ...")
    model.save("mnist_model.h5")
    print("Done. Model saved as mnist_model.h5")

    model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # to get the graph of accuracy
    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # to get the graph of loss
    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # to save the graphs
    plt.tight_layout()
    plt.savefig("Training Graph.png")
    print("Saved training curves to training_history.png")
    plt.show()



if __name__ == "__main__":
    main()