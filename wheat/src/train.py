from src.data_loader import load_dataset, get_data_generator
from src.model import build_model
from tensorflow.keras.callbacks import EarlyStopping

def main():
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Get data generator for augmentation
    datagen = get_data_generator()

    # Build the new 3-class model
    model = build_model()

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=20,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    # Save the trained model
    model.save('wheat_rust_model_severity.h5')
    print("Model saved as 'wheat_rust_model_severity.h5'")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
