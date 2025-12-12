"""
Script principal d'entraînement
Exécutez : python train.py --model vgg16 --epochs 30
"""

import argparse
import os
from src.data.preprocess import get_data_generators, compute_class_weights
from src.models.vgg16_model import build_model, compile_model, setup_fine_tuning
from src.utils.metrics import plot_training_history

def main():
    parser = argparse.ArgumentParser(description='Train FER2013 model')
    parser.add_argument('--model', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--data_dir', type=str, default='../input/fer2013', help='Data directory')
    
    args = parser.parse_args()
    
    # 1. Data
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    train_gen, val_gen = get_data_generators(train_dir, test_dir)
    class_weights = compute_class_weights(train_gen)
    
    # 2. Model
    model, base_model = build_model()
    model = compile_model(model, learning_rate=1e-4)
    
    # 3. Train
    print("🎓 Phase 1 : Entraînement de la tête")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3)
        ]
    )
    
    # 4. Fine-tuning
    print("🔥 Phase 2 : Fine-tuning block5")
    base_model = setup_fine_tuning(base_model)
    model = compile_model(model, learning_rate=1e-5)
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs // 2,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
        ]
    )
    
    # 5. Save
    model.save('models/vgg16_fer2013_final.h5')
    print("✅ Modèle sauvegardé dans models/vgg16_fer2013_final.h5")

if __name__ == '__main__':
    main()