"""
Modèle VGG16 pour FER2013
Fine-tuning avec dégel du block5
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Construit le modèle VGG16 avec fine-tuning
    """
    # 1. Charger VGG16 sans la tête
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # 2. Geler tout au début
    for layer in base_model.layers:
        layer.trainable = False
    
    # 3. Ajouter la tête de classification
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

def compile_model(model, learning_rate=1e-4):
    """
    Compile le modèle
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def setup_fine_tuning(base_model):
    """
    Débloquer le block5 pour fine-tuning
    """
    set_trainable = False
    for layer in base_model.layers:
        if "block5" in layer.name:
            set_trainable = True
        layer.trainable = set_trainable
    return base_model