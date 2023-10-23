import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import csv
import shutil
import re 
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class TomatoClasificacion:
    def __init__(self, path_train, path_valid, optimizer, epochs, batch_size, nueva_resolucion=(256, 256), num_clases=10):
        self.path_train = path_train
        self.path_valid = path_valid
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.nueva_resolucion = nueva_resolucion
        self.num_clases = num_clases


    def get_df_train(self):
        '''
        Esta funcion lee los datos de entrenamiento
        '''
        train_data = []
        for dirname, _, filenames in os.walk(self.path_train):
            for filename in filenames:
                train_data.append(os.path.join(dirname, filename))
        # creamos un dataframe con los datos de entrenamiento
        df_train = pd.DataFrame(train_data, columns=['path'])

        df_train['label'] = df_train['path'].apply(lambda x: os.path.dirname(x).split('___')[1])
        #guardamos el csv en la carpeta train
        path = 'train'
        os.makedirs(path, exist_ok=True)
        df_train.to_csv(os.path.join(path, 'train.csv'), index=False)
        return df_train
        
    def get_df_valid(self):
        '''
        Esta funcion lee los datos de validacion
        '''
        valid_data = []
        for dirname, _, filenames in os.walk(self.path_valid):
            for filename in filenames:
                valid_data.append(os.path.join(dirname, filename))
        # creamos un dataframe con los datos de validacion
        df_valid = pd.DataFrame(valid_data, columns=['path'])
        df_valid['label'] = df_valid['path'].apply(lambda x: os.path.dirname(x).split('___')[1])
        #guardamos el csv en la carpeta valid
        path = 'valid'
        os.makedirs(path, exist_ok=True)
        df_valid.to_csv(os.path.join(path, 'valid.csv'), index=False)
        return df_valid
    
    #creamos una funcion que revise si las iagenes tienen la misma resolucion
    def verificar_resolucion(self, carpeta_raiz):
            '''
            Esta función verifica la resolución de las imágenes en un directorio
            '''
            resoluciones = set()
            
            for raiz, dirs, archivos in os.walk(carpeta_raiz):
                for archivo in archivos:
                    if archivo.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')):
                        ruta_imagen = os.path.join(raiz, archivo)
                        with Image.open(ruta_imagen) as img:
                            resoluciones.add(img.size)
            #si las resoluciones son diferentes, se redimensionan
            if len(resoluciones) > 1:
                print(f'Resoluciones en la carpeta {carpeta_raiz}: {resoluciones}')
                print('Redimensionando imágenes...')
                self.redimensionar_imagenes(carpeta_raiz)
            else:
                print(f'Resoluciones en la carpeta {carpeta_raiz}: {resoluciones}')

            return None
    
    def crear_generador_aumentacion(self):
        '''
         Esta función crea un objeto ImageDataGenerator con aumentación de datos
        '''
        datagen = ImageDataGenerator(
            rotation_range=40,        # Rango de valores aleatorios para rotar la imagen (0-40 grados)
            width_shift_range=0.2,    # Rango de valores aleatorios para desplazar la imagen horizontalmente
            height_shift_range=0.2,   # Rango de valores aleatorios para desplazar la imagen verticalmente
            shear_range=0.2,          # Rango de valores aleatorios para aplicar corte (shear) a la imagen
            zoom_range=0.2,           # Rango de valores aleatorios para aplicar zoom a la imagen
            horizontal_flip=True,     # Habilitar la reflexión horizontal aleatoria
            fill_mode='nearest',      # Método para rellenar los píxeles vacíos generados por la aumentación
            brightness_range=(0.8, 1.2),  # Rango de valores aleatorios para ajustar el brillo de la imagen
            rescale=1./255            # Normalizar los valores de los píxeles al rango [0, 1]
        )
        return datagen
    
    def crear_modelo_simple(self):
        '''
         Esta función crea un modelo secuencial
        '''
        model = Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_clases, activation='softmax')
        ])

        # Compilar el modelo
        model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=[
                metrics.CategoricalAccuracy(name='accuracy'),
                tfa.metrics.F1Score(
                    num_classes=self.num_clases, average='weighted', name='f1_score'
                ),
                metrics.Recall(name='recall')
            ]
            
        )

        return model
    
    def crear_modelo_preentrenado(self):
        '''
         Esta función crea un modelo secuencial con dropout
        '''
        # Cargar el modelo preentrenado resnet50
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',  # Cargar los pesos preentrenados
            input_shape=(256, 256, 3),
            include_top=False  # No incluir la capa fully-connected al final
        )

        # Descongelar las capas del modelo base a partir de una cierta capa
        for layer in base_model.layers[-20:]:  # Descongela las últimas 20 capas
            layer.trainable = True

        # Crear el modelo secuencial
        model = Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_clases, activation='softmax')
        ])

        # Compilar el modelo
        model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=[
                metrics.CategoricalAccuracy(name='accuracy'),
                tfa.metrics.F1Score(
                    num_classes=self.num_clases, average='weighted', name='f1_score'
                ),
                metrics.Recall(name='recall')
            ]
        )

        return model
    
    #ahora usaremos los 3 modelos para compararlos
    def entrenar_modelo(self, modelo, datagen, datos_entrenamiento, datos_validacion, name):
        '''
         Esta función entrena un modelo
        '''
        train_generator = datagen.flow_from_dataframe(
            dataframe=datos_entrenamiento,
            directory=None,
            x_col="path",
            y_col="label",
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        validation_generator = datagen.flow_from_dataframe(
            dataframe=datos_validacion,
            directory=None,
            x_col="path",
            y_col="label",
            target_size=(256, 256),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'mejor_modelo_{name}.h5', save_best_only=True)


        historia = modelo.fit(
            train_generator,
            validation_data = validation_generator,
            epochs=self.epochs,
            steps_per_epoch=len(datos_entrenamiento) // self.batch_size,
            validation_steps=len(datos_validacion) // self.batch_size,
            callbacks=[early_stopping, model_checkpoint] 
        )
        
        return modelo, historia  # Retorna el modelo entrenado y la historia
        

if __name__ == '__main__':
    #proveemos la ruta a las carpetas de entrenamiento y validacion
    path_train = '/home/hnavarro/Downloads/archive_final_proyect/Plant_Diseases_Dataset/train/'
    path_valid = '/home/hnavarro/Downloads/archive_final_proyect/Plant_Diseases_Dataset/valid/'
    #creamos una instancia de la clase
    tomato = TomatoClasificacion(path_train=path_train, path_valid=path_valid, optimizer='adam', epochs=100, batch_size=32, num_clases=10)
    df_train= tomato.get_df_train()
    df_valid= tomato.get_df_valid()
    tomato.verificar_resolucion(path_train)
    tomato.verificar_resolucion(path_valid)
    #generamos la aumentacion de los datos
    datagen = tomato.crear_generador_aumentacion()

    #creamos el modelo simple
    modelo_simple = tomato.crear_modelo_simple()

    #entrenamos el modelo simple
    modelo_simple_entrenado, historia_simple = tomato.entrenar_modelo(modelo_simple, datagen, df_train, df_valid, name='simple')
    #guardamos la historia en un df
    df_historia_simple = pd.DataFrame(historia_simple.history)
    df_historia_simple.to_csv('historia_simple.csv', index=False)

    #creamos modelo preentrenado descongelado
    modelo_preentrenado = tomato.crear_modelo_preentrenado()
    modelo_descongelado, historia_descongelado = tomato.entrenar_modelo(modelo_preentrenado, datagen, df_train, df_valid, name='resnet')

    #guardamos la historia
    df_historia_desc = pd.DataFrame(historia_descongelado.history)
    df_historia_desc.to_csv('historia_descongelado.csv', index=False)
    


