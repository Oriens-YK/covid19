import numpy as np
X = np.load(r"D:\DeepLearning\covid\CLAHE\CLAHE_288_X_0_255.npy")
y = np.load(r"D:\DeepLearning\covid\CLAHE\CLAHE_288_y_0_255.npy")
X = X.astype("float16")

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify = y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=5, stratify = y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

for i in range(len(y_train)):   
  if y_train[i] == 2:
    y_train[i] = 1

for i in range(len(y_test)):
  if y_test[i] == 2:
    y_test[i] = 1
  
for i in range(len(y_val)):
  if y_val[i] == 2:
    y_val[i] = 1
    
# create model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
model = Xception(weights=None, include_top=False, input_shape=[224, 224, X.shape[3]])
model_x = model.output
x = GlobalAveragePooling2D(name='avg_pool')(model_x)
predictions = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=model.input, outputs=predictions)
model.summary()

#fit
from tensorflow.keras import optimizers
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
history = model.fit(X_train, y_train, batch_size = 1, epochs = 5,validation_data =(X_val, y_val), shuffle=True)
score = model.evaluate(X_test, y_test)