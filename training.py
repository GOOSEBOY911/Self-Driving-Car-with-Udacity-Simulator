print('Setting Up')
from utils import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split

# 1
path = 'myData'
data = importDataInfo(path)
# 2
balanceData(data, display=False)
# 3
imagesPath, steerings = loadData(path, data)
# 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images', len(xTrain))
print('Total Validation Images', len(xVal))
model = createModel()
model.summary()
# 5
history = model.fit(batGen(xTrain, yTrain, 100, 1), steps_per_epoch=500, epochs=30,
                    validation_data=batGen(xVal, yVal, 100, 0), validation_steps=200)
# 6
model.save('model.h5')
print('Model is saved')
# 7
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()