import numpy as np
from numpy import linalg as LA
from PIL import Image

# load data
train_data = []
test_data = []
for id_ in range(40):
    for pic in range(10):
        if pic < 7 :
            train_dir = str(id_+1)+'_'+str(pic+1)+'.png'
            train_img = Image.open('./data/'+train_dir)
            train_data.append(np.array(train_img).flatten())
        else: 
            test_dir = str(id_+1)+'_'+str(pic+1)+'.png'
            test_img = Image.open('./data/'+test_dir)
            test_data.append(np.array(test_img).flatten())
train_data = np.stack(train_data, axis = 0)
test_data = np.stack(test_data, axis = 0)

# find eigen face
def PCA(train_data):
    mean = np.mean(train_data, axis = 0)
    cen_train = train_data - mean
    cov = np.cov(cen_train.T)
    value, vector = LA.eig(cov)
    return np.real(vector)

# visualization : mean face/ first four eigen face/ 
eigen_face = PCA(train_data)

face_one = eigen_face[:,-1].reshape(56,46)
face_two = eigen_face[:,-2].reshape(56,46)
face_three = eigen_face[:,-3].reshape(56,46)
face_four = eigen_face[:,-4].reshape(56,46)
face_mean = np.mean(train_data, axis = 0).reshape(56,46)

#face_one *= (255.0/face_one.max())
#pil_one = Image.fromarray(face_one)
#pil_one.show()

