import numpy as np
import pickle
import streamlit as st
import cv2
from skimage.feature import hog

def get_feature_hog(data):
  features, _ = hog(data , orientations=9 , pixels_per_cell = (8,8),
                      cells_per_block = (2,2) , visualize = True ,  multichannel = True)
  features = np.expand_dims(features, axis = 0)
  return features

def main():
    st.title('Classify Cifar10 datasets with Hog and SVM')
    uploaded_file = st.file_uploader("Upload your image")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resize_img = cv2.resize(img, (32,32))
        features_predict = get_feature_hog(resize_img)
        with open('weights.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('x_test_hog.npy', 'rb') as f:
            x_test_hog = np.load(f)
        with open('x_train_hog.npy', 'rb') as f:
            x_train_hog = np.load(f)
        classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        y_pred = model.predict(features_predict)
        st.image(img, channels="BGR")
        st.write(classes[y_pred[0]])

    

if __name__ == '__main__':
    main()
