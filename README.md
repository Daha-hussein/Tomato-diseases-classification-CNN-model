<h1>Tomato Plant Disease Classification using Deep Learning</h1>

<h2>Overview</h2>

<p>This project aims to classify various diseases affecting tomato plants using deep learning techniques. The model is trained on a dataset sourced from Plant Village available on Kaggle. The classification is performed using a Convolutional Neural Network (CNN) architecture implemented with TensorFlow and Keras.</p>

<h2>Dataset</h2>

<p>The dataset consists of images depicting various tomato plant diseases along with healthy tomato plants. The dataset has been split into training, validation, and test sets using the splitfolders tool. It contains ten classes corresponding to different tomato plant diseases.</p>

<h2>Model Architecture</h2>

<p>The CNN model architecture comprises several convolutional and max-pooling layers followed by dense layers. The input images are resized to 256x256 pixels with 3 color channels (RGB). The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.</p>

<h2>Training and Evaluation</h2>

<p>The model is trained for 20 epochs, with performance metrics such as accuracy and loss monitored during training. After training, the model is evaluated on a separate test set to assess its generalization ability. The accuracy achieved on the test set is approximately 71.5%.</p>

<h2>Prediction and Inference</h2>

<p>The trained model is capable of predicting the class labels of input images. A function for inference has been implemented to predict disease classes and their corresponding confidence levels. Sample images from the test set are used to demonstrate the model's prediction capabilities.</p>

<h2>Saving the Model</h2>

<p>The trained model is saved in the HDF5 format for future use.</p>

<h2>Requirements</h2>

<ul>
  <li>Python 3.x</li>
  <li>TensorFlow</li>
  <li>Keras</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
</ul>

<h2>Usage</h2>

<ol>
  <li>Clone this repository.</li>
  <li>Install the required dependencies using <code>pip install -r requirements.txt</code>.</li>
  <li>Run <code>potato-disease-classification-mode.ipynb</code> to train the model.</li>
  <li>After training, run <code>python function predict</code> to perform inference on sample images.</li>
  <li>Optionally, customize the model architecture or hyperparameters as needed.</li>
</ol>

<h2>Credits</h2>

<ul>
  <li>Dataset: Plant Village on Kaggle</li>
  <li>Splitting Tool: splitfolders</li>
  <li>TensorFlow and Keras documentation for guidance on model implementation</li>
</ul>

<h2>Author</h2>

<p>Daha Hussein</p>
<p>Email: dahahussein9@gmail.com</p>
