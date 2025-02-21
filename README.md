# SentimentXpert  

## 🎬 IMDb Movie Review Sentiment Analysis using Bi-LSTM  

### 🚀 Project Overview  
SentimentXpert is a deep learning-based sentiment analysis model designed to classify IMDb movie reviews as positive or negative. By leveraging **Bidirectional Long Short-Term Memory (Bi-LSTM)**, our model significantly improves accuracy compared to conventional LSTM architectures.  

### 📌 Key Features  
- **Advanced NLP Techniques**: Utilizes tokenization, stop-word removal, lemmatization, and text normalization.  
- **Deep Learning Architecture**: Implements a Bi-LSTM model to capture bidirectional dependencies in text.  
- **High Accuracy**: Achieved an **89.19% accuracy**, surpassing the referenced research paper's LSTM model.  
- **Robust Model Training**: Applied stratified K-fold cross-validation for consistent performance evaluation.  

---
## 🤖 What is Bi-LSTM?  
**Bidirectional Long Short-Term Memory (Bi-LSTM)** is an advanced type of **Recurrent Neural Network (RNN)** that improves sequence processing by considering **both past and future contexts**.  

Unlike traditional **LSTM**, which only processes input sequentially from past to future, **Bi-LSTM** processes data in **two directions** (forward and backward), making it **more effective for sentiment analysis and NLP tasks**. This helps in understanding the sentiment of a sentence more accurately by capturing context from both sides of the text.  

---

## 📚 What You’ll Learn from This Project  
By working on **SentimentXpert**, you'll gain hands-on experience with:  

✅ **Natural Language Processing (NLP)** – Text preprocessing, tokenization, stop-word removal, and lemmatization.  
✅ **Deep Learning** – Understanding and implementing **LSTM** and **Bi-LSTM** for text classification.  
✅ **Hyperparameter Tuning** – Optimizing batch size, learning rates, and dropout for better model accuracy.  
✅ **Model Evaluation** – Using **accuracy, precision, recall, and F1-score** to compare different models.  
✅ **Real-World Deployment** – Deploying a machine learning model using **Flask** for live sentiment analysis.  

---

## 📂 Dataset  
- **Source**: IMDb Movie Reviews  
- **Labels**: Positive and Negative Sentiments  
- **Preprocessing**: HTML tag removal, tokenization, text encoding, and padding  

---

## 🏗 Model Architecture  
1. **Embedding Layer**: Converts words into dense vector representations.  
2. **Bidirectional LSTM**: Captures contextual dependencies in both directions.  
3. **Dense Layer**: Transforms extracted features into sentiment predictions.  
4. **Sigmoid Activation**: Outputs probabilities for classification.  
5. **Optimizer**: Adam  
6. **Loss Function**: Binary Cross-Entropy  

---

## 📊 Performance Metrics  

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|----------|--------|---------|
| **Bi-LSTM (Ours)**  | **89.19%** | **89.21%** | **89.25%** | **89.23%** |
| LSTM (Reference)    | 87%       | 81%       | 80%    | 80%    |

---

## 🎯 Impact & Applications  
✅ **For Industry**: Enhances sentiment analysis capabilities for movie reviews, customer feedback, and social media analytics.  
✅ **For Research**: Provides insights into bidirectional processing for text classification tasks.  

---

## 🛠 Tech Stack  
- **Programming Language**: Python  
- **Libraries**: TensorFlow, Keras, NLTK, Pandas, NumPy, Matplotlib, Seaborn  
- **Deployment**: Flask (for real-time model inference)  

---

## 📌 How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/yourusername/SentimentXpert.git
   ```

2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

3. **Train the model:**
  ```bash
  python train.py
  ```

4. **Test the model:** 
  ```bash
  python test.py
  ```

5. **Deploy using Flask:**
  ```bash
  python app.py
  ```
💡 *By exploring this project, you'll not only enhance your NLP and deep learning skills but also build a **strong portfolio** that showcases your ability to develop and deploy advanced AI models. Whether you're a student, researcher, or aspiring data scientist, SentimentXpert is a great way to dive deeper into the world of sentiment analysis and deep learning!* 🚀
