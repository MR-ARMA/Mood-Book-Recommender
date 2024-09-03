# Context-Aware Book Recommendation System Based on User Personality and Emotion Prediction

## Introduction

This project develops a **context-aware book recommendation system** that predicts a user's personality traits and emotional state from their text input and uses these predictions to recommend books aligned with the user's psychological profile. By integrating **Natural Language Processing (NLP)** and **machine learning** techniques, the system aims to offer highly personalized book recommendations.

## Project Objectives

1. **Personality Prediction**: Build an NLP model to predict the Big Five personality traits from user text.
2. **Emotion Detection**: Develop an NLP model to detect the user's emotional state from text.
3. **Recommendation System**: Create a context-aware recommendation system using the predicted personality traits and emotions to suggest suitable books.
4. **Integration and Evaluation**: Integrate the models and evaluate the system’s performance against baseline models.

## Methodology

### Data Collection

- **Essays Dataset**: Used for training personality prediction models. [Essays Dataset on Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **GoEmotions Dataset**: For training emotion detection models. [GoEmotions on GitHub](https://github.com/google-research/google-research/tree/master/goemotions)
- **Goodreads Dataset**: For content-based and collaborative filtering. [Goodreads Dataset on Kaggle](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)
- **Book-Crossing Dataset**: Provides additional user-item interaction data. [Book-Crossing Dataset on Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)

### Model Development

- **Personality Prediction**: Fine-tune BERT or RoBERTa on the Essays Dataset.
- **Emotion Detection**: Fine-tune a BERT-based model on the GoEmotions Dataset.
- **Recommendation System**: Implement a hybrid model using content-based and collaborative filtering techniques, enhanced with personality and emotion features.

### System Integration

- Integrate personality prediction, emotion detection, and book recommendation into a unified system.
- Develop a simple web interface using Flask or Django for user interaction.

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: Hugging Face Transformers, NLTK, spaCy, Scikit-Learn, TensorFlow, PyTorch
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask or Django

## Expected Outcomes

1. A functional personality prediction model.
2. An accurate emotion detection model.
3. A context-aware book recommendation system that integrates personality and emotion predictions.
4. A prototype demonstrating the integrated system.

## Getting Started

1. Clone the repository: `git clone https://github.com/yourusername/your-repository.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the instructions in `docs/INSTALL.md` for setup and usage.

For more details, please refer to the [project documentation](docs/README.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
