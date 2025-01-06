**IMDB Sentiment Analysis with Stacked Autoencoder and LSTM**

**Overview**

This project performs sentiment analysis on IMDB movie reviews using a
combination of a **Stacked Autoencoder** and **Long Short-Term Memory
(LSTM)** network. The goal is to classify movie reviews as either
**positive** or **negative** based on their content. The code also
includes data preprocessing, tokenization, lemmatization, and
visualization of word clouds for sentiment analysis.

**Libraries and Tools Used:**

-   **NumPy**: Used for numerical operations and data manipulation.

-   **Pandas**: For data handling and manipulation (loading datasets,
    checking for missing values, etc.).

-   **Matplotlib** & **Seaborn**: For data visualization and plotting
    (e.g., confusion matrices and word clouds).

-   **TensorFlow** & **Keras**: For building and training deep learning
    models.

-   **NLTK**: For text processing tasks like tokenization,
    lemmatization, and stopword removal.

-   **Scikit-learn**: For machine learning utilities (train-test split,
    metrics, etc.).

-   **WordCloud**: For visualizing the most frequent words in positive
    and negative reviews.

**File Structure**

-   **IMDB Dataset.csv**: The IMDB dataset containing reviews and
    sentiment labels.

-   **Anjaneyam NLP Code**: The directory containing the code used for
    data processing, modeling, and evaluation.

-   **README.md**: This file explaining the purpose of the project.

**Steps in the Code**

**1. Data Loading and Initial Exploration**

The IMDB dataset is loaded into a Pandas DataFrame and basic exploration
is done, including:

-   Checking the shape and structure of the dataset.

-   Checking for missing values.

-   Removing duplicates and handling mislabeled sentiment values.

**2. Preprocessing and Cleaning Text**

The reviews are preprocessed to:

-   Remove HTML tags, URLs, and non-alphabetic characters.

-   Convert text to lowercase and split it into words.

-   Lemmatize the words using the **WordNetLemmatizer**.

**3. Word Cloud Visualization**

The word clouds are generated for both **positive** and **negative**
reviews to visualize frequently occurring words. The positive word cloud
uses warm colors, while the negative word cloud uses cool colors.

**4. Sentiment Classification with Stacked Autoencoder (SAE) and LSTM**

The sentiment analysis is done using a Stacked Autoencoder combined with
an LSTM network:

-   The reviews are tokenized and padded to form fixed-length sequences.

-   The model architecture includes:

    -   An embedding layer for learning word representations.

    -   A dense layer for encoding the input.

    -   An LSTM layer for capturing temporal dependencies in the
        sequence.

    -   A dense output layer with sigmoid activation for binary
        classification (positive/negative sentiment).

**5. Model Training and Evaluation**

The model is trained using the training data, and evaluated on both
training and test sets. The following metrics are calculated:

-   **Accuracy**: Percentage of correct predictions.

-   **Precision**: Percentage of true positives among all positive
    predictions.

-   **Recall**: Percentage of true positives among all actual positives.

-   **F1-Score**: The harmonic mean of precision and recall.

-   **Confusion Matrix**: Visualized to assess the model\'s performance.

-   **Classification Report**: Detailed performance metrics for the
    model.

**6. 90-10 Data Split**

Another evaluation is done using a 90% training and 10% testing split to
assess the model\'s performance on a smaller test set.

**Usage**

1.  **Install Dependencies**:\
    Ensure that all the required libraries are installed, including
    tensorflow, keras, nltk, pandas, matplotlib, seaborn, etc. You can
    install them using pip:

bash

Copy code

pip install numpy pandas matplotlib seaborn tensorflow nltk scikit-learn
wordcloud

2.  **Run the Code**:

    -   Download and unzip the dataset into the appropriate directory.

    -   Execute the script to perform sentiment analysis. The model will
        train and print evaluation metrics for both the training and
        testing datasets.

3.  **Model Evaluation**: After training, the script will print out:

    -   Accuracy, Precision, Recall, and F1-Score for both training and
        testing sets.

    -   A confusion matrix and classification report for both training
        and testing datasets.

**Notes**

-   The dataset is assumed to be in a .csv format with columns review
    (movie review text) and sentiment (1 for positive and 0 for
    negative).

-   The code handles text preprocessing (lemmatization, stopword
    removal, etc.) before passing the data into the model.

-   The model uses **LSTM** combined with a **Stacked Autoencoder** to
    perform sentiment analysis, which is useful for capturing temporal
    dependencies in sequential text data.
