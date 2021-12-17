# How can you make your speech go viral?
**Authors:** Andrea Oliveri, Célina Chkroun, Kaourintin Tamine, Mattia Lazzaroni

[**Data story website**](https://kaoutamine.github.io/viralSpeech/)

## Abstract 
In a world where the internet allows speaking to millions of people, we want to identify what factors can be linked to someone effectively reaching huge audiences. 
To measure how many people are reached by a quote we will use the number of repetition of said quote in newspapers. 
By analysing the characteristics of extremely viral quotes we want to try building guidelines to optimize the chance of a quote reaching a wider range of people. 
These guidelines would constitute very interesting tools for politicians and influencers as they would allow to better choose the topic and the speaker delivering a concept to increate the likelihood of being heavily quoted. 

## Research Questions:

#### Are there any identifiable characteristics of the speaker which make it more "quotable"?
Age? Gender? Occupation? What characteristics of a speaker make it more likely to reach a larger audience, and what can be changed to increase the quotability. 

#### Does the sentiment, topic or number of words of the quote influence its success?
We will also explore if the textual content of the quote has an impact on the virality. We will mainly investigate two factors: the number of words in the quote and its topic. 

#### Is it possible to use some machine learning models to predict virality?
We want to see how effective these models are at predicting such a complex variable.


## Datasets:
We have used two separate datasets:
- [`Quotebank`](https://zenodo.org/record/4277311#.YYpVGWDMJhE): corpus of quotations from news extracted between 2015 and 2020.
- [`Wikidata`](https://www.wikidata.org/wiki/Wikidata:Main_Page): a database containing information about subjects in Wikimedia. In this project we will use it to gather information about the speaker of the quotes.


## Methods

### 1. Collect data from Wikidata
To extract all information we will need from Wikidata, the qids of all speakers in Quotebank are collected, keeping track of those for which the quote attribution was ambiguous.
With these we can filter out rows and columns we are not interested in from the provided [`speaker_attributes.parquet`](Data/speaker_attributes.parquet), and query from Wikidata the link counts of ambiguous speakers.
Finally, we can query Wikidata for human-readable English labels of relevant qids.

### 2. Dealing with large dataset
Due to the large size and heavy compression of the Quotebank dataset, working with it presents two challenges:

1) Dataset can't be entirely fit in RAM
2) Parsing dataset requires a long time

Issue 1 was addressed by decompressing and loading lines from the json files one by one.

Issue 2 was attenuated by implementing a Python decorator storing to disk results of long computation (for a detailed explaination, read the doctring of `cache_to_file_pickle` in [src/utils.py](src/utils.py)).

Note that, after filtering, [`speaker_attributes.parquet`](Data/speaker_attributes.parquet) could easily be kept in RAM while working on Quotebank line by line.

### 3. Features extraction

#### Number of occurences of quotes
We first analyzed the number of occurences of all the quotes present in Quotebank and defined the threshold above which a quote will be considered as viral.

#### Speaker's features
We explored the raw distribution of the speaker's features as well as the distributions obtained when weighting by the number of different quotes of the speaker and by the total number of occurrences of all his quotes. Further analysis looked at the co-occurences of values in features for which multiple values are given to a single speaker, such as occupations. Similarly, the co-occurences between values of different features was analyzed.

#### Length of quotes
We analyzed the distributions of lenghts of all quotes in Quotebank.

#### Quote topics
At first, we tried to extract the topics of the quote using `Latent semantic analysis`<sup>[1](https://en.wikipedia.org/wiki/Latent_semantic_analysis) [2](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) [3](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a)</sup>.
However, we found out by implementing this method that it was too computationally and memory heavy.

Upon further research, we stumbled upon a (great) package called [`BERTopic`](https://github.com/MaartenGr/BERTopic)<sup>[1](https://towardsdatascience.com/dynamic-topic-modeling-with-bertopic-e5857e29f872)</sup>, implementing all the tools we needed to quickly extract the main topics present in the Quoteback dataset and visualize them.
Moreover, part of the computations can be hugely accelerated using a GPU. Training BERTopic on 1% of the dataset we successfully extracted and visualized meaningful topics.

#### Quote sentiment
We used the (also great) package called [`vaderSentiment`](https://github.com/cjhutto/vaderSentiment) to extract the sentiment expressed in the quotes.


### 4. Training models
In order to predict the number of occurrences of a quote, the following models have been trained:

- Linear regression
- Decision tree regressors (with different post-pruning)

To predict its virality (if it was repeated more than a pre-defined number of times), the following models have been trained:

- Linear Support Vector Machine
- Decision tree classifiers (with different post-pruning)

These models were chosen because of their easy interpretability. 

To train the models, each line of Quotebank dataset is converted into a feature vector (or discarded if feature is missing) and a binary label for classifiers or an integer output for the regressor. Training is performed on a randomly sampled 70% subset of these lines (this also includes the validation set for hyperparameters tuning) and the rest is used for testing.


## Organization within the team
Within the team, the organization was as follows:

|                                                        |  Andrea  |  Célina  | Kaourintin |  Mattia  |
|--------------------------------------------------------|:--------:|:--------:|:----------:|:--------:|
| Query Wikidata for additional speaker information      | &#10003; |          |            |          |
| Join speaker information and Quotebank                 | &#10003; | &#10003; |            |          |
| Parse Quotebank to collect statistics                  |          | &#10003; |            |          |
| Create utilitary functions                             | &#10003; |          |            |          |
| Speaker feature extraction                             | &#10003; |          |            |          |
| Exploratory data analysis (plots and results analysis) |          | &#10003; |  &#10003;  | &#10003; |
| Topics detection (BERTopic training and analysis)      | &#10003; |          |  &#10003;  | &#10003; |
| Sentiment extraction                                   | &#10003; |          |            |          |
| Feature matrix generation                              | &#10003; |          |            |          |
| Relationship between features and output               |          | &#10003; |            |          |
| Choice of machine learning models                      | &#10003; |          |            |          |
| Notebook machine learning theory                       |          | &#10003; |  &#10003;  |          |
| Cross-validation results collection                    | &#10003; |          |            | &#10003; |
| Cross-validation results analysis                      |          |          |  &#10003;  | &#10003; |
| Data story layout and graphical user interface         |          | &#10003; |            |          |
| Data story writing                                     | &#10003; |          |  &#10003;  | &#10003; |
| Interactive plots data story                           |          | &#10003; |            |          |
| README                                                 | &#10003; | &#10003; |  &#10003;  | &#10003; |

## Repository structure

- [Cache](Cache): Contains results of long computations.
- [Data](Data): Contains the provided datasets (Quotebank's json files and [`speaker_attributes.parquet`](Data/speaker_attributes.parquet)).
- [src](src): Contains utilitary modules implementing:
    - Feature extraction: [feature_extraction.py](src/feature_extraction.py)
    - Visualization: [plot.py](src/plot.py)
    - Management of data: [utils.py](src/utils.py)
    - Management of sparse matrix data: [utils_sparse_matrix.py](src/utils_sparse_matrix.py)
    - Cross-validation functions: [cross_validation.py](src/cross_validation.py)
- [Project.ipynb](Project.ipynb): Notebook in which the analysis is performed.
