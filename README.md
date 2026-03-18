## Technical Note on Viewing

Due to a known issue in GitHub's web rendering engine (related to the handling of JSON interactive widget metadata in notebooks), the preview of the `.ipynb` files might show an "Invalid Notebook" error message.
The files, the code, and the training logs are perfectly intact. To properly view the project, simply download the notebooks and open them locally via Jupyter or VS Code.

# EVALITA 2026 GSI:detect - Gender Stereotype Detection and Classification

This repository contains the implementation of my university project based on the EVALITA 2026 GSI:detect challenge. The goal of the project is the identification and classification of gender stereotypes within short Italian texts. The work is structured into two main tasks, addressed in their respective notebooks.

## Part 1: Task 1 - Stereotype Degree Estimation (Task1.ipynb)

The first task focuses on evaluating the degree of stereotyping in a text. Instead of approaching the problem as a classic binary classification, I framed the task as a regression problem to predict a continuous score.

### Pre-processing Strategy
Classic NLP pipelines tend to remove fundamental textual features for the recognition of gender stereotypes. I therefore adopted an extremely conservative approach:
- **Emoji translation**: Converted into their Italian textual descriptions so as not to lose their semantic value.
- **No lemmatization or stemming**: Maintaining the masculine or feminine inflection of pronouns, articles, and adjectives is vital. Plurals were kept intact as they are strong indicators of stereotyped generalizations.
- **Keeping stop-words**: Words like "non" (not) or "tutte" (all feminine) can completely flip or define the true meaning of a sentence.
- **Original casing and punctuation**: Preserved as they are strong indicators of sentiment, sarcasm, or anger in online comments.
- **Separation between text and context**: Passing the model a single string merging the context (e.g., the article title) and the comment led to syntactic confusion. I separated them, allowing the Self-Attention mechanism to correctly distinguish the role of the context from the user's actual opinion.

### Model Choice: UmBERTo
After validating both BERT (dbmdz/bert-base-italian-cased) and UmBERTo (Musixmatch/umberto-commoncrawl-cased-v1) via Cross-Validation, the final choice fell on UmBERTo for two main reasons:
1. **Web language handling**: Being trained on CommonCrawl, it natively understands social media language, including slang and abbreviations.
2. **Sarcasm detection**: It proved more capable of catching the sarcastic tone of certain Italian expressions compared to BERT.

The only limitation that emerged during the error analysis phase is the model's difficulty in distinguishing between simple vulgarity and actual stereotypes, tending to overestimate the score in the presence of profanity.

## Part 2: Task 2 - Multiclass Classification (Task2.ipynb)

The second task requires classifying the texts into 7 categories (6 specific stereotypes and 1 "NO" stereotype class). The main obstacle was the Small Data environment (roughly 200 initial examples), combined with a strong class imbalance.

### Data Cleaning and Data Augmentation
Given the microscopic size of the dataset, the quality of every single label was crucial.
- **Human-in-the-Loop Data Cleaning**: I used an LLM based on the task guidelines to flag inconsistent original annotations. These cases were then manually reviewed to correct or remove errors.
- **Semantic Data Augmentation**: Avoiding back-translation (which degraded semantics), I used an LLM to generate synthetic syntactic variations exclusively for the minority classes. The prompt strictly enforced the retention of slang, excessive punctuation, and grammatical errors so as not to alter the original distribution.

### Pipeline Evolution
- **Baseline (BERT)**: An initial implementation with Nested Cross-Validation led to unstable loss curves and massive overfitting, with a low Macro F1-Score (~0.35).
- **Knowledge Distillation**: To mitigate overfitting, I implemented a Teacher Ensemble to produce soft labels on which to train a Student model. However, Confidence Analysis revealed that the model still struggled enormously on the minority classes and the "NO" class.
- **Contrastive Learning (SetFit)**: This was the architectural turning point. Working in a few-shot learning logic, SetFit effectively understood the semantic distances between text embeddings even with very few examples, doubling the performance (Macro F1-Score of ~0.62).
- **Cascaded Attempt**: Finally, I experimented with a cascaded pipeline (Binary Model for Stereotype/NO followed by the Multiclass model). However, the approach worsened the F1-Score, confirming the direct SetFit model as the definitive final solution.
