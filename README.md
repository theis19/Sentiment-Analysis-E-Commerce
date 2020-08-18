# Sentiment-Analysis-E-Commerce
This Repository is made to store my final project as a requirement to graduate from Data Science Job Connector Program at Purwadhika Digital Technology School.

## Project Description
This is a dataset of [Women's Clothing E-Commerce](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) that revolves around the reviews written by customers. In this project, the author tries to put himself in the E-Commerce owner's position where as the owner, he needs to understand and gain insight of the store sales, and one way to do that is by gaining it through customer's reviews. Aside from gaining insight, the author will also try to make a model to learn sentiments from the review's data. The model could then predict review's sentiment, whether Positive, Neutral, or Negative.

## Content and Feature
This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:

- Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
- Age: Positive Integer variable of the reviewers age.
- Title: String variable for the title of the review.
- Review Text: String variable for the review body.
- Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
- Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
- Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
- Division Name: Categorical name of the product high level division.
- Department Name: Categorical name of the product department name.
- Class Name: Categorical name of the product class name.

## Exploratory Data Analysis (EDA)

At this step i look through the data to see what information i could find.

- It turns out that about half of the data has a Rating of 5, this means that the customers are pretty happy with the products.
- There are some missing data in the Title and Review Text columns, i assume the customer didn't bother to write down a review but they do leave a Rating.
- The Ratings for the data where there's no Review Text (Review Text is NaN) mostly consist of Rating 5, this means the customer just doesn't want to write a review but they are somewhat happy with the products.
- The customers are consist of mostly older adult women, this is shown by the right skewed distribution on the plot. Most of the customers are between the age of 30-40 years old.
- It seems Age doesn't really play a part in the rating, since average-wise, every Rating given is about the same.
- Rating-wise it seems most of the products are really good, averaging mostly above a Rating of 4, with Bottoms being the most popular and Trend at the bottom.

![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/age_dist.png "Age Distribution")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/sentiment_count.png "Sentiment Count")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/sentiment_percentage.png "Sentiment Percentage")

## Data Preprocessing, Feature Engineering, and More EDA

In this section, we clean the text data with NLP techniques to analyze it and prepare it for the machine learning model.
The cleaning processes i did are:

1. Alpha
   - Stands for alphabet, the cleaning process done here is to get the lowercase version and alphabetical only text.
2. Stemmed
   - As the name implies, here we did stemming process to the text data, we also remove the stopwords in the data.
3. Lemmatized
   - Similar to stemmed, we remove the stopwords in the data and apply lemmatization technique to the data with the help of POS taggging

I can then look closer in which word showed up frequently in each of the processed text.

![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/processed_text.png "Processed Text")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/common_word_ex.png "Most Common Word Example")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/common_word_ex2.png "Most Common Word Example (Word Cloud)")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/common_bigram_ex.png "Most Common Bigram Example")

# Machine Learning Model

The model i choose in this project are:

- **Multinomial Naive Bayes** : since it's commonly used for text data due to it being based on the Bayes' Theorem
- **Logistic Regression** : since it's one of the simple yet powerful classification machine learning model. But because this is a multiclass classification problem, i use One vs Rest Logistic Regression.
- **Random Forest** : since it is one of the most robust machine learning model, i included Random Forest since it might be better to deal with our imbalanced classification problem.

After training each model, i  conclude that the best model we will use for this case is Multinomial Naive Bayes on Alphabet-Only Text without TF-IDF. Besides looking at the results, it's also because Multinomial Naive Bayes doesn't consume as much time as the other models we're using.

![Alt Text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/conf_title.png "Confusion Matrix Title")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/conf_review.png "Confusion Matrix Review")
![alt text](https://github.com/theis19/Sentiment-Analysis-E-Commerce/blob/master/images/conf_comb.png "Confusion Matrix Combination")

## Conclusion

- The model i chose for this project ended up being **Multinomial Naive Bayes**
- The processed text data best for this ended up being the **Alphabet-only Lowered Text Data**
- Due to Imbalanced Dataset, our model had a hard time predicting the minority classes and thus only able to achieve a **Balanced Accuracy Score of 60-70%**

## Possible Solutions

Because only reaching a 60-70% is not that good, i believe there should be other things we could do to improve this model, but because lack of time, computation power, and experience i couldn't do it. Those things are:

- Do a lexicon-based approach, where instead of doing bag of words, train the machine learning model on the meaning of the words too, to achieve better understanding of the text data
- Use Oversampling technique such as SMOTE to deal with the imbalanced dataset
- Use Deep Learning
