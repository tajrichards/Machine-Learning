# Taj Richards - some code from my Machine Learning course.
# We decided to create a spam filter from a dataset of text messages via Naive Bayesian model.

## Classification using Naive Bayes 

##--------------------------------------------------------
## Install necessary packages
##--------------------------------------------------------
install.packages("tm")            # text prep
install.packages("SnowballC")     # stemming
install.packages("e1071")         # naive bayes modeling
install.packages("gmodels")       # cross tabulation of results

##--------------------------------------------------------
## Load data
##--------------------------------------------------------

# read the sms data into the sms data frame
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

##--------------------------------------------------------
## Prepare data
##--------------------------------------------------------
# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable
table(sms_raw$type)

# build a corpus using the text mining (tm) package
# VectorSource - turn each element into a document
# Corpus - turn object into a collection of documents
library(tm)
sms_corpus <- Corpus(VectorSource(sms_raw$text))


# clean up the corpus using tm_map()
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# stem word variants (e.g. learn, learned, learning)
library(SnowballC)
corpus_clean <- tm_map(corpus_clean, stemDocument)

# examine the clean corpus 
sms_raw[1,]     # spam label
inspect(corpus_clean[1]) # text not including the spam label

# create a document-term sparse matrix
#corpus_clean <- tm_map(corpus_clean, PlainTextDocument) 

# corpus_clean <- Corpus(VectorSource(corpus_clean))
sms_dtm <- DocumentTermMatrix(corpus_clean)

# creating training and test datasets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_train_labels <- sms_raw$type[1:4169]
sms_test_labels <- sms_raw$type[4170:5559]


# check that the proportion of spam is similar
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))


# indicator features for frequent words
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
head(sms_freq_words, 20)

sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test  <- sms_dtm_test[, sms_freq_words]

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_dtm_freq_train, 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test,  2, convert_counts)

##--------------------------------------------------------
## Model data
##--------------------------------------------------------
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace=1)
sms_classifier # this can be very long

# explore classifier
model <- as.list(sms_classifier)
length(model$tables) # number of words
names(model$tables)[1:200] # first 200 words
model$tables$collect  # probabilities for the word 'collect'

##--------------------------------------------------------
## Evaluate performance
##--------------------------------------------------------
sms_test_pred <- predict(sms_classifier, sms_test)

library(caret)
confusionMatrix(sms_test_pred, sms_test_labels, positive = "spam")



##--------------------------------------------------------
## Going further
##--------------------------------------------------------
# note: for conditional probabilities use
sms_test_prob <- predict(sms_classifier, sms_test, type="raw")
round(head(sms_test_prob), 6)
