

library(quanteda)
library(stm)
library(tm)
library(NLP)
library(openNLP)
library(ggplot2)
library(ggdendro)
library(cluster)
library(fpc)  

#load data from DonaldTrumpSpeech.csv

url<-"/Users/anushiarora/Desktop/Resume website/resumeinfo.csv"
precorpus<- read.csv(url, 
                     header=TRUE, stringsAsFactors=FALSE)
dim(precorpus) 
names(precorpus)  
head(precorpus)
str(precorpus)

# Creating a corpus for speech
library(quanteda)
require(quanteda)

speechcorpus<- corpus(precorpus$Explanation,
                      docnames=precorpus$Work)
#explore the corpus of speech

names(speechcorpus)   
summary(speechcorpus)  
head(speechcorpus)

#Generate DFM
corpus<- toLower(speechcorpus, keepAcronyms = FALSE) 
cleancorpus <- tokenize(corpus, 
                        removeNumbers=TRUE,  
                        removePunct = TRUE,
                        removeSeparators=TRUE,
                        removeTwitter=FALSE,
                        verbose=TRUE)


stop_words <- c("and", "by", "the", "on", "of", "purchases", "being", "made","using", "are", "performance", "to","base", "salaries",
                "improve", "future", "improvise", "using", "homework", "flow", "department's", "exams", "500+", "students", "every","week","use", "ranking", "restaurant", "univers", "base", "salari", "rank", "universities", "use", "valu", "ms","grade", "young", "sport", "on-time",
                "keep", "track", "__it", "__for", "per", "Departmental", "with", "providing", "faculty/staff", "past", "attendance",
                "in", "without", "lot", "wit", "line", "nov", "didn", "set", "abl", "would'v", "__we","use", "doctor"
                "one", "year", "s", "t", "know", "also", "just", "like", "can", "need", "number", "say", "includ",
                "new", "go","now", "look", "back", "take", "thing", "even", "ask", "seen", "said", "put", "day",
                "anoth", "come", "use", "total", "restaurant", "project" , "wwechat", "create", "company", "trump",
                "happen", "place", "thank", "ve", "get", "much", "compani", "via", "us", "value", "speech", "identifi")
stop_words <- tolower(stop_words)

dfm<- dfm(cleancorpus, toLower = TRUE, 
          ignoredFeatures = c(stop_words, stopwords("english")),
          verbose=TRUE, 
          stem=TRUE)

# Reviewing top features

topfeatures(dfm, 200)   

#dfm with trigrams

cleancorpus1 <- tokenize(corpus, 
                         removeNumbers=TRUE,  
                         removePunct = TRUE,
                         removeSeparators=TRUE,
                         removeTwitter=FALSE, 
                         ngrams=3, verbose=TRUE)

dfm.trigram<- dfm(cleancorpus1, toLower = TRUE, 
                  ignoredFeatures = c(stop_words, stopwords("english")),
                  verbose=TRUE, 
                  stem=FALSE)
topfeatures.trigram<-topfeatures(dfm.trigram, n=50)
topfeatures.trigram

# Wordcloud for Speech

library(wordcloud)
set.seed(140)   #keeps cloud' shape fixed
dark2 <- brewer.pal(8, "Dark2")   
freq<-topfeatures(dfm, n=200)


wordcloud(names(freq), 
          freq, max.words=50, 
          min.freq = 1,
          rot.per = 0.45,
          colors=brewer.pal(8, "Dark2"))


#running topics
temp<-textProcessor(documents=precorpus$Explanation, metadata = precorpus)
names(temp)  # produces:  "documents", "vocab", "meta", "docs.removed" 
meta<-temp$meta
vocab<-temp$vocab
docs<-temp$documents
out <- prepDocuments(docs, vocab, meta)
docs<-out$documents
vocab<-out$vocab
meta <-out$meta

prevfit <-stm(docs , vocab , 
              K=3, 
              verbose=TRUE,
              data=meta, 
              max.em.its=25)

topics <-labelTopics(prevfit , topics=c(1:3))
topics   

plot.STM(prevfit, type="summary")
plot.STM(prevfit, type="perspectives", topics = c(1,3))
plot.STM(prevfit, type="perspectives", topics = c(1,2))
plot.STM(prevfit, type="perspectives", topics = c(2,3))

# to aid on assigment of labels & intepretation of topics

mod.out.corr <- topicCorr(prevfit)  #Estimates a graph of topic correlations
plot.topicCorr(mod.out.corr)

### Advanced method for Topic Modeling
#######################################


library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(openNLP)

#load .csv file with news articles

url<-"/Users/anushiarora/Desktop/Resume website/resumeinfo.csv"
precorpus<- read.csv(url, 
                     header=TRUE, stringsAsFactors=FALSE)

#passing Full Text to variable news_2015
speech<-precorpus$Explanation


#Cleaning corpus
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words <- c(stop_words, "and", "by", "the", "on", "of", "purchases", "being","based","salary", "made","using", "are", "performance", "to","base", "salaries","a","an","was","for","it", "were","as","be",
                "their", "there", "from", "which","improve", "future", "improvise", "using", "homework", "flow", "department's", "exams", "500+", "students", "every","week","use", "ranking", "restaurant", "univers", "base", "salari", "rank", "universities", "use", "valu", "ms","grade", "young", "sport", "on-time","50k")
stop_words <- tolower(stop_words)


speech <- gsub("'", "", speech) # remove apostrophes
speech <- gsub("[[:punct:]]", " ", speech)  # replace punctuation with space
speech <- gsub("[[:cntrl:]]", " ", speech)  # replace control characters with space
speech <- gsub("^[[:space:]]+", "", speech) # remove whitespace at beginning of documents
speech <- gsub("[[:space:]]+$", "", speech) # remove whitespace at end of documents
speech <- gsub("[^a-zA-Z -]", " ", speech) # allows only letters
speech <- tolower(speech)  # force to lowercase

## get rid of blank docs
speech <- speech[speech != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(speech, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

#############
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

news_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = news_for_LDA$phi, 
                   theta = news_for_LDA$theta, 
                   doc.length = news_for_LDA$doc.length, 
                   vocab = news_for_LDA$vocab, 
                   term.frequency = news_for_LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)







