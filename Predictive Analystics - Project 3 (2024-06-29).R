library(recommenderlab)
library(naniar)
library(dplyr)
library(tidyr)
library(tidyverse)
library(arulesViz)

# Load the data
Books = read.csv("Books.csv", stringsAsFactors = T, na.strings = c("","NA"))
Rate = read.csv("Ratings.csv", stringsAsFactors = T, na.strings = c("","NA"))

#Missing data check
miss_var_summary(Books)
miss_var_summary(Rate)

#Remove duplicates (books with the same 'title')
Books = Books %>% distinct(title, .keep_all = TRUE)
Rate = Rate %>% distinct(book_id, user_id, .keep_all = TRUE)

#Filtering ratings: keep only the ratings that have a a reference in the Books dataset
Rate = Rate[which(Rate$book_id %in% Books$book_id),]

#Calculate the average rating of books
average_rating_1 <- mean(Rate$rating)

#Calculate the top 10 books
top_rated_books <- Books %>%
  arrange(desc(average_rating)) %>%
  select(title, authors, average_rating) %>% 
  top_n(10) 

top_rated_books

#calculate top 10 read books
top_read_books <- Books %>%
  arrange(desc(ratings_count)) %>%
  select(title, authors, average_rating, ratings_count ) %>% 
  top_n(10) 

top_read_books

#Plot the relationship of avg rating vs total reviews

ggplot(Books, aes(x = average_rating, y = ratings_count)) +
  geom_point() +  # Add points
  geom_smooth(method = "lm", se = FALSE, color = "blue") +  # Add linear trend line
  labs(x = "Average Rating", y = "Total Reviews") +  # Label axes
  ggtitle("Relationship between Average Rating and Total Reviews") +  # Add title
  theme_minimal()  # Optional: Use a minimal theme for the plot


#Calculate the top 10 authors by average rating (min 100 reviews)
top_authors <- Books %>%
  group_by(authors) %>%
  summarize(avg_rating = mean(average_rating, na.rm = TRUE)) %>%
  arrange(desc(avg_rating)) %>%
  top_n(10)

top_authors

#Create a table of number of reviews per user
user_ratings_summary <- Rate %>%
  group_by(user_id) %>%
  summarize(total_ratings = n())

#Filter for top 20 users with at least 100 reviews
top_20_user_ratings_summary <- Rate %>%
  group_by(user_id) %>%
  summarize(total_ratings = n()) %>%
  arrange(desc(total_ratings)) %>%
  top_n(20)

# Filter so that only users with 100 reviews 
users_with_100_ratings <- user_ratings_summary %>%
  filter(total_ratings >= 100) %>%
  select(user_id) 

filtered_ratings <- Rate %>%
  filter(user_id %in% users_with_100_ratings$user_id)

#Creating utility matrix (users as rows, books as columns, ratings as entries)
#rows/columns labels are sorted according to user_id & book_id 
ratingmat = spread(select(filtered_ratings, user_id, book_id, rating), book_id, rating)

# Remove first column and convert to matrix
ratingmat = as.matrix(ratingmat[,-1])

# Add user_id's as row labels and books titles as column labels
user_labels = sort(unique(filtered_ratings$user_id))
book_labels <- Books$title[Books$book_id %in% as.numeric(colnames(ratingmat))]
dimension_names = list(user_id = user_labels, book_id = book_labels)
dimnames(ratingmat) = dimension_names
ratingmat[1:4,1:4] #Inspecting the rating matrix

#Converting rating/utility matrix to realRatingMatrix to use with recommenderlab functions
ratingMatrix = as(ratingmat, "realRatingMatrix")


#Inspection of rating matrix: 
#Visual inspection of a small portion of the rating matrix (first 50 users, first 100 books)
image(ratingMatrix[1:50,1:100], main = "Raw Ratings")

#Number of submitted reviews (for each of the first 10 users)
rowCounts(ratingMatrix[1:10,])

#Number of users who have submitted at least 200 reviews
length(rowCounts(ratingMatrix)>=500)

#Average rating submitted (for each of the first 10 users)
avg_rating_by_user = as.data.frame(rowMeans(ratingMatrix)) #automatically ignores NA's

########### Model evaluation

############ Comparing multiple models at the same time

#Example 3:
#Defining good rating (i.e. a liked item) as 4 stars or more
set.seed(123)
scheme3 = evaluationScheme(ratingMatrix, method = "split",
                           train=0.85, given=100, goodRating=4)

#Setting up a list of models to compare
algorithms = list("UBCF_20_C" =   list(name = "UBCF", param = list(nn = 20, normalize = "center", method = "Cosine")),
                  "UBCF_25_P" = list(name = "UBCF", param = list(nn = 25, method="pearson")),
                  "UBCF_25" = list(name = "UBCF", param = list(nn = 25)),
                  "UBCF_30" = list(name = "UBCF", param = list(nn = 30)),
                  "IBCF_25" = list(name = "IBCF", param = list(k = 25)),
                  "IBCF_35_P" = list(name = "IBCF", param = list(k = 35, method="pearson")))

#Evaluating models in terms of their predicted ratings:
resultsRMSE = evaluate(x=scheme3, method=algorithms, type="ratings")
avg(resultsRMSE)

#Plot the RMSE, MSE and MAE of the 5 models being compared
pdf("RMSEplot.pdf", width = 6, height = 4)
plot(resultsRMSE, legend.position = "topright", cex = 0.5, ylim = c(0,5))
dev.off()

plot(resultsRMSE, legend.position = "topright", cex = 0.5, ylim = c(0,5))

#Evaluating models in terms of their top_N recommended books:
resultsROC = evaluate(x=scheme3, method=algorithms, n=seq(10,100,10))

#ROC plot
plot(resultsROC, annotate = T, legend = "topleft", main="ROC Curve")

#Precision/Recall plot
plot(resultsROC, "prec/rec", annotate = T, legend = "bottomright", main="Precision/Recall")


##################### Recommendations for First Reviewer

### User based Collaborative Filtering
#Recommendation for First User using User Based Collaborative Filtering
recUBCF_25 = Recommender(ratingMatrix[-1,],"UBCF",
                         param = list(nn = 25))

#Top 5 recommendations for first user
predUBCFTOP = predict(recUBCF_25,ratingMatrix[1,],n=5)
as(predUBCFTOP,'list')

#recommendations for the first user (predicted ratings of the books not rated by the user)
predUBCF = predict(recUBCF_25,ratingMatrix[1,],type="ratings")
as(predUBCF,'list')

#Optional: storing predicted ratings in a data frame
predictedUBCF_ratings = as(predUBCF,'data.frame') 

### Item Based Collaborative Filtering 
#Recommendation for First User using Item Based Collaborative Filtering
recIBCF_25 = Recommender(ratingMatrix[-1,],"IBCF", param = list(k = 25))

#Top 5 recommendations for first user
predIBCFTOP = predict(recIBCF_25,ratingMatrix[1,],n=5)
as(predIBCFTOP,'list')

#recommendations for the first user (predicted ratings of the books not rated by the user)
predIBCF = predict(recUBCF_25,ratingMatrix[1,],type="ratings")
as(predIBCF,'list')

#Optional: storing predicted ratings in a data frame
predictedIBCF_ratings = as(predIBCF,'data.frame') 



################## Association Rules 

#Remove NAs by replacing them with 0's
ratingmat_arules = ratingmat
ratingmat_arules[is.na(ratingmat_arules)] <- 0
rating_arules <- ratingmat_arules > 0
rating_arules <- as.numeric(rating_arules)
dim(rating_arules) <- dim(ratingmat)
dimnames(rating_arules) = dimension_names
rating_arules

rating_trans <- as(rating_arules, "transactions")

#Get rules: when running apriori(), include the minimum support, minimum confidence
rules = apriori(rating_trans, parameter = list(supp = 0.03, conf = 0.95))
inspect(rules)

#Let's filter by lift > 1 
#(among the rules with support>0.05 and confidence>0.9, only show the ones with lift>1)
inspect(subset(rules, lift>11.5)) 

## Top 3 rules by lift
inspect(head(sort(rules, by="lift"),3))

## Visualize Top 3 rules by lift
Top3_Rules = (head(sort(rules, by="lift"),3)) 
#make interactive plots using engine=htmlwidget parameter in plot
plot(Top3_Rules, method="graph", engine="htmlwidget")
