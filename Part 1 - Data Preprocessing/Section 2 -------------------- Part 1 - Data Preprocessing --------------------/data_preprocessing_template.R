# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# This is how we are going to fill missing data, by filling with mean of the columns data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                    dataset$Age)
# same we do for salary
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#Encode Categorial data
dataset$Country = factor(dataset$Country,
                         levels=c('France','Spain','Germany'),labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         levels=c('No','Yes'),labels = c(0,1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)

split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE) # if split function returns true it will go to training set 
test_set = subset(dataset, split == FALSE) # if split function returns true it will go to test set 



# Feature Scaling

# As most of the model in machine learning are based upon Euclidean Distances
# which is distance between two points on the graph
# sqrt of (x2-x1)2 + (y2-y1)2
# So, we need to normalise data with large number with much wider range, and 
# in our case it is "Salary" column as Euclidean distance gets dominated as compared 
# "Age" column

# Now as we had done factor during encoding the result is not numeric
# so excuting following will give error
# training_set = scale(training_set)
# test_set = scale(test_set)
# Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric

# so we scale only Age and country by selecting only these columns
# As in R index starts at 1
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])