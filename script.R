## libraries
library(randomForest)
library(rpart) #for decision tree
library(rattle) # for plotting rpart tree
library(rpart.plot) # for plotting rpart tree
library(RColorBrewer) # for plotting rpart tree

## functions

# args: two instances (rows)
# returns: the hamming distance between two instances 
# eg: get.hamming.distance(dataset[1, ], dataset[3, ])
get.hamming.distance <- function(row.1, row.2) {
  cols <- 2:(ncol(row.1))
  length(which(row.1[, cols] != row.2[, cols])) 
} 

# args: an instance (row) and a distance limit (max.ham):
# returns: dataframe of all possible instances with hamming distance from row < max.ham
# eg: upto.hamming(dataset[1, ], 5)
upto.hamming <- function(row, max.ham) {
  all.instances <- hamming(row, 1)
  for (i in 2:max.ham) {
    all.instances <- rbind(all.instances, hamming(row, i))
  } 
  all.instances
}

# args: an instance (row),
#       a hamming distance (distance), 
#       the column containing the classification (asumes labeled data)
# returns: a dataframe of all posible instances with hamming distance from row == distance
# eg: hamming(dataset[1, ], lapply(dataset, levels), 2, 1)
hamming = function(row, distance = 1, classification.col = 1) {
  options = lapply(row, levels)
  perturb.cols = combn((1:ncol(row))[-classification.col], distance, simplify=FALSE)
  perturbed.dataframe = data.frame(row)
  
  for (combination in perturb.cols) {
    combos = expand.grid(options[combination])
    tmp.row = row
    for (i in 1:nrow(combos)) {
      tmp.row[combination] = combos[i,]
      if (sum(tmp.row[combination] == row[combination]) == 0) {
        perturbed.dataframe = rbind(perturbed.dataframe, data.frame(tmp.row))
      }
    }
  }
  
  perturbed.dataframe
}

# args: an instance from a dataset (instance)
# produces a plot I guess
make.a.plot <- function(instance) {
  results <- data.frame(train.d=0, test.d=0, accuracy=0)
  max.ham <- 3 
  i <- 1 
  for (train.d in 1:max.ham) {
    train.set <- upto.hamming(instance, train.d) 
    forest <- randomForest(formula=formula, data=train.set) 
    for (test.d in 1:max.ham) {
      test.set <- hamming(test.d) 
      rf.predictions = predict(forest, test.set) 
      rf.accuracy = mean(rf_predictions == test.set[1, ])
      results[i, ] <- c(train.d, test.d, rf.accuracy) 
      i <- i+1 
     } 
  } 
  plt <- ggplot(results) + 
    geom_line(aes(test.d, accuracy), group=train.d)
  print(plt)
}


## load the dataset
url_string <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
dataset <- read.csv(url(url_string))
class_col <- 1
class_col_name <- colnames(dataset)[class_col]
formula=as.formula(paste(class_col_name, "~."))

## coerce columns into factors
dataset <- data.frame(lapply(dataset, factor))

## get datset levels
dataset_levels <- lapply(dataset, levels) 

## split into training and test data
num_rows <- nrow(dataset)
split_row <- floor(num_rows/2)

# training_data <- a random sample with size = 1/2 the dataset
training_data <- sample(1:num_rows, size=num_rows-split_row)

# training_data <- the whole dataset
training_data <- 1:num_rows

## train a random forest on the training data
random_forest <- randomForest::randomForest(formula=formula, data=dataset, subset=training_data)

## check rf accuracy on test data
rf_predictions <- predict(random_forest, dataset[-training_data,], type="class")
rf_accuracy <- mean(rf_predictions == dataset[,class_col][-training_data])

make.a.plot(dataset[1,])

## train tree
#tree <- rpart::rpart(formula=formula, data=dataset, subset=training_data, method="class")
#tree_predictions <- predict(tree, dataset[-training_data,], type="class")
#tree_accuracy <- mean(tree_predictions == dataset[,class_col][-training_data])

## incomplete, unused
#get.folds <- function(no.rows, no.folds) { 
#  fold.size <- as.integer(no.rows/no.folds) 
#  folds <- list() 
#  for (i in 1:(no.folds-1)) {
#    available <- 1:no.rows 
#    for (prev.fold %in% folds) {
#      available <- setdiff(available, prev.fold) 
#    } 
#      #new.fold <- subset( 
#       # hamish stopped writing the function here
#  }
#}
  