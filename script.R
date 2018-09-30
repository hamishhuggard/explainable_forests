## libraries

library(randomForest)
library(rpart) #for decision tree
library(rattle) # for plotting rpart tree
library(rpart.plot) # for plotting rpart tree
library(RColorBrewer) # for plotting rpart tree
library(plyr) # for rbind.fill
library(ggplot2) # for ggplot

## functions

get.hamming.distance <- function(row.1, row.2) {
  # args: two instances (rows)
  # returns: the hamming distance between two instances 
  # eg: get.hamming.distance(dataset[1, ], dataset[3, ])
  cols <- 2:(ncol(row.1))
  length(which(row.1[, cols] != row.2[, cols])) 
} 


upto.hamming <- function(row, max.ham) {
  # args: an instance (row) and a distance limit (max.ham):
  # returns: dataframe of all possible instances with hamming distance from row < max.ham
  # eg: upto.hamming(dataset[1, ], 5)
  all.instances <- hamming(row, 1)
  for (i in 2:max.ham) {
    all.instances <- rbind(all.instances, hamming(row, i))
  } 
  all.instances
}

hamming.old <- function(row, distance = 1, classification.col = 1) {
  # args: an instance (row),
  #       a hamming distance (distance), 
  #       the column containing the classification (asumes labeled data)
  # returns: a dataframe of all posible instances with hamming distance from row == distance
  # eg: hamming(dataset[1, ], lapply(dataset, levels), 2, 1)
  options <- lapply(row, levels)
  perturb.cols <- combn((1:ncol(row))[-classification.col], distance, simplify=FALSE)
  perturbed.dataframe <- data.frame(row)
  
  for (combination in perturb.cols) {
    combos <- expand.grid(options[combination])
    tmp.row <- row
    for (i in 1:nrow(combos)) {
      tmp.row[combination] <- combos[i,]
      if (sum(tmp.row[combination] == row[combination]) == 0) {
        perturbed.dataframe <- rbind(perturbed.dataframe, data.frame(tmp.row))
      }
    }
  }
  
  perturbed.dataframe
}

hamming = function(row, distance = 1, classification.col = 1) {
  perturb.cols = combn((1:ncol(row))[-classification.col], distance, simplify=FALSE)
  options <- lapply(row, levels)
  # Remove values which occur in row
  for (i in 1:length(row)) {
    options[[i]] <- options[[i]][which(options[[i]]!=row[, i])]
  }
  result <- row
  for (combination in perturb.cols) {
    combos <- expand.grid(options[combination])
    combo.frame <- data.frame(row)
    combo.frame[1:nrow(combos), ] <- row
    combo.frame[, combination] <- combos
    result <- rbind(result, combo.frame)
  }
  result[2:nrow(result), ]
}

AddClassifications <- function(new.dataset) {
  # globals: random.forest, dataset
  class.col = attr(dataset, "class.col")
  rf.predictions <- predict(random.forest, new.dataset, type="class")
  new.dataset[,class.col] <- rf.predictions
  return(new.dataset)
}

GetHammingRings <- function(instance, max.ham) {
  hamming.rings <- list()
  for (i in 1:max.ham) {
    h <- hamming(instance, distance = i)
    h <- AddClassifications(h)
    hamming.rings[[i]] <- h
  }
  return(hamming.rings)
}

GetHammingCircles <- function(hamming.rings) {
  hamming.circles <- list()
  for (i in 1:length(hamming.rings)) {
    training.data <- plyr::rbind.fill(hamming.rings[1:i])
    hamming.circles[[i]] <- training.data
  }
  return(hamming.circles)
}

TrainTrees <- function(hamming.circles) {
  # globals: dataset
  class.col <- attr(dataset, "class.col")
  class.formula <- attr(dataset, "formula")
  trees <- list()
  for (i in 1:length(hamming.circles)) {
    training.data <- hamming.circles[[i]]
    if (length(unique(training.data[,class.col])) == 1) {
      trees[[i]] = NULL
    } else {
      trees[[i]] <- rpart::rpart(formula=class.formula, data=training.data, method="class")
    }
  }
  return(trees)
}

EvaluateTrees <- function(trees, hamming.circles) {
  class.col <- attr(dataset, "class.col")
  results <- data.frame(train.d=0, test.d=0, accuracy=0)
  row.num = 1
  for (i in 1:length(hamming.circles)) {
    tree = trees[[i]]
    for (j in 1:length(hamming.circles)) {
      if (!is.null(tree)) {
        test.data <- hamming.circles[[j]]
        tree.predictions <- predict(tree, test.data, type="class")
        tree.accuracy <- mean(tree.predictions == test.data[,class.col])
        results[row.num, ] <- c(i, j, tree.accuracy) 
      } else {
        results[row.num, ] <- c(i, j, 1)
      }
      row.num <- row.num+1
    }
  }
  return(results)
}

PlotResults <- function(results) {
  plt <- ggplot(results) + 
    geom_line(aes(results$train.d, results$accuracy), group=results$test.d)
  print(plt)
}

make.a.plot <- function(results) {
  plt <- ggplot(results) + 
    geom_line(aes(results$test.d, results$accuracy), group=results$train.d)
  print(plt)
}

GetDataset <- function(url_string, class.col = 1) {
  dataset <- read.csv(url(url_string))
  dataset <- data.frame(lapply(dataset, factor))  # coerce columns into factors
  attr(dataset, "class.col") <- class.col
  attr(dataset, "class.colname") <- colnames(dataset)[class.col]
  attr(dataset, "formula") <- as.formula(paste(attr(dataset, "class.colname"), "~."))
  attr(dataset, "levels") <- lapply(dataset, levels)  # get dataset levels
  return(dataset)
}


GetTrainingData <- function(dataset, sample.type = "random") {
  if (sample.type == "random") {
    # training_data <- a random sample with size = 1/2 the dataset
    num.rows <- nrow(dataset)
    split.row <- floor(num.rows/2)
    training.data <- sample(1:num.rows, size=num.rows-split.row)
  } else if (sample.type== "whole") {
    # training_data <- the whole dataset
    training.data <- 1:nrow(dataset)
  }
  return(training.data)
}

## load the dataset

url_string <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
dataset = GetDataset(url_string)

training.data <- GetTrainingData(dataset, "random")

## train a random forest on the training data
random.forest <- randomForest::randomForest(formula=attr(dataset, "formula"), data=dataset, subset=training.data)

## check rf accuracy on test data
rf.predictions <- predict(random.forest, dataset[-training.data,], type="class")
rf.accuracy <- mean(rf.predictions == dataset[,attr(dataset, "class.col")][-training.data])

instance = dataset[1,]
max.ham = 5

hamming.rings <- GetHammingRings(instance, max.ham)
hamming.circles <- GetHammingCircles(hamming.rings)
trees <- TrainTrees(hamming.circles)
results <- EvaluateTrees(trees, hamming.circles)
PlotResults(results)
