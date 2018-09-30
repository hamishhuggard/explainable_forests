## libraries

library(randomForest)
library(rpart) #for decision tree
library(rattle) # for plotting rpart tree
library(rpart.plot) # for plotting rpart tree
library(RColorBrewer) # for plotting rpart tree
library(plyr) # for rbind.fill
library(ggplot2) # for ggplot

## functions

GetHammingDistance <- function(row.1, row.2) {
  # args: two instances (rows)
  # returns: the hamming distance between two instances
  # eg: GetHammingDistance(dataset[1, ], dataset[3, ])
  cols <- 2:(ncol(row.1))
  length(which(row.1[, cols] != row.2[, cols]))
}


UptoHamming <- function(row, max.ham) {
  # args: an instance (row) and a distance limit (max.ham):
  # returns: dataframe of all possible instances with hamming distance from row < max.ham
  # eg: UptoHamming(dataset[1, ], 5)
  all.instances <- Hamming(row, 1)
  for (i in 2:max.ham) {
    all.instances <- rbind(all.instances, Hamming(row, i))
  }
  all.instances
}

HammingOld <- function(row, distance = 1) {
  # args: an instance (row),
  #       a hamming distance (distance),
  # returns: a dataframe of all posible instances with hamming distance from row == distance
  # eg: HammingOld(dataset[1, ], 5)
  classification.col <- attr(row, "class.col")
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

Hamming <- function(row, distance = 1) {
  # args: an instance (row),
  #       a hamming distance (distance),
  # returns: a dataframe of all posible instances with hamming distance from row == distance
  # eg: Hamming(dataset[1, ], 5)
  classification.col <- attr(row, "class.col")
  perturb.cols <- combn((1:ncol(row))[-classification.col], distance, simplify=FALSE)
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
  # args: a dataset
  # globals: random.forest, dataset
  # returns: a dataset with labels predicted by random.forest
  # eg: AddClassifications(test.data)
  class.col <- attr(dataset, "class.col")
  rf.predictions <- predict(random.forest, new.dataset, type="class")
  new.dataset[,class.col] <- rf.predictions
  return(new.dataset)
}

GetHammingRings <- function(instance, max.ham) {
  # args: a row from a dataset (instance)
  #       a hamming distance limit (max.ham)
  # returns: a list with max.ham items, where list[[i]]
  #   is a dataframe containing all instances exactly
  #   Hamming distance i from instance
  hamming.rings <- list()
  for (i in 1:max.ham) {
    h <- Hamming(instance, distance = i)
    h <- AddClassifications(h)
    hamming.rings[[i]] <- h
  }
  return(hamming.rings)
}

GetHammingDisks <- function(hamming.rings) {
  # args: a list (hamming.rings) where list[[i]]
  #   is a dataframe containing all instances exactly
  #   Hamming distance i from instance
  # returns: a list of length(hamming.rings), where list[[i]]
  #   is a dataframe containing all instances at most
  #   Hamming distance i from instance
  hamming.disks <- list()
  for (i in 1:length(hamming.rings)) {
    training.data <- plyr::rbind.fill(hamming.rings[1:i])
    hamming.disks[[i]] <- training.data
  }
  return(hamming.disks)
}

TrainTrees <- function(hamming.disks) {
  # args: a list (hamming.disks), where list[[i]]
  #   is a dataframe containing all instances at most
  #   Hamming distance i from an initial instance
  # globals: dataset
  # returns: a list of length(hamming.disks), where list[[i]]
  #   is either a rpart tree trained on hamming.disks[[i]], or
  #   a factor (if hamming.disks[[i]] has only one classification)
  class.col <- attr(dataset, "class.col")
  class.formula <- attr(dataset, "formula")
  trees <- list()
  for (i in 1:length(hamming.disks)) {
    training.data <- hamming.disks[[i]]
    if (length(unique(training.data[,class.col])) == 1) {
      trees[[i]] <- unique(training.data[,class.col]) # if there was only one label, store that
    } else {
      trees[[i]] <- rpart::rpart(formula=class.formula, data=training.data, method="class")
    }
  }
  return(trees)
}

EvaluateTrees <- function(trees, hamming.disks) {
  # args: a list (hamming.disks), where list[[i]]
  #   is a dataframe containing all instances at most
  #   Hamming distance i from an initial instance
  #       a list (trees), where list[[i]]
  #   is either a rpart tree trained on hamming.disks[[i]], or
  #   a factor (if hamming.disks[[i]] has only one classification)
  # globals: dataset
  # returns: a dataframe of results with columns train.d, accuracy and test.d
  class.col <- attr(dataset, "class.col")
  results <- data.frame(train.d=integer(0), test.d=integer(0), accuracy=double(0))
  row.num <- 1
  for (i in 1:length(hamming.disks)) {
    tree <- trees[[i]]
    rf.prediction <- predict(random.forest, instance, type="class")
    if (class(tree) == "rpart") {
      tree.prediction <- predict(tree, instance, type="class")
      tree.accuracy <- as.integer(tree.prediction == rf.prediction)
    } else if (class(tree) == "factor") {
      tree.accuracy <- mean(tree == rf.prediction)
    }
    results[row.num, ] <- c(i, 0, tree.accuracy)
    row.num <- row.num+1
    for (j in 1:length(hamming.disks)) {
      test.data <- hamming.disks[[j]]
      if (class(tree) == "rpart") {  # successfully trained trees have class rpart
        tree.predictions <- predict(tree, test.data, type="class")
        tree.accuracy <- mean(tree.predictions == test.data[,class.col])
      } else if (class(tree) == "factor") {  # if there was only one label, tree is a factor
        tree.accuracy <- mean(tree == test.data[,class.col])
      } else {
        print ("unexpected class for tree in EvaluateTrees")
        print (class(tree))
      }
      results[row.num, ] <- c(i, j, tree.accuracy)
      row.num <- row.num+1
    }
  }
  return(results)
}

PlotResults <- function(results) {
  # args: a dataframe (results), with columns train.d, accuracy and test.d
  # returns: a ggplot of accuracy against train.d, with a line for each test.d
  plt <- ggplot(results, aes(ymin = 0.0, xmin = 1.0)) +
    geom_line(aes(test.d, accuracy, colour = train.d, group = train.d))
  print(plt)
}

GetDataset <- function(url_string, class.col = 1, header = FALSE) {
  dataset <- read.csv(url(url_string), header = header)
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
dataset <- GetDataset(url_string, header = FALSE)

training.data <- GetTrainingData(dataset, "random")

## train a random forest on the training data
random.forest <- randomForest::randomForest(formula=attr(dataset, "formula"), data=dataset, subset=training.data)

## check rf accuracy on test data
rf.predictions <- predict(random.forest, dataset[-training.data,], type="class")
rf.accuracy <- mean(rf.predictions == dataset[,attr(dataset, "class.col")][-training.data])

instance <- dataset[1,]
max.ham <- 5

hamming.rings <- GetHammingRings(instance, max.ham)
hamming.disks <- GetHammingDisks(hamming.rings)
trees <- TrainTrees(hamming.disks)
results <- EvaluateTrees(trees, hamming.rings)
PlotResults(results)

## get training.data with only one label to test handling of that case
# training.data <- hamming.disks[[i]][hamming.disks[[i]][,1] == unique(hamming.disks[[i]][,1])[[1]],]
