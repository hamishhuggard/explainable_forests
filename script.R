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
    colnames(combo.frame) <- colnames(result)
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

PlotAndWrite <- function(results, fig.write = FALSE, fig.id = 0, fig.dir = "plots", fig.prefix = 'plot-', fig.ext = 'png', fig.width = 500, instance.n = 1) {
  PlotResults(results, instance.n)
  if (fig.write == TRUE) {
    dev.print(png, paste(fig.dir, "/", fig.prefix, fig.id, '.', fig.ext, sep = ""), width = fig.width)
  }
}

PlotResults <- function(results, instance.n=1) {
  # args: a dataframe (results), with columns train.d, accuracy and test.d
  # returns: a ggplot of accuracy against train.d, with a line for each test.d
  results[, 1] <- factor(results[, 1], levels=c('1','2','3','4','5'), ordered=TRUE)
  plt <- ggplot(results, aes(ymin = 0.0, ymax = 1.0)) +
    geom_line(aes(test.d, accuracy, colour = train.d, group = train.d)) +
    geom_point(aes(test.d, accuracy, colour = train.d, group = train.d)) +
    labs(title = paste("Instance",instance.n), x = "Testing HD", 
         y = "Accuracy", color = "Training HD")
  print(plt)
}

GetDataset <- function(url_string, class.col = 1, header = FALSE, local = FALSE) {
  if (local == TRUE) {
    string <- url_string
  } else {
    string <- url(url_string)
  }
  dataset <- read.csv(string, header = header)
  dataset <- data.frame(lapply(dataset, factor))  # coerce columns into factors
  colnames(dataset) <- col.names
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
  } else if (sample.type == "preset") {
    # training.data <- a previously generated random sample
    # x = as.integer(eval(parse(text=paste("c(", paste(c(training.data), collapse=", "), ")", sep=""))))
    training.data <- as.integer(c(105, 204, 143, 22, 50, 275, 201, 242, 125, 147, 40, 121, 103, 108, 246, 43, 137, 253, 271, 82, 224, 178, 160, 145, 221, 190, 202, 274, 208, 73, 197, 188, 61, 14, 236, 270, 161, 211, 195, 209, 130, 196, 124, 92, 166, 133, 59, 95, 7, 126, 141, 264, 33, 165, 3, 282, 265, 268, 181, 180, 157, 272, 255, 79, 42, 20, 169, 56, 128, 88, 71, 78, 261, 244, 182, 53, 194, 32, 146, 38, 237, 142, 41, 280, 104, 123, 94, 85, 192, 216, 139, 200, 240, 24, 155, 18, 247, 9, 70, 64, 11, 238, 223, 215, 259, 257, 173, 131, 263, 156, 21, 46, 154, 170, 279, 77, 167, 127, 213, 52, 135, 84, 256, 25, 45, 219, 254, 63, 69, 231, 245, 233, 149, 29, 248, 49, 109, 83, 58, 153, 222, 140, 159))
  }
  return(training.data)
}

## load the dataset

url_string <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
url_string <- "breast-cancer.data"
col.names <- c("Class", "age", "menopause", "tumor.size", "inv.nodes", "node.caps", "deg.malig", "breast", "breast.quad", "irradiat")
dataset <- GetDataset(url_string, header = FALSE, local = TRUE)

#training.data <- GetTrainingData(dataset, "random")
#training.data <- GetTrainingData(dataset, "whole")
training.data <- GetTrainingData(dataset, "preset")

## train a random forest on the training data
random.forest <- randomForest::randomForest(formula=attr(dataset, "formula"), data=dataset, subset=training.data)

## check rf accuracy on test data
rf.predictions <- predict(random.forest, dataset[-training.data,], type="class")
rf.accuracy <- mean(rf.predictions == dataset[,attr(dataset, "class.col")][-training.data])

max.ham <- 5

for (num in 1:12) {
  instance <- dataset[num,]
  hamming.rings <- GetHammingRings(instance, max.ham)
  hamming.disks <- GetHammingDisks(hamming.rings)
  trees <- TrainTrees(hamming.disks)
  results <- EvaluateTrees(trees, hamming.rings)
  PlotAndWrite(results, fig.write = TRUE, fig.prefix = "plot-whole-pretty-", fig.id = num, instance.n = num)
}

## get training.data with only one label to test handling of that case
# training.data <- hamming.disks[[i]][hamming.disks[[i]][,1] == unique(hamming.disks[[i]][,1])[[1]],]
