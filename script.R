library(randomForest)
library(rpart) #for decision tree
library(rattle) # for plotting rpart tree
library(rpart.plot) # for plotting rpart tree
library(RColorBrewer) # for plotting rpart tree
library(colorspace)
library(plyr) # for rbind.fill
library(ggplot2) # for ggplot

set.seed(42)

## functions

GetHammingDistance <- function(class.col, instance.1, instance.2) {
  # args: class.col (column number with the classification) two instances (rows)
  # returns: the hamming distance between two instances
  # eg: GetHammingDistance(dataset[1, ], dataset[3, ])
  cols <- 1:(ncol(instance.1))
  cols <- cols[cols != class.col]
  length(which(instance.1[, cols] != instance.2[, cols]))
}


UptoHamming <- function(instance, max.ham) {
  # args: an instance (row) and a distance limit (max.ham):
  # returns: dataframe of all possible instances with hamming distance from row < max.ham
  # eg: UptoHamming(dataset[1, ], 5)
  all.instances <- data.frame(instance)
  for (i in 1:max.ham) {
    all.instances <- rbind(all.instances, Hamming(instance, i))
  }
  all.instances
}

HammingOld <- function(instance, distance = 1) {
  # args: an instance (row),
  #       a hamming distance (distance),
  # returns: a dataframe of all posible instances with hamming distance from row == distance
  # eg: HammingOld(dataset[1, ], 5)
  classification.col <- attr(instance, "class.col")
  options <- lapply(instance, levels)
  perturb.cols <- combn((1:ncol(instance))[-classification.col], distance, simplify=FALSE)
  perturbed.dataframe <- data.frame()
  
  for (combination in perturb.cols) {
    combos <- expand.grid(options[combination])
    tmp.instance <- instance
    for (i in 1:nrow(combos)) {
      tmp.instance[combination] <- combos[i,]
      if (sum(tmp.instance[combination] == instance[combination]) == 0) {
        perturbed.dataframe <- rbind(perturbed.dataframe, data.frame(tmp.instance))
      }
    }
  }
  
  perturbed.dataframe
}

Hamming <- function(instance, distance = 1) {
  # args: an instance (instance),
  #       a hamming distance (distance),
  # returns: a dataframe of all posible instances with hamming distance from row == distance
  # eg: Hamming(dataset[1, ], 5)
  classification.col <- attr(instance, "class.col")
  perturb.cols <- combn((1:ncol(instance))[-classification.col], distance, simplify=FALSE)
  options <- lapply(instance, levels)
  # Remove values which occur in instance
  for (i in 1:length(instance)) {
    options[[i]] <- options[[i]][which(options[[i]]!=instance[, i])]
  }
  result <- instance
  for (combination in perturb.cols) {
    combos <- expand.grid(options[combination])
    combo.frame <- data.frame(instance)
    combo.frame[1:nrow(combos), ] <- instance
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

GetHammingDisks <- function(instance, hamming.rings) {
  # args: a list (hamming.rings) where list[[i]]
  #   is a dataframe containing all instances exactly
  #   Hamming distance i from instance
  # returns: a list of length(hamming.rings), where list[[i]]
  #   is a dataframe containing all instances at most
  #   Hamming distance i from instance
  hamming.disks <- list()
  for (i in 1:length(hamming.rings)) {
    training.data <- plyr::rbind.fill(hamming.rings[1:i])
    training.data <- rbind(training.data, instance)
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
      trees[[i]] <- rpart::rpart(formula=class.formula, data=training.data, method="class", minsplit=1, minbucket=1, cp=0, maxdepth=30)
    }
  }
  return(trees)
}

AssessAccuracy <- function(trees) {
  # globals: dataset, training.data
  for (i in 1:length(trees)) {
    tree <- trees[[i]]
    actual.classifications <- dataset[-training.data, attr(dataset, "class.col")]
    if (class(tree) == "rpart") {  # successfully trained trees have class rpart
      tree.predictions <- predict(tree, dataset[-training.data,], type="class")
    } else if (class(tree) == "factor") {  # if there was only one label, tree is a factor
      tree.predictions <- tree
    } else {
      print ("unexpected class for tree in EvaluateTrees")
      print (class(tree))
    }
    tree.accuracy <- mean(tree.predictions == actual.classifications)
    attr(trees[[i]], "accuracy") <- tree.accuracy
  }
  return(trees)
}

EvaluateTree <- function(tree, hamming.disks) {
  # globals: dataset, random.forest
  class.col <- attr(dataset, "class.col")
  results <- data.frame(train.d=integer(0), test.d=integer(0), accuracy=double(0))
  instance.num <- 1
  ## assess fidelity of ordinary tree
  # first evaluate the fidelity on the original instance
  rf.prediction <- predict(random.forest, instance, type="class")
  original.prediction <- predict(tree, instance, type="class")
  tree.fidelity <- as.integer(original.prediction == rf.prediction)
  results[instance.num, ] <- c(0, 0, tree.fidelity)
  instance.num <- instance.num+1
  # then evaluate the fidelity on all the hamming datasets
  for (j in 1:length(hamming.disks)) {
    hamming.data <- hamming.disks[[j]]
    hamming.predictions <- predict(tree, hamming.data, type="class")
    tree.fidelity <- mean(hamming.predictions == hamming.data[,class.col])
    results[instance.num, ] <- c(0, j, tree.fidelity)
    instance.num <- instance.num+1
  }
  return(results)
}

EvaluateTrees <- function(trees, hamming.disks) {
  # args:
  # a list (hamming.disks), where list[[i]]
  #   is a dataframe containing all instances at most
  #   Hamming distance i from an initial instance
  # a list (trees), where list[[i]]
  #   is either a rpart tree trained on hamming.disks[[i]], or
  #   a factor (if hamming.disks[[i]] has only one classification)
  # globals: dataset
  # returns: a dataframe of results with columns train.d, accuracy and test.d
  class.col <- attr(dataset, "class.col")
  results <- data.frame(train.d=integer(0), test.d=integer(0), accuracy=double(0))
  instance.num <- 1
  ## assess fidelity of the hamming trees
  for (i in 1:length(trees)) {
    tree <- trees[[i]]
    # first evaluate the fidelity on the original instance
    rf.prediction <- predict(random.forest, instance, type="class")
    if (class(tree) == "rpart") {  # successfully trained trees have class rpart
      original.prediction <- predict(tree, instance, type="class")
      tree.fidelity <- as.integer(original.prediction == rf.prediction)
    } else if (class(tree) == "factor") {  # if there was only one label, tree is a factor
      tree.fidelity <- mean(tree == rf.prediction)
    } else {
      print ("unexpected class for tree in EvaluateTrees")
      print (class(tree))
    }
    results[instance.num, ] <- c(i, 0, tree.fidelity)
    instance.num <- instance.num+1
    # then evaluate the fidelity on the hamming datasets
    for (j in 1:length(hamming.disks)) {
      hamming.data <- hamming.disks[[j]]
      if (class(tree) == "rpart") {  # successfully trained trees have class rpart
        hamming.predictions <- predict(tree, hamming.data, type="class")
        tree.fidelity <- mean(hamming.predictions == hamming.data[,class.col])
      } else if (class(tree) == "factor") {  # if there was only one label, tree is a factor
        tree.fidelity <- mean(tree == hamming.data[,class.col])
      } else {
        print ("unexpected class for tree in EvaluateTrees")
        print (class(tree))
      }
      results[instance.num, ] <- c(i, j, tree.fidelity)
      instance.num <- instance.num+1
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
  results[, 1] <- factor(results[, 1], levels=c(0:max.ham), ordered=TRUE)
  plt <- ggplot(results, aes(ymin = 0.0, ymax = 1.0)) +
    scale_color_brewer(palette="Spectral") +
    geom_line(aes(test.d, accuracy, colour = train.d, group = train.d)) +
    geom_point(aes(test.d, accuracy, colour = train.d, group = train.d)) +
    labs(title = paste("Instance",instance.n), x = "Testing HD", 
         y = "Fidelity", color = "Training HD")
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
  }
  return(training.data)
}

## load the breast cancer dataset
url_string <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
#url_string <- "breast-cancer.data"
col.names <- c("Class", "age", "menopause", "tumor.size", "inv.nodes", "node.caps", "deg.malig", "breast", "breast.quad", "irradiat")

## OR load the lymphography dataset
#url_string <- "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data"
#col.names <- c("Class", "lymphatics", "block.of.affere", "bl.of.lymph.c", "bl.of.lymph.s", "by.pass", "extravasates", "regeneration.of",
#                "early.uptake.in", "lym.nodes.dimin", "lym.nodes.enlar", "changes.in.lym", "defect.in.node", "changes.in.node", "changes.in.stru",
#                "special.forms", "dislocation.of", "exclusion.of.no", "no.of.nodes.in")

dataset <- GetDataset(url_string, header = FALSE, local = FALSE)

#training.data <- GetTrainingData(dataset, "random")
#training.data <- GetTrainingData(dataset, "whole")
training.data <- GetTrainingData(dataset, "random")

## train a random forest on the training data
random.forest <- randomForest::randomForest(formula=attr(dataset, "formula"), data=dataset, subset=training.data)

## check rf accuracy on test data
rf.predictions <- predict(random.forest, dataset[-training.data,], type="class")
levels(rf.predictions) <- levels(dataset[,attr(dataset, "class.col")])
rf.accuracy <- mean(rf.predictions == dataset[,attr(dataset, "class.col")][-training.data])

rf.labels <- predict(random.forest, dataset[training.data,], type="class")
# df2 <- dataset[training.data,]
# df2[attr(dataset, "class.col")] <- rf.labels
# global.model <- rpart::rpart(formula=attr(dataset, "formula"), data=df2, method="class", minsplit=1, minbucket=1, cp=0, maxdepth=5)
# 
# global.model.predictions <- predict(global.model, dataset[-training.data,], type="class")
# global.model.fidelity <- mean(rf.predictions == global.model.predictions)
# 
# fancyRpartPlot(global.model)

# ## train an ordinary tree on the training data
# ordinary.tree <- rpart::rpart(formula=attr(dataset, "formula"), data=dataset[training.data,], method="class")
# 
# ## check oordinary tree accuracy on test data
# ordinary.tree.predictions <- predict(ordinary.tree, dataset[-training.data,], type="class")
# ordinary.tree.accuracy <- mean(ordinary.tree.predictions == dataset[,attr(dataset, "class.col")][-training.data])
# 
# ## evaluate ordinary tree fidelity on test data
# ordinary.tree.test.fidelity <- mean(ordinary.tree.predictions == rf.predictions)

#####################
## good boy global ##
#####################
instance <- dataset[1,]
hamming.rings <- GetHammingRings(instance, max.ham)
hamming.disks <- GetHammingDisks(instance, hamming.rings)
h8 <- hamming.disks[[8]]
#for (i in 1:10) {
global.training.d <- sample(1:nrow(h8), nrow(h8)*0.1)
global.model <- rpart::rpart(formula=attr(dataset, "formula"), data=h8[global.training.d,], method="class")
#global.model.predictions <- predict(global.model, h8[-global.training.d,], type="class")
#rf.predictions <- predict(random.forest, h8[-global.training.d,], type="class")
#global.model.fidelity <- mean(rf.predictions == global.model.predictions)
#print(global.model.fidelity) 
#}
#fancyRpartPlot(global.model)


all.trees.per.instance <- list()
summary.results = data.frame()
max.ham = 5
instance.num = 45
for (instance.num in 1:nrow(dataset)) {
  print(paste("Evaluating instance", instance.num, "of", nrow(dataset)))
  instance <- dataset[instance.num,]
  hamming.rings <- GetHammingRings(instance, max.ham)
  hamming.disks <- GetHammingDisks(instance, hamming.rings)
  trees <- TrainTrees(hamming.disks)
  results <- EvaluateTrees(trees, hamming.rings)
  
  #ordinary.tree.results <- EvaluateTree(ordinary.tree, hamming.rings)
  #results <- rbind(results, ordinary.tree.results)
  global.model.results <- EvaluateTree(global.model, hamming.rings)
  results <- rbind(results, global.model.results)
  trees <- AssessAccuracy(trees) # adds accuracy as an attribute to each tree
  all.trees.per.instance[[instance.num]] <- trees
  
  summary.results <- rbind(summary.results, cbind(instance.num, results))
  
  PlotAndWrite(results, fig.write = TRUE, fig.prefix = "test-", fig.id = instance.num, instance.n = instance.num, fig.width = 800)
}

#setwd("Desktop/CS760")

#saveRDS(summary.results, file="breast_cancer_results.data")
#saveRDS(all.trees.per.instance, file="breast_cancer_all_trees_per_instance.data")

saveRDS(summary.results, file="lymphography_results.data")
saveRDS(all.trees.per.instance, file="evaluation_all_trees_per_instance.data")

# summary.results = readRDS(file="XXX.data")




## get training.data with only one label to test handling of that case
# training.data <- hamming.disks[[i]][hamming.disks[[i]][,1] == unique(hamming.disks[[i]][,1])[[1]],]

# PLOT ERROR BARS

error.bars.and.shit.helper <- function(instances, plt.title="Bitches vs Problems") {
  results <- data.frame(train.d=integer(0), test.d=integer(0), 
                        accuracy.mean=double(0), accuracy.sd=double(0))
  results[, 1] <- factor(results[, 1], levels=c(0:max.ham), ordered=TRUE)
  results.sub <- results
  for (train.hd in 1:max.ham) {
    i <- 1
    for (test.hd in 0:max.ham) {
      xxx <- subset(instances, train.d==train.hd & test.d==test.hd)
      fucking.mean <- mean(xxx$accuracy)
      fucking.sd <- sd(xxx$accuracy)
      results.sub[i, ] <- c(train.hd, test.hd, fucking.mean, fucking.sd)
      i <- i+1
    }
    plt <- ggplot(results.sub, aes(ymin = 0.0, ymax = 1.0)) +
      geom_line(aes(test.d, accuracy.mean)) +
      geom_point(aes(test.d, accuracy.mean)) +
      labs(title = paste(paste(plt.title,'Training d =',train.hd)), x = "Testing HD", 
           y = "Accuracy") +
      geom_errorbar(aes(x = test.d, ymin=accuracy.mean-accuracy.sd, ymax=pmin(1,accuracy.mean+accuracy.sd)))
    print(plt)
    write.plt(paste0(plt.title,'_d_equals_',train.hd))
    
    results <- rbind(results, results.sub)
  }
  plt <- ggplot(results, aes(ymin = 0.0, ymax = 1.0)) +
    geom_line(aes(test.d, accuracy.mean, colour = train.d, group = train.d)) +
    geom_point(aes(test.d, accuracy.mean, colour = train.d, group = train.d)) +
    labs(title = paste(paste(plt.title)), x = "Testing HD", 
         y = "Accuracy", color = "Training HD")
  print(plt)
  write.plt(plt.title)
}

write.plt <- function(name) {
  dev.print(png, paste0(name,".png"), width = 500)
}

error.bars.and.shit <- function(no.instances) {
  training.data.inds <- GetTrainingData(dataset, "random")

  training.instances <- summary.results[1, ]
  for (i in training.data.inds) {
    training.instances <- rbind(training.instances, subset(summary.results, instance.num==i))
  }
  training.instances <- training.instances[2:nrow(training.instances), ]
  error.bars.and.shit.helper(training.instances, "Training-Set")
  #error.bars.and.shit.helper.2(training.instances, "Training Set,")

  test.instances <- summary.results[1, ]
  test.data.inds <- setdiff(1:nrow(dataset), training.data.inds)
  for (i in test.data.inds) {
    test.instances <- rbind(test.instances, subset(summary.results, instance.num==i))
  }
  test.instances <- test.instances[2:nrow(test.instances), ]
  error.bars.and.shit.helper(test.instances, "Test-Set")
  #error.bars.and.shit.helper.2(test.instances, "Test Set,")
}

error.bars.and.shit()



explain.a.test.instance <- function(index) {
  training.data.inds <- GetTrainingData(dataset, "random")
  test.data.inds <- setdiff(1:nrow(dataset), training.data.inds)
  instance <- dataset[training.data.inds[index], ]
  
  hamming.rings <- GetHammingRings(instance, max.ham)
  hamming.disks <- GetHammingDisks(hamming.rings)
  trees <- TrainTrees(hamming.disks)
  results <- EvaluateTrees(trees, hamming.rings, instance)
  
  # LEMONify
  results[, 'accuracy'] <- results$accuracy * 2^(-results$test.d)
  lemon.results <- data.frame(training.hd=1:5, score=0, complexity=0)
  for (i in 1:5) {
    lemon.results[i, 'score'] <- sum( subset(results, train.d==i)$accuracy )
    if (class(trees[[i]]) == "rpart") {
      nodes <- as.numeric(rownames(trees[[i]]$frame))
      lemon.results[i, 'complexity'] <- max(rpart:::tree.depth(nodes))
    }
  }
  print(lemon.results)

  # Plot
  results[, 1] <- factor(results[, 1], levels=c(0:max.ham), ordered=TRUE)
  plt <- ggplot(results, aes(ymin = 0.0, ymax = 1.0)) +
    geom_line(aes(test.d, accuracy, colour = train.d, group = train.d)) +
    geom_point(aes(test.d, accuracy, colour = train.d, group = train.d)) +
    labs(title = "To Explain", x = "Testing HD", 
         y = "Accuracy", color = "Training HD")
  
  # Add red line
  red.line <- data.frame(x=seq(0,5,0.1))
  red.line[, 'y'] <- 2^(-red.line$x)
  plt <- plt + geom_line(data=red.line, aes(x, y), linetype="dashed", colour="red")
  
  # Print a write plot
  print(plt)
  dev.print(png, "score_plot.png", width = 1000)
  
  # Draw best explanation
  best.expl <- trees[[which.max(lemon.results$score)]]
  fancyRpartPlot(best.expl)
  png("tree_plt.png")
  fancyRpartPlot(best.expl)
  dev.off()
}

to.explain <- 20 #20th test instance
index <- to.explain

explain.a.test.instance( to.explain  )
