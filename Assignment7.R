library(recommenderlab)
install.packages(c("reshape2","ggplot2","data.table","recosystem"))
library(reshape2)
library(ggplot2)
library(data.table)
library(recosystem)


#Function to calculate RMSE given the error
rmse <- function(error)
{
  sqrt(mean(error^2))
}



#---------Load csv files---------------------

tr<-read.csv("D:/Personal Data/MS/Post Module3/RS/Assignment 7/train.csv",header=TRUE)
ts<-read.csv("D:/Personal Data/MS/Post Module3/RS/Assignment 7/test.csv",header=TRUE)

#------- Change Col Names for the files -----------------
cols <- c("user_index","item_index","rating")
colnames(ts) <- cols
colnames(tr) <- cols

#----------Matrix Factorization Start ----------------------------


#------Creating duplicate training and test data set

trmf =  tr
tsmf =  ts


#---------Data Manipulation of training and test dataset------------------


#Training Data Set modified

#Getting list of unique user ids
tr_useridx =  unique(trmf$user_index)
#Convert unique user ids as data frame
tr_useridx = as.data.frame(tr_useridx)
#Creating sequence of numbers for every user id to be used in building model and prediction funcion
tr_useridx$user_idx = seq(1:nrow(tr_useridx))
#replacing the sequence of numbers for the user ids in the actual training set
trmf$user_index = tr_useridx$user_idx[match(trmf$user_index,tr_useridx$tr_useridx)]
#Getting list of unique business(item) ids
tr_itemidx =  unique(trmf$item_index)
#Convert unique business(item) ids as data frame
tr_itemidx = as.data.frame(tr_itemidx)
#Creating sequence of numbers for every business(item) id to be used in building model and prediction funcion
tr_itemidx$itemidx = seq(1:nrow(tr_itemidx))
#replacing the sequence of numbers for the business(item) ids in the actual training set
trmf$item_index = tr_itemidx$itemidx[match(trmf$item_index,tr_itemidx$tr_itemidx)]


#Training Data Set modified

#Getting list of unique user ids in the training dataset
ts_useridx =  unique(tsmf$user_index)
#Convert unique user ids as data frame
ts_useridx = as.data.frame(ts_useridx)
#Creating sequence of numbers for every user id to be used in building model and prediction funcion
ts_useridx$user_idx = seq(1:nrow(ts_useridx))

tsmf$user_index = ts_useridx$user_idx[match(tsmf$user_index,ts_useridx$ts_useridx)]

ts_itemidx =  unique(tsmf$item_index)

ts_itemidx = as.data.frame(ts_itemidx)

ts_itemidx$itemidx = seq(1:nrow(ts_itemidx))

tsmf$item_index = ts_itemidx$itemidx[match(tsmf$item_index,ts_itemidx$ts_itemidx)]


#Creating an object of Recommender

r = Reco()

# Creating data required for Reco object
train_set<- data_memory(trmf$user_index,trmf$item_index,trmf$rating,index1 = TRUE)
test_data<-data_memory(tsmf$user_index,tsmf$item_index,NULL,index1 = TRUE)


#Providing tuning options for the model - Default options provided
opt = list(dim = c(10, 20, 30),                        # 4
                 costp_l1 = 0, costp_l2 = 0.01,   # 5
                 costq_l1 = 0, costq_l2 = 0.01,   # 6
                 niter = 20,                      # 7
                 nthread = 4) 
opts_tune = r$tune(train_set,opt)


#Training the model
r$train(train_set, opts = opts_tune$min)

# Running the model
pred = r$predict(test_data, out_pred = out_memory())
pred = as.data.frame(pred)
# including output in data set test for further calculations
Test<- cbind(tsmf,pred)
error = Test$pred - Test$rating
error
# Calculate RMSE for MF
rmse(error) 

# Converting the sequence of numbers back to the respective user ids provided in original dataset
Test$user_index = ts_useridx$ts_useridx[match(Test$user_index,ts_useridx$user_idx)]
# Converting the sequence of numbers back to the respective business(item) ids provided in original dataset
Test$item_index = ts_itemidx$ts_itemidx[match(Test$item_index,ts_itemidx$itemidx)]
#Removing original rating column
Test = within(Test, rm(rating))

#----------Matrix Factorization End ----------------------------


#---------User Based CF Start ----------------------------

#----------------Creating Sparse Matrix--------------------

#Training Data Set Sparse Matrix Creation

tr_sparse<- acast(tr,user_index~item_index,value.var="rating", fun.aggregate = max, na.rm=TRUE)
#Test Data Set Sparse Matrix Creation
ts_sparse<- acast(ts,user_index~item_index,value.var="rating", fun.aggregate = max, na.rm=TRUE)

#Replacing Inf values with NA
tr_sparse[!is.finite(tr_sparse)] <- NA
ts_sparse[!is.finite(ts_sparse)] <- NA


# The CF model would work only where the columns match as part of the matrix 
# The idea is to get difference between the two datasets.
# The Traing dataset has 440 items and Test data has 412 columns.
# the difference of 28 columns will be added to the Test data set 
# with NA values to be able to run a CG model

# Getting difference in columns between Training and Testing dataset
x.diff <- setdiff(colnames(tr_sparse), colnames(ts_sparse))

#Creating a dataframe of the columns not available in Test dataset and assigning NA values to them
ts_diff = data.frame(matrix(NA, nrow = nrow(ts_sparse), ncol = length(x.diff)))

#
row.names(ts_diff) = row.names(ts_sparse)
colnames(ts_diff) = x.diff

ts_new = cbind(ts_sparse,ts_diff)
ts_new<-as.matrix(ts_new)

#Converting Sparse Matrix to Real Rating Matrix
tr_sparse<-as.matrix(tr_sparse)
tr_sparse<-as(tr_sparse, "realRatingMatrix")
dim(tr_sparse)
ts_sparse<-as.matrix(ts_sparse)
ts_new<-as(ts_new, "realRatingMatrix")
dim(ts_new)


#Creating User Based Colloborative Filtering Model
ubcf_train <- Recommender(tr_sparse, method="UBCF",param=list(normalize = "Z-score",method="Cosine",nn=3, minRating=1 ))
#Columns generated from the modelcreated
names(getModel(ubcf_train))
#Predctions generated using the training and test data
ubcf_pred <- predict(ubcf_train, ts_new, type ="ratings")

ubcf_pred<- as(ubcf_pred, "data.frame")

e <- evaluationScheme(ts_new, method="split", tr_sparse,)



#-------------- Hybrid of MF and UBCF -----------
predsmf=  Test
predscf =  ubcf_pred

predsmf = within(predsmf, rm(X))
predscf = within(predscf, rm(X))

cols <- c("user_index","item_index","rating_mf")
colnames(predsmf) <- cols
cols <- c("user_index","item_index","rating_cf")
colnames(predscf) <- cols

testcf = merge(predscf,predsmf,by = c("user_index" = "user_index", "item_index" = "item_index"))

testcf = merge(testcf,ts,by = c("user_index" = "user_index", "item_index" = "item_index"))

rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Initial Variables for Loopign through Alphas and geenrating RMSE values
cfalpha = 1.0
cnt = 1
alpha = 0
mfalpha = 0.0
rmsealphas = numeric(11)
for(i in 0:10) 
{
  
  cfalpha = 0.0
  mfalpha = 1.0
  # Value of Alpha
  alpha =  (0.1 * i)
  #Alpha value for UBCF 
  cfalpha =  cfalpha + alpha
  #Alpha value for MF 
  mfalpha =  mfalpha -  alpha
  cnt = cnt + i
  #Average Hybrid Rating for Combination for UBCF and MF
  testcf$avg_pred_rating = (testcf$rating_cf * cfalpha) +  (testcf$rating_mf * (mfalpha))
  #Error of the Average Hybrid Rating calculated with ratings from Test Data Set
  error = testcf$avg_pred_rating - testcf$rating
  # Calculate RMSE for Average Hybrid Rating
  rmsealphas[i+1] =  rmse(error) 
  print (rmse(error))
}

#Plot the RMSE values calculated for the Hybrid ratings created
plot(rmsealphas, type= 'b', ylab = "RMSE Values", xlab= "Iteration for Alphas",main="Hybrid Rating- Weighted Average")

#------------End of Hybrid Rating------------------#