library(lubridate)  
library(dplyr)    
library(DMwR)
library(tidyr)
library(caTools)
library(Hmisc)
library(Metrics)
library(outliers)
library(OutlierDetection)
library(randomForest)
library(rpart)
library(ggplot2)
#import datasets
setwd("C:/Users/Akshay Prabhu/Desktop/edwisor project")
train_fare = read.csv("train_cab.csv", header=T, stringsAsFactors = FALSE)
test_fare = read.csv("test.csv", header = T, stringsAsFactors = FALSE)
#summary
summary(train_fare)

#replace missing values
#actual value 3
#mean 2.62
#median 1
#we will freeze median method to impute missing val
train_fare$fare_amount=as.numeric(as.character(train_fare$fare_amount))
train_fare$passenger_count[is.na(train_fare$passenger_count)]=median(train_fare$passenger_count, na.rm=T)
train_fare$fare_amount[is.na(train_fare$fare_amount)]=median(train_fare$fare_amount, na.rm=T)

#we need to remove the passenger count greater than 7
train_fare=filter(train_fare, passenger_count<8)

#we also need to remove fare_amount <=0
train_fare=filter(train_fare, fare_amount>0)

#we also need to set range of latitudes and longitudes
train_fare=filter(train_fare, pickup_latitude<=90, pickup_latitude>=-90,
                     pickup_longitude<=180, pickup_longitude>=-180,
                     dropoff_latitude<=90, pickup_longitude>=-90,
                     dropoff_longitude<=180, pickup_longitude>=-90)

#The first step to analyse how the fares have changed over time, 
#is to create features like hour, day of the week, day, month, year 
#from pickup datetime.


datetxt=as.Date(train_fare$pickup_datetime)

train_fare$Year = as.numeric(format(datetxt, format = "%Y"))
train_fare$Month = as.numeric(format(datetxt, format = "%m"))
train_fare$Day = as.numeric(format(datetxt, format = "%d"))
train_fare$Time <- format(as.POSIXct(train_fare$`pickup_datetime`, "%Y-%m-%d %H:%M:%S", tz = ""), format = "%H:%M:%S")
train_fare$Days_of_week=wday(datetxt, label = TRUE)

datetxt=as.Date(test_fare$pickup_datetime)

test_fare$Year = as.numeric(format(datetxt, format = "%Y"))
test_fare$Month = as.numeric(format(datetxt, format = "%m"))
test_fare$Day = as.numeric(format(datetxt, format = "%d"))
test_fare$Time <- format(as.POSIXct(test_fare$`pickup_datetime`, "%Y-%m-%d %H:%M:%S", tz = ""), format = "%H:%M:%S")
test_fare$Days_of_week=wday(datetxt, label = TRUE)

#we need to remove datetime value of 43
train_fare=train_fare[-c(1315),]

#distance
fun_dist <-function(lat1, lat2, lon1,lon2){
  p = 0.017453292519943295 # Pi/180
  a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
  return (0.6213712 * 12742 * asin(sqrt(a)))
}
i = 1
dis = ''
while (i<=16041) {
  lat1 = train_fare$pickup_latitude[i]
  lat2 = train_fare$dropoff_latitude[i]
  lon1 = train_fare$pickup_longitude[i]
  lon2 = train_fare$dropoff_longitude[i]
  x = fun_dist(lat1,lat2, lon1,lon2)
  dis[[i]]<-x
  i=i+1
}
train_fare$Distance <- dis
train_fare$Distance=as.numeric(as.character(train_fare$Distance))

i = 1
dis_test = ''
while (i<=9914) {
  lat1 = test_fare$pickup_latitude[i]
  lat2 = test_fare$dropoff_latitude[i]
  lon1 = test_fare$pickup_longitude[i]
  lon2 = test_fare$dropoff_longitude[i]
  y = fun_dist(lat1,lat2, lon1,lon2)
  dis_test[[i]]<-y
  i=i+1
}

test_fare$Distance <- dis_test
test_fare$Distance=as.numeric(as.character(test_fare$Distance))


#Removing zero distance value
dis_zero = which(train_fare$Distance == 0)
train_fare = train_fare [-dis_zero, ]


#outliers
variable1=subset(train_fare, select=-c(2,3,4,5,6,7,8,9,10,11,12))
OutlierDetection(variable1, k = 0.05 * nrow(variable1), cutoff = 0.95,
                 Method = "euclidean", rnames = FALSE, depth = FALSE,
                 dense = FALSE, distance = FALSE, dispersion = FALSE)


#removing distance>2000
dis_2k=which(train_fare$Distance>2000)
train_fare=train_fare[-dis_2k,]


#removing fare_amount greater than 4k
fa_4k=which(train_fare$fare_amount>4000)
train_fare=train_fare[-fa_4k,]

#lets find correlation with p-values between variables
data_cor=subset(train_fare, select=-c(2,3,4,5,6,11))
data_cor=sapply(data_cor, as.numeric)
cor_cof=rcorr(as.matrix(data_cor))
cor_cof
#we can see that Distance,Month and Year are highly correlated with Fare_amount. We can neglect other variables
train_datamodel=subset(train_fare, select=-c(2,3,4,5,6,7,10,11,12))
train_datamodel=train_datamodel[,c(3,2,4,1)]

#line graph of year vs fare amount
var_year=group_by(train_fare, Year)

var_summary=summarise(var_year, minFare = min(fare_amount), maxFare = max(fare_amount), meanFare = mean(fare_amount))

ggplot()+
  geom_line(aes(x=var_summary$Year, y=var_summary$meanFare),color='blue')+
  ggtitle('Average Fare amount vs Year')+
  xlab('year')+
  ylab('Average fare_amount')
#We see higher fares in more recent years, which is expected assuming inflation.
#But there is also downfall in fare_amount in 2015.

#line graph of month vs fare amount
var_month=group_by(train_fare, Month)

var_summary_month=summarise(var_month, minFare = min(fare_amount), maxFare = max(fare_amount), meanFare = mean(fare_amount))

ggplot()+
  geom_line(aes(x=var_summary_month$Month, y=var_summary_month$meanFare),color='blue')+
  ggtitle('Average Fare amount vs Month')+
  xlab('month')+
  ylab('Average fare_amount')
#We see lower fares in the months of January, February and July.


#regression
#lets split the train_data into trainset and testset
set.seed(123)
train_index=sample.split(train_datamodel, SplitRatio = 0.8)
train_trainFair=subset(train_datamodel, train_index==TRUE)
train_testFair=subset(train_datamodel, train_index==FALSE)

#lets run the linear regression model
lm_model=lm(fare_amount~., data=train_trainFair)
summary(lm_model)

y_pred=predict(lm_model, newdata = train_testFair)

#lets plot the results of linear regression model on training set
ggplot()+
  geom_point(aes(x=train_trainFair$Distance, y=train_trainFair$fare_amount),color='red')+
  geom_line(aes(x=train_trainFair$Distance, y=predict(lm_model, newdata = train_trainFair)),color='blue')+
  ggtitle('Fare amount vs Distance(train set)')+
  xlab('Distance')+
  ylab('fare_amount')

#lets plot the results of linear regression model on test set
ggplot()+
  geom_point(aes(x=train_testFair$Distance, y=train_testFair$fare_amount),color='red')+
  geom_line(aes(x=train_trainFair$Distance, y=predict(lm_model, newdata = train_trainFair)),color='blue')+
  ggtitle('Fare amount vs Distance(test set)')+
  xlab('Distance')+
  ylab('fare_amount')

rmse(train_testFair$fare_amount,y_pred)
#rmse for linear regression model is 9.3

#decision tree
fit=rpart(fare_amount~., data = train_trainFair, method = "anova")
dt_predict=predict(fit,train_testFair[-4])
rmse(train_testFair$fare_amount,dt_predict)

#random forest
RF_model=randomForest(fare_amount~.,train_trainFair,importance=TRUE, ntree=500)
RF_prediction=predict(RF_model, train_testFair[-4])

rmse(train_testFair$fare_amount,RF_prediction)



#lets impute predicted values from random forest into test data.
test_dataModel=subset(test_fare, select=-c(1,2,3,4,5,6,9,10,11))
pred_value=predict(RF_model, test_dataModel)
pred_value<-data.frame(pred_value)
test_fare$predicted_fairamount<-pred_value[1]

  