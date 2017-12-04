#Script to fill missing values in a dataset
library(lattice)
library(VIM)
library(mice)

#This section of the script combines all the 5 years data into one single file
i<-2012
Site_ID <- "674_ALL_YEAR="
filename <- paste(Site_ID,toString(i),".csv",sep = "")
data <- read.csv(file=filename,header = TRUE,sep = ",")
i<-i+1
while(i<2017){
  filename <- paste(Site_ID,toString(i),".csv",sep = "")
  datai <- read.csv(file=filename,header = TRUE,sep = ",")
  data <- rbind(data,datai)
  i<-i+1
}

#Save the file intermediate level
write.csv(data, '674_final_data_2017.csv')
final_data<-read.csv('674_final_data_2017.csv')

#This section replaces all missing values represented either by -99.90 or -99.9 and replaces them with NA
num_row <- nrow(final_data)
num_col <- ncol(final_data)
i<-1
count <- 0
while(i<=num_col){
  count <- 0
  print(i)
  if(any(final_data[i]=='-99.90')){
    j<-1
    #count <- 0
    while(j<=num_row){
      if (final_data[j,i]=='-99.90'){
        count <- count+1
        final_data[j,i] <- as.numeric('NA')
      }
      else{
        final_data[j,i] <- as.numeric(final_data[j,i])
      }
      j<-j+1
    }
    print(count)
  }
  if(any(final_data[i]=='-99.9')){
    j<-1
    #count <- 0
    while(j<=num_row){
      if (final_data[j,i]=='-99.9'){
        count <- count+1
        final_data[j,i] <- as.numeric('NA')
      }
      else{
        final_data[j,i] <- as.numeric(final_data[j,i])
      }
      j<-j+1
    }
    print(count)
  }
  i<-i+1
}

#again save a file at this stage
write.csv(final_data, '674_final_data_startall_1217.csv')

#remove the siteid, date, hour colums, also remove if any columns which returned more than 10% of missing values from above step
#example column number 22 is removed because it had nearly 90% missing values
final_data<-final_data[c(5:21,23:28)]


#Saving only the genuine features
write.csv(final_data, '674_final_data_NA_1216.csv')
final_data<-read.csv('674_final_data_NA_1216.csv')

#use this section of the code if one has to employ interpollation, moving average, kalman, etc for filling in the missing values
#
k<-1
while(k<ncol(final_data)){
  final_data[k] <- na.interpolation(final_data[k], option = "linear")
  k<-k+1
}
write.csv(mice_final, '674_interpolation.csv')

# exponential moving average
t<-1
while(t<ncol(final_data)){
  final_data[t] <- round(na.ma(final_data[t],  k = 24, weighting = "exponential"), digits = 2)
  t<-t+1
}
write.csv(mice_final, '674_exponential_ma.csv')

#Plot distribution of NA per feature using kaman filter
final_data_kal<-na.kalman(final_data, model ="StructTS", smooth = TRUE) 
num_col <- ncol(final_data_kal)
i<-1
while(i<=num_col){
  fname <- paste("./674_final_data_kal",toString(i),".png", sep = "")
  X11(type="cairo")
  plotNA.distribution(final_data_kal[,i])
  i<-i+1
  savePlot(file=fname)
  dev.off()
}
write.csv(mice_final, '674_kalman.csv')
final_data<-read.csv('674_final_data_NA_1217.csv')
final_data <- final_data[,c(2:24)]
final_data<-final_data[,c(1:3,5:8,12,14:23)]
final_data_mice_18_18 <- mice(final_data, m=18, maxit = 18)
final_data_mice_18<-with(final_data_mice_18_18,lm(PREC.I.1..in.~PREC.I.2..in.+TOBS.I.1..degC.++SMS.I.1..8..pct....silt.+SMS.I.1..20..pct....silt.+STO.I.1..2..degC.+STO.I.1..8..degC.+SAL.I.1..20..gram.+RDC.I.1..8..unit.+RDC.I.1..20..unit.+BATT.I.1..volt.+BATT.I.2..volt.+WDIRV.H.1..degr.+WSPDX.H.1..mph.+WSPDV.H.1..mph.+SRADV.H.1..watt.+RHUMV.H.1..pct.+RHUMX.H.1..pct.))
                                                  
#Use the pool() function to combine the results of all the models
combo_18_model<-pool(final_data_mice_18)

i<-1
final_data1818 <- final_data
new_final_data_1818_mice <-0
while(i<=ncol(final_data1818)){
  fname <- paste("674_final_data_mice_",toString(i), sep = "")
  assign(fname, complete(final_data_mice_18_18,i))
  i<-i+1
}
mice_final <- round((`674_final_data_mice_1`+`674_final_data_mice_2`+`674_final_data_mice_3`+`674_final_data_mice_4`+`674_final_data_mice_5`+`674_final_data_mice_6`+`674_final_data_mice_7`+`674_final_data_mice_8`+`674_final_data_mice_9`+`674_final_data_mice_10`+`674_final_data_mice_11`+`674_final_data_mice_12`+`674_final_data_mice_13`+`674_final_data_mice_14`+`674_final_data_mice_15`+`674_final_data_mice_16`+`674_final_data_mice_17`+`674_final_data_mice_18`)/18, digit=2)
write.csv(mice_final, '674_final_data_mice.csv')