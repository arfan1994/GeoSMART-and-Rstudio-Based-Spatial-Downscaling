########################################################### R Script: RFgwr Downscaling with Loop Over All Time Steps##########################################################

library(xgboost)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(terra)
library(openxlsx)
library(readr)
library(Metrics)
library(sp)
library(GWmodel)      ## GW models
library(plyr)         ## Data management
library(sp)           ## Spatial Data management
library(spdep)        ## Spatial autocorrelation
library(RColorBrewer) ## Visualization
library(classInt)     ## Class intervals
library(raster)       ## spatial data
library(grid)         ## plot
library(gridExtra)    ## Multiple plot
library(ggplot2)      #  plotting
library(tidyverse)    # data 
library(SpatialML)    # Geographically weighted regression
library(h2o)
library(caret)
library(randomForest)


# Load coarse and fine datasets
coarse_data <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
fine_data   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")
coarse_data <- dplyr::rename(coarse_data,
                             TWSA        = lwe_thickness,
                             Temperature = `Air temperature`,
                             ET          = Evapotranspiration_openet,
                             Elevation = Elevation,
                             NDVI         = `NDVI`,
                             LST         = `Land surface temperature`,
                             GPP          = `gross primary production`,
                             Prcp          = `Precipitation`
)


fine_data <- dplyr::rename(fine_data,
                           Temperature = `Air temperature`,
                           ET          = Evapotranspiration_openet,
                           Elevation = Elevation,
                           NDVI         = `NDVI`,
                           LST         = `Land surface temperature`,
                           GPP          = `gross primary production`,
                           Prcp          = `Precipitation`
)


coarse_data[is.na(coarse_data)] <- -999
fine_data[is.na(fine_data)]     <- -999

# Extract Year and Month
coarse_data <- coarse_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

fine_data <- fine_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

# Define start and end year/month
# Define start and end
start_year <- 2007
start_month <- "Mar"
end_year <- 2012
end_month <- "Dec"

# Ensure MonthName is an ordered factor
month_levels <- month.abb
time_steps <- unique(dplyr::select(coarse_data, Year, MonthName)) %>%
  mutate(
    MonthName = factor(MonthName, levels = month.abb, ordered = TRUE)
  ) %>%
  arrange(Year, MonthName) %>%
  filter(
    (Year > start_year | (Year == start_year & MonthName >= start_month)) &
      (Year < end_year | (Year == end_year & MonthName <= end_month))
  )


# Initialize empty data frames to collect results
all_train_results <- data.frame()
all_test_results <- data.frame()
all_whole_results <- data.frame()


# Loop over each time step
for (i in 1:nrow(time_steps)) {
  current_year <- time_steps$Year[i]
  current_month <- as.character(time_steps$MonthName[i])
  message(paste0("🔄 Processing ", current_year, "-", current_month, "..."))
  
  df_train <- coarse_data %>%
    filter(Year == current_year, MonthName == current_month)
  df_train<-data.frame(df_train)
  
  df_fine <- fine_data %>%
    filter(Year == current_year, MonthName == current_month)
  
  df_fine<-data.frame(df_fine)
  
  
  
  set.seed(456)
  splitIndex <- createDataPartition(df_train$TWSA, p = 0.8, list = FALSE)
  train_data <- df_train[splitIndex, ]
  test_data <- df_train[-splitIndex, ]
  
  
  
  # define coordinates
  Coords <- train_data[, c("X", "Y")]
  #apply RF_gw model
  library(grf)
  grf.model <- grf(TWSA ~ Temperature + Elevation + ET + NDVI + 
                     LST + GPP + Prcp, 
                   dframe = train_data,
                   bw = 50,                  # Bandwidth for adaptive kernel
                   ntree = 400,              # Number of trees
                   mtry = 2,                 # Number of variables tried at each split
                   kernel = "adaptive",      # Kernel type
                   forests = TRUE,           # Save local forests
                   coords = Coords)          # Coordinate matrix or data frame
  
  
  # Make predictions on the training set
  
  x <- train_data$X
  y <- train_data$Y
  Temperature <- train_data$Temperature
  Elevation   <- train_data$Elevation
  ET          <- train_data$ET
  NDVI         <- train_data$NDVI
  LST         <- train_data$LST
  GPP        <- train_data$GPP
  Prcp          <- train_data$Prcp
  
  
  df_training <- data.frame(
    X = x,
    Y = y,
    Temperature,
    Elevation,
    ET,
    NDVI,
    LST,
    GPP,
    Prcp
  )
  
  train_predictions <- predict.grf(grf.model, df_training, x.var.name = "X", y.var.name = "Y", local.w = 0, global.w = 1)
  # Make predictions on the testing set
  
  x <- test_data$X
  y <- test_data$Y
  Temperature <- test_data$Temperature
  Elevation   <- test_data$Elevation
  ET          <- test_data$ET
  NDVI         <- test_data$NDVI
  LST         <- test_data$LST
  GPP        <- test_data$GPP
  Prcp          <- test_data$Prcp
  
  df_testing <- data.frame(
    X = x,
    Y = y,
    Temperature,
    Elevation,
    ET,
    NDVI,
    LST,
    GPP,
    Prcp
  )
  
  test_predictions <- predict.grf(grf.model, df_testing, x.var.name = "X", y.var.name = "Y", local.w = 0, global.w = 1)
  
  
  
  # Make predictions on the whole dataset
  
  x <- df_train$X
  y <- df_train$Y
  Temperature <- df_train$Temperature
  Elevation   <- df_train$Elevation
  ET          <- df_train$ET
  NDVI         <- df_train$NDVI
  LST         <- df_train$LST
  GPP        <- df_train$GPP
  Prcp          <- df_train$Prcp
  
  df_testing_wholedataset <- data.frame(
    X = x,
    Y = y,
    Temperature,
    Elevation,
    ET,
    NDVI,
    LST,
    GPP,
    Prcp
  )
  
  test_predictions_whole <- predict.grf(
    grf.model,
    df_testing_wholedataset,
    x.var.name = "X",
    y.var.name = "Y",
    local.w = 0,
    global.w = 1
  )
  
  
  FIPS.xy_CoarseRes <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = test_predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  
  
  coordinates(FIPS.xy_CoarseRes) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes) <- TRUE
  r_pred <- raster(FIPS.xy_CoarseRes, layer = "Predicted_Tws")
  r_obs  <- raster(FIPS.xy_CoarseRes, layer = "Observed_TWS")
  r_res  <- raster(FIPS.xy_CoarseRes, layer = "Residuals")
  plot(r_obs)
  plot(r_pred)
  
  
  # Predict on fine resolution
  
  #####develop model for downscalng based on model cross-valiated in training and testing stage
  
  
  # define coordinates
  Coords <- df_train[, c("X", "Y")]
  #apply RF_gw model
  library(grf)
  grf.model_whole <- grf(
    TWSA ~ Temperature + Elevation + ET + NDVI + GPP+ LST + Prcp,
    dframe = df_train,
    bw = 50,                 # Bandwidth for adaptive kernel
    ntree = 100,             # Number of trees
    mtry = 2,                # Number of variables tried at each split
    kernel = "adaptive",     # Kernel type: adaptive or fixed
    forests = TRUE,          # Save all local forests
    coords = Coords          # X, Y coordinate matrix/data frame
  )
  
  
  x  <- df_train$X
  y  <- df_train$Y
  Temperature <- df_train$Temperature
  Elevation   <- df_train$Elevation
  ET          <- df_train$ET
  NDVI         <- df_train$NDVI
  LST         <- df_train$LST
  GPP        <- df_train$GPP
  Prcp          <- df_train$Prcp
  
  df_testing_whole <- data.frame(
    X = x,
    Y = y,
    Temperature,
    Elevation,
    ET,
    NDVI,
    LST,
    GPP,
    Prcp
  )
  
  predictions_whole <- predict.grf(
    grf.model_whole,
    df_testing_whole,
    x.var.name = "X",
    y.var.name = "Y",
    local.w = 0,
    global.w = 1
  )
  
  
  FIPS.xy_CoarseRes_1 <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  coordinates(FIPS.xy_CoarseRes_1) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes_1) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes_1) <- TRUE
  r_pred_calibrated <- raster(FIPS.xy_CoarseRes_1, layer = "Predicted_Tws")
  r_res_calibrated  <- raster(FIPS.xy_CoarseRes_1, layer = "Residuals")
  
  
  x  <- df_fine$X
  y  <- df_fine$Y
  Temperature <- df_fine$Temperature
  Elevation   <- df_fine$Elevation
  ET          <- df_fine$ET
  NDVI         <- df_fine$NDVI
  LST         <- df_fine$LST
  GPP        <- df_fine$GPP
  Prcp          <- df_fine$Prcp
  
  
  df_testing_fine <- data.frame(
    X = x,
    Y = y,
    Temperature,
    Elevation,
    ET,
    NDVI,
    LST,
    GPP,
    Prcp
  )
  
  fine_pred <- predict.grf(
    grf.model_whole,
    df_testing_fine,
    x.var.name = "X",
    y.var.name = "Y",
    local.w = 0,
    global.w = 1
  )
  
  df_fine$fine_pred<-fine_pred
  
  
  
  
  xmin <- min(df_fine$X)
  xmax <- max(df_fine$X)
  ymin <- min(df_fine$Y)
  ymax <- max(df_fine$Y)
  res <- 0.00449
  
  # 1. Define raster with known extent and resolution
  r_template <- raster(xmn = xmin, xmx = xmax, ymn = ymin,  ymx = ymax, resolution = res, crs = "+proj=longlat +datum=WGS84")
  
  # 2. Convert point data to SpatialPointsDataFrame
  coordinates(df_fine) <- ~X + Y
  proj4string(df_fine) <- CRS("+proj=longlat +datum=WGS84")
  
  # 3. Rasterize using field 'fine_pred'
  r_fine_pred <- rasterize(df_fine, r_template, field = "fine_pred", fun = mean)
  
  
  r_fine_res<- resample(r_res_calibrated, r_fine_pred, method="bilinear")
  
  
  
  corrected_downscaled_TWSA <- r_fine_pred + r_fine_res
  
  plot(r_obs)
  plot(r_pred)
  plot(r_fine_res)
  plot(r_fine_pred)
  plot(corrected_downscaled_TWSA)
  
  # Save rasters
  
  
  save_base1 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/ModelEvaluationTrainTesting/TrainingTesting/RFgwr_TWSA_", current_year, "_", current_month)
  save_base2 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/ModelEvaluationTrainTesting/Observed/RFgwr_TWSA_", current_year, "_", current_month)
  save_base3 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/ModelEvaluationTrainTesting/Residuals/RFgwr_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred, paste0(save_base1, "_RFgwr_PRED25km.tif"), overwrite = TRUE)
  writeRaster(r_obs, paste0(save_base2, "_RFgwr_OBS25km.tif"), overwrite = TRUE)
  writeRaster(r_res, paste0(save_base3, "_RFgwr_RES25km.tif"), overwrite = TRUE)
  
  save_base4 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/FinePredictionDownscalingCorrection/Predicted/RFgwr_TWSA_", current_year, "_", current_month)
  save_base5 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/FinePredictionDownscalingCorrection/PredtiveError25km/RFgwr_TWSA_", current_year, "_", current_month)
  save_base6 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/FinePredictionDownscalingCorrection/PredtiveError1km/RFgwr_TWSA_", current_year, "_", current_month)
  save_base7 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/FinePredictionDownscalingCorrection/Downscaled/RFgwr_TWSA_", current_year, "_", current_month)
  save_base8 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/FinePredictionDownscalingCorrection/DownscaledCorrected/RFgwr_TWSA_", current_year, "_", current_month)
  
  
  writeRaster(r_pred_calibrated, paste0(save_base4, "_RFgwr_PREDCalibrated25km.tif"), overwrite = TRUE)
  writeRaster(r_res_calibrated, paste0(save_base5, "_RFgwr_RESCalibrated25km.tif"), overwrite = TRUE)
  writeRaster(r_fine_res, paste0(save_base6, "_RFgwr_RESCALIBRATED1km.tif"), overwrite = TRUE)
  writeRaster(r_fine_pred, paste0(save_base7, "_RFgwr_Downscaled1km.tif"), overwrite = TRUE)
  writeRaster(corrected_downscaled_TWSA, paste0(save_base8, "_RFgwr_DownscaledCorrected1km.tif"), overwrite = TRUE)
  
  
  
  ###Export data in excel
  
  # Create Date column
  date_str <- paste0(current_year, "-", current_month)
  
  # Collect Training Results
  
  EXPORT_TRAINING <- data.frame(
    X = train_data$X,
    Y = train_data$Y,
    Observed = train_data$TWSA,
    Predicted = train_predictions,
    Date = date_str
  )
  
  # Collect Testing Results
  
  EXPORT_TESTING <- data.frame(
    X = test_data$X,
    Y = test_data$Y,
    Observed = test_data$TWSA,
    Predicted = test_predictions,
    Date = date_str
  )
  
  # Collect Whole Results
  EXPORT_WHOLE <- data.frame(
    X = df_train$X,
    Y = df_train$Y,
    Observed = df_train$TWSA,
    Predicted = test_predictions_whole,
    Date = date_str
  )
  
  # Append to full results
  all_train_results <- rbind(all_train_results, EXPORT_TRAINING)
  all_test_results  <- rbind(all_test_results, EXPORT_TESTING)
  all_whole_results <- rbind(all_whole_results, EXPORT_WHOLE)
  
  
  message("✅ Done.")
}




# Create workbook
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "Training_performance")
openxlsx::writeData(wb, "Training_performance", all_train_results)

openxlsx::addWorksheet(wb, "Testing_performance")
openxlsx::writeData(wb, "Testing_performance", all_test_results)

openxlsx::addWorksheet(wb, "Whole_data_performance")
openxlsx::writeData(wb, "Whole_data_performance", all_whole_results)

openxlsx::saveWorkbook(
  wb,
  file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/RFgwr_model_performance_all_months_2.xlsx",
  overwrite = TRUE
)

print("✅ Excel workbook saved for all months.")


library(ggplot2)
library(Metrics)

# === Evaluation plots ===
rsq <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# Plotting function
plot_perf <- function(obs, pred, title_text) {
  df <- data.frame(Observed = obs, Predicted = pred)
  r2 <- round(1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2), 3)
  rmse_val <- round(rmse(obs, pred), 2)
  bias_val <- round(mean(pred - obs), 2)
  
  ggplot(df, aes(x = Observed, y = Predicted)) +
    xlim(-250, 250) + ylim(-250, 250) +
    geom_point(color = "blue", size = 4, shape = 21, fill = "skyblue", stroke = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -250, y = 250,  # 👈 Move stats box to top-right
             label = paste0("R² = ", r2, "\nRMSE = ", rmse_val, "\nBias = ", bias_val),
             size = 5, hjust = 0, vjust = 1, fontface = "bold") +
    labs(title = title_text,
         x = "Observed TWSA (mm)",
         y = "Predicted TWSA (mm)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 13),
      panel.border = element_rect(color = "black", fill = NA, size = 1.2)
    ) +
    coord_fixed(ratio = 1.5)
}


plot_perf(all_train_results$Observed, all_train_results$Predicted, "RFgwr_: Training Set")
plot_perf(all_test_results$Observed, all_test_results$Predicted, "RFgwr_: Testing Set")
plot_perf(all_whole_results$Observed, all_whole_results$Predicted, "RFgwr_: Whole Dataset")




###save model output
FIPS.xy_OUT<-df_fine[,1:3]
FIPS.xy_OUT$fine_pred<- fine_pred
results<-as.data.frame(FIPS.xy_OUT)

#Save output files in excel
# Save to CSV
write.csv(results, file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RFgwr/RFGWR_TWS_DOWNSCALED_2002new.csv", row.names = FALSE) 



########################################################### R Script: RF (random forest) Downscaling with Loop Over All Time Steps##########################################################

# Load required libraries
library(xgboost)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(keras)
library(caret)
library(openxlsx)
library(xlsx)
library(raster)
library(readr)
library(Metrics)
library(sp)



# Load coarse and fine datasets
coarse_data <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
fine_data   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")
coarse_data <- dplyr::rename(coarse_data,
                             TWSA        = lwe_thickness,
                             Temperature = `Air temperature`,
                             ET          = Evapotranspiration_openet,
                             Elevation = Elevation,
                             NDVI         = `NDVI`,
                             LST         = `Land surface temperature`,
                             GPP          = `gross primary production`,
                             Prcp          = `Precipitation`
)


fine_data <- dplyr::rename(fine_data,
                           Temperature = `Air temperature`,
                           ET          = Evapotranspiration_openet,
                           Elevation = Elevation,
                           NDVI         = `NDVI`,
                           LST         = `Land surface temperature`,
                           GPP          = `gross primary production`,
                           Prcp          = `Precipitation`
)


coarse_data[is.na(coarse_data)] <- -999
fine_data[is.na(fine_data)]     <- -999

# Extract Year and Month
coarse_data <- coarse_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

fine_data <- fine_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

# Define start and end year/month
# Define start and end
start_year <- 2007
start_month <- "Mar"
end_year <- 2007
end_month <- "Jun"

# Ensure MonthName is an ordered factor
month_levels <- month.abb
time_steps <- unique(dplyr::select(coarse_data, Year, MonthName)) %>%
  mutate(
    MonthName = factor(MonthName, levels = month.abb, ordered = TRUE)
  ) %>%
  arrange(Year, MonthName) %>%
  filter(
    (Year > start_year | (Year == start_year & MonthName >= start_month)) &
      (Year < end_year | (Year == end_year & MonthName <= end_month))
  )


# Initialize empty data frames to collect results
all_train_results <- data.frame()
all_test_results <- data.frame()
all_whole_results <- data.frame()


# Loop over each time step
for (i in 1:nrow(time_steps)) {
  current_year <- time_steps$Year[i]
  current_month <- as.character(time_steps$MonthName[i])
  message(paste0("🔄 Processing ", current_year, "-", current_month, "..."))
  
  df_train <- coarse_data %>%
    filter(Year == current_year, MonthName == current_month)
  df_train<-data.frame(df_train)
  
  df_fine <- fine_data %>%
    filter(Year == current_year, MonthName == current_month)
  
  df_fine<-data.frame(df_fine)
  
  
  
  set.seed(456)
  splitIndex <- createDataPartition(df_train$TWSA, p = 0.8, list = FALSE)
  train_data <- df_train[splitIndex, ]
  test_data <- df_train[-splitIndex, ]
  
  
  # Random Forest (replacing GRF)
  library(randomForest)
  rf.model <- randomForest(
    TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp,
    data = train_data,
    ntree = 400,
    mtry  = 2,
    importance = TRUE,
    na.action = na.omit
  )
  
  
  
  
  # Make predictions on the training set
  train_predictions <- predict(rf.model, newdata = train_data)
  # Make predictions on the testing set
  
  test_predictions <- predict(rf.model, newdata = test_data)
  
  # Make predictions on the whole dataset

  test_predictions_whole <- predict(rf.model, newdata = df_train)

  
  FIPS.xy_CoarseRes <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = test_predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  
  
  coordinates(FIPS.xy_CoarseRes) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes) <- TRUE
  r_pred <- raster(FIPS.xy_CoarseRes, layer = "Predicted_Tws")
  r_obs  <- raster(FIPS.xy_CoarseRes, layer = "Observed_TWS")
  r_res  <- raster(FIPS.xy_CoarseRes, layer = "Residuals")
  plot(r_obs)
  plot(r_pred)
  
  
  # Predict on fine resolution
  
  #####develop model for downscalng based on model cross-valiated in training and testing stage
  
 rf.model_whole <- randomForest(
    TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp,
    data = df_train,
    ntree = 400,
    mtry  = 2,
    importance = TRUE,
    na.action = na.omit
  )
  

 

  predictions_whole <- predict(rf.model_whole, newdata = df_train)
  
  
  FIPS.xy_CoarseRes_1 <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  coordinates(FIPS.xy_CoarseRes_1) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes_1) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes_1) <- TRUE
  r_pred_calibrated <- raster(FIPS.xy_CoarseRes_1, layer = "Predicted_Tws")
  r_res_calibrated  <- raster(FIPS.xy_CoarseRes_1, layer = "Residuals")
  
  

  fine_pred <- predict(rf.model_whole, newdata = df_fine)
  
  df_fine$fine_pred<-fine_pred
  
  
  
  
  xmin <- min(df_fine$X)
  xmax <- max(df_fine$X)
  ymin <- min(df_fine$Y)
  ymax <- max(df_fine$Y)
  res <- 0.00449
  
  # 1. Define raster with known extent and resolution
  r_template <- raster(xmn = xmin, xmx = xmax, ymn = ymin,  ymx = ymax, resolution = res, crs = "+proj=longlat +datum=WGS84")
  
  # 2. Convert point data to SpatialPointsDataFrame
  coordinates(df_fine) <- ~X + Y
  proj4string(df_fine) <- CRS("+proj=longlat +datum=WGS84")
  
  # 3. Rasterize using field 'fine_pred'
  r_fine_pred <- rasterize(df_fine, r_template, field = "fine_pred", fun = mean)
  
  
  r_fine_res<- resample(r_res_calibrated, r_fine_pred, method="bilinear")
  
  
  
  corrected_downscaled_TWSA <- r_fine_pred + r_fine_res
  
  plot(r_obs)
  plot(r_pred)
  plot(r_fine_res)
  plot(r_fine_pred)
  plot(corrected_downscaled_TWSA)
  
  # Save rasters
  
  # Save rasters (RF version)
  
  save_base1 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/ModelEvaluationTrainTesting/TrainingTesting/RF_TWSA_", current_year, "_", current_month)
  save_base2 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/ModelEvaluationTrainTesting/Observed/RF_TWSA_", current_year, "_", current_month)
  save_base3 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/ModelEvaluationTrainTesting/Residuals/RF_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred, paste0(save_base1, "_RF_PRED25km.tif"), overwrite = TRUE)
  writeRaster(r_obs,  paste0(save_base2, "_RF_OBS25km.tif"),  overwrite = TRUE)
  writeRaster(r_res,  paste0(save_base3, "_RF_RES25km.tif"),  overwrite = TRUE)
  
  save_base4 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/FinePredictionDownscalingCorrection/Predicted/RF_TWSA_", current_year, "_", current_month)
  save_base5 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/FinePredictionDownscalingCorrection/PredtiveError25km/RF_TWSA_", current_year, "_", current_month)
  save_base6 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/FinePredictionDownscalingCorrection/PredtiveError1km/RF_TWSA_", current_year, "_", current_month)
  save_base7 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/FinePredictionDownscalingCorrection/Downscaled/RF_TWSA_", current_year, "_", current_month)
  save_base8 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/FinePredictionDownscalingCorrection/DownscaledCorrected/RF_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred_calibrated,      paste0(save_base4, "_RF_PREDCalibrated25km.tif"),     overwrite = TRUE)
  writeRaster(r_res_calibrated,       paste0(save_base5, "_RF_RESCalibrated25km.tif"),      overwrite = TRUE)
  writeRaster(r_fine_res,             paste0(save_base6, "_RF_RESCALIBRATED1km.tif"),       overwrite = TRUE)
  writeRaster(r_fine_pred,            paste0(save_base7, "_RF_Downscaled1km.tif"),          overwrite = TRUE)
  writeRaster(corrected_downscaled_TWSA, paste0(save_base8, "_RF_DownscaledCorrected1km.tif"), overwrite = TRUE)
  
  
  ###Export data in excel
  
  # Create Date column
  date_str <- paste0(current_year, "-", current_month)
  
  # Collect Training Results
  
  EXPORT_TRAINING <- data.frame(
    X = train_data$X,
    Y = train_data$Y,
    Observed = train_data$TWSA,
    Predicted = train_predictions,
    Date = date_str
  )
  
  # Collect Testing Results
  
  EXPORT_TESTING <- data.frame(
    X = test_data$X,
    Y = test_data$Y,
    Observed = test_data$TWSA,
    Predicted = test_predictions,
    Date = date_str
  )
  
  # Collect Whole Results
  EXPORT_WHOLE <- data.frame(
    X = df_train$X,
    Y = df_train$Y,
    Observed = df_train$TWSA,
    Predicted = test_predictions_whole,
    Date = date_str
  )
  
  # Append to full results
  all_train_results <- rbind(all_train_results, EXPORT_TRAINING)
  all_test_results  <- rbind(all_test_results, EXPORT_TESTING)
  all_whole_results <- rbind(all_whole_results, EXPORT_WHOLE)
  
  
  message("✅ Done.")
}




# Create workbook
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "Training_performance")
openxlsx::writeData(wb, "Training_performance", all_train_results)

openxlsx::addWorksheet(wb, "Testing_performance")
openxlsx::writeData(wb, "Testing_performance", all_test_results)

openxlsx::addWorksheet(wb, "Whole_data_performance")
openxlsx::writeData(wb, "Whole_data_performance", all_whole_results)

openxlsx::saveWorkbook(
  wb,
  file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/RF/RF_model_performance_all_months_2.xlsx",
  overwrite = TRUE
)


print("✅ Excel workbook saved for all months.")


library(ggplot2)
library(Metrics)

# === Evaluation plots ===
rsq <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# Plotting function
plot_perf <- function(obs, pred, title_text) {
  df <- data.frame(Observed = obs, Predicted = pred)
  r2 <- round(1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2), 3)
  rmse_val <- round(rmse(obs, pred), 2)
  bias_val <- round(mean(pred - obs), 2)
  
  ggplot(df, aes(x = Observed, y = Predicted)) +
    xlim(-250, 250) + ylim(-250, 250) +
    geom_point(color = "blue", size = 4, shape = 21, fill = "skyblue", stroke = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -250, y = 250,  # 👈 Move stats box to top-right
             label = paste0("R² = ", r2, "\nRMSE = ", rmse_val, "\nBias = ", bias_val),
             size = 5, hjust = 0, vjust = 1, fontface = "bold") +
    labs(title = title_text,
         x = "Observed TWSA (mm)",
         y = "Predicted TWSA (mm)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 13),
      panel.border = element_rect(color = "black", fill = NA, size = 1.2)
    ) +
    coord_fixed(ratio = 1.5)
}


plot_perf(all_train_results$Observed, all_train_results$Predicted, "RFgwr_: Training Set")
plot_perf(all_test_results$Observed, all_test_results$Predicted, "RFgwr_: Testing Set")
plot_perf(all_whole_results$Observed, all_whole_results$Predicted, "RFgwr_: Whole Dataset")





########################################################### R Script: xgboost Downscaling with Loop Over All Time Steps##########################################################

# Load required libraries
library(xgboost)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(keras)
library(caret)
library(openxlsx)
library(xlsx)
library(raster)
library(readr)
library(Metrics)
library(sp)



# Load coarse and fine datasets
coarse_data <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
fine_data   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")
coarse_data <- dplyr::rename(coarse_data,
                             TWSA        = lwe_thickness,
                             Temperature = `Air temperature`,
                             ET          = Evapotranspiration_openet,
                             Elevation = Elevation,
                             NDVI         = `NDVI`,
                             LST         = `Land surface temperature`,
                             GPP          = `gross primary production`,
                             Prcp          = `Precipitation`
)


fine_data <- dplyr::rename(fine_data,
                           Temperature = `Air temperature`,
                           ET          = Evapotranspiration_openet,
                           Elevation = Elevation,
                           NDVI         = `NDVI`,
                           LST         = `Land surface temperature`,
                           GPP          = `gross primary production`,
                           Prcp          = `Precipitation`
)


coarse_data[is.na(coarse_data)] <- -999
fine_data[is.na(fine_data)]     <- -999

# Extract Year and Month
coarse_data <- coarse_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

fine_data <- fine_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

# Define start and end year/month
# Define start and end
start_year <- 2007
start_month <- "Mar"
end_year <- 2007
end_month <- "Jun"

# Ensure MonthName is an ordered factor
month_levels <- month.abb
time_steps <- unique(dplyr::select(coarse_data, Year, MonthName)) %>%
  mutate(
    MonthName = factor(MonthName, levels = month.abb, ordered = TRUE)
  ) %>%
  arrange(Year, MonthName) %>%
  filter(
    (Year > start_year | (Year == start_year & MonthName >= start_month)) &
      (Year < end_year | (Year == end_year & MonthName <= end_month))
  )


# Initialize empty data frames to collect results
all_train_results <- data.frame()
all_test_results <- data.frame()
all_whole_results <- data.frame()


# Loop over each time step
for (i in 1:nrow(time_steps)) {
  current_year <- time_steps$Year[i]
  current_month <- as.character(time_steps$MonthName[i])
  message(paste0("🔄 Processing ", current_year, "-", current_month, "..."))
  
  df_train <- coarse_data %>%
    filter(Year == current_year, MonthName == current_month)
  df_train<-data.frame(df_train)
  
  df_fine <- fine_data %>%
    filter(Year == current_year, MonthName == current_month)
  
  df_fine<-data.frame(df_fine)
  
  
  
  set.seed(456)
  splitIndex <- createDataPartition(df_train$TWSA, p = 0.8, list = FALSE)
  train_data <- df_train[splitIndex, ]
  test_data <- df_train[-splitIndex, ]
  
  
  library(caret)
  
  xgbTree <- train(
    TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp,
    data = train_data,
    method = "xgbTree",
    trControl = trainControl(method = "none"),
    tuneGrid = data.frame(
      nrounds = 300,        # like your example
      max_depth = 3,
      eta = 0.1,            # learning rate (set as needed)
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    ),
    metric = "RMSE"
  )
  
  
  
  # Make predictions on the training set
  train_predictions <- predict(xgbTree, newdata = train_data)
  # Make predictions on the testing set
  
  test_predictions <- predict(xgbTree, newdata = test_data)
  
  # Make predictions on the whole dataset
  
  test_predictions_whole <- predict(xgbTree, newdata = df_train)
  
  
  FIPS.xy_CoarseRes <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = test_predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  
  
  coordinates(FIPS.xy_CoarseRes) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes) <- TRUE
  r_pred <- raster(FIPS.xy_CoarseRes, layer = "Predicted_Tws")
  r_obs  <- raster(FIPS.xy_CoarseRes, layer = "Observed_TWS")
  r_res  <- raster(FIPS.xy_CoarseRes, layer = "Residuals")
  plot(r_obs)
  plot(r_pred)
  
  
  # Predict on fine resolution
  
  #####develop model for downscalng based on model cross-valiated in training and testing stage
  

 xgbTree_whole <- train(
    TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp,
    data = df_train,
    method = "xgbTree",
    trControl = trainControl(method = "none"),
    tuneGrid = data.frame(
      nrounds = 300,        # like your example
      max_depth = 3,
      eta = 0.1,            # learning rate (set as needed)
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    ),
    metric = "RMSE"
  )
  
  
  
  
  
  predictions_whole <- predict(xgbTree_whole, newdata = df_train)
  
  
  FIPS.xy_CoarseRes_1 <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  coordinates(FIPS.xy_CoarseRes_1) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes_1) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes_1) <- TRUE
  r_pred_calibrated <- raster(FIPS.xy_CoarseRes_1, layer = "Predicted_Tws")
  r_res_calibrated  <- raster(FIPS.xy_CoarseRes_1, layer = "Residuals")
  
  
  
  fine_pred <- predict(xgbTree_whole, newdata = df_fine)
  
  df_fine$fine_pred<-fine_pred
  
  
  
  
  xmin <- min(df_fine$X)
  xmax <- max(df_fine$X)
  ymin <- min(df_fine$Y)
  ymax <- max(df_fine$Y)
  res <- 0.00449
  
  # 1. Define raster with known extent and resolution
  r_template <- raster(xmn = xmin, xmx = xmax, ymn = ymin,  ymx = ymax, resolution = res, crs = "+proj=longlat +datum=WGS84")
  
  # 2. Convert point data to SpatialPointsDataFrame
  coordinates(df_fine) <- ~X + Y
  proj4string(df_fine) <- CRS("+proj=longlat +datum=WGS84")
  
  # 3. Rasterize using field 'fine_pred'
  r_fine_pred <- rasterize(df_fine, r_template, field = "fine_pred", fun = mean)
  
  
  r_fine_res<- resample(r_res_calibrated, r_fine_pred, method="bilinear")
  
  
  
  corrected_downscaled_TWSA <- r_fine_pred + r_fine_res
  
  plot(r_obs)
  plot(r_pred)
  plot(r_fine_res)
  plot(r_fine_pred)
  plot(corrected_downscaled_TWSA)
  
  # Save rasters
  
  # Save rasters (XGBoost version)
  
  save_base1 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/ModelEvaluationTrainTesting/TrainingTesting/XGB_TWSA_", current_year, "_", current_month)
  save_base2 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/ModelEvaluationTrainTesting/Observed/XGB_TWSA_", current_year, "_", current_month)
  save_base3 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/ModelEvaluationTrainTesting/Residuals/XGB_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred, paste0(save_base1, "_XGB_PRED25km.tif"), overwrite = TRUE)
  writeRaster(r_obs,  paste0(save_base2, "_XGB_OBS25km.tif"),  overwrite = TRUE)
  writeRaster(r_res,  paste0(save_base3, "_XGB_RES25km.tif"),  overwrite = TRUE)
  
  save_base4 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/FinePredictionDownscalingCorrection/Predicted/XGB_TWSA_", current_year, "_", current_month)
  save_base5 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/FinePredictionDownscalingCorrection/PredtiveError25km/XGB_TWSA_", current_year, "_", current_month)
  save_base6 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/FinePredictionDownscalingCorrection/PredtiveError1km/XGB_TWSA_", current_year, "_", current_month)
  save_base7 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/FinePredictionDownscalingCorrection/Downscaled/XGB_TWSA_", current_year, "_", current_month)
  save_base8 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/FinePredictionDownscalingCorrection/DownscaledCorrected/XGB_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred_calibrated,         paste0(save_base4, "_XGB_PREDCalibrated25km.tif"),        overwrite = TRUE)
  writeRaster(r_res_calibrated,          paste0(save_base5, "_XGB_RESCalibrated25km.tif"),         overwrite = TRUE)
  writeRaster(r_fine_res,                paste0(save_base6, "_XGB_RESCALIBRATED1km.tif"),          overwrite = TRUE)
  writeRaster(r_fine_pred,               paste0(save_base7, "_XGB_Downscaled1km.tif"),             overwrite = TRUE)
  writeRaster(corrected_downscaled_TWSA, paste0(save_base8, "_XGB_DownscaledCorrected1km.tif"),    overwrite = TRUE)
  
  
  ###Export data in excel
  
  # Create Date column
  date_str <- paste0(current_year, "-", current_month)
  
  # Collect Training Results
  
  EXPORT_TRAINING <- data.frame(
    X = train_data$X,
    Y = train_data$Y,
    Observed = train_data$TWSA,
    Predicted = train_predictions,
    Date = date_str
  )
  
  # Collect Testing Results
  
  EXPORT_TESTING <- data.frame(
    X = test_data$X,
    Y = test_data$Y,
    Observed = test_data$TWSA,
    Predicted = test_predictions,
    Date = date_str
  )
  
  # Collect Whole Results
  EXPORT_WHOLE <- data.frame(
    X = df_train$X,
    Y = df_train$Y,
    Observed = df_train$TWSA,
    Predicted = test_predictions_whole,
    Date = date_str
  )
  
  # Append to full results
  all_train_results <- rbind(all_train_results, EXPORT_TRAINING)
  all_test_results  <- rbind(all_test_results, EXPORT_TESTING)
  all_whole_results <- rbind(all_whole_results, EXPORT_WHOLE)
  
  
  message("✅ Done.")
}




# Create workbook
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "Training_performance")
openxlsx::writeData(wb, "Training_performance", all_train_results)

openxlsx::addWorksheet(wb, "Testing_performance")
openxlsx::writeData(wb, "Testing_performance", all_test_results)

openxlsx::addWorksheet(wb, "Whole_data_performance")
openxlsx::writeData(wb, "Whole_data_performance", all_whole_results)

openxlsx::saveWorkbook(
  wb,
  file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XGB/XGB_model_performance_all_months_2.xlsx",
  overwrite = TRUE
)


print("✅ Excel workbook saved for all months.")


library(ggplot2)
library(Metrics)

# === Evaluation plots ===
rsq <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# Plotting function
plot_perf <- function(obs, pred, title_text) {
  df <- data.frame(Observed = obs, Predicted = pred)
  r2 <- round(1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2), 3)
  rmse_val <- round(rmse(obs, pred), 2)
  bias_val <- round(mean(pred - obs), 2)
  
  ggplot(df, aes(x = Observed, y = Predicted)) +
    xlim(-250, 250) + ylim(-250, 250) +
    geom_point(color = "blue", size = 4, shape = 21, fill = "skyblue", stroke = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -250, y = 250,  # 👈 Move stats box to top-right
             label = paste0("R² = ", r2, "\nRMSE = ", rmse_val, "\nBias = ", bias_val),
             size = 5, hjust = 0, vjust = 1, fontface = "bold") +
    labs(title = title_text,
         x = "Observed TWSA (mm)",
         y = "Predicted TWSA (mm)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 13),
      panel.border = element_rect(color = "black", fill = NA, size = 1.2)
    ) +
    coord_fixed(ratio = 1.5)
}


plot_perf(all_train_results$Observed, all_train_results$Predicted, "RFgwr_: Training Set")
plot_perf(all_test_results$Observed, all_test_results$Predicted, "RFgwr_: Testing Set")
plot_perf(all_whole_results$Observed, all_whole_results$Predicted, "RFgwr_: Whole Dataset")


########################################################### R Script: svm Downscaling with Loop Over All Time Steps##########################################################

# Load required libraries
library(xgboost)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(keras)
library(caret)
library(openxlsx)
library(xlsx)
library(raster)
library(readr)
library(Metrics)
library(sp)
library(readxl)
library(e1071) # Required for SVM
library(dplyr)
library(caret)
library(openxlsx)




# Load coarse and fine datasets
coarse_data <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
fine_data   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")
coarse_data <- dplyr::rename(coarse_data,
                             TWSA        = lwe_thickness,
                             Temperature = `Air temperature`,
                             ET          = Evapotranspiration_openet,
                             Elevation = Elevation,
                             NDVI         = `NDVI`,
                             LST         = `Land surface temperature`,
                             GPP          = `gross primary production`,
                             Prcp          = `Precipitation`
)


fine_data <- dplyr::rename(fine_data,
                           Temperature = `Air temperature`,
                           ET          = Evapotranspiration_openet,
                           Elevation = Elevation,
                           NDVI         = `NDVI`,
                           LST         = `Land surface temperature`,
                           GPP          = `gross primary production`,
                           Prcp          = `Precipitation`
)


coarse_data[is.na(coarse_data)] <- -999
fine_data[is.na(fine_data)]     <- -999

# Extract Year and Month
coarse_data <- coarse_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

fine_data <- fine_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

# Define start and end year/month
# Define start and end
start_year <- 2007
start_month <- "Mar"
end_year <- 2007
end_month <- "Jun"

# Ensure MonthName is an ordered factor
month_levels <- month.abb
time_steps <- unique(dplyr::select(coarse_data, Year, MonthName)) %>%
  mutate(
    MonthName = factor(MonthName, levels = month.abb, ordered = TRUE)
  ) %>%
  arrange(Year, MonthName) %>%
  filter(
    (Year > start_year | (Year == start_year & MonthName >= start_month)) &
      (Year < end_year | (Year == end_year & MonthName <= end_month))
  )


# Initialize empty data frames to collect results
all_train_results <- data.frame()
all_test_results <- data.frame()
all_whole_results <- data.frame()


# Loop over each time step
for (i in 1:nrow(time_steps)) {
  current_year <- time_steps$Year[i]
  current_month <- as.character(time_steps$MonthName[i])
  message(paste0("🔄 Processing ", current_year, "-", current_month, "..."))
  
  df_train <- coarse_data %>%
    filter(Year == current_year, MonthName == current_month)
  df_train<-data.frame(df_train)
  
  df_fine <- fine_data %>%
    filter(Year == current_year, MonthName == current_month)
  
  df_fine<-data.frame(df_fine)
  
  
  
  set.seed(456)
  splitIndex <- createDataPartition(df_train$TWSA, p = 0.8, list = FALSE)
  train_data <- df_train[splitIndex, ]
  test_data <- df_train[-splitIndex, ]
  
  
  library(caret)
  
  svm.model <- caret::train(TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp, 
                            data=train_data, 
                            
                            kernel = "radial", cost = 10, gamma = 0.1)
  
  
  
  # Make predictions on the training set
  train_predictions <- predict(svm.model, newdata = train_data)
  # Make predictions on the testing set
  
  test_predictions <- predict(svm.model, newdata = test_data)
  
  # Make predictions on the whole dataset
  
  test_predictions_whole <- predict(svm.model, newdata = df_train)
  
  
  FIPS.xy_CoarseRes <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = test_predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  
  
  coordinates(FIPS.xy_CoarseRes) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes) <- TRUE
  r_pred <- raster(FIPS.xy_CoarseRes, layer = "Predicted_Tws")
  r_obs  <- raster(FIPS.xy_CoarseRes, layer = "Observed_TWS")
  r_res  <- raster(FIPS.xy_CoarseRes, layer = "Residuals")
  plot(r_obs)
  plot(r_pred)
  
  
  # Predict on fine resolution
  
  #####develop model for downscalng based on model cross-valiated in training and testing stage
  
  

  
  svm.model_whole <- caret::train(TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp, 
                            data=df_train, 
                            
                            kernel = "radial", cost = 10, gamma = 0.1)
  
  
  
  predictions_whole <- predict(svm.model_whole, newdata = df_train)
  
  
  FIPS.xy_CoarseRes_1 <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  coordinates(FIPS.xy_CoarseRes_1) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes_1) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes_1) <- TRUE
  r_pred_calibrated <- raster(FIPS.xy_CoarseRes_1, layer = "Predicted_Tws")
  r_res_calibrated  <- raster(FIPS.xy_CoarseRes_1, layer = "Residuals")
  
  
  
  fine_pred <- predict(svm.model_whole, newdata = df_fine)
  
  df_fine$fine_pred<-fine_pred
  
  
  
  
  xmin <- min(df_fine$X)
  xmax <- max(df_fine$X)
  ymin <- min(df_fine$Y)
  ymax <- max(df_fine$Y)
  res <- 0.00449
  
  # 1. Define raster with known extent and resolution
  r_template <- raster(xmn = xmin, xmx = xmax, ymn = ymin,  ymx = ymax, resolution = res, crs = "+proj=longlat +datum=WGS84")
  
  # 2. Convert point data to SpatialPointsDataFrame
  coordinates(df_fine) <- ~X + Y
  proj4string(df_fine) <- CRS("+proj=longlat +datum=WGS84")
  
  # 3. Rasterize using field 'fine_pred'
  r_fine_pred <- rasterize(df_fine, r_template, field = "fine_pred", fun = mean)
  
  
  r_fine_res<- resample(r_res_calibrated, r_fine_pred, method="bilinear")
  
  
  
  corrected_downscaled_TWSA <- r_fine_pred + r_fine_res
  
  plot(r_obs)
  plot(r_pred)
  plot(r_fine_res)
  plot(r_fine_pred)
  plot(corrected_downscaled_TWSA)
  
  # Save rasters
  
  # Save rasters (XGBoost version)
  
  # Save rasters (SVM version)
  
  save_base1 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/ModelEvaluationTrainTesting/TrainingTesting/SVM_TWSA_", current_year, "_", current_month)
  save_base2 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/ModelEvaluationTrainTesting/Observed/SVM_TWSA_", current_year, "_", current_month)
  save_base3 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/ModelEvaluationTrainTesting/Residuals/SVM_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred, paste0(save_base1, "_SVM_PRED25km.tif"), overwrite = TRUE)
  writeRaster(r_obs,  paste0(save_base2, "_SVM_OBS25km.tif"),  overwrite = TRUE)
  writeRaster(r_res,  paste0(save_base3, "_SVM_RES25km.tif"),  overwrite = TRUE)
  
  save_base4 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/FinePredictionDownscalingCorrection/Predicted/SVM_TWSA_", current_year, "_", current_month)
  save_base5 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/FinePredictionDownscalingCorrection/PredtiveError25km/SVM_TWSA_", current_year, "_", current_month)
  save_base6 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/FinePredictionDownscalingCorrection/PredtiveError1km/SVM_TWSA_", current_year, "_", current_month)
  save_base7 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/FinePredictionDownscalingCorrection/Downscaled/SVM_TWSA_", current_year, "_", current_month)
  save_base8 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/FinePredictionDownscalingCorrection/DownscaledCorrected/SVM_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred_calibrated,         paste0(save_base4, "_SVM_PREDCalibrated25km.tif"),        overwrite = TRUE)
  writeRaster(r_res_calibrated,          paste0(save_base5, "_SVM_RESCalibrated25km.tif"),         overwrite = TRUE)
  writeRaster(r_fine_res,                paste0(save_base6, "_SVM_RESCALIBRATED1km.tif"),          overwrite = TRUE)
  writeRaster(r_fine_pred,               paste0(save_base7, "_SVM_Downscaled1km.tif"),             overwrite = TRUE)
  writeRaster(corrected_downscaled_TWSA, paste0(save_base8, "_SVM_DownscaledCorrected1km.tif"),    overwrite = TRUE)
  
  
  
  ###Export data in excel
  
  # Create Date column
  date_str <- paste0(current_year, "-", current_month)
  
  # Collect Training Results
  
  EXPORT_TRAINING <- data.frame(
    X = train_data$X,
    Y = train_data$Y,
    Observed = train_data$TWSA,
    Predicted = train_predictions,
    Date = date_str
  )
  
  # Collect Testing Results
  
  EXPORT_TESTING <- data.frame(
    X = test_data$X,
    Y = test_data$Y,
    Observed = test_data$TWSA,
    Predicted = test_predictions,
    Date = date_str
  )
  
  # Collect Whole Results
  EXPORT_WHOLE <- data.frame(
    X = df_train$X,
    Y = df_train$Y,
    Observed = df_train$TWSA,
    Predicted = test_predictions_whole,
    Date = date_str
  )
  
  # Append to full results
  all_train_results <- rbind(all_train_results, EXPORT_TRAINING)
  all_test_results  <- rbind(all_test_results, EXPORT_TESTING)
  all_whole_results <- rbind(all_whole_results, EXPORT_WHOLE)
  
  
  message("✅ Done.")
}




# Create workbook
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "Training_performance")
openxlsx::writeData(wb, "Training_performance", all_train_results)

openxlsx::addWorksheet(wb, "Testing_performance")
openxlsx::writeData(wb, "Testing_performance", all_test_results)

openxlsx::addWorksheet(wb, "Whole_data_performance")
openxlsx::writeData(wb, "Whole_data_performance", all_whole_results)

openxlsx::saveWorkbook(
  wb,
  file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/SVM/SVM_model_performance_all_months_2.xlsx",
  overwrite = TRUE
)


print("✅ Excel workbook saved for all months.")


library(ggplot2)
library(Metrics)

# === Evaluation plots ===
rsq <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# Plotting function
plot_perf <- function(obs, pred, title_text) {
  df <- data.frame(Observed = obs, Predicted = pred)
  r2 <- round(1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2), 3)
  rmse_val <- round(rmse(obs, pred), 2)
  bias_val <- round(mean(pred - obs), 2)
  
  ggplot(df, aes(x = Observed, y = Predicted)) +
    xlim(-250, 250) + ylim(-250, 250) +
    geom_point(color = "blue", size = 4, shape = 21, fill = "skyblue", stroke = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -250, y = 250,  # 👈 Move stats box to top-right
             label = paste0("R² = ", r2, "\nRMSE = ", rmse_val, "\nBias = ", bias_val),
             size = 5, hjust = 0, vjust = 1, fontface = "bold") +
    labs(title = title_text,
         x = "Observed TWSA (mm)",
         y = "Predicted TWSA (mm)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 13),
      panel.border = element_rect(color = "black", fill = NA, size = 1.2)
    ) +
    coord_fixed(ratio = 1.5)
}


plot_perf(all_train_results$Observed, all_train_results$Predicted, "SVM: Training Set")
plot_perf(all_test_results$Observed, all_test_results$Predicted, "SVM: Testing Set")
plot_perf(all_whole_results$Observed, all_whole_results$Predicted, "SVM: Whole Dataset")






########################################################### R Script: CART Downscaling with Loop Over All Time Steps##########################################################

# Load required libraries
library(xgboost)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(keras)
library(caret)
library(openxlsx)
library(xlsx)
library(raster)
library(readr)
library(Metrics)
library(sp)
library(readxl)
library(e1071) # Required for SVM
library(dplyr)
library(caret)
library(openxlsx)




# Load coarse and fine datasets
coarse_data <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
fine_data   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")
coarse_data <- dplyr::rename(coarse_data,
                             TWSA        = lwe_thickness,
                             Temperature = `Air temperature`,
                             ET          = Evapotranspiration_openet,
                             Elevation = Elevation,
                             NDVI         = `NDVI`,
                             LST         = `Land surface temperature`,
                             GPP          = `gross primary production`,
                             Prcp          = `Precipitation`
)


fine_data <- dplyr::rename(fine_data,
                           Temperature = `Air temperature`,
                           ET          = Evapotranspiration_openet,
                           Elevation = Elevation,
                           NDVI         = `NDVI`,
                           LST         = `Land surface temperature`,
                           GPP          = `gross primary production`,
                           Prcp          = `Precipitation`
)


coarse_data[is.na(coarse_data)] <- -999
fine_data[is.na(fine_data)]     <- -999

# Extract Year and Month
coarse_data <- coarse_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

fine_data <- fine_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

# Define start and end year/month
# Define start and end
start_year <- 2007
start_month <- "Mar"
end_year <- 2007
end_month <- "Jun"

# Ensure MonthName is an ordered factor
month_levels <- month.abb
time_steps <- unique(dplyr::select(coarse_data, Year, MonthName)) %>%
  mutate(
    MonthName = factor(MonthName, levels = month.abb, ordered = TRUE)
  ) %>%
  arrange(Year, MonthName) %>%
  filter(
    (Year > start_year | (Year == start_year & MonthName >= start_month)) &
      (Year < end_year | (Year == end_year & MonthName <= end_month))
  )


# Initialize empty data frames to collect results
all_train_results <- data.frame()
all_test_results <- data.frame()
all_whole_results <- data.frame()


# Loop over each time step
for (i in 1:nrow(time_steps)) {
  current_year <- time_steps$Year[i]
  current_month <- as.character(time_steps$MonthName[i])
  message(paste0("🔄 Processing ", current_year, "-", current_month, "..."))
  
  df_train <- coarse_data %>%
    filter(Year == current_year, MonthName == current_month)
  df_train<-data.frame(df_train)
  
  df_fine <- fine_data %>%
    filter(Year == current_year, MonthName == current_month)
  
  df_fine<-data.frame(df_fine)
  
  
  
  set.seed(456)
  splitIndex <- createDataPartition(df_train$TWSA, p = 0.8, list = FALSE)
  train_data <- df_train[splitIndex, ]
  test_data <- df_train[-splitIndex, ]
  
  
  # install.packages("rpart")
  library(rpart)
  
 
  cart <- caret::train(TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp, 
                           data=train_data, method="rpart2", 
                           trControl=caret::trainControl(method="none"), 
                           tuneGrid=data.frame(maxdepth=6),
                           metric="RMSE")
  
  
  
  # Make predictions on the training set
  train_predictions <- predict(cart, newdata = train_data)
  # Make predictions on the testing set
  
  test_predictions <- predict(cart, newdata = test_data)
  
  # Make predictions on the whole dataset
  
  test_predictions_whole <- predict(cart, newdata = df_train)
  
  
  FIPS.xy_CoarseRes <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = test_predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  
  
  coordinates(FIPS.xy_CoarseRes) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes) <- TRUE
  r_pred <- raster(FIPS.xy_CoarseRes, layer = "Predicted_Tws")
  r_obs  <- raster(FIPS.xy_CoarseRes, layer = "Observed_TWS")
  r_res  <- raster(FIPS.xy_CoarseRes, layer = "Residuals")
  plot(r_obs)
  plot(r_pred)
  
  
  # Predict on fine resolution
  
  #####develop model for downscalng based on model cross-valiated in training and testing stage
  
  
  
  cart_whole <- caret::train(TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp, 
                       data=df_train, method="rpart2", 
                       trControl=caret::trainControl(method="none"), 
                       tuneGrid=data.frame(maxdepth=6),
                       metric="RMSE")
  
  
  
  predictions_whole <- predict(cart_whole, newdata = df_train)
  
  
  FIPS.xy_CoarseRes_1 <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  coordinates(FIPS.xy_CoarseRes_1) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes_1) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes_1) <- TRUE
  r_pred_calibrated <- raster(FIPS.xy_CoarseRes_1, layer = "Predicted_Tws")
  r_res_calibrated  <- raster(FIPS.xy_CoarseRes_1, layer = "Residuals")
  
  
  
  fine_pred <- predict(cart_whole, newdata = df_fine)
  
  df_fine$fine_pred<-fine_pred
  
  
  
  
  xmin <- min(df_fine$X)
  xmax <- max(df_fine$X)
  ymin <- min(df_fine$Y)
  ymax <- max(df_fine$Y)
  res <- 0.00449
  
  # 1. Define raster with known extent and resolution
  r_template <- raster(xmn = xmin, xmx = xmax, ymn = ymin,  ymx = ymax, resolution = res, crs = "+proj=longlat +datum=WGS84")
  
  # 2. Convert point data to SpatialPointsDataFrame
  coordinates(df_fine) <- ~X + Y
  proj4string(df_fine) <- CRS("+proj=longlat +datum=WGS84")
  
  # 3. Rasterize using field 'fine_pred'
  r_fine_pred <- rasterize(df_fine, r_template, field = "fine_pred", fun = mean)
  
  
  r_fine_res<- resample(r_res_calibrated, r_fine_pred, method="bilinear")
  
  
  
  corrected_downscaled_TWSA <- r_fine_pred + r_fine_res
  
  plot(r_obs)
  plot(r_pred)
  plot(r_fine_res)
  plot(r_fine_pred)
  plot(corrected_downscaled_TWSA)
  
  # Save rasters

  # Save rasters (CART version)
  
  save_base1 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/ModelEvaluationTrainTesting/TrainingTesting/CART_TWSA_", current_year, "_", current_month)
  save_base2 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/ModelEvaluationTrainTesting/Observed/CART_TWSA_", current_year, "_", current_month)
  save_base3 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/ModelEvaluationTrainTesting/Residuals/CART_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred, paste0(save_base1, "_CART_PRED25km.tif"), overwrite = TRUE)
  writeRaster(r_obs,  paste0(save_base2, "_CART_OBS25km.tif"),  overwrite = TRUE)
  writeRaster(r_res,  paste0(save_base3, "_CART_RES25km.tif"),  overwrite = TRUE)
  
  save_base4 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/FinePredictionDownscalingCorrection/Predicted/CART_TWSA_", current_year, "_", current_month)
  save_base5 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/FinePredictionDownscalingCorrection/PredtiveError25km/CART_TWSA_", current_year, "_", current_month)
  save_base6 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/FinePredictionDownscalingCorrection/PredtiveError1km/CART_TWSA_", current_year, "_", current_month)
  save_base7 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/FinePredictionDownscalingCorrection/Downscaled/CART_TWSA_", current_year, "_", current_month)
  save_base8 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/FinePredictionDownscalingCorrection/DownscaledCorrected/CART_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred_calibrated,         paste0(save_base4, "_CART_PREDCalibrated25km.tif"),        overwrite = TRUE)
  writeRaster(r_res_calibrated,          paste0(save_base5, "_CART_RESCalibrated25km.tif"),         overwrite = TRUE)
  writeRaster(r_fine_res,                paste0(save_base6, "_CART_RESCALIBRATED1km.tif"),          overwrite = TRUE)
  writeRaster(r_fine_pred,               paste0(save_base7, "_CART_Downscaled1km.tif"),             overwrite = TRUE)
  writeRaster(corrected_downscaled_TWSA, paste0(save_base8, "_CART_DownscaledCorrected1km.tif"),    overwrite = TRUE)
  
  
  
  
  ###Export data in excel
  
  # Create Date column
  date_str <- paste0(current_year, "-", current_month)
  
  # Collect Training Results
  
  EXPORT_TRAINING <- data.frame(
    X = train_data$X,
    Y = train_data$Y,
    Observed = train_data$TWSA,
    Predicted = train_predictions,
    Date = date_str
  )
  
  # Collect Testing Results
  
  EXPORT_TESTING <- data.frame(
    X = test_data$X,
    Y = test_data$Y,
    Observed = test_data$TWSA,
    Predicted = test_predictions,
    Date = date_str
  )
  
  # Collect Whole Results
  EXPORT_WHOLE <- data.frame(
    X = df_train$X,
    Y = df_train$Y,
    Observed = df_train$TWSA,
    Predicted = test_predictions_whole,
    Date = date_str
  )
  
  # Append to full results
  all_train_results <- rbind(all_train_results, EXPORT_TRAINING)
  all_test_results  <- rbind(all_test_results, EXPORT_TESTING)
  all_whole_results <- rbind(all_whole_results, EXPORT_WHOLE)
  
  
  message("✅ Done.")
}



# Create workbook
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "Training_performance")
openxlsx::writeData(wb, "Training_performance", all_train_results)

openxlsx::addWorksheet(wb, "Testing_performance")
openxlsx::writeData(wb, "Testing_performance", all_test_results)

openxlsx::addWorksheet(wb, "Whole_data_performance")
openxlsx::writeData(wb, "Whole_data_performance", all_whole_results)

openxlsx::saveWorkbook(
  wb,
  file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/CART/CART_model_performance_all_months_2.xlsx",
  overwrite = TRUE
)


print("✅ Excel workbook saved for all months.")


library(ggplot2)
library(Metrics)

# === Evaluation plots ===
rsq <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# Plotting function
plot_perf <- function(obs, pred, title_text) {
  df <- data.frame(Observed = obs, Predicted = pred)
  r2 <- round(1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2), 3)
  rmse_val <- round(rmse(obs, pred), 2)
  bias_val <- round(mean(pred - obs), 2)
  
  ggplot(df, aes(x = Observed, y = Predicted)) +
    xlim(-250, 250) + ylim(-250, 250) +
    geom_point(color = "blue", size = 4, shape = 21, fill = "skyblue", stroke = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -250, y = 250,  # 👈 Move stats box to top-right
             label = paste0("R² = ", r2, "\nRMSE = ", rmse_val, "\nBias = ", bias_val),
             size = 5, hjust = 0, vjust = 1, fontface = "bold") +
    labs(title = title_text,
         x = "Observed TWSA (mm)",
         y = "Predicted TWSA (mm)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 13),
      panel.border = element_rect(color = "black", fill = NA, size = 1.2)
    ) +
    coord_fixed(ratio = 1.5)
}


plot_perf(all_train_results$Observed, all_train_results$Predicted, "CART: Training Set")
plot_perf(all_test_results$Observed, all_test_results$Predicted, "CART: Testing Set")
plot_perf(all_whole_results$Observed, all_whole_results$Predicted, "CART: Whole Dataset")




########################################################### R Script: ANN (single-hidden-layer) Downscaling with Loop Over All Time Steps##########################################################

# Load required libraries
library(xgboost)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(keras)
library(caret)
library(openxlsx)
library(xlsx)
library(raster)
library(readr)
library(Metrics)
library(sp)
library(readxl)
library(e1071) # Required for SVM
library(dplyr)
library(caret)
library(openxlsx)




# Load coarse and fine datasets
coarse_data <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
fine_data   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")
coarse_data <- dplyr::rename(coarse_data,
                             TWSA        = lwe_thickness,
                             Temperature = `Air temperature`,
                             ET          = Evapotranspiration_openet,
                             Elevation = Elevation,
                             NDVI         = `NDVI`,
                             LST         = `Land surface temperature`,
                             GPP          = `gross primary production`,
                             Prcp          = `Precipitation`
)


fine_data <- dplyr::rename(fine_data,
                           Temperature = `Air temperature`,
                           ET          = Evapotranspiration_openet,
                           Elevation = Elevation,
                           NDVI         = `NDVI`,
                           LST         = `Land surface temperature`,
                           GPP          = `gross primary production`,
                           Prcp          = `Precipitation`
)


coarse_data[is.na(coarse_data)] <- -999
fine_data[is.na(fine_data)]     <- -999

# Extract Year and Month
coarse_data <- coarse_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

fine_data <- fine_data %>%
  mutate(MonthName = sub(".*-", "", Month),
         Year = as.numeric(sub("-.*", "", Month)) + 2000)

# Define start and end year/month
# Define start and end
start_year <- 2007
start_month <- "Mar"
end_year <- 2007
end_month <- "Jun"

# Ensure MonthName is an ordered factor
month_levels <- month.abb
time_steps <- unique(dplyr::select(coarse_data, Year, MonthName)) %>%
  mutate(
    MonthName = factor(MonthName, levels = month.abb, ordered = TRUE)
  ) %>%
  arrange(Year, MonthName) %>%
  filter(
    (Year > start_year | (Year == start_year & MonthName >= start_month)) &
      (Year < end_year | (Year == end_year & MonthName <= end_month))
  )


# Initialize empty data frames to collect results
all_train_results <- data.frame()
all_test_results <- data.frame()
all_whole_results <- data.frame()


# Loop over each time step
for (i in 1:nrow(time_steps)) {
  current_year <- time_steps$Year[i]
  current_month <- as.character(time_steps$MonthName[i])
  message(paste0("🔄 Processing ", current_year, "-", current_month, "..."))
  
  df_train <- coarse_data %>%
    filter(Year == current_year, MonthName == current_month)
  df_train<-data.frame(df_train)
  
  df_fine <- fine_data %>%
    filter(Year == current_year, MonthName == current_month)
  
  df_fine<-data.frame(df_fine)
  
  
  
  set.seed(456)
  splitIndex <- createDataPartition(df_train$TWSA, p = 0.8, list = FALSE)
  train_data <- df_train[splitIndex, ]
  test_data <- df_train[-splitIndex, ]
  
  

  ANN <- caret::train(TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp,
                                  data=train_data, method="nnet",
                                  trControl=caret::trainControl(method="cv", number=5),
                                  preProcess=c("center","scale"),
                                  tuneGrid=expand.grid(size=c(8,12,20), decay=c(1e-3,1e-2,1e-1)),
                                  metric="RMSE", linout=TRUE, trace=FALSE, MaxNWts=50000, maxit=2000)
  
  
  
  # Make predictions on the training set
  train_predictions <- predict(ANN, newdata = train_data)
  # Make predictions on the testing set
  
  test_predictions <- predict(ANN, newdata = test_data)
  
  # Make predictions on the whole dataset
  
  test_predictions_whole <- predict(ANN, newdata = df_train)
  
  
  FIPS.xy_CoarseRes <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = test_predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  
  
  coordinates(FIPS.xy_CoarseRes) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes) <- TRUE
  r_pred <- raster(FIPS.xy_CoarseRes, layer = "Predicted_Tws")
  r_obs  <- raster(FIPS.xy_CoarseRes, layer = "Observed_TWS")
  r_res  <- raster(FIPS.xy_CoarseRes, layer = "Residuals")
  plot(r_obs)
  plot(r_pred)
  
  
  # Predict on fine resolution
  
  #####develop model for downscalng based on model cross-valiated in training and testing stage
  
  
  
  ANN_whole <- caret::train(TWSA ~ Temperature + Elevation + ET + NDVI + LST + GPP + Prcp,
                             data=df_train, method="nnet",
                             trControl=caret::trainControl(method="cv", number=5),
                             preProcess=c("center","scale"),
                             tuneGrid=expand.grid(size=c(8,12,20), decay=c(1e-3,1e-2,1e-1)),
                             metric="RMSE", linout=TRUE, trace=FALSE, MaxNWts=50000, maxit=2000)
  
  
  
  predictions_whole <- predict(ANN_whole, newdata = df_train)
  
  
  FIPS.xy_CoarseRes_1 <- df_train %>%
    dplyr::select(X, Y) %>%
    mutate(Predicted_Tws = predictions_whole,
           Observed_TWS = df_train$TWSA,
           Residuals = Observed_TWS - Predicted_Tws)
  coordinates(FIPS.xy_CoarseRes_1) <- ~X + Y
  proj4string(FIPS.xy_CoarseRes_1) <- CRS("+proj=longlat +datum=WGS84")
  gridded(FIPS.xy_CoarseRes_1) <- TRUE
  r_pred_calibrated <- raster(FIPS.xy_CoarseRes_1, layer = "Predicted_Tws")
  r_res_calibrated  <- raster(FIPS.xy_CoarseRes_1, layer = "Residuals")
  
  
  
  fine_pred <- predict(ANN_whole, newdata = df_fine)
  
  df_fine$fine_pred<-fine_pred
  
  
  
  
  xmin <- min(df_fine$X)
  xmax <- max(df_fine$X)
  ymin <- min(df_fine$Y)
  ymax <- max(df_fine$Y)
  res <- 0.00449
  
  # 1. Define raster with known extent and resolution
  r_template <- raster(xmn = xmin, xmx = xmax, ymn = ymin,  ymx = ymax, resolution = res, crs = "+proj=longlat +datum=WGS84")
  
  # 2. Convert point data to SpatialPointsDataFrame
  coordinates(df_fine) <- ~X + Y
  proj4string(df_fine) <- CRS("+proj=longlat +datum=WGS84")
  
  # 3. Rasterize using field 'fine_pred'
  r_fine_pred <- rasterize(df_fine, r_template, field = "fine_pred", fun = mean)
  
  
  r_fine_res<- resample(r_res_calibrated, r_fine_pred, method="bilinear")
  
  
  
  corrected_downscaled_TWSA <- r_fine_pred + r_fine_res
  
  plot(r_obs)
  plot(r_pred)
  plot(r_fine_res)
  plot(r_fine_pred)
  plot(corrected_downscaled_TWSA)
  
  # Save rasters
  
  # Save rasters (CART version)
  
  # Save rasters (ANN version)
  
  save_base1 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/ModelEvaluationTrainTesting/TrainingTesting/ANN_TWSA_", current_year, "_", current_month)
  save_base2 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/ModelEvaluationTrainTesting/Observed/ANN_TWSA_", current_year, "_", current_month)
  save_base3 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/ModelEvaluationTrainTesting/Residuals/ANN_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred, paste0(save_base1, "_ANN_PRED25km.tif"), overwrite = TRUE)
  writeRaster(r_obs,  paste0(save_base2, "_ANN_OBS25km.tif"),  overwrite = TRUE)
  writeRaster(r_res,  paste0(save_base3, "_ANN_RES25km.tif"),  overwrite = TRUE)
  
  save_base4 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/FinePredictionDownscalingCorrection/Predicted/ANN_TWSA_", current_year, "_", current_month)
  save_base5 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/FinePredictionDownscalingCorrection/PredtiveError25km/ANN_TWSA_", current_year, "_", current_month)
  save_base6 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/FinePredictionDownscalingCorrection/PredtiveError1km/ANN_TWSA_", current_year, "_", current_month)
  save_base7 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/FinePredictionDownscalingCorrection/Downscaled/ANN_TWSA_", current_year, "_", current_month)
  save_base8 <- paste0("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/FinePredictionDownscalingCorrection/DownscaledCorrected/ANN_TWSA_", current_year, "_", current_month)
  
  writeRaster(r_pred_calibrated,         paste0(save_base4, "_ANN_PREDCalibrated25km.tif"),        overwrite = TRUE)
  writeRaster(r_res_calibrated,          paste0(save_base5, "_ANN_RESCalibrated25km.tif"),         overwrite = TRUE)
  writeRaster(r_fine_res,                paste0(save_base6, "_ANN_RESCALIBRATED1km.tif"),          overwrite = TRUE)
  writeRaster(r_fine_pred,               paste0(save_base7, "_ANN_Downscaled1km.tif"),             overwrite = TRUE)
  writeRaster(corrected_downscaled_TWSA, paste0(save_base8, "_ANN_DownscaledCorrected1km.tif"),    overwrite = TRUE)
  
  
  
  
  
  ###Export data in excel
  
  # Create Date column
  date_str <- paste0(current_year, "-", current_month)
  
  # Collect Training Results
  
  EXPORT_TRAINING <- data.frame(
    X = train_data$X,
    Y = train_data$Y,
    Observed = train_data$TWSA,
    Predicted = train_predictions,
    Date = date_str
  )
  
  # Collect Testing Results
  
  EXPORT_TESTING <- data.frame(
    X = test_data$X,
    Y = test_data$Y,
    Observed = test_data$TWSA,
    Predicted = test_predictions,
    Date = date_str
  )
  
  # Collect Whole Results
  EXPORT_WHOLE <- data.frame(
    X = df_train$X,
    Y = df_train$Y,
    Observed = df_train$TWSA,
    Predicted = test_predictions_whole,
    Date = date_str
  )
  
  # Append to full results
  all_train_results <- rbind(all_train_results, EXPORT_TRAINING)
  all_test_results  <- rbind(all_test_results, EXPORT_TESTING)
  all_whole_results <- rbind(all_whole_results, EXPORT_WHOLE)
  
  
  message("✅ Done.")
}



# Create workbook
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "Training_performance")
openxlsx::writeData(wb, "Training_performance", all_train_results)

openxlsx::addWorksheet(wb, "Testing_performance")
openxlsx::writeData(wb, "Testing_performance", all_test_results)

openxlsx::addWorksheet(wb, "Whole_data_performance")
openxlsx::writeData(wb, "Whole_data_performance", all_whole_results)

openxlsx::saveWorkbook(
  wb,
  file = "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/ANN/ANN_model_performance_all_months_2.xlsx",
  overwrite = TRUE
)


print("✅ Excel workbook saved for all months.")


library(ggplot2)
library(Metrics)

# === Evaluation plots ===
rsq <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# Plotting function
plot_perf <- function(obs, pred, title_text) {
  df <- data.frame(Observed = obs, Predicted = pred)
  r2 <- round(1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2), 3)
  rmse_val <- round(rmse(obs, pred), 2)
  bias_val <- round(mean(pred - obs), 2)
  
  ggplot(df, aes(x = Observed, y = Predicted)) +
    xlim(-250, 250) + ylim(-250, 250) +
    geom_point(color = "blue", size = 4, shape = 21, fill = "skyblue", stroke = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -250, y = 250,  # 👈 Move stats box to top-right
             label = paste0("R² = ", r2, "\nRMSE = ", rmse_val, "\nBias = ", bias_val),
             size = 5, hjust = 0, vjust = 1, fontface = "bold") +
    labs(title = title_text,
         x = "Observed TWSA (mm)",
         y = "Predicted TWSA (mm)") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 13),
      panel.border = element_rect(color = "black", fill = NA, size = 1.2)
    ) +
    coord_fixed(ratio = 1.5)
}


plot_perf(all_train_results$Observed, all_train_results$Predicted, "ANN: Training Set")
plot_perf(all_test_results$Observed, all_test_results$Predicted, "ANN: Testing Set")
plot_perf(all_whole_results$Observed, all_whole_results$Predicted, "ANN: Whole Dataset")

