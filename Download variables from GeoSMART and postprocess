# ===============================================================
# USER GUIDELINES:
# This R script extracts and compiles coarse and fine-resolution 
# environmental raster variables for selected months from 2002 to 2011. 
# It reads GeoTIFF files from designated folders, extracts data 
# at specified XY locations, and skips any months where fewer 
# than 5 raster layers are available (to avoid incomplete data).
# 
# Missing months are explicitly defined in the script based on data availability.
# Output CSV files contain only the first 14 relevant columns (XY coordinates, 
# key variables, and metadata). Be sure to update file paths as needed.
# ===============================================================



# Step 1: === Download Data from GeoSMART =======================================
     # Go to GeoSMART tool and download coarse and fine scale variables 
     #coarsevariables folder contains all raster images of coarse variables (e.g.., 110 km) of GRACE fields (TWS/GWS as target variable) as well as predictors (e.g., NDVI, LST, rainfall, air temperature) 
     #finevariables folder contains all raster images of fine scale  (e.g.., 1 km) predictors (e.g., NDVI, LST, rainfall, air temperature)

# Step 2: === extract raster variables to csv files in a appropriate format for preparing input to ML/AI models =======================================

library(terra)
library(dplyr)
library(readr)
library(stringr)

# === Paths to coarse and fine scale variables ===
coarse_folder <- "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/coarsevariables"
fine_folder   <- "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/finevariables"


#Preparing coarse and fine coordinates
		#Go to ArcGIS, open one fine scale and one coarse scale raster image. convert both images from raster to point using ArcGIS tool. it will generate points (coarse and fine) centered to girds/pixels. 
		#Open the attribute tables of points and then insert a two new columns of X and Y coordinates and generate coordinates (latitude and longitude). 
		#Now export point shapefile of coarse and fine scale to csv. 
		#XYcoordinatesCoarse.csv file contains ObjectID, Xcoord and Ycoord columns of coarse variables. 
		#XYcoordinatesFine.csv file contains ObjectID, Xcoord and Ycoord columns of fine variables. 
		#Remember each Xcoord and Ycoord represents center of the grid/pixel 

# === Paths to coarse and fine scale variables coordinates ===
coords_coarse <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XYcoordinatesCoarse.csv")
coords_fine   <- read_csv("E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/XYcoordinatesFine.csv")

# === Function to extract variables by date tag ===
extract_variables_by_date <- function(date_tag, is_fine = FALSE) {
  folder <- if (is_fine) fine_folder else coarse_folder
  pattern <- paste0(date_tag, "_.*\\.tif$")
  files <- list.files(folder, pattern = pattern, full.names = TRUE)
  
  if (is_fine) {
    files <- files[!grepl("lwe_thickness", files)]
  }
  
  if (length(files) < 5) {
    message("⛔ Skipping ", date_tag, " — not enough rasters.")
    return(NULL)
  }
  
  coords_df <- if (is_fine) coords_fine else coords_coarse
  
  rasters <- lapply(files, function(f) {
    r <- tryCatch(rast(f), error = function(e) NULL)
    if (!is.null(r)) {
      var_name <- sub(paste0(".*", date_tag, "_"), "", basename(f)) |> sub("\\.tif$", "", x = _)
      names(r) <- var_name
    }
    return(r)
  })
  
  rasters <- Filter(Negate(is.null), rasters)
  if (length(rasters) == 0) return(NULL)
  
  stack <- do.call(c, rasters)
  points <- vect(coords_df, geom = c("X", "Y"), crs = "EPSG:4326")
  values <- terra::extract(stack, points)[, -1]
  colnames(values) <- names(stack)
  coords_df <- bind_cols(coords_df, values)
  
  year <- as.numeric(substr(date_tag, 1, 4))
  month_num <- as.numeric(substr(date_tag, 5, 6))
  month_abbr <- month.abb[month_num]
  coords_df$Month <- paste0(year - 2000, "-", month_abbr)
  
  return(coords_df)
}

# === Remember to mention missing months of GRACE data ===If you are extracting data ahead of 2012, remember to include any missing months to the list below
missing_months <- list(
  `2002` = c(1, 2, 3, 6, 7),
  `2003` = c(6),
  `2004` = integer(0),
  `2005` = integer(0),
  `2006` = integer(0),
  `2007` = integer(0),
  `2008` = integer(0),
  `2009` = integer(0),
  `2010` = integer(0),
  `2011` = c(1, 6, 12),
  `2012` = c(5, 10)
)

# === Generate valid date tags using base R only ===
years <- 2002:2012
all_tags <- unlist(lapply(years, function(y) {
  miss <- missing_months[[as.character(y)]]
  if (is.null(miss)) miss <- integer(0)
  months <- setdiff(1:12, miss)
  sprintf("%d%02d", y, months)
}))

# === Extract all available months
coarse_list <- lapply(all_tags, extract_variables_by_date, is_fine = FALSE)


fine_list   <- lapply(all_tags, extract_variables_by_date, is_fine = TRUE)

coarse_combined <- bind_rows(Filter(Negate(is.null), coarse_list))
fine_combined   <- bind_rows(Filter(Negate(is.null), fine_list))

coarse_combined_subset <- coarse_combined %>% select(1:14)
fine_combined_subset <- fine_combined %>% select(1:14)


# === Save outputs
write_csv(coarse_combined_subset, "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_coarse_2002_2012.csv")
write_csv(fine_combined_subset,   "E:/Study (F)/NCAR Postdoc Work/Analysis and Results/GRACE DOWNSCALING/GRACETWSandVARIABLES/test_fine_2002_2012.csv")

message("✅ Export completed for 2002–20112(skipping missing months).")



