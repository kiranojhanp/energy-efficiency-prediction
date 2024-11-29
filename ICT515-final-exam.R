# ------------------------------------------------------------------------------------------------
# Load Required Packages and Data
# ------------------------------------------------------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(readxl, tidyverse, corrplot, randomForest, caret, gridExtra, knitr)

# Set working directory and load data
setwd("~/Documents/murdoch/s2/ICT515/final-exam")
energy_data <- read_excel("ENB2012_data.xlsx")


# ------------------------------------------------------------------------------------------------
# Data Overview and Pre-processing
# ------------------------------------------------------------------------------------------------
# Initial data checks
cat("Dataset Overview:\n")
kable(data.frame(
  "Columns" = names(energy_data),
  "Type" = sapply(energy_data, class)
), caption = "Data Overview (Columns and Data Types)")

cat("\nDataset Dimensions: ", dim(energy_data), "\n")

# Check for missing values
missing_values <- colSums(is.na(energy_data))
if (sum(missing_values) > 0) {
  kable(data.frame(
    "Variable" = names(missing_values),
    "Missing Count" = missing_values
  ), caption = "Missing Value Summary")
} else {
  cat("No missing values found.\n")
}


# ------------------------------------------------------------------------------------------------
# Correlation Matrix and Visualization
# ------------------------------------------------------------------------------------------------
# Compute Spearman correlation matrix and Pearson p-values
correlation_matrix <- cor(energy_data, method = "spearman")
p_values <- cor.mtest(energy_data, method = "pearson")$p

# Display correlation matrix as a table
kable(round(correlation_matrix, 2), caption = "Spearman Correlation Matrix")

# Visualize correlation matrix
colors <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(correlation_matrix,
         col = colors(200),
         method = "circle", 
         type = "upper", 
         order = "alphabet",
         addCoef.col = "black", 
         tl.col = "black", 
         tl.srt = 45, 
         number.cex = 0.8, 
         is.corr = FALSE, 
         title = "Correlation Matrix of Energy Efficiency Data", 
         mar = c(0, 0, 1, 0),
         p.mat = p_values, sig.level = 0.05,diag=FALSE)


# ------------------------------------------------------------------------------------------------
# Cross-validation Setup
# ------------------------------------------------------------------------------------------------
set.seed(123)

# Create 10-fold CV indices
folds <- createFolds(energy_data$Y1, k = 10, list = TRUE)

# Initialize storage for cross-validation results
cv_results <- list(
  lm_Y1 = vector("numeric", 10),
  lm_Y2 = vector("numeric", 10),
  rf_Y1 = vector("numeric", 10),
  rf_Y2 = vector("numeric", 10)
)

# Function to calculate all metrics
calculate_metrics <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  mae <- mean(abs(actual - predicted))
  c(MSE = mse, RMSE = rmse, R2 = r2, MAE = mae)
}

# Create storage for all predictions
all_predictions <- data.frame(
  Fold = numeric(),
  Model = character(),
  Target = character(),
  Actual = numeric(),
  Predicted = numeric()
)

# Perform 10-fold cross-validation
for(i in seq_along(folds)) {
  # Split data into training and validation sets
  train_data <- energy_data[-folds[[i]], ]
  valid_data <- energy_data[folds[[i]], ]
  
  # Train models
  lm_model_Y1 <- lm(Y1 ~ ., data = train_data)
  lm_model_Y2 <- lm(Y2 ~ ., data = train_data)
  rf_model_Y1 <- randomForest(Y1 ~ ., data = train_data, ntree = 500)
  rf_model_Y2 <- randomForest(Y2 ~ ., data = train_data, ntree = 500)
  
  # Make predictions
  pred_lm_Y1 <- predict(lm_model_Y1, valid_data)
  pred_lm_Y2 <- predict(lm_model_Y2, valid_data)
  pred_rf_Y1 <- predict(rf_model_Y1, valid_data)
  pred_rf_Y2 <- predict(rf_model_Y2, valid_data)
  
  # Store results
  cv_results$lm_Y1[i] <- calculate_metrics(valid_data$Y1, pred_lm_Y1)["RMSE"]
  cv_results$lm_Y2[i] <- calculate_metrics(valid_data$Y2, pred_lm_Y2)["RMSE"]
  cv_results$rf_Y1[i] <- calculate_metrics(valid_data$Y1, pred_rf_Y1)["RMSE"]
  cv_results$rf_Y2[i] <- calculate_metrics(valid_data$Y2, pred_rf_Y2)["RMSE"]
  
  # Store predictions for plotting
  fold_predictions <- rbind(
    data.frame(
      Fold = i,
      Model = "Linear Regression",
      Target = "Y1",
      Actual = valid_data$Y1,
      Predicted = pred_lm_Y1
    ),
    data.frame(
      Fold = i,
      Model = "Linear Regression",
      Target = "Y2",
      Actual = valid_data$Y2,
      Predicted = pred_lm_Y2
    ),
    data.frame(
      Fold = i,
      Model = "Random Forest",
      Target = "Y1",
      Actual = valid_data$Y1,
      Predicted = pred_rf_Y1
    ),
    data.frame(
      Fold = i,
      Model = "Random Forest",
      Target = "Y2",
      Actual = valid_data$Y2,
      Predicted = pred_rf_Y2
    )
  )
  
  all_predictions <- rbind(all_predictions, fold_predictions)
}


# ------------------------------------------------------------------------------------------------
# Create Visualizations
# ------------------------------------------------------------------------------------------------
# Function to create scatter plot with additional statistics
create_scatter_plot <- function(data, model_name, target) {
  subset_data <- data[data$Model == model_name & data$Target == target,]
  
  # Calculate R-squared
  r_squared <- round(cor(subset_data$Actual, subset_data$Predicted)^2, 3)
  
  # Calculate RMSE
  rmse <- round(sqrt(mean((subset_data$Actual - subset_data$Predicted)^2)), 3)
  
  ggplot(subset_data, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.5, color = "blue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    theme_minimal() +
    labs(
      title = paste(model_name, "-", target),
      subtitle = paste("R² =", r_squared, "| RMSE =", rmse),
      x = "Actual Values",
      y = "Predicted Values"
    ) +
    coord_fixed(ratio = 1) +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10))
}

# Create individual plots
plot_lm_Y1 <- create_scatter_plot(all_predictions, "Linear Regression", "Y1")
plot_lm_Y2 <- create_scatter_plot(all_predictions, "Linear Regression", "Y2")
plot_rf_Y1 <- create_scatter_plot(all_predictions, "Random Forest", "Y1")
plot_rf_Y2 <- create_scatter_plot(all_predictions, "Random Forest", "Y2")

# Arrange all plots in a grid
grid.arrange(
  plot_lm_Y1, plot_rf_Y1,
  plot_lm_Y2, plot_rf_Y2,
  ncol = 2,
  top = "Actual vs Predicted Values Across All CV Folds"
)


# ------------------------------------------------------------------------------------------------
# Create Residual Plots
# ------------------------------------------------------------------------------------------------
# Add residuals to the prediction data
all_predictions$Residuals <- all_predictions$Predicted - all_predictions$Actual

# Create residual plots
ggplot(all_predictions, aes(x = Predicted, y = Residuals, color = Model)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~Target, scales = "free") +
  theme_minimal() +
  labs(
    title = "Residual Plots by Model and Target",
    x = "Predicted Values",
    y = "Residuals"
  )


# ------------------------------------------------------------------------------------------------
# Summarize Cross-validation Results
# ------------------------------------------------------------------------------------------------
summary_stats <- all_predictions %>%
  group_by(Model, Target) %>%
  summarise(
    RMSE = sqrt(mean((Predicted - Actual)^2)),
    R_squared = cor(Predicted, Actual)^2,
    MAE = mean(abs(Predicted - Actual)),
    Mean_Residual = mean(Residuals),
    SD_Residual = sd(Residuals)
  ) %>%
  ungroup()

# Display summary statistics
kable(summary_stats, 
      caption = "Model Performance Summary Across All CV Folds",
      digits = 4)


# ------------------------------------------------------------------------------------------------
# Train Final Models on Full Dataset
# ------------------------------------------------------------------------------------------------
# Train final models using all data
final_lm_Y1 <- lm(Y1 ~ ., data = energy_data)
final_lm_Y2 <- lm(Y2 ~ ., data = energy_data)
final_rf_Y1 <- randomForest(Y1 ~ ., data = energy_data, ntree = 500)
final_rf_Y2 <- randomForest(Y2 ~ ., data = energy_data, ntree = 500)

# ------------------------------------------------------------------------------------------------
# Feature Importance and Visualization
# ------------------------------------------------------------------------------------------------
# Random Forest Feature Importance
par(mfrow = c(1, 2))
varImpPlot(final_rf_Y1, main = "Feature Importance for Heating Load (Y1)")
varImpPlot(final_rf_Y2, main = "Feature Importance for Cooling Load (Y2)")


# ------------------------------------------------------------------------------------------------
# Outlier Detection and Boxplots
# ------------------------------------------------------------------------------------------------
# Function to create boxplots for a dependent variable (Y1 or Y2)
create_boxplot <- function(data, dependent_var, title_suffix) {
  data %>%
    pivot_longer(cols = starts_with("X"), names_to = "Feature", values_to = "Value") %>%
    ggplot(aes(x = Feature, y = .data[[dependent_var]])) +
    geom_boxplot(aes(fill = Feature), outlier.colour = "red", outlier.shape = 16) +
    theme_minimal() +
    labs(
      title = paste("Boxplot of X1 to X8 Against", title_suffix),
      x = "Independent Variables (X1 to X8)",
      y = paste("Dependent Variable", dependent_var)
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Create boxplots for both Y1 and Y2
boxplot_Y1 <- create_boxplot(energy_data, "Y1", "Y1 (Heating Load)")
boxplot_Y2 <- create_boxplot(energy_data, "Y2", "Y2 (Cooling Load)")

# Arrange the plots vertically
grid.arrange(boxplot_Y1, boxplot_Y2, ncol = 1)
