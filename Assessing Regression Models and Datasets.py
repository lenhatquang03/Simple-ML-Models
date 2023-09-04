import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Type
import patsy

from ISLP.models import (ModelSpec as MS, poly)
from ISLP import load_data

# Diagnostic Plots
class LinearRegDiagnostic():
    def __init__(self, results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        if isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError("result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")
        # statsmodels.regression.linear_model.RegressionResultWrapper is a class summarizing the fit of a linear regression model. It handles the output of contrasts, estimates of covariance, etc.
        
        self.results = maybe_unwrap_results(results)
        # This function is used to get raw results back from wrapped results so that they can be used in plotting functions or other post-estimation type routines.
        
        self.y_true = self.results.model.endog
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names
        # Endogenous: caused by factors within the system -> response
        # Exogenous: caused by factors outside the system -> predictors
        self.y_predict = self.results.fittedvalues
        
        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        # This instance contains methods to calculate the main influence and outlier measures for the OLS regression.
        self.standardized_resid = influence.resid_studentized_internal
        # Standardized residuals using variance from OLS. 
        self.leverage = influence.hat_matrix_diag
        # Diagonal of the hat_matrix for OLS. It is defined as the matrix that converts values from the observed variable into estimators obtained with the least squares method.
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)
        # params are the linear coefficients that minimize the least squares criterion
        self.nresids = len(self.standardized_resid)

    
    def __call__(self, plot_context = 'seaborn-paper', **kwargs):
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (13, 13))
            self.residual_plot(ax = ax[0][0])
            self.qq_plot(ax = ax[0][1])
            self.scale_location_plot(ax = ax[1][0])
            self.leverage_plot(ax = ax[1][1], high_leverage_threshold = kwargs.get('high_leverage_threshold'), cooks_threshold = kwargs.get('cooks_threshold'))
            plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
            fig.suptitle('Diagnostic Plots', fontweight = 'bold')
            plt.show()
        return self.vif_table(), fig, ax
    
    
    def residual_plot(self, ax = None):
        """
        Residals versus Fitted Values
        Graphical tool to identify non-linearity
        Horizontal blue line is an indicator that the residual has a linear pattern
        """
        if ax is None: # To independently make this individually
            fig, ax = plt.subplots()
            
        sns.residplot(x = self.y_predict, y = self.residual, lowess = True, scatter_kws = {'color': 'red', 'linewidths': 1,'edgecolors': 'black', 'alpha': 0.5}, line_kws = {'color': 'blue', 'lw': 1, 'ls': '-'}, ax = ax)
        # Lowess smoothing (locally weighted scatterplot smoothing) uses a robust weighting scheme to reduce the impact of outliers.
        
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.argsort(residual_abs), 0)
        # np.flip(array-like, axis = None): reverse the order of elements in an array along the given axis, the shape of the array is preserved.
        # np.argsort(a, kind = 'quicksort): returns the indices the would sort an array from smallest to largest
        abs_resid_top_3 = abs_resid[:3]
        for i in abs_resid_top_3:
            ax.annotate(i, xy = (self.y_predict[i], self.residual[i]), color = 'black')
            # Annotate the point xy with text "text"
        ax.set_title('Residuals vs Fitted', fontweight = 'bold')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        return ax
    
    def qq_plot(self, ax = None):
        '''
        Standardized Residual versus Theoretical Quantile plot
        Used to visually check if the residuals are normally distributed, thus support the linear assumption that the error term follows a normal distribution.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        
        QQ = ProbPlot(self.standardized_resid)
        # ProbPlot(data, dist=<scipy.stats._continuous_distns.norm_gen object>) compares "data" against "dist" to generate Q-Q and P-P Probability Plots.
        figure = QQ.qqplot(markerfacecolor = 'red', markeredgewidth = 1, markeredgecolor = 'black', alpha = 0.5, ax = ax)
        sm.qqline(figure.axes[1], line = '45', color='blue', linestyle='dashed', lw = 1)
        # line: {None, '45', 's', 'r', 'q'} = Options for the reference line to which the data is compared.
        abs_norm_resid = np.flip(np.argsort(np.abs(self.standardized_resid)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for i, x, y in self.__qq_top_resid(QQ.theoretical_quantiles, abs_norm_resid_top_3):
            ax.annotate(i, xy = (x, y), ha = 'right', color = 'black')
            # ha = horizontal alignment
        ax.set_title('Normal Q-Q', fontweight = 'bold')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax
    
    def scale_location_plot(self, ax = None):
        '''
        Sqrt(Standardized Residual) versus Fitted Values
        Used to check homoscedascity (constant or similar variance) of the residuals  
        '''
        if ax is None:
            fig, ax = plt.subplots()
        
        residual_norm_abs_sqrt = np.sqrt(np.abs(self.standardized_resid))
        ax.scatter(self.y_predict, residual_norm_abs_sqrt, color = 'red', edgecolors = 'black', linewidths= 1, alpha = 0.5)
        sns.regplot(x = self.y_predict, y = residual_norm_abs_sqrt, scatter = False, ci = False, lowess = True, line_kws = {'color': 'blue', 'lw': 1}, ax = ax)
        # ci: size of the confidence interval for the regression estimate. For large datasets, it may be advisable to avoid that computation by setting this parameter to None.
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(i, xy = (self.y_predict[i], residual_norm_abs_sqrt[i]), color = 'black')
        
        ax.set_title('Scale-Location', fontweight = 'bold')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
        return ax
    
    def leverage_plot(self, ax = None, high_leverage_threshold = False, cooks_threshold = 'baseR'):
        '''
        Residuals versus Leverage plot
        Points falling outside Cook's distance curves are considered influential observations
        '''
        if ax is None:
            fig, ax = plt.subplots()
            
        ax.scatter(self.leverage, self.standardized_resid, color = 'red', linewidths= 1, edgecolors= 'black', alpha= 0.5)
        sns.regplot(x = self.leverage, y = self.standardized_resid, scatter = False, lowess = True, ci = False, line_kws = {'color': 'blue', 'lw': 1}, ax = ax)
        
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(i, xy = (self.leverage[i], self.standardized_resid[i]), color = 'black')
        
        factors = []
        if cooks_threshold == 'baseR' or cooks_threshold is None:
            factors = [1, 0.5]
        elif cooks_threshold == 'convention':
            factors = [4/self.nresids]
        elif cooks_threshold == 'dof':
            factors = [4/(self.nresids - self.nparams)]
        else:
            raise ValueError('threshold_method must be one if the following: "convention", "dof" or "baseR" (default)')
        
        for i, factor in enumerate(factors):
            label = "Cook's distance" if i == 0 else None
            xtemp, ytemp = self.__cooks_dist_line(factor)
            ax.plot(xtemp, ytemp, label = label, lw = 1.25, ls = '--', color = 'green')
            ax.plot(xtemp, np.negative(ytemp), lw = 1.25, ls = '--', color = 'green')
        
        if high_leverage_threshold:
            high_leverage = 2*self.nparams/self.nresids
            if max(self.leverage) > high_leverage:
                ax.axvline(high_leverage, label = 'High leverage', ls = '-.', color = 'purple', lw = 1)
                ax.axhline(0, ls = 'dotted', color = 'black', lw = 1.25)
        
        ax.set_xlim(0, max(self.leverage) + 0.01)
        ax.set_ylim(min(self.standardized_resid) - 0.1, max(self.standardized_resid) + 0.1)
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.set_title('Residuals vs Leverage', fontweight = 'bold')
        plt.legend(loc = 'best')
        return ax
    
    def vif_table(self):
        '''
        VIF, the variance inflation factor, is a measure of multicollinearity. 
        VIF > 5 for a variable indicates that it is highly collinear with other input variables.
        '''
        vif_df = pd.DataFrame()
        vif_df['Features'] = self.xvar_names
        vif_df['VIF Factor'] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]
        return (vif_df.sort_values('VIF Factor').round(2))
    
    def __cooks_dist_line(self, factor):
        '''
        Helper function for plotting Cook's distance curves
        '''
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1-x))/x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y
    
    def __qq_top_resid(self, quantiles, top_residual_indices):
        '''Helper generator function yielding the index and coordinates.
        '''
        offset = 0
        quant_index = 0
        previous_is_negative = None
        for resid_index in top_residual_indices:
            y = self.standardized_resid[resid_index]
            is_negative = y < 0
            if previous_is_negative == None or previous_is_negative == is_negative:
                offset += 1
            else:
                quant_index -= offset
            x = quantiles[quant_index] if is_negative else np.flip(quantiles, 0)[quant_index]
            quant_index += 1
            previous_is_negative = is_negative
            yield resid_index, x, y

def percentage_error(regression_model, y, n, p):
    residual_squared = [i**2 for i in list(regression_model.resid)]
    RSE = np.sqrt(sum(residual_squared)/(n-p-1))
    return RSE/(y.mean())

# Exercise 1: The use of a simple linear regression model on the Auto data set.

auto = load_data('Auto')
cols = list(auto.columns)
y, X = auto['mpg'], auto['horsepower']
X = sm.add_constant(X)
# The function will take a DataFrame or a Series and add an initial column "const" with 1.0 in it. This ensures that our regression line will pass through the y-axis and provides more accurate estimates of our coefficients since without an intercept, the regression line will be forced to pass through the origin.

linear_model = sm.OLS(y, X).fit()
print(linear_model.summary())
print(percentage_error(linear_model, y, 392, 1))


# The p-values generated from the F-statistic and those associated with the intercept term and the mpg coefficient are virtually zero. Hence, they indicate there exists a relationship between the predictor and the response.

# The percentage error is 20.93%, along with the R squared value of approximately 61% show a strong relationship between the predictor and the response.

# The horsepower coefficient is -0.1578. So obviously the relationship between the predictor and the response is negative.

# The predicted mpg associated with a horsepower of 98 is 24.4715. The associated 95% confidence interval and prediction interval are [23.973079, 24.961075] and [14.809396, 34.124758].

prediction = linear_model.get_prediction(exog = [1, 98])
print(prediction.summary_frame())
print(prediction.predicted_mean) 

# Plotting the response and the variable 
sns.lmplot(x = 'horsepower', y = 'mpg', data = auto, scatter_kws = {'color': 'red', 'edgecolors': 'black', 'alpha': 0.5})


# Diagnostic plots
diag_plts = LinearRegDiagnostic(linear_model)
vif, fig, ax = diag_plts()
print(vif)

  # The Residuals vs Fitted plot shows a slight u-shaped, which indicates non-linearity in the data.
  # The Residuals vs Leverage chart shows some possible outliers, we can confirm by using the studentized residuals to find observation with values greater than 3.
results = maybe_unwrap_results(linear_model)
influence = results.get_influence()
studentized_residuals = influence.resid_studentized_external
auto['Studentized Residuals'] = studentized_residuals
print(auto[auto['Studentized Residuals'] > 3])

# Exercise 2: The use of multiple linear regression on the Auto data set

# A scatterplot matrix which includes all of the variables in the data set
sns.pairplot(data = auto, diag_kind = 'hist')

# The matrix of correlations 
print(auto.corr())

# Multiple linear regression fit
X, y = auto.drop(['mpg', 'name'], axis = 1), auto['mpg']
X = MS(X).fit_transform(auto)
# MS() creates a transform object, then the fit() method takes the original array and may fo some computaions on it to fit it in the dataset. The transform() method applies the fitted transformation to the array of data and produces the model matrix. 
mul_model = sm.OLS(y, X).fit()
print(mul_model.summary())
   # The p-value generated from the F-statistic is virtually zero, which indicates a relationship between the predictors and the response
   # Based on each predictor's p-value, one can see that the "displacement", "weight", "year" and "origin" predictors appear to have a statistically significant relationship to the response.
   # The coefficient of the "year" predictor suggests that a one-unit increase in "year" will result in an increase of 0.7508 in "mpg", holding all other features fixed.

# Diagnostic plots 
diag_plots = LinearRegDiagnostic(mul_model)
vif, fig, ax = diag_plots()
print(vif)
   # The Residuals vs Fitted plot does suggest unusually large outliers, namely observations 320, 323 and 324. It also indicates non-linearity in our dataset.
   # The scale-location plot shows heteroscedascity. Could be improved by plotting a concave function of the response.
   # The Residuals vs Leverage plot identifies many observations with unsually high leverage, namely observations 13.
   # The VIF tables detects collinearity between predictors such as "horsepower", "cylinders", "weight" and "displacement"
   
# Model adjustment
adjusted_X = MS(cols[1:-1] + [('displacement', 'weight'), ('horsepower', 'cylinders')])
adjusted_model = sm.OLS(y, adjusted_X)
print(adjusted_model.summary())
   # The R-squared value increase from 0.821 to 0.866, meaning that 25% more unexplained variance has been explained by our model
   # Both interactions added are statistically significant.

# Polynomial Regression
poly_X = MS(cols[1:-1]+[poly('cylinders', degree=2)] + [('displacement', 'weight'), ('horsepower', 'cylinders')] ).fit_transform(auto)
poly_model = sm.OLS(y, poly_X).fit()
print(poly_model.summary())

# Transformed predictors (not polynomial though)
f = 'mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin + displacement*weight + cylinders*horsepower + np.log(acceleration)'
transfomed_y, transformed_X = patsy.dmatrices(f, auto, return_type = 'dataframe')
model = sm.OLS(y, X).fit()
print(model.summary())