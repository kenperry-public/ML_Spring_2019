import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split

import functools

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import random 
from sklearn import linear_model


import pdb

class Transformation_Helper():
    def __init__(self, **params):
        return

    def gen_data(self, m=30, mean=30, std=10, slope=5,
                 random_seed=None,
                 attrs=None
                 ):
        if random_seed is not None:
            np.random.seed(random_seed)
       
        length = mean + std*np.random.randn(m) 
        width  = mean + std*np.random.randn(m) 

        area = length * width

        # Clip data
        area = np.where( area < 350, 350, area)
        area = np.where( area > 2200, 2200, area)

        noise =   ( std**2  * np.random.randn(m) ) * slope/2
        price = noise + slope * area

        df = pd.DataFrame( { "length": length,
                             "width":  width,
                             "area":   area,
                             "price":  price
                             }
                           )

        df = df.sort_values(by=["area"])

        if attrs is not None:
            df = df[ attrs ]

        return df

    def LinearSeparate_1d_example(self, max_val=10, num_examples=50, visible=True):
        # Generate one dimensional samples
        rng = np.random.RandomState(42)
        X = rng.uniform(-max_val, +max_val, num_examples)

        # Assign class 1 (Positive) to examples at extremes, 0 (Negative) to middle
        y = np.zeros_like(X)
        y[ X < - max_val//2] = 1
        y[ X >   max_val//2 ] = 1

        # Plot 1d data
        pos_X, neg_X = X[ y >  0 ], X[ y <=  0 ]
        fig_raw,ax_raw = plt.subplots(1,1, figsize=(12,6))
        ax_raw.scatter( neg_X, np.zeros_like(neg_X), color="red",   label="Negative")
        ax_raw.scatter( pos_X, np.zeros_like(pos_X), color="green", label="Positive")
        ax_raw.set_xlabel("$\mathbf{x}_1$")
        ax_raw.legend()
        
        if not visible:
            plt.close(fig_raw)

        # Add second dimension that is square of the first, and plot
        fig_trans,ax_trans = plt.subplots(1,1, figsize=(12,6))
        X2 = X**2
        pos_X2, neg_X2 = X2[ y > 0 ], X2[ y <= 0 ]
        ax_trans.scatter( neg_X, neg_X2, color="red",    label="Negative")
        ax_trans.scatter( pos_X, pos_X2, color="green",  label="Positive")

        ax_trans.set_xlabel("$\mathbf{x}_1$")
        ax_trans.set_ylabel("$\mathbf{x}_1^2$", rotation=0)
        
        if not visible:
            plt.close(fig_trans)

        return fig_raw, ax_raw, fig_trans, ax_trans

class InfluentialPoints_Helper():
    def __init__(self, Debug=False, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        self.Debug = Debug
        
        return

    def fit(self, x,y):
        # Fit the model to x, y
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        
        y_pred = regr.predict(x)
        return (x, y_pred, regr.coef_[0])

    def gen_data(self,num=10):
        rng = self.rng
        
        x = np.linspace(-10, 10, num).reshape(-1,1)
        y =  x + rng.normal(size=num).reshape(-1,1)

        self.x, self.y = x, y
        
        return(x,y)

    
    def plot_update(self, x,y, fitted):
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        plt.ion()

        # Scatter plot of data point
        _ = ax.scatter(x, y, color='black', label="true")

        # Line of best fit
        ax.plot( fitted[0], fitted[1], color='red', label='fit')

        slope = fitted[2]

        ax.set_title("Slope = {s:.2f}".format(s=slope[0]))
        # count += 1
        # if count % max_count == 0:
        fig.canvas.draw()


    def fit_update(self, x_l, y_l):
        Debug = self.Debug
        
        # Retrieve original data
        x, y, fitted = self.x, self.y, self.fitted

        if Debug:
            print("called with ", x_l, y_l)
            
        # Update a single y_value
        # - the element of array y at index x_l will be changed to y_l
        x_update, y_update = x.copy(), y.copy()
        y_update[ x_l ] = y_l

        # Fit the model to the updated data
        fitted = self.fit(x_update,y_update)

        # Plot the points and the fit, on the updated data
        self.plot_update(x_update, y_update, fitted)

    def create_plot_updater(self):
        # Return a plot update function
        # NOTES:
        # - functools.partial does not return an object of the necessary type (bound method_
        # - See:
        # --- https://stackoverflow.com/questions/16626789/functools-partial-on-class-method
        # --- https://stackoverflow.com/questions/48337392/how-to-bind-partial-function-as-method-on-python-class-instances
        # --- so this is our own version of partial
        # - the "interact" method has *named* arguments (e.g., x_l, y_l)
        # -- the function we create must use the *same* names for its formal parameters
        def fn(x_l, y_l):
            return self.fit_update(x_l,y_l)
    
        return fn

    def plot_init(self):
        x, y = self.x, self.y

        # Fit a line to the x,y data
        fitted = self.fit(x,y)

        # Save the fitted data
        self.fitted = fitted

        # Create  function to update the fit and the plot
        plot_updater = self.create_plot_updater()
        return plot_updater
    

    def plot_interact(self, fit_update_fn):
        x, y = self.x, self.y
        num = len(x)
        
        interact(fit_update_fn,
                 x_l=widgets.IntSlider(min=0,  max=num-1,  step=1,
                                       value=int(num/2),
                                       continous_update=False,
                                       description="Index of pt:"
                                       ),
                 y_l=widgets.IntSlider(min=y.min(), max=y.max(), step=1, 
                                       value=y[ int(num/2)], 
                                       continous_update=False,
                                       description="New value:"
                                       )
                 )

class ShiftedPrice_Helper():
    def __init__(self, **params):
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def gen_data(self, m=None):
        th = Transformation_Helper()
        
        dfs = [ th.gen_data(random_seed=42+i, m=m) for i in range(0,4) ]

        return dfs

    def plot_data(self, dfs, xlabel="Size", ylabel="Price", ax=None, nolabels=False):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(12,6) )

        x0, y0 = dfs[0]["area"],  dfs[0]["price"]
        x1, y1 = dfs[1]["area"],  dfs[1]["price"] +2000

        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        
        if nolabels:
            x, y = pd.concat( [x0, x1]), pd.concat( [y0, y1] )
            _= ax.scatter(x,y)
        else:
            _= ax.scatter(x0,  y0, label="$time_0$")
            _= ax.scatter(x1,  y1, label="$time_1$")
            _= ax.legend()

        _= ax.set_xlabel(xlabel)
        _= ax.set_ylabel(ylabel)
        
        return ax

class RelativePrice_Helper():
    def __init__(self, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def relative_price(self):
        return (1, 1, 7, 8)
        # return (1, 1, 2, 3)
    
    def gen_data(self, attrs=None):
        # Generate a set of random, near linear data (number of series: num_series)
        num_series = 3
        th = Transformation_Helper()
        dfs = [ th.gen_data(random_seed=42+i, attrs=attrs) for i in range(0,num_series+1) ]

        # Inflate each series by a different amount (rel_price)
        rel_price = self.relative_price()

        # dfs_i re the inflates series
        dfs_i = []
        for i in range(0, len(dfs)):
            df = dfs[i].copy()
            df["price"] *= rel_price[i]
            dfs_i.append(df)

        return dfs_i

    def plot_data(self, dfs, labels, attrs=None,
                  ax=None, xlabel=None,  ylabel=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(12,6))
            
        for i in [0,2,3]:
            _= ax.scatter( dfs[i]["area"], dfs[i]["price"], label=labels[i] )
        
        _= ax.legend()
        if xlabel is not None:
            _= ax.set_xlabel(xlabel)

        if ylabel is not None:
            _= ax.set_ylabel(ylabel)

        return ax
                           
class StockPrice_shiftedMean_Helper():
    def __init__(self, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def gen_prices(self, return_mean, return_std, level_means, num_returns=30):
        """
        Generate a price series consisting of segments.

        Each segment is same length (num_returns) but different average level.
        Between segments: there is a jump in the price level (implemented as an extra point with unusual return)

        Parameters
        ----------
        return_mean: Scalar.  Average daily return
        return_std:  Scalar.  Standard deviation of daily returns
        level_means: Array.   level_means[i] is the average Price level for ties 0..((i+1)*num_returns -1)

        Returns
        -------
        DataFrame with colums "Level", "Return" for price levels and returns
        """

        num_segments = len(level_means)
        num_pts = num_segments*num_returns + (num_segments -1)
        # Generate returns
        rng = self.rng

        # Generate returns from normal distribution with mean return_mean and std deviation return_std
        x = np.linspace(0, num_pts ).reshape(-1,1)
        rets =  return_mean + rng.normal(size=num_pts) * return_std

        # Change the returns between segments to implement the change in mean level
        level_means_np = np.array(level_means)
        level_pcts = level_means_np[1:]/level_means_np[:-1] -1

        for i in np.arange(level_pcts.shape[0]):
            rets[ (i+1)*num_returns + i ] = level_pcts[i]

        # Cumulative returns
        cum_rets = np.cumprod(1 + rets)

        # Cumulative returns to prices
        levels = level_means[0] * cum_rets

        self.num_returns = num_returns
        self.num_segments = num_segments
        
        return pd.DataFrame( list(zip(levels, rets)), columns=[ "Level", "Return" ]) 

    def plot_data(self, df, fig=None, axs=None, visible=True):
        """
        Convenience method to plot the DataFrame with levels and returns

        Parameters
        ----------
        DataFrame with columns "Level", "Return"

        Returns
        -------
        fig, axs: Matplot lib plot
        """

        if axs is None:
            fig, axs = plt.subplots(1,2, figsize=(12,6) )

        axs[0].plot( df["Level"])
        axs[0].set_title("Level")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Price")
        
        axs[1].plot( df["Return"])
        axs[1].set_title("Return")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Return")


        if not visible:
            plt.close(fig)

        return fig, axs

    def plot_segments(self, df, fig=None, axs=None, visible=True):
        """
        Convenience method to plot Returns of each segment the DataFrame

        Parameters
        ----------
        DataFrame with columns "Level", "Return"

        Returns
        -------
        fig, axs: Matplot lib plot
        """

        num_returns, num_segments = self.num_returns, self.num_segments
        
        if axs is None:
            fig, axs = plt.subplots(1, num_segments, figsize=(12,6) )


        for i in range(num_segments):
            jump_idx = (i+1)*num_returns + i
            rets_series = df.iloc[ jump_idx - num_returns :jump_idx ]["Return"]
            
            axs[i].hist(rets_series)
            axs[i].set_title(f"Return segment {i:d}")

        if not visible:
            plt.close(fig)

        return fig, axs


class StockPrice_drift_Helper():
    def __init__(self, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        
        return

    def gen_prices(self, return_mean, return_std, drift=.001, level_init=10, num_returns=30):
        """
        Generate a price series where the prices drift via an additive constant daily return amount (drift)

        Parameters
        ----------
        return_mean: Scalar.  Average daily return EXCLUDING drift
        return_std:  Scalar.  Standard deviation of daily returns

        The "de-meaned" returns have mean return_mean and std dev. return_std

        drift:       Scalar,  Constant return added to the daily return

        The final returns have mean (drift + return_mean) and std dev. return_std
        level_init:  Scalar, the initial Price

        Prices are derived by computing cumulative returns and multiplying by level_init

        Returns
        -------
        DataFrame with colums "Level", "Return" for price levels and returns
        """

        # Generate de-meaned returns
        rng = self.rng
        
        # Generate returns from normal distribution with mean return_mean and std deviation return_std
        rets =  (return_mean + drift) + rng.normal(size=num_returns) * return_std

        # Cumulative returns
        cum_rets = np.cumprod(1 + rets)

        # Cumulative returns to prices
        levels = level_init * cum_rets
        
        return pd.DataFrame( list(zip(levels, rets)), columns=[ "Level", "Return" ]) 

    def plot_data(self, df, fig=None, axs=None, visible=True):
        """
        Convenience method to plot the DataFrame with levels and returns

        Parameters
        ----------
        DataFrame with columns "Level", "Return"

        Returns
        -------
        fig, axs: Matplot lib plot
        """

        if axs is None:
            fig, axs = plt.subplots(1,2, figsize=(12,6) )

        axs[0].plot( df["Level"])
        axs[0].set_title("Level")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Price")
        
        axs[1].plot( df["Return"])
        axs[1].set_title("Return")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Return")


        if not visible:
            plt.close(fig)

        return fig, axs

    def plot_rolling(self, df, window=10, fig=None, axs=None, visible=True):
        """
        Convenience method to plot the rolling mean of the DataFrame with levels and returns

        Parameters
        ----------
        DataFrame with columns "Level", "Return"

        Returns
        -------
        fig, axs: Matplot lib plot
        """

        if axs is None:
            fig, axs = plt.subplots(1,2, figsize=(12,6) )

        rolling_df = df.rolling(window)

        rolling_mean_df = rolling_df.mean()
        rolling_std_df  = rolling_df.std()
        
        axs[0].plot( rolling_mean_df["Level"])
        axs[0].set_title("Rolling mean: Level")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Price")
        
        axs[1].plot( rolling_mean_df["Return"])
        axs[1].set_title("Rolling mean: Return")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Return")


        if not visible:
            plt.close(fig)

        return fig, axs


class StockReturn_Pooling_Helper():
    def __init__(self, **params):
        # Random number generator
        rng = np.random.RandomState(42)
        self.rng = rng
        
        self.colors = [ "blue", "orange"]
        return

    def gen_returns(self, return_means, return_stds, beta=1.4, num_returns=30):
        """
        Generate a return series consisting of segments.

        Each segment is same length (num_returns) but different average level.
        Between segments: there is a jump in the price level (implemented as an extra point with unusual return)

        Parameters
        ----------
        These are arrays: element i defines segment i
        return_means: Array.  Average daily return
        return_std:   Array.  Standard deviation of daily returns
        

        Returns
        -------
        DataFrame with colums "Return" for price levels and returns
        """

        num_segments = len(return_means)
    
        
        # Generate normalized returns
        rng = self.rng
        rets_norm = rng.normal(size=num_returns)

        dfs = []
        # Generate returns from normal distribution with mean return_mean and std deviation return_std
        for seg in range(num_segments):
            ind =  return_means[seg] +  rets_norm * return_stds[seg]
            noise = rng.normal(size=len(ind)) * return_stds[seg] * .05
            
            dep =  beta * ind + noise 
            
            df = pd.DataFrame( { "ind": ind, 
                                 "dep": dep,
                                 "Dt": [ (seg*num_returns) + i for i in range(num_returns) ],
                                 "Segment": [seg] * num_returns }
                             )
            df.set_index("Dt", inplace=True)
            
            dfs.append(df)

           
        big_df = pd.concat( dfs, axis=0)
        
        return big_df
    
    def normalize_data(self, df, Debug=False):
        """
        Return normalized verson of DataFrame df
        - subtract mean, divide by standard deviation
        
        Parameters
        ----------
        df: DataFrame
        
        Returns:
        DataFrame
        """
        
        dfs = []

        # Normalize each segment
        # using the mean and standard deviation off each segment
        segments = df["Segment"].unique()
        
        for seg in segments:
            df_seg = df.loc[df["Segment"] == seg,:].copy()
            
            # Compute mean and std deviation of each variable
            seg_ind_mean, seg_ind_std = df_seg.loc[:,"ind"].mean(), df_seg.loc[:,"ind"].std()
            seg_dep_mean, seg_dep_std = df_seg.loc[:,"dep"].mean(), df_seg.loc[:,"dep"].std()

            if Debug:
                print(f"Time {seg:d}: mean_ind={seg_ind_mean:5.3f}, std_ind={seg_ind_std:5.3f}")

            # Standardize
            df_seg.loc[:,"ind"] = (df_seg.loc[:,"ind"] - seg_ind_mean)/seg_ind_std
            df_seg.loc[:,"dep"] = (df_seg.loc[:,"dep"] - seg_dep_mean)/seg_dep_std

            dfs.append(df_seg)

        df_new = pd.concat(dfs, axis=0)

        return df_new

    def plot_data(self, df, fig=None, axs=None, visible=True):
        """
        Convenience method to plot the DataFrame with levels and returns

        Parameters
        ----------
        DataFrame with columns "Level", "Return"

        Returns
        -------
        fig, axs: Matplot lib plot
        """

        segments = df["Segment"].unique()
        num_segments = len(segments)
        
        if axs is None:
            fig = plt.figure(figsize=(12,6), constrained_layout=True)
            gs = GridSpec(4, 2, figure=fig)
            axs = [ fig.add_subplot( spec ) for spec in [ gs[0:2,:], 
                                                          gs[2,0], gs[2,1],
                                                          gs[3,0], gs[3,1]
                                                        ] 
                  ]
        
        colors = self.colors
        
        # Joint plot
        for i, seg in enumerate(segments):
            df_seg = df[ df["Segment"] == seg ]
            
            _= axs[0].scatter( df_seg["ind"], df_seg["dep"], label=f'$time_{i:d}$', color=colors[i])
            
        _= axs[0].set_xlabel("ind")
        _= axs[0].set_ylabel("dep") 
        
        _= axs[0].legend()
        
        # Individual histograms
        spec_num = 1
        for j, key in enumerate( ["dep", "ind"] ):
            # all segment should share the same range on horizontal axis
            xmin, xmax = df[key].min(), df[key].max()
            
            for i, seg in enumerate(segments):
                df_seg = df[ df["Segment"] == seg ]

                _= axs[spec_num].hist(df_seg[key], color=colors[i])
                _= axs[spec_num].set_title(f'$time_{i:d}$ {key}')
                _= axs[spec_num].set_xlim([xmin,xmax])
                spec_num += 1
                 
        if not visible:
            plt.close(fig)

        return fig, axs

    def plot_segments(self, df, fig=None, axs=None, visible=True):
        """
        Convenience method to plot Returns of each segment the DataFrame

        Parameters
        ----------
        DataFrame with columns "ind", "dep", "Segment"

        Returns
        -------
        fig, axs: Matplot lib plot
        """

        segments = df["Segment"].unique()
        num_segments = len(segments)
        
        if axs is None:
            fig, axs = plt.subplots(1, num_segments, figsize=(12,6) )


        colors = self.colors
        
        for i, seg in enumerate(segments):
            df_seg = df[ df["Segment"] == seg ]
            
            _= axs[i].scatter( df_seg["ind"], df_seg["dep"], label=f'$time_{seg:d}$', color=colors[i])
            _= axs[i].set_xlabel("ind")
            _= axs[i].set_ylabel("dep")
            
            _= axs[i].legend()
            
        if not visible:
            plt.close(fig)

        return fig, axs
