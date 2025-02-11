#%%########## USEFUL LIBRARIES ##########

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from flask_caching import Cache
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dateutil.relativedelta import relativedelta
from statsmodels.stats.stattools import jarque_bera
from statsmodels.regression.rolling import RollingOLS
from dash import html, dcc, Input, Output, dash_table
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey

#%%########## INITIALIZING GLOBAL VARIABLES ##########

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

tabs_styles = {'height': '44px'}
tab_style = {'borderBottom': '1px solid #d6d6d6',
             'padding': '6px', 'fontWeight': 'bold'}
tab_selected_style = {'borderTop': '1px solid #d6d6d6', 'borderBottom': '1px solid #d6d6d6',
                      'backgroundColor': 'grey', 'color': 'white', 'padding': '6px'}
drivers_cols = ["const", "rate", "ccy_equ",
                "usd_equ", "tot", "esi", "inflation"]
assumps_cols = ["Homos.", "No autoc.", "Normal."]
tables_conditional_styles = [{'if': {'column_id': col},
                              'color': 'gainsboro'} for col in drivers_cols] + [
    {'if': {'column_id': ''},
     'fontWeight': 'bold'}] + [
    {'if': {'filter_query': f'{{{col}}} contains "*"', 'column_id': col},
     'backgroundColor': 'lightgreen', 'fontWeight': 'bold', 'color': 'black'} for col in drivers_cols] + [
    {'if': {'filter_query': f'{{{col}}} contains "**"', 'column_id': col},
     'backgroundColor': 'limegreen', 'fontWeight': 'bold', 'color': 'black'} for col in drivers_cols] + [
    {'if': {'column_id': "R^2"},
     'fontWeight': 'bold', 'backgroundColor': 'whitesmoke'}] + [
    {'if': {'column_id': col},
     'backgroundColor': 'oldlace'} for col in assumps_cols] + [
    {'if': {'filter_query': f'{{{col}}} contains "true"', 'column_id': col},
     'fontWeight': 'bold', 'color': 'springgreen'} for col in assumps_cols] + [
    {'if': {'filter_query': f'{{{col}}} contains "false"', 'column_id': col},
     'fontWeight': 'bold', 'color': 'red'} for col in assumps_cols] + [
    {'if': {'column_id': "R2 Adj."},
     'fontWeight': 'bold', 'backgroundColor': 'whitesmoke'}]

ccy_list = ["EUR", "CHF", "GBP", "NOK", "SEK", "JPY", "AUD", "NZD", "CAD"]
factors_list = ["const", "rate", "ccy_equ",
                "usd_equ", "tot", "esi", "inflation"]
windows_dict = {1: "6M", 2: "9M", 3: "12M", 4: "18M", 5: "24M"}
trading_periods = ["1W", "2W", "1M", "3M", "6M",
                   "9M", "12M", "18M", "24M", "48M", "60M"]
tickers_dict = {"EUR": {"RATE": "GDBR10 Index PX_LAST", "EQUITY": "SX5T Index PX_LAST", "INFLATION": None},
                "USD": {"RATE": "USGG10YR Index PX_LAST", "EQUITY": "SPXT Index PX_LAST", "INFLATION": "USSWIT1 Curncy PX_LAST"},
                "GBP": {"RATE": "GUKG10 Index PX_LAST", "EQUITY": "TUKXG Index PX_LAST", "INFLATION": "BPSWIT1 Curncy PX_LAST"},
                "CHF": {"RATE": "GSWISS10 Index PX_LAST", "EQUITY": "SMIMC Index PX_LAST", "INFLATION": None},
                "CAD": {"RATE": "GCAN10YR Index PX_LAST", "EQUITY": "0000AR Index PX_LAST", "INFLATION": None},
                "NOK": {"RATE": "GNOR10YR Index PX_LAST", "EQUITY": "OBX Index PX_LAST", "INFLATION": None},
                "SEK": {"RATE": "GTSEK10Y Govt PX_LAST", "EQUITY": "OMXSGI Index PX_LAST", "INFLATION": None},
                "NZD": {"RATE": "GNZGB10 Index PX_LAST", "EQUITY": "NZSE50FG Index PX_LAST", "INFLATION": None},
                "JPY": {"RATE": "GJGB10 Index PX_LAST", "EQUITY": "NKYTR Index PX_LAST", "INFLATION": "JYSWIT1 Curncy PX_LAST"},
                "AUD": {"RATE": None, "EQUITY": "ASA52 Index PX_LAST", "INFLATION": "AUSWIT1 Curncy PX_LAST"}}

#%%######## DASHBOARD CLASS ########

class Dashboard():
    
    """
    This class allows the creation of the layout, the management of callbacks, 
    and the updates of graphs/tables of the dashboard.

    Attributes
    ----------
    app : dash.Dash
        The Dash application instance that is used to create the dashboard.
    data : pd.DataFrame
        A DataFrame containing the imported data used for analysis .
    """
    
    
    def __init__(self, app:dash.Dash) -> None: 
        
        """
        Initializing the Dash application, importing data, and setting the layout.

        Parameters
        ----------
        app : dash.Dash
            The Dash application instance.
        """
        
        # Importing Data
        self.data = self.import_data()
        
        # Application
        self.app = app
        self.app.layout = self.dash_layout()   
        self.app.callback(
            Output('global_reg_res', 'data'),
            Output('significance_table', 'data'),
            Output('roll_betas_per_ccy_fig', 'figure'),
            Output('roll_betas_per_fac_fig', 'figure'),
            Output('drivers_reg_res_table', 'data'),
            Output('cumul_errors_tables', 'data'),
            Output('trading_period_fig', 'figure'),
            Output('global_reg_res_title', 'children'),
            Output('drivers_reg_res_title', 'children'),
            Output('significance_table_title', 'children'),
            Output('roll_betas_per_ccy_title', 'children'),
            Output('roll_betas_per_fac_title', 'children'),
            Output('cum_error_per_wndw_title', 'children'),
            Input('lookback_window', 'value'),
            Input('roll_betas_ccy_pair_selection', 'value'),
            Input('roll_betas_factor_selection', 'value'),
            Input('trading_period_selection', 'value'))(self.update_dashboard)   
        
        
    def import_data(self) -> pd.DataFrame:
        
        """
        Importing the data from an Excel file (Github link) and filtering by the last 10 years.

        Returns
        -------
        data : pd.DataFrame
            Bloomberg data (fx, rates, equities...).
        """
        cached_data = cache.get('data_cache')  # Utilisation du cache Flask-Caching
        if cached_data:
            return cached_data

        try:
            path = "https://github.com/NicolasHurbin/FX-Faire-Value-Model/raw/refs/heads/main/DATA.xlsx"
            data = pd.read_excel(path, index_col=0)
        except Exception as e:
            raise ValueError(f"Error while reading the file: {e}")
            
        # Filtering by the last 10 years
        start_date = max(data.index) - relativedelta(years=10)
        data = data.loc[start_date:, :]
        
        # Adding Nan column for missing values
        data[None] = np.nan
        
        cache.set('data_cache', data, timeout=60*60*24)
        
        return data   
    
    
    def dash_layout(self) -> app.layout:
        
        """
        Generating the layout for the Dash application, including the title, window selection,
        regression results, rolling betas, driver regression results, and errors/trading sections.
        
        Returns
        -------
        layout : app.layout
            The complete layout of the application as a Dash component.
        """
        
        layout = dbc.Container([
            
            ### --- TITLE OF THE DASHBOARD --- ###
            html.H1("- FX FAIR VALUE MODEL -"),
            html.Hr(),
          
            dbc.Row([
                ### --- WINDOW SELECTION --- ###
                dbc.Col([html.H4("Window: ")], width=1),
                dbc.Col([dcc.Slider(id="lookback_window", min=1, max=5,
                                    step=1, value=3, marks=windows_dict)], width=11)]),

            dbc.Row([
                dbc.Col([
                ### --- GLOBAL REGRESSION RESULTS TABLE --- ###
                html.Br(),
                html.Div(id="global_reg_res_title"),
                dash_table.DataTable(id='global_reg_res',
                                     style_cell={'font_size':'10px', "textAlign":"center"},
                                     style_table={'height': '300px'},
                                     style_header={'fontWeight': 'bold'},
                                     style_data_conditional=tables_conditional_styles)], width=6),
                dbc.Col([
                ### --- ROLLING REGRESSIONS RESULTS TABS --- ###
                dcc.Tabs(value='signif_tab', children=[
                    # FIRST TAB: Significance Ratio
                    dcc.Tab(label='Significance', value='signif_tab', style=tab_style, selected_style=tab_selected_style,
                            children=[html.H6(""),
                                      html.Div(id="significance_table_title"),
                                      dash_table.DataTable(id='significance_table',
                                                           style_cell={'font_size':'10px', "textAlign":"center"},
                                                           style_header={'fontWeight': 'bold'},
                                                           style_data_conditional=tables_conditional_styles)]),
                     # SECOND TAB: Rolling Betas per ccy Graph
                     dcc.Tab(label='Rolling Betas per ccy', value='rolling_betas_ccy_tab', style=tab_style, selected_style=tab_selected_style,
                             children=[html.H4(""),
                                       dbc.Row([dbc.Col([html.H6(""), html.Div(id="roll_betas_per_ccy_title")], width=6),
                                                 dbc.Col([dcc.Dropdown(ccy_list, 'EUR', id='roll_betas_ccy_pair_selection')], width=6)]),
                                       dcc.Graph(id='roll_betas_per_ccy_fig', config={'displayModeBar': False})]),
                      # THIRD TAB: Rolling Betas per factor Graph
                      dcc.Tab(label='Rolling Betas per factor', value='rolling_betas_fac_tab', style=tab_style, selected_style=tab_selected_style,
                              children=[html.H4(""),
                                        dbc.Row([dbc.Col([html.H6(""), html.Div(id="roll_betas_per_fac_title")], width=6),
                                                 dbc.Col([dcc.Dropdown(factors_list, 'rate', id='roll_betas_factor_selection')], width=6)]),
                                        dcc.Graph(id='roll_betas_per_fac_fig', config={'displayModeBar': False})])], style=tabs_styles)], width=6)]),
                                                         
            html.Hr(),                                                                                
            dbc.Row([
                dbc.Col([
                    ### --- DRIVERS REGRESSION RESULTS TABLE --- ###
                    html.Br(),
                    html.Div(id="drivers_reg_res_title"),
                    dash_table.DataTable(id='drivers_reg_res_table',
                                         style_cell={'font_size':'10px', "textAlign":"center"},
                                         style_table={'height': '100px'},
                                         style_header={'fontWeight': 'bold'},
                                         style_data_conditional=tables_conditional_styles)], width=4),
                dbc.Col([
                    ### --- ERRORS AND TRADING TABS --- ###              
                    dcc.Tabs(value='errors_tables_tab', children=[
                        # FIRST TAB: Errors Tables
                        dcc.Tab(label='Errors Tables', value='errors_tables_tab', style=tab_style, selected_style=tab_selected_style,
                                children=[html.H6("Cumulative Errors Tables per ccy"),
                                          dash_table.DataTable(id='cumul_errors_tables', 
                                                              style_cell={'font_size':'10px', "textAlign":"center"},
                                                              style_table={'height': '100px'},
                                                              style_header={'fontWeight': 'bold'},
                                                              style_data_conditional=tables_conditional_styles)]),
                        # SECOND TAB: Errors Graphs
                        dcc.Tab(label='Errors Graphs', value='errors_graphs_tab', style=tab_style, selected_style=tab_selected_style,
                                children=[html.H4(""),
                                          dbc.Row([dbc.Col([html.H6(""), html.Div(id="cum_error_per_wndw_title")], width=6),
                                                    dbc.Col([dcc.Dropdown(trading_periods, '3M', id='trading_period_selection')], width=6)]),
                                          dcc.Graph(id='trading_period_fig', config={'displayModeBar': False})])])], width=8)]),
                      
            ], className="text-center bg-light", fluid=True)
      
        return layout
  
        
    def update_dashboard(self, window_select: str, ccy_select: str, factor_select: str, trading_period_select: str) -> tuple:
        
        """
        Updating the dashboard by performing regression and trading analysis, and formatting the results.
     
        Parameters
        ----------
        window_select : str
            The selected window size.
        ccy_select : str
            The selected currency for the rolling betas graphs.
        factor_select : str
            The selected factor for the rolling betas graphs.
        trading_period_select : str
            The selected trading period for cumulative error analysis.
     
        Returns
        -------
        tuple
            A tuple containing the formatted tables, figures, and titles for the dashboard layout.
        """  

        window = windows_dict[window_select]
        
        # Initializing Other Classes
        trading = Trading()
        reg_models = Regressions(self.data)
        formatting = Formatting(window, ccy_select, factor_select, trading_period_select)

        # Identifying Drivers
        global_results = reg_models.drivers_identification(window)

        # Analysing Drivers
        drivers_results = reg_models.drivers_analysis(global_results["drivers"])

        # Trading Analysis
        trading_results = trading.trading_analysis(drivers_results)

        #### Outputs Formatting ####
        layout_outputs = formatting.output_layout(global_results, drivers_results, trading_results)
        
        return layout_outputs     

#%%######## REGRESSIONS CLASS ########  

class Regressions():
    
    """
    This class is called to perform nregression analysis, including Ordinary Least Squares (OLS) and
    rolling OLS regressions. This class is mainly used to identify and analyze FX drivers for different currency pairs.

    Attributes
    ----------
    data : pd.DataFrame
        The pandas DataFrame containing all the Bloomberg data.
    usd_rate : pd.Series
        The USD 10y yield rate time series.
    usd_indx : pd.Series
        The USD equity index (SPXT Index) time series, log-transformed.
    usd_infl : pd.Series
        The USD inflation rate (1Y expectations) time series.
    usd_tot : pd.Series
        The USD Commodities Terms of Trades time series.
    usd_esi : pd.Series
        The USD Economic Surprise Index time series.
    """
    
    def __init__(self, data: pd.DataFrame) -> None:
        
        """
        Initializing all the USD factors and data.
        
        Parameters:
        ----------
        data : pd.DataFrame
            The input data containting all the time series.
        """
                
        self.data = data
        self.usd_rate = self.data[tickers_dict["USD"]["RATE"]]
        self.usd_indx = np.log(self.data[tickers_dict["USD"]["EQUITY"]])
        self.usd_infl = self.data[tickers_dict["USD"]["INFLATION"]]
        self.usd_tot = self.data["CTOTUSD Index PX_LAST"]
        self.usd_esi = self.data["CESIUSD Index PX_LAST"]
        
    
    def drivers_identification(self, window: str) -> dict:
        
        """
        Identify and analyze financial drivers using both OLS and rolling OLS regression. 
        The results are stored in a dictionary.

        Parameters:
        ----------
        window : str
            The rolling window size for regression, e.g., "12M" for a 12-month window.

        Returns:
        -------
        results : dict
            A dictionary containing:
            - "drivers": A dictionary with "names" and "features" for each currency.
            - "ols_results": Betas from the OLS regression.
            - "significance_ratios": Significance ratios from the rolling regressions.
            - "rolling_results": Rolling betas and p-values from the rolling regressions.
        """
        
        months = int(window[:-1])
        
        results = {}
        results["drivers"] = {}
        results["drivers"]["names"] = {}
        results["drivers"]["features"] = {}
        
        identif_betas = pd.DataFrame()
        signif_ratios = pd.DataFrame()
        
        signif_ratios.index = factors_list
        identif_betas.index = factors_list + assumps_cols + ["R2 Adj."]
             
        rolling_betas_dict, rolling_pvals_dict = {}, {}

        for ccy in ccy_list:

            ccy_pair = f"{ccy}USD"
            
            features = pd.DataFrame(index=self.data.index)
            features["y"] = np.log(self.data[f"{ccy_pair} Curncy PX_LAST"]) * 100
            features["tot"] = self.data[f"CTOT{ccy} Index PX_LAST"] - self.usd_tot
            features["esi"] = self.data[f"CESI{ccy} Index PX_LAST"] -self.usd_esi
            features["rate"] = self.data[tickers_dict[ccy]["RATE"]] - self.usd_rate
            features["ccy_equ"] = np.log(self.data[tickers_dict[ccy]["EQUITY"]]) * 100
            features["usd_equ"] = self.usd_indx * 100
            features["inflation"] = self.data[tickers_dict[ccy]["INFLATION"]] - self.usd_infl

            features = features.diff() 
            features.dropna(axis=0, how="all", inplace=True)
            features.dropna(axis=1, how="all", inplace=True)
            features = features.resample("W-MON").sum()
        
            ### SIMPLE OLS ###

            end_date = features.index.max() - relativedelta(weeks=1)
            start_date = end_date - relativedelta(months=months)
            start_indx = features.index.get_indexer([start_date], method='nearest')[0]
            start_date = features.index[start_indx]
            
            ols_features = features.loc[start_date:end_date, :]
            ols_regression = self.run_ols_regression(ols_features, 
                                                     check=["assumptions", "significance", "drivers"], 
                                                     constant=True)
        
            identif_betas[ccy] = ols_regression["betas"]
            results["drivers"]["names"][ccy] = ols_regression["drivers"]
                        
            ### ROLLING OLS ###
            
            rolling_window = len(ols_features)
            rolling_features = features.loc[:end_date, :]
            
            rolling_regression = self.run_rolling_regression(features, 
                                                             rolling_window, 
                                                             constant=True)
            results["drivers"]["features"][ccy] = rolling_features

            signif_ratios[ccy] = self.compute_significance_ratio(rolling_regression["pvals"])
            
            temp_betas = {(ccy, factor) : rolling_regression["betas"][factor].values 
                          for factor in rolling_regression["betas"].columns}
            temp_pvals = {(ccy, factor) : rolling_regression["pvals"][factor].values 
                          for factor in rolling_regression["pvals"].columns}
            
            rolling_betas_dict.update(temp_betas)
            rolling_pvals_dict.update(temp_pvals)
            
        multi_index = pd.MultiIndex.from_tuples(rolling_betas_dict.keys(), names=['currency', 'factor'])
        
        betas_data = list(rolling_betas_dict.values())
        pvals_data = list(rolling_pvals_dict.values())
        
        betas_data = np.array(betas_data).T
        pvals_data = np.array(pvals_data).T                                                     

        rolling_betas = pd.DataFrame(betas_data, index=rolling_regression["betas"].index, columns=multi_index)
        rolling_pvals = pd.DataFrame(pvals_data, index=rolling_regression["pvals"].index, columns=multi_index)

        ### Adding results to output dict
        results["ols_results"] = identif_betas.T
        results["significance_ratios"] = signif_ratios
        results["drivers"]["window"] = rolling_window
        results["rolling_results"] = {"rolling_betas" : rolling_betas, "rolling_pvals" : rolling_pvals}

        return results 
        

    def drivers_analysis(self, drivers_data: dict) -> dict:
        
        """
        Performing OLS and ROlling OLS regressions on the identified drivers only.

        Parameters:
        ----------
        drivers_data : dict
            A dictionary containing the following keys:
            - "names": A dictionary with the names of the drivers for each currency.
            - "features": A dictionary with the features data for each currency.
            - "window": The rolling window size for the regression.

        Returns
        -------
        drivers_results : dict
            A dictionary containing:
            - "betas": The rolling regression betas for each currency.
            - "features": The features data used for the regression for each currency.
            - "drivers_table": A table with the final betas and significance (with */** for significant values).
        """
        
        drivers_names = drivers_data["names"]
        features_dict = drivers_data["features"]
        rolling_window = drivers_data["window"]
                
        # Initializing results dictionary
        drivers_results = {"betas": {},
                           "features": features_dict,
                           "drivers_table": pd.DataFrame(index=factors_list)}
                
        for ccy in ccy_list:
            
            drivers_list = drivers_names[ccy]
        
            # Check if intercept belong to drivers
            constant = "const" in drivers_list
            if constant : drivers_list.remove("const")
                         
            features_cols = ["y"] + drivers_list
            features = features_dict[ccy][features_cols]
            
            # Run Rolling OLS regression on drivers only
            roll_reg_results =  self.run_rolling_regression(features, 
                                                            rolling_window, 
                                                            constant=constant)
            roll_betas = roll_reg_results["betas"]
            roll_pvals = roll_reg_results["pvals"]
            roll_r2 = roll_reg_results["r2"]
            
            drivers_results["betas"][ccy] = roll_betas
            
            # Retrieving the last values of the rolling regression
            end_date = max(roll_betas.index)
            last_betas = roll_betas.loc[end_date,:].round(4)
            last_pvals = roll_pvals.loc[end_date,:]
            last_r2 = round(roll_r2.loc[end_date] * 100, 2)
            
            # Significance Masks
            mask_5pct = last_pvals < .05
            mask_1pct = last_pvals < .01
            last_betas = last_betas.where(~mask_5pct, last_betas.astype(str) + '*')
            last_betas = last_betas.where(~mask_1pct, last_betas.astype(str) + '*')
                        
            drivers_results["drivers_table"][ccy] = last_betas
            drivers_results["drivers_table"].loc["R2 Adj.", ccy] = str(last_r2) + "%"

        drivers_results["drivers_table"] = drivers_results["drivers_table"].T

        return drivers_results
            

    @staticmethod
    @cache.memoize()
    def run_ols_regression(features: pd.DataFrame, check: list, constant: bool) -> dict:
        
        """
        This method performs an OLS regression on the given features. It can also provide a number of 
        results, such as OLS assumptions checks, significance checks, and drivers identifications.

        Parameters:
        ----------
        features : pd.DataFrame
            The dataframe containing the dependent variable 'y' and the features.
        check : list
            A list of checks to perform, such as "significance", "assumptions", or "drivers".
        constant : bool
            Whether to include a constant (intercept) in the regression model.

        Returns
        -------
        dict
            A dictionary containing the regression results:
            - "betas": The estimated coefficients of the regression.
            - "drivers": A list of drivers with p-values less than 0.05.
        """
                
        y = features["y"]
        X = features.drop(columns=["y"])

        # Add constant if required
        if constant: X = sm.add_constant(X)
    
        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        betas = model.params.round(4)
        pvals = model.pvalues
        resid = model.resid
        
        # Perform significance check if required
        if "significance" in check:
            mask_5pct = pvals < .05
            mask_1pct = pvals < .01
            betas = betas.where(~mask_5pct, betas.astype(str) + '*')
            betas = betas.where(~mask_1pct, betas.astype(str) + '*')
            
        # Perform OLS assumptions tests if required
        if "assumptions" in check:
            
            checker = {}
            # Homoscedasticity
            white_test = het_white(resid, X)
            checker["Homos."] = white_test[1] > 0.05
            # Autocorrelation
            breusch_godfrey_test = acorr_breusch_godfrey(model, nlags=4)
            checker["No autoc."] = breusch_godfrey_test[1] > 0.05
            # Normality Test
            jarque_bera_test = jarque_bera(resid)
            checker["Normal."] = jarque_bera_test[1] > 0.05
            checker = pd.Series(checker).astype("boolean")
  
            betas = pd.concat([betas, checker])
            
        # Identify significant drivers if required
        if "drivers" in check:
            drivers = pvals[pvals < .05].index.tolist()
        else:
            drivers = []
                        
        betas["R2 Adj."] = f"{round(model.rsquared_adj * 100, 2)}%"

        return {"betas" : betas, "drivers" : drivers}    


    @staticmethod
    @cache.memoize()
    def run_rolling_regression(features: pd.DataFrame, rolling_window: str, constant: bool) -> dict:
        
        """
        This method performs an OLS rolling regression on the given features.
     
        Parameters:
        ----------
        features : pd.DataFrame
            The dataframe containing the dependent variable 'y' and the features.
        rolling_window : str
            The window size for the rolling regression (e.g., '12W' for 12 weeks).
        constant : bool
            Whether to include a constant (intercept) in the regression model.
     
        Returns
        -------
        dict
            A dictionary containing the following keys:
            - "betas": The regression betas for each rolling window.
            - "pvals": The p-values for the regression coefficients.
            - "r2": The adjusted R-squared values for each rolling window.
        """
                
        y = features["y"]
        X = features.drop(columns=["y"])
        
        # Add constant if required
        if constant: X = sm.add_constant(X)
        
        rolling_model = RollingOLS(y, X, window=rolling_window).fit()
        rolling_betas = rolling_model.params
        rolling_pvals = pd.DataFrame(rolling_model.pvalues, index=rolling_betas.index, columns=rolling_betas.columns)
        rolling_r2 = rolling_model.rsquared_adj.dropna()

        rolling_betas.dropna(inplace=True)
        rolling_pvals.dropna(inplace=True)

        return {"betas" : rolling_betas, "pvals" : rolling_pvals, "r2" : rolling_r2}

         
    
    def compute_significance_ratio(self, pvalues: pd.DataFrame) -> pd.Series:
        
        """
        This method computes the ratio of significant p-values (p-value < 0.05) in the given DataFrame.
        
        Parameters:
        ----------
        pvalues : pd.DataFrame
            A pandas DataFrame containing p-values from a rolling regression analysis.
        
        Returns
        -------
        significance_ratio : pd.Series
            The percentage of significant p-values (p-value < 0.05) for each factors
        """

        mask = pvalues < .05
        # Ratio of significant dates over total period
        significance_ratio = mask.sum() / len(mask)
        significance_ratio = (significance_ratio * 100).round(2)
        significance_ratio = significance_ratio.astype(str) + "%"
        
        return significance_ratio

#%%######## TRADING CLASS ########

class Trading():
    
    """ This class is used to retrieve errors analysis between the market and the FX models. """

    def trading_analysis(self, drivers_data: dict) -> dict:
        
        """
        This method analyzes the trading errors based on drivers and calculates cumulative errors over different periods.
        
        Parameters:
        ----------
        drivers_data : dict
            A dictionary containing rolling "betas" and "features" for each currency.
        
        Returns:
        -------
        cumulative_error_dict : dict
            A dictionary containing cumulative errors for different trading periods.
        """
        
        # Initialize result DataFrame
        errors_results = pd.DataFrame(columns=ccy_list)
        cumulative_error_last = pd.DataFrame()
        cumulative_error_dict = {}

        for ccy in ccy_list:
            
            betas = drivers_data["betas"][ccy].shift(1).dropna()
            features = drivers_data["features"][ccy].reindex(betas.index)
            
            if "const" in betas.columns: features["const"] = 1
            
            # Calculating predicted values and error
            y = features["y"]
            X = features[betas.columns]            
            y_hat = np.multiply(betas, X).sum(axis=1)
            error = y_hat - y
            errors_results[ccy] = error

        for period in trading_periods:
            
            freq = period[-1]  # "W" or "M"
            count = int(period[:-1])  # Number of weeks or months
            end_date = max(errors_results.index)
        
            if freq == "W":
                start_date = end_date - relativedelta(weeks=count)
            elif freq == "M":
                start_date = end_date - relativedelta(months=count)
                start_indx = errors_results.index.get_indexer([start_date], method='nearest')[0]
                start_date = errors_results.index[start_indx]

            # Resample and calculate cumulative error if the date is valid
            errors_resampled = errors_results.loc[start_date:, :].copy()
            errors_resampled.loc[start_date, :] = 0
            cumulative_error = errors_resampled.cumsum()
            cumulative_error_dict[period] = cumulative_error
            cumulative_error_last[period] = cumulative_error.loc[end_date, :]
        
        cumulative_error_dict["last"] = cumulative_error_last.round(4)

        return cumulative_error_dict
    
    
    def ornstein_uhlenbeck_calib(self, residuals: dict) -> pd.DataFrame:
        
        """
        Calibrates an Ornstein-Uhlenbeck process on the residuals for each currency in the model.
      
        This method computes the parameters of the Ornstein-Uhlenbeck process for each currency, 
        including the mean reversion rate, long-term mean, disturbance term (noise), and half-life 
        (time to revert to the long-term mean).
      
        Parameters:
        -----------
        residuals : pd.DataFrame
            A DataFrame containing the residuals for each currency.
      
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the calibrated parameters for each currency: Mean Reversion Rate, 
            Long Term Mean, Disturbance Term, and Half-Life.
        """
        
        dt = 1 / 52
        calibration_results = pd.DataFrame()  
        
        for ccy in ccy_list:
            
            y = self.residuals[ccy].diff().dropna()
            X = self.residuals[ccy].reindex(y.index)
            X = sm.add_constant(X)
            
            calib_model = sm.OLS(y, X).fit()
            inter = calib_model.params["const"]    
            slope = calib_model.params[ccy]   
            error = calib_model.resid
            
            mr_rate = - slope / dt
            lt_mean = inter / (mr_rate * dt)
            noise   = np.sqrt(error.var() / dt)
            half_life = np.log(2) / mr_rate
            
            calibration_results.loc["Mean Rev. Rate", ccy] = mr_rate
            calibration_results.loc["Long Term Mean", ccy] = lt_mean
            calibration_results.loc["Disturb. Term", ccy] = noise
            calibration_results.loc["Half Life", ccy] = half_life
        
        return calibration_results
        
           
#%%######## FORMATTING CLASS ########     
 
class Formatting() :
    
    """ This class is used to format tables and figures under the Dash format. """
    
    def __init__(self, window, ccy_select, factor_select, trading_period_select):
        self.window = window
        self.ccy_select = ccy_select
        self.factor_select = factor_select
        self.trading_period_select = trading_period_select
    
    def format_table(self, table: pd.DataFrame) -> list:
        
        """
        Formatting a pandas DataFrame into the  dash_table format for displaying.

        Parameters:
        ----------
        table : pd.DataFrame
            The pandas DataFrame to be formatted.
        
        Returns:
        -------
        list of dict
            Formatted data
        """
        
        table.index.name = ""
        table = table.reset_index()

        return table.to_dict("records")
    
    
    def format_figure(self, data: dict, graph: str) -> go.Figure:
        
        """
        Formatting and generating a Plotly figure based on the provided data and graph type.

        Parameters
        ----------
        data : dict
            A dictionary containing the data to be plotted, including rolling betas and p-values.
        ccy : str
            The currency for which the rolling betas and p-values are to be plotted (used when graph type is 'ccy').
        factor : str
            The factor for which the rolling betas and p-values are to be plotted (used when graph type is 'factor').
        trading_period : str
            The trading period for which the graph is generated (used when graph type is 'trading').
        graph : str
            The type of graph to generate. Can be one of 'trading', 'ccy', or 'factor'.

        Returns
        -------
        go.Figure
            A Plotly figure object containing the generated graph.
        """

        # If errors trading graph
        if graph == "trading":
            graph_data = data[self.trading_period_select] 
            fig = px.line(graph_data, x=graph_data.index, y=graph_data.columns)
            fig.update_layout(dict(autosize=False, height=250, width=975, 
                                   margin=dict(l=5, r=5, b=4, t=15, pad=4)))
            return fig
        
        fig = go.Figure()
        
        # If rolling betas per ccy graph
        if graph == "ccy":
            graph_betas = data["rolling_betas"][self.ccy_select]
            graph_pvals = data["rolling_pvals"][self.ccy_select]
            
        # If rolling betas per factor graph            
        elif graph == "factor":
            graph_betas = data["rolling_betas"].swaplevel(axis=1)[self.factor_select]
            graph_pvals = data["rolling_pvals"].swaplevel(axis=1)[self.factor_select]
            
        for col in graph_betas.columns:
            col_beta = graph_betas[col]     
            col_mask = graph_pvals[col] < 0.05

            col_mask = np.where((col_mask.shift(1) == True) & (col_mask == False), True, col_mask)

            signif_yes = np.where(col_mask, col_beta, np.nan)
            signif_not = np.where(~col_mask, col_beta, np.nan)

            data_yes = pd.Series(signif_yes, index=col_beta.index)
            data_not = pd.Series(signif_not, index=col_beta.index)

            fig.add_trace(go.Scatter(x=data_yes.index, y=data_yes.values, name=col))
            fig.add_trace(go.Scatter(x=data_not.index, y=data_not.values, showlegend=False, line=dict(color='lightgrey')))
            
        fig.update_layout(dict(autosize=False, height=250, width=730, 
                               margin=dict(l=5, r=5, b=4, t=15, pad=4)))
    
        return fig
    
    
    def output_layout(self, global_: dict, drivers_: dict, trading_: dict) -> tuple:
        
        """ 
        Generating the layout elements for the dashboard, including formatted tables, figures, and titles.
        
        Parameters
         ----------
         global_ : dict
             A dictionary containing the global regressions results.
         drivers_ : dict
             A dictionary containing drivers' results.
         trading_ : dict
             A dictionary containing trading data.
        
         Returns
         -------
         tuple
             A tuple containing the following elements:
             - tab_ols_results: Formatted table of OLS global regression results.
             - tab_signif_ratio: Formatted table of significance ratios.
             - fig_betas_ccy: Figure for rolling betas by currency.
             - fig_betas_fac: Figure for rolling betas by factor.
             - tab_drivers_betas: Formatted table of drivers' betas.
             - tab_cumul_errors: Formatted table of cumulative trading errors.
             - fig_trad_period: Figure for cumulative errors by trading period.
             - tit_signif_ratio: Title for significance ratios.
             - tit_betas_ccy: Title for rolling betas by currency.
             - tit_betas_fac: Title for rolling betas by factor.
             - tit_errors_fig: Title for cumulative errors graph.
        """
        
        tab_ols_results  = self.format_table(global_["ols_results"])
        tab_signif_ratio = self.format_table(global_["significance_ratios"])
        fig_betas_ccy = self.format_figure(global_["rolling_results"], graph="ccy")
        fig_betas_fac = self.format_figure(global_["rolling_results"], graph="factor")
        fig_trad_period = self.format_figure(trading_, graph="trading")
        
        tab_drivers_betas = self.format_table(drivers_["drivers_table"])
        tab_cumul_errors  = self.format_table(trading_['last'])
        
        tit_global_res = f"Global Regression Results - {self.window} window"
        tit_signif_ratio = f"Ratio of Significant Periods - {self.window} window"
        tit_betas_ccy = f"Rolling Betas for {self.ccy_select} ccy - {self.window} window"
        tit_betas_fac = f"Rolling Betas for \"{self.factor_select}\" factor - {self.window} window"
        tit_drivers_res = f"Drivers Regression Results - {self.window} window"
        tit_errors_fig = f"Cumulative Error for the past {self.trading_period_select} "
        
        return (tab_ols_results, tab_signif_ratio,  fig_betas_ccy, 
                fig_betas_fac, tab_drivers_betas, tab_cumul_errors, 
                fig_trad_period, tit_global_res, tit_signif_ratio, 
                tit_betas_ccy, tit_betas_fac, tit_drivers_res,
                tit_errors_fig)
       
#%%######## RUNNING APPLICATION ########

if __name__ == '__main__': 
    run = Dashboard(app)    
    run.app.run_server(debug=False, port=8004)
   
   
        
        
        
        
        
        
        
        
        