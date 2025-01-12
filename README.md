# How to Use HM and MC

In legs = [{"strike_price": {}, "initial_price": {}, "contract_size": {}, "option_type": "{}"}, you can put you values if you have less than four legs just list contract_size to 0, option_type is set to either Call or Put

Under def current_pnl(leg) there is "underlying_price = {}", you will have to change this to the current price of the underlying, do the same for this "underlying = {}".

The steps above are for both HM and MC, but the steps below are only for HM

These parameters you can change to your desire: drift = {}
                                                days_to_expiry = {}
                                                T = days_to_expiry / {}
                                                underlying_price = {}
Where drift is the current rates, days to expiry is self explanotory and so is the underlying price. 


  
  
