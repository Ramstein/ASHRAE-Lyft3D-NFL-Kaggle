import numpy as np
import pandas as pd
import datetime, math


def preprocess(train, b_meta, w_train, test=False):
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    from pandas.api.types import is_categorical_dtype

    def reduce_mem_usage(df, use_float16=False):
        """
        Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

        for col in df.columns:
            if is_datetime(df[col]) or is_categorical_dtype(df[col]):
                continue
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype("category")

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

        return df

    # reducing memory usage
    train = reduce_mem_usage(train, use_float16=True)
    b_meta = reduce_mem_usage(b_meta, use_float16=True)
    w_train = reduce_mem_usage(w_train, use_float16=True)

    # merging b_meta and w_train to train
    train = train.merge(b_meta, on="building_id", how="left")
    train = train.merge(w_train, on=["site_id", "timestamp"], how="left")

    def primary_use(x):
        p_use = {"Religious worship":3.91,"Retail":26.61,"Other":35.10,"Warehouse/storage":70.93,"Technology/science":227.89,
                 "Food sales and service":304.76,"Lodging/residential":307.98,"Entertainment/public assembly":320.39,"Parking":321.10,
                 "Public services":373.27,"Utility":538.77,"Manufacturing/industrial":549.41,"Office":752.07,"Healthcare":820.03,
                 "Education":2456.70,"Services":10026.19}
        for key in p_use.keys():
            if x == key: return p_use.get(key)
    train['primary_use']=train['primary_use'].apply(lambda x: primary_use(x))
    # floor_count
    train['floor_count_ifnan'] = train.floor_count.isnull().astype('int')

    # dew_temperature
    train['dew_temperature'] = train['dew_temperature'].fillna(23)
    train['dew_temperature_k'] = train['dew_temperature'].apply(lambda x: x + 273.15)

    # air_temperature
    train["air_temperature"] = train["air_temperature"].fillna(35)
    train['air_temperature_k'] = train['air_temperature'].apply(lambda x: x + 273.15)

    # precip_depth_1_hr
    train['precip_depth_1_hr_ifnan'] = train.precip_depth_1_hr.isnull().astype("int")
    train['precip_depth_1_hr'] = train['precip_depth_1_hr'].fillna(300)

    # sea_level_pressure
    train['sea_level_pressure'] = train['sea_level_pressure'].fillna(980)
    train['sea_level_pressure_atm'] = train['sea_level_pressure'].apply(lambda x: x / 1013.25)

    # wind_direction
    train['wind_direction'] = train['wind_direction'].fillna(0)

    # wind_speed
    train['wind_speed'] = train['wind_speed'].fillna(15)

    # timestamp
    train.timestamp = pd.to_datetime(train.timestamp, format="%Y-%m-%d %H:%M:%S")
    train.square_feet = np.log1p(train.square_feet)

    if not test:
        train.sort_values("timestamp", inplace=True)
        train.reset_index(drop=True, inplace=True)

    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]

    # year_build
    train['year_built_ifnan'] = train.year_built.isnull().astype('int')
    train['year_built'] = train['year_built'].fillna(2015)

    train["hour"] = train.timestamp.dt.hour
    train['year'] = train['timestamp'].dt.year
    train['month'] = train['timestamp'].dt.month
    train['day'] = train['timestamp'].dt.day
    train["weekday"] = train.timestamp.dt.weekday
    train['age'] = (train['year'] - train['year_built'])
    train["is_holiday"] = (train.timestamp.dt.date.astype("str").isin(holidays)).astype(int)

    # relative_humidity
    train['relative_humidity'] = 100 - 5 * (train['air_temperature_k'] - train['dew_temperature_k'])

    # air_density
    train['air_density'] = (train['sea_level_pressure_atm'] * 14.67) / (train['air_temperature_k'] * 0.0821)

    # saturated_vapour_density
    vapour_density = 7.2785
    train['saturated_vapour_density'] = train['relative_humidity'] * vapour_density



    # Building surface area
    def build_sur_area(x,h):
        side = x**(0.5)
        return 2*(side*side + side*h + side*h)

    train['building_height'] = train['floor_count_ifnan'] * 3  # each floor 3m high
    train['building_vol'] = train['building_height'] * train['square_feet']

    train['build_sur_area'] = build_sur_area(train['square_feet'], train['building_height'] )

    # evaporation rate
    def evaporated_water(x0, x1, x2, x3, x4, x5, time):
        xs = 0.622*(((x0*x1*0.026325) / x3)-1)  # maxm humidity ratio of saturated air
        x = 0.622 * (((x0 * x2 * 0.026325) / x3) - 1)  # humidity ratio of dry air
        if time=='hour': return x4 * x5 *(xs-x)
        if time=='minute': return (x4 * x5 *(xs-x))/60
        if time=='sec': return (x4 * x5 *(xs-x))/3600

    train['evap_coeff'] = 25+19*train['wind_speed']

    train['evap_per_hour'] = evaporated_water(train['relative_humidity'], train['dew_temperature_k'], train['air_temperature_k'],
                                                     train['sea_level_pressure'], train['evap_coeff'], train['build_sur_area'], time='hour')
    train['evap_per_minute'] = evaporated_water(train['relative_humidity'], train['dew_temperature_k'], train['air_temperature_k'],
                                                     train['sea_level_pressure'], train['evap_coeff'], train['build_sur_area'], time='minute')
    train['evap_per_sec'] = evaporated_water(train['relative_humidity'], train['dew_temperature_k'], train['air_temperature_k'],
                                                     train['sea_level_pressure'], train['evap_coeff'], train['build_sur_area'], time='sec')
    train['condense_per_hour'] = train['evap_per_hour']*train['relative_humidity']*100
    train['condense_per_minute'] = train['evap_per_minute']*train['relative_humidity']*100
    train['condense_per_sec'] = train['evap_per_sec']*train['relative_humidity']*100


    # heat required by building
    def required_by_build(x0, x1, x2):
        air_layer = (x0**(1/3)+1)**3 - x0
        air_mass = air_layer*x1
        L = 2256  # latent heat of vaporization of water = 2256 kj/kg
        return  (L*air_mass) * x2

    train['heat_lost_by_build'] = required_by_build(train['building_vol'], train['air_density'], train['evap_per_sec'])

    # dropping some features
    drop_features = ["timestamp"]

    train.drop(drop_features, axis=1, inplace=True)
    # train.dropna(axis=0, how='any')
    train['meter_reading'] = train['meter_reading'].apply(lambda x:np.log1p(x))
    return train
    # if test:
    #     row_ids = train.row_id
    #     train.drop("row_id", axis=1, inplace=True)
    #     return train, row_ids
    # else:
    #     # train.drop("meter_reading", axis=1, inplace=True)
    #     return train, y_train



