import pandas as pd
import numpy as np


def abe_says_hi():
    print("abraham is very happy")

def clean_data(df, print_results=True):
    if print_results:
        rows_before = df.shape[0]
        print(f'Rows before cleaning: {rows_before}')

    ## Remove lists with no amenities
    df = df[df['amenities'].notnull()].copy()

    ## Remove lists with no latitude
    df = df[df['latitude'].notnull()].copy()

    ## Remove Egregious prices that are obviously incorrect
    df = df[df['price'] < 5000].copy()

    if print_results:
        rows_after = df.shape[0]
        print(f'Rows after cleaning: {rows_after}')

    return df


def set_up_features(df):
    ## Go through each column and fix type

    ## body = Text length
    df['body_length'] = df['body'].str.len()

    ## amenities = create dummies for a few amenities
    df['amenities_parking'] = df['amenities'].str.contains('parking', case=False) * 1
    df['amenities_gym'] = df['amenities'].str.contains('gym', case=False) * 1
    df['amenities_pool'] = df['amenities'].str.contains('pool', case=False) * 1
    df['amenities_washer'] = df['amenities'].str.contains('washer', case=False) * 1

    ## has_photo = dummies
    df['photo_thumbnail'] = df['has_photo'].str.contains('thumbnail', case=False) * 1

    ## pets_allowed = dummies
    df['pets_dogs'] = df['pets_allowed'].str.contains('dog', case=False) * 1
    df['pets_cat'] = df['pets_allowed'].str.contains('cat', case=False) * 1

    ## state = create dummies
    df = pd.get_dummies(df,
                        prefix=['state'],
                        prefix_sep='_',
                        columns=['state'],
                        dtype=int)

    ## Replace NaNs with zero
    df = df.fillna(0)

    return df


def create_price_bins(df, print_results=True, category_names = None):
    ## Create buckets of prices
    # Define the conditions for price categories
    conditions = [
        (df['price'] < 1000),
        (df['price'] >= 1000) & (df['price'] < 1200),
        (df['price'] >= 1200) & (df['price'] < 1500),
        (df['price'] >= 1500) & (df['price'] < 2000),
        (df['price'] >= 2000) & (df['price'] < 3000),
        (df['price'] >= 3000)
    ]

    # Define the corresponding price categories
    if category_names is None:
        category_names = [
            '01 - Very Low Price (<$1K)',
            '02 - Low Price ($1K to $1.2K)',
            '03 - Medium Price ($1.2K to $1.5K)',
            '04 - High Price ($1.5K to $2K)',
            '05 - Very High Price ($2K to $3K)',
            '06 - Highly Premium ($3K+)'
        ]

    # Use np.select to assign categories based on conditions
    df['price_cat'] = np.select(conditions, category_names, default='Unknown')

    if print_results:
        display(df['price_cat'].value_counts())

    return df


def get_city_features(df):
    ## Add distance to nearest city
    city_locations = pd.read_csv('/Users/abrahambaldenegro/Documents/House Prediction Model/City Locations.csv')

    cities = city_locations["City"]

    ## To store results
    dist_data = pd.DataFrame(index=df.index)

    for city in cities:
        city_lat = city_locations[city_locations["City"] == city]["LATITUDE"].values[0]
        city_lon = city_locations[city_locations["City"] == city]["LONGITUDE"].values[0]

        city_col = city.replace(".", "")
        city_col = city_col.replace(" ", "_")
        city_col = city_col.lower()
        city_col = 'city_' + city_col
        # print(city_col)

        ## Create columns for each city
        dist_data[city_col] = np.sqrt(
            np.square((df['latitude'] - city_lat)) + np.square((df['longitude'] - city_lon)))

    ## Get final metrics
    dist_data = pd.DataFrame({'closest_city': dist_data.idxmin(axis=1),
                              'distance_to_closest_city': dist_data.min(axis=1)})

    df = pd.concat([df, dist_data], axis=1)

    df = pd.get_dummies(df,
                        prefix=[''],
                        prefix_sep='',
                        columns=['closest_city'],
                        dtype=int)

    return df
