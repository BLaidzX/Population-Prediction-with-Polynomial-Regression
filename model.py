import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def get_country_index(df, input_country):
    return df.index[df['Country Name'] == input_country].tolist()

def predict_population(input_country, year, x, y):
    poly = PolynomialFeatures(degree=4, include_bias=False)
    poly_x = poly.fit_transform(x)

    linear_reg = LinearRegression()
    linear_reg.fit(poly_x, y)

    input_year = linear_reg.predict(poly.transform([[year]]))[0]
    accuracy = linear_reg.score(poly_x, y) * 100

    print(f"{input_country} had a population of {input_year:.2f} in the year {year}.")
    print(f"Accuracy: {accuracy:.2f}%")

    plt.scatter(x, y, color='red')
    plt.plot(x, linear_reg.predict(poly_x), color='blue')
    plt.show()

def main():
    file_path = r"C:\Users\Ahmad\Downloads\Book 1.xlsx"
    df = load_data(file_path)

    countries = df['Country Name'].tolist()

    while True:
        input_country = input("Please input the country you want (type 'quit' to exit): ").strip()
        if input_country.lower() == "quit":
            break

        index_list = get_country_index(df, input_country)
        if not index_list:
            print("Not found in the dataframe. Please check your spelling.")
        else:
            index = index_list[0]
            dict1 = df.iloc[index].to_dict()
            del dict1['Country Name']

            new_df = pd.DataFrame(dict1.items(), columns=['X', 'Y'])
            x = new_df["X"].to_numpy(dtype='int64')
            y = new_df["Y"].to_numpy(dtype='int64')

            predict_year = int(input(f"Please input a year to predict the population of {input_country} during that year: "))
            predict_population(input_country, predict_year, x.reshape(-1, 1), y)

if __name__ == "__main__":
    main()
