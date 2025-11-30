import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect

# Load the dataset and model
df = pd.read_csv('Housing.csv')
column_names = df.columns.drop('price')
dropdown_columns = ['mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

dropdown_columns_values = {}
for column_name in column_names:
    if column_name in dropdown_columns:
        dropdown_columns_values[column_name] = sorted(set(df[column_name]))

# Create a Flask app
app = Flask(__name__)
# Define the prediction function

def predict(data):
    new_df = pd.DataFrame(columns=column_names)
    new_df.loc[0] = data
    model = pickle.load(open("best_house_prediction_model.pkl", 'rb'))
    predicted_price = model.predict(new_df)[0]
    return f'₹{predicted_price:,.2f}'

# Define routes 


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect('/predict')
    return render_template('index.html', column_names=column_names, dropdown_columns_values=dropdown_columns_values)


@app.route("/predict", methods=["GET", "POST"])
def prediction():
    if request.method == 'GET':
        return redirect('/')
    data = []
    for column_name in column_names:
        input_value = request.form.get(column_name)
        data.append(input_value)
    price = str(predict(data))
    return render_template('pred.html', price=price)


if __name__ == '__main__':
    app.run(debug=True)
