from django.shortcuts import render
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load the data and split into features and target
    data = pd.read_csv(r"diabetes.csv")
    X = data.drop("Outcome", axis=1)
    Y = data["Outcome"]

    # Train-test split and model training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)

    # Retrieve user inputs
    val1, val2, val3, val4 = float(request.GET['n1']), float(request.GET['n2']), float(request.GET['n3']), float(request.GET['n4'])
    val5, val6, val7, val8 = float(request.GET['n5']), float(request.GET['n6']), float(request.GET['n7']), float(request.GET['n8'])

    # Convert user input to DataFrame for SHAP compatibility
    user_input = pd.DataFrame([[val1, val2, val3, val4, val5, val6, val7, val8]], columns=X.columns)

    # Make prediction
    pred = model.predict(user_input)
    result1 = "Oops! You have DIABETES. Be more active and eat healthy foods." if pred == [1] else "Great! You DON'T have diabetes."

    # Initialize SHAP explainer and calculate SHAP values
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(user_input)

    # Prepare explanation output
    feature_names = X.columns.tolist()
    shap_values_list = shap_values[0].tolist()
    explanation = {feature_names[i]: round(shap_values_list[i], 2) for i in range(len(feature_names))}
    sorted_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
    explanation_text = "Positive values increase the risk, while negative values decrease it.\n" + \
                       "\n".join([f"{feature}: {impact}" for feature, impact in sorted_explanation.items()])

    return render(request, "predict.html", {'result2': result1, 'explanation': explanation_text})
