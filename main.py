import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math

def main():
    df = read_csv_file()
    x_train, y_train, df = get_train_data(df)
    visualize_data(df)

    w_in = np.array([0., 0.])
    b_in = 0
    alpha = 0.00001
    num_iters = 1500

    w, b, J_hist = gradient_descent(x_train, y_train, w_in, b_in, compute_cost, compute_gradient, alpha, num_iters)
    print("w, b values found by gradient descent : ")
    print("w : ", w)
    print("b : ", b)

    visualize_J(J_hist)
    visualize_model(w, b, df)

    prediction = predict(w, b)
    print("Sales predicted by the model : ", prediction)

def read_csv_file():
    df = pd.read_csv("Marketing_Data.csv")
    return df

def get_train_data(df):
    yt = df["youtube"]
    newsp = df["newspaper"]
    sales = df["sales"]

    x_train = np.column_stack((yt, newsp))  # Construct the 2D array directly

    y_train = np.array(sales)

    return x_train, y_train, df

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i]) ** 2
    
    cost = cost * (1 / (2 * m))

    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        err_i = f_wb_i - y[i]

        for j in range(n):
            dj_dw[j] += err_i * X[i, j]

        dj_db += err_i
            
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_func, gradient_func, alpha, num_iters):
    J_hist = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_func(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_hist.append(cost_func(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}")

    return w, b, J_hist

def visualize_data(df):
    ax = plt.axes(projection="3d")

    ax.set_title("Advertisement - Sales")
    ax.set_xlabel("Youtube Advertisement")
    ax.set_ylabel("Newspaper Advertisement")
    ax.set_zlabel("Sales")

    yt_ad = np.array(df["youtube"])
    newsp_ad = np.array(df["newspaper"])
    sales = np.array(df["sales"])

    ax.scatter3D(yt_ad, newsp_ad, sales)
    plt.show()

def visualize_J(J_hist):
    plt.plot(J_hist)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function over Iterations")
    plt.show()

def visualize_model(w, b, df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["youtube"], df["newspaper"], df["sales"], label='Actual Data')

    yt_range = np.linspace(df["youtube"].min(), df["youtube"].max(), 100)
    newsp_range = np.linspace(df["newspaper"].min(), df["newspaper"].max(), 100)
    yt_range, newsp_range = np.meshgrid(yt_range, newsp_range)
    sales_predicted = w[0] * yt_range + w[1] * newsp_range + b

    ax.plot_wireframe(yt_range, newsp_range, sales_predicted, color='red', alpha=0.5, label='Model Plane')

    ax.set_xlabel('Youtube Advertisement')
    ax.set_ylabel('Newspaper Advertisement')
    ax.set_zlabel('Sales')
    ax.set_title('3D Model of Linear Regression')
    ax.legend()

    plt.show()

def predict(w, b):
    yt_ad = float(input("Please enter Youtube advertisement : "))
    newsp_ad = float(input("Please enter Newspaper advertisement : "))
    
    pred = np.array([yt_ad, newsp_ad])

    prediction = np.dot(pred, w) + b
    return prediction

main()
