# customer_segmentation_dashboard_full.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

st.title("Customer Segmentation Dashboard (Upload CSV + Plots)")

# ---------- Upload CSV ----------
uploaded_file = st.file_uploader("Upload your customer CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Check required columns
    required_cols = ["Age", "Income", "SpendingScore"]
    if all(col in df.columns for col in required_cols):

        # ---------- Clustering ----------
        n_clusters = st.slider("Select number of clusters for KMeans", 2, 10, 3)
        clustering_algo = st.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN"])

        if clustering_algo == "KMeans":
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = cluster_model.fit_predict(df[required_cols])
        else:
            cluster_model = DBSCAN(eps=10, min_samples=2)
            df['Cluster'] = cluster_model.fit_predict(df[required_cols])

        st.subheader("Clustered Data")
        st.dataframe(df)

        # ---------- KNN for new customer ----------
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(df[required_cols], df['Cluster'])

        st.subheader("Predict Cluster for New Customer")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        income = st.number_input("Income (k$)", min_value=0, max_value=1000, value=40)
        spending = st.number_input("Spending Score", min_value=0, max_value=100, value=60)

        if st.button("Predict Cluster"):
            new_customer = [[age, income, spending]]
            cluster = knn.predict(new_customer)[0]
            st.success(f"The new customer belongs to **Cluster {cluster}**")

        # ---------- Visualizations ----------
        st.subheader("Visualizations")
        plot_type = st.selectbox("Choose plot type", [
            "2D Scatter (Age vs Income)",
            "2D Scatter (Income vs SpendingScore)",
            "3D Scatter (Age, Income, SpendingScore)",
            "Bar Chart (Average SpendingScore by Cluster)",
            "Histogram (Income Distribution)",
            "Boxplot (Age by Cluster)"
        ])

        if plot_type == "2D Scatter (Age vs Income)":
            plt.figure(figsize=(8,5))
            sns.scatterplot(x="Age", y="Income", hue="Cluster", data=df, palette="Set2", s=100)
            plt.title("Age vs Income by Cluster")
            st.pyplot(plt)

        elif plot_type == "2D Scatter (Income vs SpendingScore)":
            plt.figure(figsize=(8,5))
            sns.scatterplot(x="Income", y="SpendingScore", hue="Cluster", data=df, palette="Set2", s=100)
            plt.title("Income vs SpendingScore by Cluster")
            st.pyplot(plt)

        elif plot_type == "3D Scatter (Age, Income, SpendingScore)":
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(df["Age"], df["Income"], df["SpendingScore"], c=df["Cluster"], cmap="Set2", s=50)
            ax.set_xlabel("Age")
            ax.set_ylabel("Income")
            ax.set_zlabel("SpendingScore")
            plt.title("3D Cluster Visualization")
            st.pyplot(fig)

        elif plot_type == "Bar Chart (Average SpendingScore by Cluster)":
            bar_data = df.groupby('Cluster')['SpendingScore'].mean().reset_index()
            plt.figure(figsize=(8,5))
            sns.barplot(x='Cluster', y='SpendingScore', data=bar_data, palette="Set2")
            plt.title("Average SpendingScore by Cluster")
            st.pyplot(plt)

        elif plot_type == "Histogram (Income Distribution)":
            plt.figure(figsize=(8,5))
            sns.histplot(df['Income'], bins=10, kde=True, color='skyblue')
            plt.title("Income Distribution")
            st.pyplot(plt)

        elif plot_type == "Boxplot (Age by Cluster)":
            plt.figure(figsize=(8,5))
            sns.boxplot(x='Cluster', y='Age', data=df, palette="Set2")
            plt.title("Age Distribution by Cluster")
            st.pyplot(plt)

    else:
        st.warning(f"CSV must contain columns: {required_cols}")
else:
    st.info("Please upload a CSV file to start clustering and visualizations.")
