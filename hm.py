import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Utility Functions
# =========================
def preprocess_data(df):
    """Encode categorical vars, scale numeric features, return X, scaler, encoders"""
    df = df.copy()

    # Identify categorical and numeric features
    cat_cols = ["Extracurricular", "Internet", "Gender", "LearningStyle",
                "OnlineCourses", "EduTech"]
    num_cols = ["StudyHours", "Attendance", "Motivation", "AssignmentCompletion",
                "ExamScore", "StressLevel", "Age", "Discussions", "Resources"]

    # Encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Scale numerics
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols + cat_cols])  # full feature set

    return X_num, scaler, encoders


def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def run_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def describe_clusters(df, labels):
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    summary = df_clustered.groupby("Cluster").mean()
    return summary


def assign_single_student(input_dict, scaler, encoders, kmeans, pca):
    """Take single student dict, preprocess same way, assign cluster"""
    df_input = pd.DataFrame([input_dict])

    # Encode categoricals
    for col, le in encoders.items():
        df_input[col] = le.transform(df_input[col].astype(str))

    # Scale
    X_input = scaler.transform(df_input)

    # PCA
    X_input_pca = pca.transform(X_input)

    # Cluster assignment
    cluster = kmeans.predict(X_input_pca)[0]
    return cluster


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Student Learning Clustering", layout="wide")

st.title("🎓 Student Learning Behavior Clustering with PCA + K-Means")

# File upload
uploaded_file = st.file_uploader("Upload student_performance.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    X, scaler, encoders = preprocess_data(df)

    # Sidebar for clustering options
    st.sidebar.header("Clustering Options")
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    n_components = st.sidebar.radio("PCA Components", [2, 3], index=0)

    # PCA
    X_pca, pca = perform_pca(X, n_components=n_components)

    # K-Means
    labels, kmeans = run_kmeans(X_pca, n_clusters)
    df["Cluster"] = labels

    # Visualize PCA clusters
    st.subheader("PCA Cluster Visualization")
    fig, ax = plt.subplots()
    if n_components == 2:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", ax=ax)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    else:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="Set2")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
    st.pyplot(fig)

    # Cluster summaries
    st.subheader("Cluster Profiles")
    cluster_summary = describe_clusters(df.drop(columns=["FinalGrade"]), labels)
    st.dataframe(cluster_summary)

    # Compare with FinalGrade (if available)
    if "FinalGrade" in df.columns:
        st.subheader("Final Grade Distribution by Cluster")
        fig, ax = plt.subplots()
        sns.boxplot(x="Cluster", y="FinalGrade", data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

    # Single student input
    st.subheader("🔍 Classify a New Student into a Cluster")

    input_dict = {}
    # Numeric features
    input_dict["StudyHours"] = st.number_input("Study Hours", 0, 80, 10)
    input_dict["Attendance"] = st.slider("Attendance (%)", 0, 100, 75)
    input_dict["Motivation"] = st.slider("Motivation (1-10)", 1, 10, 5)
    input_dict["AssignmentCompletion"] = st.slider("Assignment Completion (1-10)", 1, 10, 7)
    input_dict["ExamScore"] = st.slider("Exam Score", 0, 100, 60)
    input_dict["StressLevel"] = st.slider("Stress Level (1-10)", 1, 10, 5)
    input_dict["Age"] = st.number_input("Age", 18, 30, 21)
    input_dict["Discussions"] = st.slider("Discussions Participation", 0, 10, 3)
    input_dict["Resources"] = st.slider("Resource Usage (1-10)", 1, 10, 5)

    # Categorical features
    input_dict["Extracurricular"] = st.selectbox("Extracurricular", ["Yes", "No"])
    input_dict["Internet"] = st.selectbox("Internet Access", ["Yes", "No"])
    input_dict["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    input_dict["LearningStyle"] = st.selectbox("Learning Style",
                                               ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"])
    input_dict["OnlineCourses"] = st.selectbox("Online Courses", ["Yes", "No"])
    input_dict["EduTech"] = st.selectbox("EduTech Usage", ["Yes", "No"])

    if st.button("Assign Cluster"):
        cluster = assign_single_student(input_dict, scaler, encoders, kmeans, pca)
        st.success(f"This student belongs to Cluster {cluster}")

        # Show cluster profile
        st.write("### Cluster Characteristics")
        st.write(cluster_summary.loc[cluster])
