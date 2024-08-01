import streamlit as st
import pandas as pd
import io
import time
from streamlit_chartjs.st_chart_component import st_chartjs
from langchain_community.chat_models import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import json

# 環境変数の読み込み
load_dotenv()

# ファイルの有効期限（秒）
FILE_EXPIRATION_TIME = 3600  # 60分

st.title('LLM-powered Chart.js Configuration Generator with File Upload')

# セッション状態の初期化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_upload_time' not in st.session_state:
    st.session_state.file_upload_time = None
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# ファイルの有効期限をチェックし、必要に応じてリセットする関数
def check_file_expiration():
    if st.session_state.file_upload_time is not None:
        current_time = time.time()
        if current_time - st.session_state.file_upload_time > FILE_EXPIRATION_TIME:
            st.session_state.df = None
            st.session_state.file_upload_time = None
            st.session_state.uploaded_file_content = None
            st.session_state.uploaded_file_name = None
            st.warning("Uploaded file has expired. Please upload a new file.")

# ファイルアップロード
uploaded_file = st.file_uploader('Upload a file for analysis', type=['csv', 'xlsx'])

if uploaded_file is not None:
    # 新しいファイルがアップロードされた場合
    file_contents = uploaded_file.read()
    if (st.session_state.uploaded_file_content != file_contents or 
        st.session_state.uploaded_file_name != uploaded_file.name):
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(io.BytesIO(file_contents))
        elif uploaded_file.name.endswith('.xlsx'):
            st.session_state.df = pd.read_excel(io.BytesIO(file_contents))
        st.session_state.file_upload_time = time.time()
        st.session_state.uploaded_file_content = file_contents
        st.session_state.uploaded_file_name = uploaded_file.name
elif st.session_state.uploaded_file_content is not None:
    # 保存されているファイルを使用
    if st.session_state.uploaded_file_name.endswith('.csv'):
        st.session_state.df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_content))
    elif st.session_state.uploaded_file_name.endswith('.xlsx'):
        st.session_state.df = pd.read_excel(io.BytesIO(st.session_state.uploaded_file_content))

# ファイルの有効期限をチェック
check_file_expiration()

if st.session_state.df is not None:
    st.write("Data Preview:")
    st.dataframe(st.session_state.df.head())

    # Chart.js設定のためのPydanticモデル
    class ChartJSConfig(BaseModel):
        labels: list = Field(description="Labels for the chart (must be an existing column name)")
        datasets: list = Field(description="Datasets for the chart (must be existing column names)")
        chart_type: str = Field(description="Type of the chart (e.g., 'bar', 'line', 'pie', etc.)")
        title: str = Field(description="Title of the chart")

    # 出力パーサーの設定
    parser = PydanticOutputParser(pydantic_object=ChartJSConfig)

    # プロンプトテンプレートの作成
    template = """
    Based on the user's request and the available columns, generate a Chart.js configuration.
    Available columns: {columns}
    User's request: {user_request}

    {format_instructions}

    Make sure to include appropriate labels, datasets, chart type, and title.
    IMPORTANT: Use ONLY the column names provided in the 'Available columns' list for labels and datasets.
    The output should be compatible with the following structure:
    {{
        "labels": ["column_name"],
        "datasets": [
            {{
                "label": "column_name",
                "data": "column_name",
                "borderWidth": 1
            }}
        ]
    }}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["user_request", "columns"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # AzureOpenAI の設定
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    # LLMChainの作成
    chain = LLMChain(llm=llm, prompt=prompt)

    # ユーザー入力
    user_request = st.text_area("Describe the chart you want to create:", height=100)

    if st.button("Generate Chart"):
        if user_request:
            with st.spinner("Generating chart configuration..."):
                try:
                    # LLMを使用して設定を生成
                    response = chain.run(user_request=user_request, columns=st.session_state.df.columns.tolist())
                    config = parser.parse(response)
                    
                    # 生成された設定の表示
                    st.subheader("Generated Chart.js Configuration:")
                    st.json(config.dict())
                    
                    # データの準備（エラーハンドリング付き）
                    if config.labels[0] not in st.session_state.df.columns:
                        st.error(f"Error: '{config.labels[0]}' is not a valid column name.")
                    else:
                        labels = st.session_state.df[config.labels[0]].tolist()
                        datasets = []
                        for dataset in config.datasets:
                            if dataset["data"] not in st.session_state.df.columns:
                                st.error(f"Error: '{dataset['data']}' is not a valid column name.")
                            else:
                                datasets.append({
                                    "label": dataset["label"],
                                    "data": st.session_state.df[dataset["data"]].tolist(),
                                    "borderWidth": dataset.get("borderWidth", 1)
                                })
                        
                        if datasets:  # データセットが少なくとも1つ有効な場合にのみグラフを表示
                            # Chart.jsを使用してグラフを表示
                            st_chartjs(data={"labels": labels, "datasets": datasets}, 
                                        chart_type=config.chart_type, 
                                        title=config.title)
                        else:
                            st.error("No valid datasets found. Unable to generate chart.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a description for the chart you want to create.")
else:
    st.warning("Please upload a file before starting the analysis.")

# ファイルリセットボタン
if st.button("Reset uploaded file"):
    st.session_state.df = None
    st.session_state.file_upload_time = None
    st.session_state.uploaded_file_content = None
    st.session_state.uploaded_file_name = None
    st.success("Uploaded file has been reset.")
    st.experimental_rerun()