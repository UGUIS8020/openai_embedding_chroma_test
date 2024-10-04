from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
import json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# OpenAI API キーの設定
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI Embeddings の初期化
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Chroma の初期化
persist_directory = "./chroma_db"
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def process_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension in ['.json', '.txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_extension == '.json':
            content = json.dumps(json.loads(content))  # JSON を整形された文字列に変換
        
        embedding = embedding_model.embed_query(content)
        return content, embedding
    
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        # 画像ファイルの処理
        # ここでは画像の内容を文字列として扱います
        return f"Image file: {file_path}", None
    
    else:
        return None, None

def main():
    data_directory = "./data/jpg"
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        content, embedding = process_file(file_path)
        
        if content is not None:
            metadata = {"filename": filename, "type": os.path.splitext(filename)[1][1:]}
            
            # Documentオブジェクトを作成
            doc = Document(page_content=content, metadata=metadata)
            
            # add_documents メソッドを使用
            chroma_db.add_documents([doc], ids=[filename])
    
    print("処理が完了しました。ベクトルストアに保存されました。")

if __name__ == "__main__":
    main()
