import os
import pickle  # Chromaファイルを読み込むのに使用
import uuid  # UUIDファイルを扱うのに使用
import numpy as np  # ランダムベクトルを作成するために使用
from pinecone import Pinecone, ServerlessSpec  # PineconeクラスとServerlessSpecのインポート

# PineconeのAPIキーと環境変数の設定
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Pineconeクラスのインスタンスを作成
pc = Pinecone(api_key=pinecone_api_key)

# インデックス名の設定
index_name = "shibuya"

# インデックスが存在しない場合は作成
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Chromaベクトルの次元に合わせる
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # クラウドプロバイダーの設定
            region="us-east-1"  # 使用するリージョン
        )
    )

# インデックスに接続
index = pc.Index(index_name)  # pc インスタンスからインデックスに接続

# ChromaファイルとUUIDファイルがあるフォルダのパス
folder_path = "./chroma_db/shibuya"

# ChromaファイルとUUIDファイルの読み込みとPineconeへのアップロード
for filename in os.listdir(folder_path):
    # Chromaファイルを処理
    if filename.endswith(".chroma"):
        chroma_path = os.path.join(folder_path, filename)
        with open(chroma_path, 'rb') as file:
            # Chromaファイルをデシリアライズしてベクトル化
            chroma_data = pickle.load(file)
            # Chromaデータをそのまま使える形に整える
            vector = chroma_data['embedding']  # 仮のキー名、実際のChromaファイル構造に合わせる

            # UUIDファイルを処理（対応するUUIDファイルがあると仮定）
            uuid_filename = filename.replace(".chroma", ".uuid")
            uuid_path = os.path.join(folder_path, uuid_filename)
            with open(uuid_path, 'r') as uuid_file:
                uuid_str = uuid_file.read().strip()

            # Pineconeにアップロード
            index.upsert(vectors=[
                {
                    "id": uuid_str,  # UUIDをIDとして使用
                    "values": vector,  # Chromaファイルから取得したベクトル
                    "metadata": {"filename": filename}
                }
            ])

# ランダムなベクトルでクエリを作成（1536次元のランダムベクトル）
query_vector = np.random.rand(1536).tolist()

# Pineconeにクエリを送信
results = index.query(queries=[query_vector], top_k=3, include_metadata=True)

# 結果の表示
for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Filename: {match['metadata']['filename']}\n")