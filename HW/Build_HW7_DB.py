# import streamlit as st
# from openai import OpenAI
# import sys
# from pathlib import Path
# import pandas as pd
# import hashlib

# # A fix for working with ChromaDB on Streamlit Community Cloud
# __import__("pysqlite3")
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# import chromadb


# #### OPENAI CLIENT ####

# if "openai_client" not in st.session_state:
#     st.session_state.openai_client = OpenAI(
#         api_key=st.secrets["OPENAI_API_KEY"]
#     )


# #### ADD DOCUMENT TO CHROMADB ####

# def add_to_collection(collection, text, document_id, metadata):

#     client = st.session_state.openai_client

#     response = client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )

#     embedding = response.data[0].embedding

#     collection.add(
#         documents=[text],
#         ids=[document_id],
#         embeddings=[embedding],
#         metadatas=[metadata]
#     )


# #### FILE LOCATIONS ####

# base_path = Path(__file__).resolve().parent.parent

# data_path = base_path / "news.csv"

# db_path = base_path / "ChromaDB_for_HW7"


# #### MAIN PAGE ####

# st.title("Build HW7 News Database")

# if not data_path.exists():
#     st.error(
#         "Place news.csv in the same folder as Build_HW7_DB.py."
#     )
#     st.stop()


# #### READ CSV ####

# news = pd.read_csv(data_path)

# news = news.drop_duplicates().reset_index(drop=True)

# st.write("Rows after removing exact duplicates:", len(news))


# #### CREATE OR OPEN COLLECTION ####

# chroma_client = chromadb.PersistentClient(
#     path=str(db_path)
# )

# collection = chroma_client.get_or_create_collection(
#     "HW7Collection"
# )

# st.write("Documents currently in database:", collection.count())


# #### BUILD DATABASE ####

# if st.button("Build ChromaDB"):

#     existing_ids = set(
#         collection.get(include=[])["ids"]
#     )

#     progress = st.progress(0)
#     status = st.empty()

#     for index, row in news.iterrows():

#         text = (
#             "Company: " + str(row["company_name"])
#             + "\nDate: " + str(row["Date"])
#             + "\nArticle: " + str(row["Document"])
#             + "\nURL: " + str(row["URL"])
#         )

#         document_id = hashlib.sha256(
#             text.encode("utf-8")
#         ).hexdigest()

#         # Skip documents already saved.
#         if document_id not in existing_ids:

#             metadata = {
#                 "company_name": str(row["company_name"]),
#                 "date": str(row["Date"]),
#                 "url": str(row["URL"]),
#                 "days_since_2000": int(row["days_since_2000"])
#             }

#             add_to_collection(
#                 collection,
#                 text,
#                 document_id,
#                 metadata
#             )

#             existing_ids.add(document_id)

#         progress.progress(
#             (index + 1) / len(news)
#         )

#         status.write(
#             f"Processed {index + 1} of {len(news)} rows"
#         )

#     st.success("ChromaDB creation complete!")

#     st.write(
#         "Documents in vector DB:",
#         collection.count()
#     )


# #### DOWNLOAD SAVED DATABASE ####

# # Close the app before using this downloaded copy elsewhere.
# if collection.count() == len(news):

#     import shutil
#     import tempfile

#     if st.button("Prepare database download"):

#         # A database backup should be made while no writes are running.
#         with tempfile.TemporaryDirectory() as temp_folder:

#             archive_path = shutil.make_archive(
#                 str(Path(temp_folder) / "ChromaDB_for_HW7"),
#                 "zip",
#                 root_dir=str(base_path),
#                 base_dir="ChromaDB_for_HW7"
#             )

#             st.session_state["HW7_DB_zip"] = (
#                 Path(archive_path).read_bytes()
#             )

#     if "HW7_DB_zip" in st.session_state:

#         st.download_button(
#             label="Download ChromaDB_for_HW7.zip",
#             data=st.session_state["HW7_DB_zip"],
#             file_name="ChromaDB_for_HW7.zip",
#             mime="application/zip"
#         )