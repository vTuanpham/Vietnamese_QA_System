import os
import sys
sys.path.insert(0, './')
import txtai

from setup_db import \
    (setup_database, drop_tables, query, insert_data)


sample_queries = (
    "Cho tôi một sự thật về vũ khí",
    "Thành phố nào ở Việt Nam có mật độ dân số cao?",
    "Tell me about natural landscapes in Canada",
    "What's the famous landmark in Paris?",
    "When did the first moon landing happen?",
    "What is the Great Wall of China?",
    "Describe the biodiversity in the Amazon Rainforest.",
    "Thủ đô của Việt Nam?",
    "TPHCM có bao nhiêu quận?",
    "Vinfast là công ty sản xuất gì?",
    "Đồ ăn gì ngon nhất ở Việt Nam?",
    "Bánh mỳ được làm từ gì?",
    "Công thức để nấu phở bò ngon miệng",
    "Chiều nay ăn gì?",
    "Landmark 81 nằm ở đâu?",
    "Tell me about the history of the Eiffel Tower.",
    "Kể về lịch sử của tháp Eiffel.",
    "What are some traditional dishes in Japan?",
    "Một số món ăn truyền thống của Nhật Bản là gì?",
    "Explain the concept of climate change and its impact on the environment.",
    "Hãy giải thích khái niệm biến đổi khí hậu và tác động của nó lên môi trường.",
    "Who is the current President of the United States?",
    "Tổng thống hiện tại của Hoa Kỳ là ai?",
    "What is the significance of the Taj Mahal in Indian history and culture?",
    "Taj Mahal có ý nghĩa gì trong lịch sử và văn hóa Ấn Độ?",
    "What are the top tourist attractions in New York City?",
    "Những điểm du lịch hàng đầu ở New York City là gì?",
    "How does photosynthesis work in plants?",
    "Quá trình quang hợp hoạt động như thế nào trong cây cỏ?",
    "Can you provide a brief overview of the Industrial Revolution?",
    "Bạn có thể cung cấp một cái nhìn tổng quan ngắn gọn về Cách mạng Công nghiệp không?",
    "Tell me about the traditions and customs of the Maasai people in Africa.",
    "Hãy kể về truyền thống và phong tục của người Maasai ở châu Phi.",
    "What is the Hubble Space Telescope, and what has it discovered in space?",
    "Kính thiên văn Hubble là gì và nó đã phát hiện gì trong không gian?",
    "Explain the theory of relativity by Albert Einstein.",
    "Hãy giải thích lý thuyết tương đối của Albert Einstein.",
    "What are the famous festivals in India?",
    "Các lễ hội nổi tiếng nào ở Ấn Độ?",
    "Tell me about the cultural significance of the Great Barrier Reef in Australia.",
    "Hãy kể về ý nghĩa văn hóa của Rạn san hô Great Barrier ở Úc.",
    "How does the human digestive system work?",
    "Hệ tiêu hóa của con người hoạt động như thế nào?",
    "What is the current status of space exploration and upcoming missions to Mars?",
    "Tình hình hiện tại của khám phá không gian và các nhiệm vụ sắp tới tới Sao Hỏa là gì?",
    "Explain the concept of renewable energy sources.",
    "Hãy giải thích khái niệm các nguồn năng lượng tái tạo.",
    "Who is William Shakespeare, and what are his famous works?",
    "William Shakespeare là ai, và tác phẩm nổi tiếng của ông là gì?",
    "Tell me about the history and cultural significance of the pyramids in Egypt.",
    "Hãy kể về lịch sử và ý nghĩa văn hóa của các kim tự tháp ở Ai Cập.",
    "How do electric cars work, and what are their environmental benefits?",
    "Ô tô điện hoạt động như thế nào và lợi ích môi trường của chúng là gì?",
    "Explain the concept of artificial intelligence and its applications in modern technology.",
    "Hãy giải thích khái niệm trí tuệ nhân tạo và các ứng dụng của nó trong công nghệ hiện đại."
)
# insert_data("inference_pipeline/dbs/documents.db",
#             table_name='docs',
#             data=fake_data)
data = query("inference_pipeline/dbs/documents.db",
              query_string='''SELECT * FROM documents''',
              fetch_size=50000)
# print(data)
data_str = []
for row in data:
    data_str.append({"id": row[0], "text": row[1], "source": row[2]})
    # print({"id": row[0], "text": row[1], "source": row[2]})
# embeddings_MiniLM = txtai.Embeddings(hybrid=True,
#                                      content=True,
#                                      path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# embeddings_mpnet = txtai.Embeddings(hybrid=True,
#                                     content=True,
#                                     path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Create an index for the list of text
# embeddings_MiniLM.index(data_str)
# embeddings_MiniLM.save("./inference_pipeline/embeddings_index/mini_lm")
# embeddings_mpnet.index(data_str)
# embeddings_mpnet.save("./inference_pipeline/embeddings_index/mpnet")

embeddings_MiniLM = txtai.Embeddings()
embeddings_MiniLM.load("./inference_pipeline/embeddings_index/mini_lm")
embeddings_mpnet = txtai.Embeddings()
embeddings_mpnet.load("./inference_pipeline/embeddings_index/mpnet")

# Run an embeddings search for each query
for query_str in sample_queries:
    semantic_MiniLM = embeddings_MiniLM.search(query_str, 1)[0]
    uid_paraphrase, score_paraphrase = semantic_MiniLM['id'], semantic_MiniLM['score']
    semantic_mpnet = embeddings_mpnet.search(query_str, 1)[0]
    uid_qa, score_qa = semantic_mpnet['id'], semantic_mpnet['score']
    data = query("inference_pipeline/dbs/documents.db",
                 query_string=f'''SELECT doc FROM documents WHERE id = {uid_paraphrase}''',
                 fetch_size=1)[0]
    print(f"Query: {query_str} \nRelevant docs {score_paraphrase}: {data}\n\n")
    data = query("inference_pipeline/dbs/documents.db",
                 query_string=f'''SELECT doc FROM documents WHERE id = {uid_qa}''',
                 fetch_size=1)[0]
    print(f"Query: {query_str} \nRelevant docs {score_qa}: {data}\n\n")
    if uid_paraphrase == uid_qa and score_paraphrase + score_qa > 0.4:
        data = query("inference_pipeline/dbs/documents.db",
                     query_string=f'''SELECT doc FROM documents WHERE id = {uid_paraphrase}''',
                     fetch_size="all")[0][0]
        print("\nMatch relevant docs: ")
        print(f"Query: {query_str} \nRelevant docs {score_qa + score_paraphrase}: {data}\n\n")

# drop_tables("inference_pipeline/dbs/documents.db",
#             tables_to_drop=["docs"])
