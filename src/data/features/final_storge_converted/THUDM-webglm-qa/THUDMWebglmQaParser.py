import sys
import random
sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from datasets import load_dataset

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "WebglmQA"


class WebglmQA(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=True,
                         )
        self.target_config = AdvanceInstructSample
        self.target_fields = ["question_text", "orig_answer_texts"]

    def read(self):
        super(WebglmQA, self).read()
        self.data_read = load_dataset("THUDM/webglm-qa")

        return None

    def convert(self):
        super(WebglmQA, self).convert()

        data_converted = []
        lfqa_prefixs = [
            "\n\n Here are some relevant documents, which may or may not be applicable to the previous question. If you use this information, please indicate 'Based on the provided documents':\n",
            "\n\n Below are some pertinent documents, which may or may not relate to the previous question. If you utilize this information, kindly mention 'In reference to the provided documents':\n",
            "\n\n The following documents may or may not be relevant to the previous question. If you choose to incorporate this information, please acknowledge with 'Based on the provided documents':\n",
            "\n\n Here are some documents that could be useful for the question at hand. It's up to you whether or not to use them. If you do, please state 'Based on the documents provided':\n",
            "\n\n These documents may or may not have relevance to the previous question. If you decide to use them, kindly acknowledge with 'In reference to the provided documents':\n",
            "\n\n Here are some documents that might be of interest in relation to the previous question. If you opt to use them, please mention 'Based on the provided documents':\n",
            "\n\n The following documents may or may not pertain to the previous question. If you incorporate this information, kindly indicate 'In reference to the provided documents':\n",
            "\n\n Here are some relevant documents, which may or may not have relevance to the previous question. If you use this information, please acknowledge with 'Based on the provided documents':\n",
            "\n\n Below are some pertinent documents that may or may not be applicable to the previous question. If you choose to incorporate this information, please state 'In reference to the provided documents':\n",
            "\n\n The following documents may or may not relate to the previous question. If you decide to use them, kindly mention 'Based on the provided documents':\n",
            "\n\n Here are some documents that could be useful for the question at hand. It's up to you whether or not to use them. If you do, please acknowledge with 'In reference to the provided documents':\n",
            "\n\n These documents may or may not have relevance to the previous question. If you opt to use them, please state 'Based on the documents provided':\n",
            "\n\n Here are some documents that might be of interest in relation to the previous question. If you choose to use them, kindly indicate 'In reference to the provided documents':\n",
            "\n\n The following documents may or may not pertain to the previous question. If you incorporate this information, please acknowledge with 'Based on the provided documents':\n",
            "\n\n Here are some relevant documents, which may or may not have relevance to the previous question. If you use this information, please indicate 'In reference to the provided documents':\n",
            "\n\n Below are some pertinent documents that may or may not be applicable to the previous question. If you opt to use this information, please state 'Based on the provided documents':\n",
            "\n\n The following documents may or may not relate to the previous question. If you decide to use them, kindly mention 'In reference to the provided documents':\n",
            "\n\n Here are some documents that could be useful for the question at hand. It's up to you whether or not to use them. If you do, please acknowledge with 'Based on the documents provided':\n",
            "\n\n These documents may or may not have relevance to the previous question. If you choose to use them, please indicate 'In reference to the provided documents':\n",
            "\n\n Here are some documents that might be of interest in relation to the previous question. If you opt to use them, kindly state 'Based on the provided documents':\n",
        ]
        lfqa_system_prompts = [
            "You are an AI assistant specializing in Question Answering. Please answer the following question based on the provided documents.",
            "Incorporate the information from the documents into your response.",
            "Consider the relevance of the provided documents in your answer.",
            "Your response should take into account the information contained in the documents.",
            "Use the documents as a reference when responding to the question.",
            "Incorporate the relevant information from the provided documents.",
            "Base your response on the information presented in the documents.",
            "Make sure to address the question with the help of the provided documents.",
            "Take the information from the documents into consideration when answering.",
            "Your answer should be influenced by the information in the provided documents.",
            "Ensure that your response is informed by the contents of the documents.",
            "Integrate the information from the documents into your reply.",
            "Refer to the documents when formulating your response.",
            "Keep the information in the documents in mind when answering.",
            "The documents are there to assist you in your response.",
            "Use the information from the documents to support your answer.",
            "Your response should reflect the content of the provided documents.",
            "Take advantage of the information in the documents when answering.",
            "In your response, consider the information from the documents.",
            "The provided documents can be a valuable resource in your answer.",
            "As an AI assistant specialized in Question Answering, analyze the provided documents and answer the question accordingly.",
            "Utilize the knowledge from the documents as a Question Answering AI assistant to address the question.",
            "Based on your specialization in Question Answering, make sure to use the provided documents in your response.",
            "Your expertise as a Question Answering AI assistant should guide you in utilizing the provided documents effectively.",
            "Your role as a specialized Question Answering AI assistant makes it essential to refer to the documents in your response.",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]
        lfqa_response_prefixs = [
            "Based on the document, ",
            "In reference to the provided documents, ",
            "Considering the information in the documents, ",
            "Taking into account the relevant content in the documents, ",
            "With the documents as a reference, ",
            "Incorporating information from the documents, ",
            "Utilizing the provided documents, ",
            "Drawing upon the content in the documents, ",
            "Referencing the documents, ",
            "Having reviewed the documents, ",
            "In light of the information in the documents, ",
            "In accordance with the documents, ",
            "Considering the materials provided, ",
            "Bearing in mind the content in the documents, ",
            "With the documents as a resource, ",
            "Incorporating knowledge from the documents, ",
            "Referring to the documents, ",
            "Taking the information in the documents into consideration, ",
            "Using the documents as a source, ",
            "Incorporating data from the documents, ",
            "",
            "",
            "",
            ""
        ]
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                # Randomly assign generic system prompt to data
                data_dict['system_prompt'] = random.choice(lfqa_system_prompts)
                data_dict['qas_id'] = self.id_generator(size=6)

                lfqa_prefix = random.choice(lfqa_prefixs)
                data_dict['question_text'] = data['question'] + lfqa_prefix
                for ref in data['references']:
                    data_dict['question_text'] += ref + "\n\n"
                lfqa_response_prefix = random.choice(lfqa_response_prefixs)
                data_dict['orig_answer_texts'] = lfqa_response_prefix + data['answer']

                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    webglm_qa_parser = WebglmQA(r"src/data/features/final_storge_converted/THUDM-webglm-qa/dummy.txt",
                                r"src/data/features/final_storge_converted/THUDM-webglm-qa")
    webglm_qa_parser.read()
    webglm_qa_parser.convert()
    webglm_qa_parser.save
