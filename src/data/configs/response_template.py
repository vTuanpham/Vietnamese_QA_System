import random
import sys
import warnings
from functools import partialmethod
from abc import ABC, abstractmethod
sys.path.insert(0, r'./')

from typing import List, Callable
from dataclasses import dataclass

from src.utils.utils import set_seed


NO_ANS_RESPONSE1 = f"Tôi xin lỗi, tôi không thể tìm thấy câu trả lời cho câu hỏi '[QUESTION]'," \
                   f" trong dữ liệu của tôi không chứa câu trả lời cho vấn đề này." \
                   f" Bạn có thể hỏi cụ thể hơn hoặc hỏi tôi một câu hỏi khác được không ?"
NO_ANS_RESPONSE2 = f"Tôi không tìm thấy đáp án cho câu trả lời của bạn trong database," \
                   f" bạn có thể đưa cho tôi thêm các tài liệu liên quan để tôi bổ sung được không ?"
NO_ANS_RESPONSE3 = f"Tôi không thể trả lời câu hỏi của bạn, có thể do dữ liệu của tôi trong database bị thiếu hụt." \
                   f" Bạn có thể hỏi tôi một câu hỏi khác được không ?"
NO_ANS_RESPONSE4 = f"Câu trả lời không tìm thấy trong dữ liệu hiện có của tôi," \
                   f" bạn có thể hỏi tôi một câu hỏi khác hoặc cung cấp thêm context cho tôi được không ?"
NO_ANS_RESPONSE5 = f"Xin lỗi, tôi không có thông tin cần thiết để trả lời câu hỏi '[QUESTION]'." \
                   f" Có thể bạn muốn thử đặt câu hỏi khác hoặc cung cấp thêm context để tôi hiểu rõ hơn."
NO_ANS_RESPONSE6 = f"Xin lỗi, nhưng tôi không thể cung cấp câu trả lời cho câu hỏi này '[QUESTION]'." \
                   f" Bạn có thể thử đặt câu hỏi khác cho tôi không?"
NO_ANS_RESPONSE7 = f"Tôi rất tiếc, nhưng tôi không có thông tin cụ thể nào liên quan đến câu hỏi '[QUESTION]'." \
                   f" Có câu hỏi khác nào tôi có thể giúp đỡ?"
NO_ANS_RESPONSE8 = f"Câu trả lời cho câu hỏi này '[QUESTION]' không có trong database của tôi." \
                   f" Hãy thử hỏi điều gì đó khác nhé!"
NO_ANS_RESPONSE9 = f"Tôi không thể tìm thấy thông tin liên quan đến câu hỏi '[QUESTION]' trong nguồn dữ liệu hiện có." \
                   f" Có thể bạn muốn thử hỏi điều gì đó khác?"
NO_ANS_RESPONSE10 = f"Rất tiếc, tôi không có đủ thông tin để trả lời câu hỏi của bạn." \
                    f" Bạn có thể đưa thêm chi tiết hoặc hỏi một câu hỏi khác."
NO_ANS_RESPONSE11 = f"Tôi không thể cung cấp câu trả lời chính xác cho câu hỏi này '[QUESTION]'." \
                    f" Bạn có thể cung cấp thêm thông tin để tôi hiểu rõ hơn không?"
NO_ANS_RESPONSE12 = f"Xin lỗi, tôi không có thông tin cần thiết để đáp ứng yêu cầu của bạn." \
                    f" Bạn có thể thử hỏi một câu hỏi khác hay không?"
NO_ANS_RESPONSE13 = f"Tôi không thể tìm thấy câu trả lời trong database của mình." \
                    f" Hãy thử đưa ra câu hỏi khác để tôi có thể giúp bạn."
NO_ANS_RESPONSE14 = f"Tôi không thể cung cấp câu trả lời đầy đủ cho câu hỏi của bạn." \
                    f" Bạn có thể cung cấp thêm thông tin hoặc hỏi câu hỏi khác."
NO_ANS_RESPONSE15 = f"Xin lỗi, nhưng tôi không thể tìm thấy câu trả lời." \
                    f" Bạn có thể đưa thêm ngữ cảnh hoặc hỏi một câu hỏi khác được không?"
NO_ANS_RESPONSE16 = f"Rất tiếc, tôi không thể tìm thấy câu trả lời mà bạn đang tìm kiếm." \
                    f" Có câu hỏi khác nào tôi có thể giúp đỡ?"
NO_ANS_RESPONSE17 = f"Tôi không có thông tin cụ thể về câu hỏi này '[QUESTION]'." \
                    f" Bạn có thể đưa thêm thông tin chi tiết hoặc hỏi một câu hỏi khác."
NO_ANS_RESPONSE18 = f"Tôi không thể trả lời câu hỏi của bạn dựa trên dữ liệu hiện có." \
                    f" Hãy thử đưa thêm ngữ cảnh hoặc hỏi một câu hỏi khác."
NO_ANS_RESPONSE19 = f"Xin lỗi, tôi không có đủ thông tin để cung cấp câu trả lời cho câu hỏi này '[QUESTION]'." \
                    f" Bạn có thể hỏi một câu hỏi khác không?"
NO_ANS_RESPONSE20 = f"Tôi không thể tìm thấy câu trả lời cho câu hỏi của bạn." \
                    f" Bạn có thể cung cấp thêm thông tin hoặc hỏi một câu hỏi khác."


TRIVIAL_ANS1 = f"Tôi không thể tìm được tài liệu liên quan cho câu hỏi [QUESTION] tuy nhiên," \
               f" theo tôi được biết [ANSWER]"
TRIVIAL_ANS2 = f"Kiến thức trong database không chứa tài liệu liên quan," \
               f" nhưng tôi có thể trả lời câu hỏi này [ANSWER]"
TRIVIAL_ANS3 = f"Theo tôi được biết [ANSWER]"
TRIVIAL_ANS4 = f"[ANSWER]"
TRIVIAL_ANS5 = f"Trong database của tôi không có tài liệu liên quan," \
               f"nhưng cảu hỏi này tôi có thể trả lời như sau: [QUESTION]"
TRIVIAL_ANS6 = f"Tôi không tìm thấy thông tin cụ thể về câu hỏi [QUESTION], tuy nhiên," \
               f"theo kiến thức của tôi, [ANSWER]"
TRIVIAL_ANS7 = f"Có vẻ như tôi không thể truy xuất được tài liệu về [QUESTION]," \
               f" nhưng đáp án có thể là [ANSWER]"
TRIVIAL_ANS8 = f"Xin lỗi, tôi không thể tìm thấy thông tin về [QUESTION]. Tuy nhiên," \
               f"đáp án có thể là [ANSWER]"
TRIVIAL_ANS9 = f"Theo kiến thức hiện có của tôi, câu trả lời cho [QUESTION] có thể là [ANSWER]"
TRIVIAL_ANS10 = f"Không có thông tin cụ thể về câu hỏi [QUESTION], nhưng tôi cho rằng [ANSWER]"
TRIVIAL_ANS11 = f"Tôi không tìm thấy tài liệu nào liên quan đến [QUESTION]," \
                f"nhưng theo kiến thức của tôi, câu trả lời có thể là [ANSWER]"
TRIVIAL_ANS12 = f"Rất tiếc, không có thông tin cụ thể về câu hỏi [QUESTION]. " \
                f"Tuy nhiên, theo tôi được biết, [ANSWER]"
TRIVIAL_ANS13 = f"Không tìm thấy tài liệu liên quan đến [QUESTION]. Tuy vậy," \
                f"tôi nghĩ câu trả lời có thể là [ANSWER]"
TRIVIAL_ANS14 = f"Xin lỗi, tôi không thể tìm thấy thông tin về [QUESTION]," \
                f" nhưng dựa vào kiến thức hiện có, câu trả lời là [ANSWER]"
TRIVIAL_ANS15 = f"Trong database của tôi không có tài liệu liên quan đến [QUESTION]. " \
                f"Nhưng theo tôi được biết, [ANSWER]"
TRIVIAL_ANS16 = f"Xin lỗi, tôi không thể tìm thấy thông tin cụ thể về câu hỏi [QUESTION]. " \
                f"Tuy nhiên, có thể [ANSWER]"
TRIVIAL_ANS17 = f"Không có thông tin cụ thể về [QUESTION] trong database của tôi. " \
                f"Tuy vậy, tôi nghĩ câu trả lời là [ANSWER]"
TRIVIAL_ANS18 = f"Tôi đã kiểm tra nhưng không có tài liệu liên quan đến [QUESTION]. " \
                f"Theo tôi được biết, [ANSWER]"
TRIVIAL_ANS19 = f"Tôi không tìm thấy thông tin về câu hỏi [QUESTION]. " \
                f"Nhưng dựa vào kiến thức hiện có, câu trả lời có thể là [ANSWER]"
TRIVIAL_ANS20 = f"Rất tiếc, không có tài liệu liên quan đến [QUESTION]. " \
                f"Nhưng tôi cho rằng [ANSWER]"

RESPONSE1 = f"Dựa vào thông tin hiện có, tôi nghĩ câu trả lời có thể là: [ANSWER]"

RESPONSE2 = f"[ANSWER]"

RESPONSE3 = f"Theo tôi, có thể câu trả lời là: [ANSWER]"

RESPONSE4 = f"Điều mà tôi có thể kết luận là: [ANSWER]"

RESPONSE5 = f"[ANSWER]"

RESPONSE6 = f"Có nhiều khả năng câu trả lời là: [ANSWER]"

RESPONSE7 = f"Dựa vào dữ kiện, tôi suy luận câu trả lời là: [ANSWER]"

RESPONSE8 = f"Với những thông tin hiện có, tôi đánh giá là: [ANSWER]"

RESPONSE9 = f"Tôi có cảm giác câu trả lời có thể là: [ANSWER]"

RESPONSE10 = f"Dựa vào kiến thức, tôi đưa ra dự đoán là: [ANSWER]"

RESPONSE11 = f"Theo những gì tôi biết, có thể câu trả lời là: [ANSWER]"

RESPONSE12 = f"Từ dữ liệu có sẵn, tôi cho rằng câu trả lời là: [ANSWER]"

RESPONSE13 = f"[ANSWER]"

RESPONSE14 = f"Nhìn vào dữ liệu, tôi đánh giá là: [ANSWER]"

RESPONSE15 = f"[ANSWER]"

RESPONSE16 = f"Không chắc chắn, nhưng dựa vào thông tin hiện có, câu trả lời có thể là: [ANSWER]"

RESPONSE17 = f"Theo tôi, câu trả lời có thể nằm ở: [ANSWER]"

RESPONSE18 = f"Dựa vào dữ liệu có sẵn, tôi đánh giá rằng câu trả lời là: [ANSWER]"

RESPONSE19 = f"Xét về khả năng, tôi cho rằng câu trả lời là: [ANSWER]"

RESPONSE20 = f"Dựa vào thông tin hiện tại, tôi suy đoán câu trả lời là: [ANSWER]"


PROMPT_INPUT1 = f"Dựa vào context sau: [CONTEXT] bạn hãy trả lời câu hỏi sau [QUESTION], nếu không có câu trả lời " \
                f"bạn có thể trả lời dựa trên kiến thức của bạn hoặc trả lời không tìm thấy " \
                f"ví dụ như: 'không tìm thấy cảu trả lời cho câu hỏi của bạn' [EOS]"

PROMPT_INPUT2 = f"Dựa vào kiến thức lấy được từ database: [CONTEXT], hãy trích xuất thông tin ra và trả lời câu hỏi " \
                f"sau [QUESTION], nếu như không tìm thấy câu trả lời, bạn có thể trả lời 'không biết,...'" \
                f"hoặc cố gắng trả lời với kiến thức của bạn [EOS]"

PROMPT_INPUT3 = f"Người dùng hỏi câu hỏi sau: [QUESTION], dựa trên kiến thức lấy được từ database: [CONTEXT], bạn hãy " \
                f"trả lời câu hỏi đó. Nếu như không trả lời được, bạn có thể yêu cầu thêm dữ liệu hoặc cố" \
                f"trả lời với kiến thức của bạn [EOS]"

PROMPT_INPUT4 = f"Bây giờ, hãy tập trung vào câu hỏi sau: [QUESTION]. Context cho câu hỏi này là [CONTEXT]." \
                f" Nếu không tìm thấy câu trả lời, bạn có thể nói 'Câu trả lời không tìm thấy trong dữ liệu hiện " \
                f"có của tôi, bạn có thể hỏi tôi một câu hỏi khác hoặc cung cấp thêm context cho tôi được không?' " \
                f"và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. Hoặc cố trả lời" \
                f"với kiến thức của bạn nếu như câu hỏi đó dễ [EOS]"

PROMPT_INPUT5 = f"Hãy xem xét câu hỏi này: [QUESTION]. Bạn có thể tìm câu trả lời trong dữ liệu này: [CONTEXT]." \
                f" Nếu không có câu trả lời trong dữ liệu, bạn có thể nói 'Xin lỗi, tôi không có thông tin cần thiết để trả lời " \
                f"câu hỏi của bạn. Có thể bạn muốn thử đặt câu hỏi khác hoặc cung cấp thêm context để tôi hiểu " \
                f"rõ hơn.' và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT6 = f"Xin hãy giúp trả lời câu hỏi sau: [QUESTION]. Dựa vào dữ kiện sau đây: [CONTEXT], bạn có thể "    \
                f"trả lời câu hỏi này. Nếu không có câu trả lời, bạn có thể nói 'Xin lỗi, nhưng tôi không thể "     \
                f"cung cấp câu trả lời cho câu hỏi này.' và yêu cầu người dùng cung cấp thông tin bổ sung hoặc "    \
                f"hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT7 = f"Giúp tôi trả lời câu hỏi sau: [QUESTION], dựa vào dữ liệu lấy từ: [CONTEXT]. Nếu không biết, bạn "    \
                f"có thể nói 'Tôi rất tiếc, nhưng tôi không có thông tin cụ thể nào liên quan đến câu hỏi của bạn.' "  \
                f"và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT8 = f"Trả lời câu hỏi sau: [QUESTION], dựa vào thông tin sau: [CONTEXT]. Nếu không có câu trả lời trong" \
                f"dữ liệu bạn có thể cố trả lời với kiến thức của bạn, hoặc "    \
                f"bạn có thể nói 'Câu trả lời cho câu hỏi này không có trong database của tôi.' và yêu cầu người "  \
                f"dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT9 = f"Dựa vào kiến thức từ database: [CONTEXT], hãy trả lời câu hỏi sau: [QUESTION]. "  \
                f"Nếu không trả lời được, bạn có thể nói 'Tôi không thể tìm thấy thông tin liên quan đến câu hỏi " \
                f"của bạn trong nguồn dữ liệu hiện có.' và yêu cầu người dùng cung cấp thông tin bổ sung hoặc " \

PROMPT_INPUT10 = f"Đối với câu hỏi sau: [QUESTION], bạn có thể tìm câu trả lời trong dữ liệu này: [CONTEXT]. "  \
                f"Nếu không biết, bạn có thể nói 'Rất tiếc, tôi không có đủ thông tin để trả lời câu hỏi của bạn.' "   \
                f"và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT11 = f"Bạn hãy trả lời câu hỏi sau: [QUESTION], dựa vào kiến thức lấy được từ database: [CONTEXT]. "    \
                f"Nếu không trả lời được, bạn có thể nói 'Tôi không thể cung cấp câu trả lời chính xác cho câu "    \
                f"hỏi này.' và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT12 = f"Hãy giúp trả lời câu hỏi sau: [QUESTION], dựa vào thông tin sau: [CONTEXT]. "    \
                f"Nếu không có câu trả lời, bạn có thể nói 'Xin lỗi, tôi không có thông tin cần thiết để đáp ứng "  \
                f"yêu cầu của bạn.' và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT13 = f"Trả lời câu hỏi sau: [QUESTION], dựa vào kiến thức từ database: [CONTEXT]. " \
                f"Nếu không trả lời được, bạn có thể nói 'Tôi không thể tìm thấy câu trả lời trong database của mình.' "    \
                f"và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT14 = f"Dựa vào thông tin sau: [CONTEXT], bạn có thể cung cấp câu trả lời đầy đủ cho câu hỏi sau: [QUESTION] "   \
                f"không? Nếu không có câu trả lời, bạn có thể nói 'Tôi không thể cung cấp câu trả lời đầy đủ cho câu "  \
                f"hỏi của bạn.' và yêu cầu người dùng cung cấp thông tin bổ sung hoặc hỏi câu hỏi khác. [EOS]"

PROMPT_INPUT15 = f"Xem xét thông tin sau: [CONTEXT]. "  \
                f"Hãy cố gắng tìm kiếm câu trả lời cho câu hỏi sau: [QUESTION]. "   \
                f"Nếu bạn cần thêm thông tin, đừng ngần ngại yêu cầu hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT16 = f"Xem xét thông tin sau: [CONTEXT]. "  \
                f"Tìm kiếm cẩn thận để giúp tôi trả lời câu hỏi sau: [QUESTION]. "  \
                f"Nếu không tìm thấy, đừng lo lắng, cứ hỏi thêm hoặc đưa ra câu hỏi khác. [EOS]"

PROMPT_INPUT17 = f"Xem xét thông tin sau: [CONTEXT]. "  \
                f"bạn hãy cố gắng giúp tôi tìm câu trả lời cho câu hỏi sau: [QUESTION]. "    \
                f"Nếu thông tin không đủ, hãy cung cấp thêm chi tiết hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT18 = f"Xem xét thông tin sau: [CONTEXT]. "  \
                f"Tôi sẽ cùng bạn tìm kiếm câu trả lời cho câu hỏi này: [QUESTION]. "   \
                f"Nếu cần thêm dữ liệu, hãy yêu cầu hoặc hỏi một câu hỏi khác. [EOS]"

PROMPT_INPUT19 = f"Xem xét thông tin sau: [CONTEXT]. "  \
                f"Hãy giúp tôi tìm câu trả lời cho câu hỏi sau: [QUESTION]. "   \
                f"Nếu không tìm thấy, bạn có thể bảo user hỏi một câu hỏi khác [EOS]"

PROMPT_INPUT20 = f"Xem xét thông tin sau: [CONTEXT]. "  \
                f"Hãy cùng tôi tìm câu trả lời cho câu hỏi sau: [QUESTION]. "   \
                f"Nếu cần thông tin bổ sung, hãy yêu cầu hoặc hỏi một câu hỏi khác. [EOS]"

NO_DOCS_MESSAGE1 = f" Không documents nào có điểm đủ cao để query cho câu hỏi. "
NO_DOCS_MESSAGE2 = f" Database không chứa documents nào phù hợp cho câu hỏi. "

@dataclass
class TEMPLATE(ABC):
    all_attr: dict = globals().items()
    max_template: int = 20

    def __post_init__(self):
        pass

    @abstractmethod
    def get(self, id: int, type: str=None, **kwargs):
        assert id <= self.max_template, "Invalid template id"
        assert type is not None, "Please specified the type of template"
        existed = False
        for existed_type in dict(self.all_attr).keys():
            if type in existed_type:
                existed = True
        assert existed, "The template type provided does not exist"
        pass

    @classmethod
    @property
    def get_random_id(cls) -> int:
        return random.randint(1, cls.max_template)


class QA_TEMPLATE(TEMPLATE):
    def get(self, id: int, type: str, question: str=None,
            context: str=None, answer: str=None):
        super(QA_TEMPLATE, self).get(id=id, type=type)
        template = dict(super(QA_TEMPLATE, self).all_attr)[type+str(id)]

        if question:
            template = template.replace("[QUESTION]", question)
        if context:
            template = template.replace("[CONTEXT]", context)
        if answer:
            template = template.replace("[ANSWER]", answer)

        if "[QUESTION]" in template or "[CONTEXT]" in template or "[ANSWER]" in template:
            warnings.warn("Missing field(s) in template!")
        return template

    get_prompt = partialmethod(get, answer=None, type="PROMPT_INPUT")
    get_neg_response = partialmethod(get, answer=None, context=None, type="NO_ANS_RESPONSE")
    get_trivial_response = partialmethod(get, context=None, type="TRIVIAL_ANS")
    get_norm_response = partialmethod(get, question=None, context=None, type="RESPONSE")
    get_no_docs_msg = partialmethod(get, question=None, context=None, answer=None, type="NO_DOCS_MESSAGE")

    get_random_prompt = partialmethod(get_prompt, id=TEMPLATE.get_random_id)
    get_random_neg_response = partialmethod(get_neg_response, id=TEMPLATE.get_random_id)
    get_random_trivial_response = partialmethod(get_trivial_response, id=TEMPLATE.get_random_id)
    get_random_norm_response = partialmethod(get_norm_response, id=TEMPLATE.get_random_id)


if __name__ == "__main__":
    python_questions = [
        "How do you comment a single line in Python?",
        "Explain the difference between Python 2 and Python 3.",
        "How can you generate a random number in Python?",
        "What is the purpose of the if __name__ == '__main__': statement in Python?",
        "How do you open and read a file in Python?",
        "Explain the usage of the range() function in Python.",
        "What are Python decorators, and how are they used?",
        "How do you handle exceptions in Python?",
        "Explain the differences between lists and tuples in Python.",
        "What is a virtual environment in Python, and why is it useful?",
        "How can you remove duplicates from a list in Python?",
        "Describe the differences between append(), extend(), and insert() in Python lists.",
        "How do you iterate over a dictionary in Python?",
        "Explain the concept of a lambda function in Python.",
        "How can you find the length of a string in Python?",
        "What is the purpose of the pass statement in Python?",
        "How do you create a class and define methods in Python?",
        "Explain the difference between shallow and deep copy of objects in Python.",
        "How do you sort a dictionary by its values in Python?",
        "What is the use of the zip() function in Python?"
    ]
    python_question_contexts = [
        "To comment a single line in Python, you use the '#' symbol. Anything after the '#' on the same line is considered a comment and is ignored by the Python interpreter.",
        "Python 2 and Python 3 are two different versions of the Python programming language. Python 3 introduced several backward-incompatible changes to the language to improve its design and fix inconsistencies.",
        "To generate a random number in Python, you can use the 'random' module. Import the module and then use functions like 'random.random()' or 'random.randint()' depending on your requirements.",
        "The 'if __name__ == '__main__':' statement is used to determine whether the Python script is being run as the main program or if it is being imported as a module into another program.",
        "To open and read a file in Python, you can use the 'open()' built-in function in combination with various file modes. After reading, close the file using the 'close()' method of the file object.",
        "The 'range()' function generates a sequence of numbers in Python. It can be used in 'for' loops or to create lists of numbers within a specified range.",
        "Decorators in Python are a powerful way to modify or extend the behavior of functions or methods without changing their actual code. They use the '@' symbol and can be used to add functionality like logging, caching, etc.",
        "Exception handling in Python is done using 'try', 'except', 'else', and 'finally' blocks. It allows you to gracefully handle errors and exceptions that may occur during program execution.",
        "Lists and tuples are both used to store collections of items in Python, but lists are mutable (can be changed), whereas tuples are immutable (cannot be changed).",
        "A virtual environment in Python is a self-contained directory that contains its own Python interpreter and installed packages. It allows you to work on different projects with different dependencies without conflicts.",
        "To remove duplicates from a list in Python, you can convert it to a 'set' and then back to a 'list', as sets automatically remove duplicate elements.",
        "In Python lists, 'append()' adds an element to the end of the list, 'extend()' appends elements from another iterable, and 'insert()' inserts an element at a specified index.",
        "To iterate over a dictionary in Python, you can use a 'for' loop, which by default, iterates over the keys. You can also use the 'items()' method to loop over both keys and values.",
        "A lambda function in Python is a small anonymous function defined using the 'lambda' keyword. It can have any number of arguments but can only have one expression.",
        "You can find the length of a string in Python using the built-in 'len()' function, which returns the number of characters in the string.",
        "The 'pass' statement in Python is a null operation; it does nothing. It is used as a placeholder where syntactically some code is required, but you want to skip its execution.",
        "In Python, you create a class using the 'class' keyword, and you define methods (functions within the class) to perform actions or return information about the object.",
        "Shallow copy creates a new object, but it references the same elements as the original object. Deep copy creates a completely independent copy of the object and its elements.",
        "To sort a dictionary by its values in Python, you can use the 'sorted()' function with a lambda function as the 'key' argument or utilize the 'collections' module.",
        "The 'zip()' function in Python is used to combine multiple iterables into a single iterable of tuples. Each tuple contains elements from corresponding positions in the input iterables."
    ]
    # TEMPLATE().get_prompt(10, question="Why is the sky red ?")
    # TEMPLATE(seed=45).get_random_prompt(question="Why is the sky blue ?")

    # prompts = TEMPLATE().get_random_norm_response_batch(answers=python_question_contexts)
    # for prompt in prompts:
    #     print(prompt+"\n")

    # prompt = QA_TEMPLATE().get_trivial_response(id=20,
    #                                             question=python_questions[0],
    #                                             answer=python_question_contexts[0])
    # print(prompt)

    set_seed(575338)
    prompt = QA_TEMPLATE().get_random_prompt(question=python_questions[0],
                                             context=python_question_contexts[0])

    print(prompt)
