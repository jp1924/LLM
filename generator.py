import argparse
import os
import warnings
from pathlib import Path

import pandas as pd
from accelerate.utils import release_memory
from setproctitle import setproctitle

from transformers import AutoTokenizer


# Use aphrodite-engine or vLLM
try:
    from aphrodite import LLM, SamplingParams

    print("- Using aphrodite-engine")

except ImportError:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    print("- Using vLLM")

setproctitle("LogicKor Evaluation")


# NOTE: 경로 문제 발생하는게 거슬려서, question.jsonl 파일을 python코드에 직접 추가함. 그래서 코드 하나만 들고 다녀도, 전부 처리할 수 있게 만듬.
# df_questions = pd.read_json(
#     "LogicKor/questions.jsonl", orient="records", encoding="utf-8-sig", lines=True
# )
df_questions = [
    {
        "id": 1,
        "category": "추론(Reasoning)",
        "questions": [
            "각국의 법률에서는 정의라는 개념이 자주 등장하며, 법령의 형성과 해석에 있어 매우 중요한 부분을 차지한다. 하지만 정의란 명확히 규정할 수 없는 개념이기에 해석의 논란이 있을 수 있다. 그렇다면 사회구성원의 대다수가 납득할 수 있는 보편적 정의를 입증하는 방법은 무엇일지 생각해보아라.",
            "위 답변을 영어 문장 3개로 요약한 후. 해당 문장에 있는 단어 3개를 선택하여 단어의 의미를 설명해라.",
        ],
        "references": [None, None],
    },
    {
        "id": 2,
        "category": "추론(Reasoning)",
        "questions": [
            "ABC 기업은 곧 $100의 단가로 신제품을 출시할 예정이며, 연간 예상 수요는 10,000개입니다. 고정비용은 $200,000이고, 제품 단위당 변동 비용은 $60으로 계산됩니다. 최근 시장 조사에 따르면, 경쟁사가 유사한 제품을 $80에 출시할 계획입니다. 이러한 상황에서 ABC 기업이 취할 수 있는 본원적 전략을 기반으로 한 대응 방안을 제안하세요. 특히, 제품 차별화, 비용 절감, 시장 세분화, 혁신 및 연구개발, 전략적 제휴 및 파트너십 중 어느 전략이 가장 효과적일지 분석하고, 그 이유를 설명해 주세요.",
            "지금까지 등장한 모든 숫자를 더한 후, 더한 값의 각 자릿수 숫자들의 합을 구해보세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 3,
        "category": "추론(Reasoning)",
        "questions": [
            "민수에게는 빨간색 셔츠 한 벌과 노란색 셔츠 다섯 벌이 있어. 민수가 노란색 바지 3벌을 새로 구매하고 노란색 바지 두 벌을 친구에게 선물했을 때 민수는 몇 벌의 노란색 셔츠를 가지고 있을까?",
            "민수가 가지고 있는 셔츠들의 색상을 모두 섞으면 무슨 색이 될까? 물감의 경우와 빛의 경우를 모두 설명해줘.",
            "민수가 가지고있는 셔츠들의 색상을 모두 섞으면 무슨 색이 될까? 물감의 경우와, 빛의 경우를 모두 설명해줘.",
        ],
        "references": ["다섯 벌", None],
    },
    {
        "id": 4,
        "category": "추론(Reasoning)",
        "questions": [
            "A, B, C 세 사람 중 한 명이 유리를 깨뜨렸습니다. 경찰이 찾아와 범인을 찾으려 합니다. 세 사람 중 한 명은 거짓말을 하고 나머지 두 명은 진실을 말하고 있습니다. 범인은 누구일까요?\nA: '범인은 C에요.'\nB: '제가 범인이에요.\nC: '저는 범인이 아니에요.''",
            "이런 문제에 대해서 어떻게 생각하나요? 한번 비슷한 문제를 만들고 풀이까지 제시해보세요.",
        ],
        "references": ["B", None],
    },
    {
        "id": 5,
        "category": "추론(Reasoning)",
        "questions": [
            "라면 2개를 끓일 때 물의 양이 550x2=1100이 아니라 더 적게 넣으라고 하는 이유가 뭐야?",
            "물의 증발량에 영향을 미치는 요인들에는 뭐가 있을까?",
        ],
        "references": [
            "물의 증발량은 라면 1개를 끓일 때와 2개를 끓일 때 거의 같다. 따라서, 라면을 2개를 끓일 때는 1100mL보다 적은 양의 물을 사용해도 된다.",
            None,
        ],
    },
    {
        "id": 6,
        "category": "추론(Reasoning)",
        "questions": [
            "민수의 책상 위에는 연필 12자루가 있다. 지아가 연필 한 자루를 빌려 갔고 현수도 그랬다. 민수는 남은 연필의 반절을 가방에 넣었다. 민수의 책상에는 몇 개의 연필이 남았을까?",
            "비슷한 문제를 출제하고 해당 문제의 풀이 방법을 제시해줘.",
        ],
        "references": ["5개", None],
    },
    {
        "id": 7,
        "category": "추론(Reasoning)",
        "questions": [
            "SNS의 등장 이후, SNS를 자아실현의 수단으로써 활용하며 젊은 세대에서 SNS에 자랑하기 위한 소비가 많아지는 경향을 띄고 있다. 이러한 현상이 발생하게 된 부가적 요인들을 생각해보고, 해결 방안을 상세히 제시해보시오.",
            "자본주의적 관점에서 이러한 현상이 문제가 없다는 주장을 논리적으로 전개해 보시오.",
        ],
        "references": [None, None],
    },
    {
        "id": 8,
        "category": "수학(Math)",
        "questions": [
            "한 회사에는 100명의 직원이 있습니다. 회사에서는 각 직원에게 1부터 100까지의 고유한 번호를 부여했습니다. 회사는 100개의 캐비닛을 가지고 있는데, 각 캐비닛 안에는 1부터 100까지의 번호 중 하나가 무작위로 쓰여 있습니다. 각 번호와 캐비닛은 일대일대응을 이룹니다. 회사는 모든 직원이 자기 번호가 할당된 캐비닛을 찾는 것을 목표로 합니다. 각 직원은 최대 50개의 캐비닛을 열어볼 수 있습니다. 한 직원이 캐비닛을 확인한 후에는 다음 직원이 와서 캐비닛을 확인하기 전에 모든 캐비닛을 닫습니다. 모든 직원이 자기 번호가 할당된 캐비닛을 찾을 확률을 최대화하려면 어떤 전략을 사용해야 할까요?",
            "직원의 수가 30명이고 15개의 상자를 열 수 있을 때에도 해당 전략이 유효할까요?",
        ],
        "references": [
            "이 문제는 원순열(순환이 이루어지는 순열)을 이용해서 풀 수 있다. 해당 전략을 이용하면 (1/2)^100의 확률을 약 0.31까지 올릴 수 있다.",
            "유효하다.",
        ],
    },
    {
        "id": 9,
        "category": "수학(Math)",
        "questions": [
            "점 (0, 2), (4, 5)를 동시에 지나는 직선의 방정식을 구하고, 두 점을 지나는 원의 방정식을 하나로 정의하는 데 필요한 조건이 무엇인지를 설명하시오.",
            "직선의 방정식을 x에 대한 식으로 나타내보아라.",
        ],
        "references": [
            "직선의 방정식은 y = 3/4x + 2이며, 원의 방정식을 하나로 정의하기 위해서는 원이 지나는 점 하나를 더 정의하거나, 원의 반지름을 두 점 사이의 거리의 반으로 정의해야 한다. 단, 이때 원이 지나는 점 세 개는 한 직선 위에 있어서는 안 된다. (이 외의 답변도 오류가 없다면 허용)",
            "x = 4/3(y - 2) 또는 x = 4/3y - 8/3",
        ],
    },
    {
        "id": 10,
        "category": "수학(Math)",
        "questions": [
            "다음 두 집합 A, B에 대해 A 교집합 B = {1, 2, 3, 4, 5}이고, A 합집합 B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}일 때, A와 B의 원소의 개수 범위를 구하시오.",
            "집합 C = {2, 6, 3, 1} 일 때 (A 교집합 B) 합집합 C를 구하시오.",
        ],
        "references": [
            "A의 원소의 개수는 5개 이상 10개 이하, B의 원소의 개수는 5개 이상 10개 이하이다.",
            "(A 교집합 B) 합집합 C = {1, 2, 3, 4, 5, 6}. 같은 원소끼리 합할경우 중복하여 쓰지 않는다.",
        ],
    },
    {
        "id": 11,
        "category": "수학(Math)",
        "questions": [
            "한 상자에는 빨간 공이 4개, 파란 공이 6개 들어있습니다. 이 상자에서 공을 무작위로 2개를 꺼낼 때, 두 공의 색깔이 같을 확률은?",
            "세 공의 색깔이 모두 같을 확률은?",
        ],
        "references": ["7/15 또는 46.7%", "1/5 또는 20%"],
    },
    {
        "id": 12,
        "category": "수학(Math)",
        "questions": ["벡터 A(4, 5, 2), B(1,3,2)를 내적해라.", "두 벡터의 코사인 유사도를 구해라."],
        "references": ["23", "답은 0.9163 또는 23 / (3sqrt(70)). 단, 계산 과정 및 개념에 집중하여 평가할 것"],
    },
    {
        "id": 13,
        "category": "수학(Math)",
        "questions": [
            "3의 98제곱의 일의 자리 숫자를 구해보세요.",
            "3의 98제곱은 몇 자리 숫자일까요? (단, 로그 3은 0.5)",
        ],
        "references": ["9", "50자리 숫자"],
    },
    {
        "id": 14,
        "category": "수학(Math)",
        "questions": ["f(x) = 3x^3 + 2x^2 + 58을 미분하시오.", "미분한 식을 한 번 더 미분해보시오."],
        "references": ["9x^2 + 4x", "18x + 4"],
    },
    {
        "id": 15,
        "category": "글쓰기(Writing)",
        "questions": [
            "당신은 라이트노벨을 작성하는 작가입니다. 이번에 새롭게 쓸 작품의 제목은 '히로인이 미래에서 온 시간 여행자'입니다. 제목을 참고하여 작품의 개요를 작성해주고, 너무 뻔한 작품이 되지 않도록 하기 위해서는 어떠한 노력을 할 수 있을지 생각해보세요.",
            "작품의 초반 전개를 흥미진진하게 400자 내외로 작성해보세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 16,
        "category": "글쓰기(Writing)",
        "questions": [
            "당신은 지구를 떠나 우주로 떠난 미래의 우주인입니다. 지구에 대한 그리움을 담아 우주인의 일기를 쓰려고 합니다. 우주인의 일기를 쓰기 위해서는 어떠한 접근 방법이 필요할지 생각해보고, 일기의 내용을 구체적으로 작성해보세요.",
            "우주인의 일기를 쓰는 것이 지구인의 일기를 쓰는 것보다 어려운 점은 무엇인지 설명해보세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 17,
        "category": "글쓰기(Writing)",
        "questions": [
            "온라인 서비스에서 광고를 없애는 '프리미엄 요금제'가 회사 매출에 긍정적 영향을 일으킬 수 있다는 주제로 프리미엄 요금제 도입을 촉구하는 글을 작성해보아라.",
            "프리미엄 요금제를 도입하면서 기존의 무료 서비스를 이용하던 사용자들이 어떠한 반응을 보일 수 있는지 예상해보아라.",
        ],
        "references": [None, None],
    },
    {
        "id": 18,
        "category": "글쓰기(Writing)",
        "questions": [
            "당신은 방금 역사에 남을법한 발견을 한 과학자입니다. 놀란 감정을 담아 동료 과학자에게 발견한 내용을 설명하는 이메일을 작성해보세요. 이메일은 '이보게! 제임스'로 시작해야 합니다.",
            "만약에 이메일이 아닌 편지를 작성해야 하는 상황에서는 어떠한 부분을 다르게 작성할 수 있을지 생각해보고, 다시 작성해보세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 19,
        "category": "글쓰기(Writing)",
        "questions": [
            "새로 오픈한 카페를 홍보하려고 합니다. 홍보 대상, 홍보 전략을 선정하고 대상을 설득할 수 있을 만한 홍보 메시지를 만들어보세요.",
            "다른 홍보 대상을 선정하고, 그 대상에게 맞는 홍보 전략과 메시지를 작성해보세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 20,
        "category": "글쓰기(Writing)",
        "questions": [
            "기자인 당신은 한 게임 A를 수년간 플레이해오며 게임 A에 대한 광범위한 지식을 가지고 있습니다. 최근 출시된 게임 B가 게임 A의 핵심 요소를 모방했다는 논란이 있습니다. 게임 B의 개발사에 인터뷰를 요청하는 과정에서, 게임 B의 개발자들로부터 그들의 의도와 게임 디자인 결정에 대한 깊이 있는 이해를 얻고자 합니다. 이를 바탕으로 게임 B의 개발사에 인터뷰를 요청하는 메일을 작성해보세요.",
            "인터뷰 요청 메일을 받은 게임 B의 개발사는 어떠한 반응을 보일 수 있을지 예상해보세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 21,
        "category": "글쓰기(Writing)",
        "questions": [
            "자본론을 쓴 카를 마르크스가 오늘날의 비트코인 폭등과 같은 경제 현상을 어떻게 바라보았을지 상상해보고, 그의 관점에서 현대 사회의 문제를 분석해라.",
            "카를 마르크스의 관점에서 현대 사회의 문제를 분석하는 것이 어려운 점은 무엇인지 설명해라.",
        ],
        "references": [None, None],
    },
    {
        "id": 22,
        "category": "코딩(Coding)",
        "questions": [
            "시간 복잡도를 어떻게 구할 수 있는지 설명해주고, 많이 쓰이는 알고리즘 중에 최적화를 통해 시간 복잡도를 줄인 예시를 알려줘.",
            "공간 복잡도라는 용어도 있던데 뭐가 다른 거야?",
        ],
        "references": [None, None],
    },
    {
        "id": 23,
        "category": "코딩(Coding)",
        "questions": [
            "'XSS가 무엇이고 어떻게 악용될 수 있는지'를 설명하는 웹사이트의 HTML을 작성해보세요.",
            "XSS를 방지하는 방법은 무엇이 있나요? 이번에는 HTML이 아닌 마크다운 형식으로 작성해주세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 24,
        "category": "코딩(Coding)",
        "questions": [
            "Python에서 멀티프로세스 및 멀티스레드를 구현할 수 있는 기본적인 방법 두 가지를 설명하고, 두 가지 방법의 차이점을 자세히 설명하시오.",
            "multiprocessing을 활용하여 변수 a에 1을 1억 번 더하는 코드를 작성해보시오.",
        ],
        "references": [
            "multiprocessing, threading이 있다. 차이점은 메모리 공유 여부나 GIL 제한 여부 등이 있다.",
            None,
        ],
    },
    {
        "id": 25,
        "category": "코딩(Coding)",
        "questions": [
            "프로그래밍 언어별 특징이 궁금합니다. C++, JavaScript, Go, Rust, Swift 언어별 특징을 간단히 설명해주세요.",
            "각 언어의 특징을 바탕으로, 상황에 따라 어떤 언어를 사용하면 좋을지 예시를 들어 설명해주세요.",
        ],
        "references": [None, None],
    },
    {
        "id": 26,
        "category": "코딩(Coding)",
        "questions": [
            "PyTorch를 사용해서 간단한 CNN을 구현해봐.",
            "ReLU가 뭔지 설명해주고, 비슷한 역할을 하는 다른 함수들을 알려줘.",
        ],
        "references": [None, None],
    },
    {
        "id": 27,
        "category": "코딩(Coding)",
        "questions": [
            "HTTP/1.1 및 2의 주요 메서드 5개와 그들의 역할에 대하여 설명해보시오.",
            "HTTP 메서드 중 PUT과 PATCH의 차이는 무엇인가?",
        ],
        "references": [
            "GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD, TRACE, CONNECT 중 5개를 설명",
            "PUT은 전체를 업데이트하는 반면, PATCH는 일부를 업데이트한다.",
        ],
    },
    {
        "id": 28,
        "category": "코딩(Coding)",
        "questions": [
            "코딩 문제\n주어진 리스트에서 중복되지 않는 첫 번째 문제를 반환하는 함수를 작성해라.\n함수명: find_unique_character\n매개변수: characters (list)\n반환값: 중복되지 않는 첫 번째 문자\n예시:\n입력: ['a', 'b', 'c', 'a', 'd']\n출력: 'b'\n입력: ['a', 'b', 'a', 'b', 'c']\n출력: 'c'\n입력: ['a', 'b', 'c', 'd', 'e']\n출력: 'a'\n언어는 자유롭게 사용 할수 있다.",
            "주어진 문제의 난이도를 높일 방법을 제시해라.",
        ],
        "references": [None, None],
    },
    {
        "id": 29,
        "category": "이해(Understanding)",
        "questions": [
            "정보가 풍부하고 주의력이 짧아진 시대에, 복잡한 문제를 깊이 파고들어 의미 있고 고품질의 작업을 수행하는 능력은 점점 더 가치 있게 되었습니다. 컴퓨터 과학 교수이자 저자인 칼 뉴포트는 이 능력을 '심층 작업'이라고 명명합니다. 뉴포트는 집중력이 분산되지 않는 상태에서 집중적이고 끊임없는 주의를 기울여 개인이 인지 능력의 한계를 밀어붙일 수 있게 하며, 새로운 가치를 창출하고, 기술을 향상하며, 자신이 하는 일을 모방하기 어렵게 만든다고 주장합니다.\n\n심층 작업은 주의가 산만한 상태에서 수행되는 비인지적 요구가 많지 않고, 논리적 성격의 작업인 '얕은 작업'과 대조됩니다. 이러한 작업은 필요하긴 하지만, 세상이나 개인의 개인적 성장, 만족감 또는 경력 발전에 크게 기여하지 않습니다. 뉴포트의 관점에서 심층 작업의 능력은 단지 하나의 기술이 아니라 우리 경제에서의 초능력입니다.\n\n가설에 따르면 기술이 발전함에 따라 대부분의 얕은 작업은 자동화되거나 아웃소싱될 수 있으며, 심층 작업이 성공적인 개인을 구별하는 결정적인 기술로 남게 됩니다. 그러나 이 기술을 익히고 수행하는 것은 그리 쉽지만은 않습니다. 이는 규율, 구조화된 작업 환경 및 사회적 미디어와 이메일과 같은 현대 생활에서 만연한 산만함의 유혹을 저항할 수 있는 능력을 요구합니다.\n\n또한, 뉴포트는 심층 작업이 생산성을 향상시킬 뿐만 아니라, 만족감과 웰빙에도 기여한다고 주장합니다. 심층 작업 상태는 시간이 날아가는 것처럼 느껴지고, 한가지 작업에 완전히 몰두하는 몰입 상태와 유사합니다. 이러한 몰입과 강렬한 집중은 일과 생활에서 더 큰 만족감을 이끌어낼 수 있습니다.\n\n결론적으로, 심층 작업가설은 산만함과 표면적인 참여로 가득 찬 세계에서, 자기 일에 깊이 몰두할 수 있는 능력은 비범한 성취뿐만 아니라 의미 있고 만족스러운 삶으로 가는 길이라고 제시합니다.\n\n지문을 잘 읽고, 심층 작업이 가능하게 하는 것 중 하나가 아닌 것은 무엇인지 골라보세요. 그리고 뉴포트의 심층 작업에 대한 관점을 설명해주세요.\nA) 생산성 향상.\nB) 작업의 자동화 증가.\nC) 만족감과 웰빙의 감각.\nD) 일과 생활에서의 더 큰 만족감.",
            "오늘날의 교육 체계가 학생들이 심층 작업 능력을 개발할 수 있도록 어떻게 지원하고 있는지, 혹은 지원하지 못한다면, 어떤 요소들이 심층 작업 기술의 습득을 방해하고 있는지에 관해 설명해주세요.",
        ],
        "references": ["B", None],
    },
    {
        "id": 30,
        "category": "이해(Understanding)",
        "questions": [
            "농업은 약 12,000년 전, 신석기 혁명이 시작된 이래로 인류 문명의 기반이 되어왔습니다. 이 중대한 시기는 유목 생활을 하는 수렵채집 공동체에서 정착한 농경 사회로의 전환을 나타냈습니다. 농사의 시작과 동물의 가축화는 인류 사회, 경제, 환경에 중대한 변화를 가져왔습니다.\n\n농업의 가장 초기 형태는 비옥한 초승달 지대, 즉 풍부한 토양과 유리한 기후로 유명한 중동 지역에서 기원한 것으로 여겨집니다. 이곳에서 고대 농부들은 밀, 보리, 콩을 재배하기 시작했으며, 염소, 양, 소와 같은 동물을 가축화했습니다. 이에 따라 사람들이 더 이상 음식을 찾아 떠돌아다니지 않아도 되었고, 정착 공동체의 출현으로 이어졌습니다.\n\n농업 실천이 퍼져감에 따라, 다양한 지역에서는 자신들의 지역 환경에 맞게 이러한 방법을 적용했습니다. 예를 들어, 아메리카 대륙에서는 원주민들이 옥수수, 콩, 호박과 같은 작물을 재배하며, 지속 가능한 농업에 대한 초기 이해를 보여주는 쓰리 시스터즈 기술을 사용했습니다. 한편, 동아시아에서는 벼와 기장의 재배가 대규모 인구를 부양하고 복잡한 사회의 발달로 이어졌습니다.\n\n18세기와 19세기의 산업혁명은 기계화된 농업을 도입했습니다. 이 변화는 식량 생산을 대폭 증가시켰지만, 토양 악화와 삼림 파괴와 같은 심각한 환경적 결과를 초래했습니다. 오늘날, 농업은 증가하는 세계 인구를 먹여 살리면서 동시에 환경 발자국을 최소화해야 하는 이중 과제에 직면해 있습니다. 생명공학 혁신, 지속 가능한 농업 실천, 도시 농업이 이러한 도전을 해결하기 위한 최전선에 있습니다.\n\n그럼에도 불구하고, 농업은 사회적 이슈와 깊이 연관되어 있습니다. 농경지의 분배, 자원에 대한 접근, 기후변화의 영향은 중요한 관심사입니다. 세계가 발전함에 따라, 농업의 지속가능성과 변화하는 환경에 적응할 수 있는 능력은 식량 안보와 환경 관리를 보장하는 데 있어 매우 중요할 것입니다.\n\n질문: 산업 혁명이 농업에 미친 영향을 이 글에서는 무엇이라고 이야기하고 있나? 답을 제시하고 해당 답을 도출해낸 근거를 설명해라.\nA) 노동력 부족으로 인해 식량 생산이 감소했다.\nB) 유전자 변형을 통해 작물의 영양 가치를 개선했다.\nC) 기계화된 농업을 도입하여 식량 생산을 증가시켰지만 환경 문제를 야기했다.\nD) 유기농 농법의 광범위한 채택으로 이어졌다.",
            "이 글에 대해서 어떻게 생각해? 이해한 내용을 바탕으로 평가, 비판해줘.",
        ],
        "references": ["C", None],
    },
    {
        "id": 31,
        "category": "이해(Understanding)",
        "questions": [
            "현대 사회는 급속도로 변화하고 있습니다. 기술의 발전, 세계화, 인구 구조 변화, 환경 문제 등 다양한 요인들이 복잡하게 상호작용하며 우리의 일상과 사회 전반에 지대한 영향을 미치고 있습니다.\n\n우선 디지털 혁명은 우리 삶의 모습을 근본적으로 바꾸어 놓았습니다. 인터넷, 스마트폰, 소셜 미디어의 보편화로 인해 정보 습득과 소통 방식이 크게 달라졌으며, 이는 경제, 교육, 문화 등 사회 전 영역에 걸쳐 변화를 가져왔습니다. 또한 인공지능, 사물인터넷, 빅데이터 등의 기술은 산업 구조와 고용 시장의 판도를 바꾸어 놓고 있습니다. 이러한 변화는 개인의 삶의 질을 향상하고 사회의 효율성을 높이는 긍정적 측면이 있지만, 디지털 격차, 개인정보 침해, 일자리 감소 등의 부작용도 야기하고 있습니다.\n\n세계화의 진전도 현대 사회의 주요한 특징 중 하나입니다. 교통과 통신의 발달로 인해 국가 간 교류가 활발해졌으며, 이는 경제, 문화, 정치 등 여러 방면에서 상호의존성을 높였습니다. 다국적 기업의 성장, 국제 무역의 확대, 문화 교류의 활성화 등은 세계화가 가져온 긍정적 변화입니다. 하지만 동시에 국가 간 경쟁 심화, 문화적 획일화, 불평등 심화 등의 문제점도 대두되고 있습니다.\n\n인구 구조의 변화 또한 현대 사회가 직면한 중요한 도전 중 하나입니다. 선진국을 중심으로 고령화가 빠르게 진행되고 있으며, 이는 노동력 감소, 사회 복지 비용 증가, 세대 간 갈등 등의 문제를 야기하고 있습니다. 또한 저출산 문제는 인구 감소와 경제 성장 둔화로 이어질 수 있어 사회적 대응이 필요한 상황입니다.\n\n환경 문제는 인류가 직면한 가장 심각한 도전 중 하나로 떠오르고 있습니다. 기후 변화, 자원 고갈, 생물 다양성 감소, 환경 오염 등은 지속 가능한 발전을 위협하는 요인들입니다. 이에 대한 국제 사회의 공조와 각국 정부의 정책적 노력이 요구되는 한편, 개인 차원에서의 실천도 중요성이 커지고 있습니다.\n\n이 외에도 양극화 심화, 다문화 사회로의 전환, 4차 산업혁명의 진전 등 현대 사회를 특징짓는 다양한 현상들이 존재합니다. 이러한 변화의 소용돌이 속에서 우리는 사회 구성원 모두의 삶의 질 향상과 지속 가능한 발전이라는 목표를 향해 나아가야 할 것입니다. 이를 위해서는 사회 각 주체 간 소통과 협력, 장기적 관점에서의 전략 수립, 그리고 개인의 책임감 있는 행동이 필요할 것입니다.\n\nQ: 이 지문에서 '장기적 관점에서의 전략 수립이 요구된다.'라는 내용을 담은 문단은 몇 번째 문단인지 알려주고, 글의 내용을 요약해보세요.",
            "환경 문제 해결을 위해 필요한 것으로 볼 수 없는 것은 무엇이고, 나머지 선택지들이 환경 문제 해결에 어떻게 기여 할 수 있는지 설명해보세요.\n(A) 국제 사회의 공조\n(B) 각국 정부의 정책적 노력\n(C) 개인 차원에서의 실천\n(D) 경제 성장 우선주의",
        ],
        "references": ["6번째 문단(마지막 문단)", "(D) 경제 성장 우선주의"],
    },
    {
        "id": 32,
        "category": "이해(Understanding)",
        "questions": [
            "아래 3개의 지문을 읽고, 각 지문에 관한 질문에 답해보세요.\na) 경제:\n한국은행이 2022년 5월 26일 발표한 '경제전망보고서'에 따르면, 한국 경제는 2022년에 2.7% 성장할 것으로 예상됩니다. 이는 지난해 11월 전망치인 3.0%에서 0.3%p 하향 조정된 수치입니다. 성장 전망 하향 조정의 주된 요인으로는 러시아-우크라이나 전쟁에 따른 글로벌 공급망 차질, 원자재 가격 상승, 그리고 중국의 경기 둔화 등이 꼽혔습니다. 한편, 소비자물가 상승률은 4.5%로 전망되어 기존 전망치 2.3%를 크게 상회할 것으로\n보입니다. 이는 국제 유가와 원자재 가격 상승, 공급망 병목에 따른 비용 상승 압력이 소비자물가로 전가되고 있기 때문입니다.\nb) 사회:\n우리나라의 1인 가구 비율이 빠르게 증가하고 있습니다. 통계청이 2021년 9월 29일 발표한 '인구주택총조사'에 따르면 2020년 기준 1인 가구 비율은 31.7%로 나타났습니다. 이는 2015년의 27.2%에 비해 4.5%p 상승한 수치로, 5년 사이에 큰 폭으로 증가하였음을 보여줍니다. 1인 가구 증가의 주된 요인으로는 고령화, 만혼 및 비혼 인구 증가, 그리고 젊은 층의 독립 선호 경향 등이 지목되고 있습니다. 1인 가구의 증가는 주택, 소비,\n복지 등 다양한 분야에 영향을 미치고 있어 이에 대한 정책적 대응이 필요한 상황입니다.\nc) 인문:\n20세기의 대표적인 철학자인 장 폴 사르트르는 실존주의 철학을 통해 개인의 주체성과 자유에 주목했습니다. 그의 1943년 저서 '존재와 무(L'Être et le néant)'에서 사르트르는 인간을 '자유로 선고된 존재'로 규정하였습니다. 이는 인간이 어떤 본질이나 목적에 의해 규정되는 것이 아니라, 스스로 선택하고 행동함으로써 자신의 존재를 만들어간다는 의미입니다. 사르트르는 또한 '타인은 지옥이다'라는 유명한 문구를 통해, 타인의\n시선으로 인해 개인의 자유가 위협받는 상황을 지적하였습니다. 그의 사상은 전후 실존주의 운동에 지대한 영향을 끼쳤으며, 개인의 주체성과 책임을 강조하는 철학으로 평가받고 있습니다. 한편, 사르트르는 문학에서도 왕성한 활동을 펼쳤는데, 1945년 발표된 연극 '출구 없는 방(Huis clos)'은 실존주의 철학을 극적으로 구현한 작품으로 유명합니다. 이 작품에서 그는 죽은 후에도 자유를 추구하는 인물들을 통해 실존의 문제를 날카롭게\n파헤칩니다.\n이제 아래 질문에 대한 응답을 각각 작성해보세요.\n'a)' 지문에서의 2022년 한국 경제 성장률 전망치는 얼마인가요?\n'b)' 지문에서 언급된 1인 가구 비율은 2020년 기준 얼마인가요?\n'c)' 지문에서 사르트르가 인간을 규정한 표현은 무엇인가요?",
            "세 가지 지문 사이에 공통점이 있나요?",
        ],
        "references": ["2.7%, 31.7%, '자유로 선고된 존재", None],
    },
    {
        "id": 33,
        "category": "이해(Understanding)",
        "questions": [
            "내가 당신을 사랑하는 것은 까닭이 없는 것은 아닙니다.\n다른 사람들은 나의 홍안만을 사랑하지만은 당신은 나의 백발도 사랑하는 까닭입니다.\n\n내가 당신을 사랑하는 것은 까닭이 없는 것은 아닙니다.\n다른 사람들은 나의 미소만을 사랑하지만은 당신은 나의 눈물도 사랑하는 까닭입니다.\n\n내가 당신을 사랑하는 것은 까닭이 없는 것은 아닙니다.\n다른 사람들은 나의 건강만을 사랑하지만은 당신은 나의 죽음도 사랑하는 까닭입니다.\n\n위 글의 화자는 '당신'을 사랑하는 까닭이 있는가?",
            "위에서 주어진 글과 같은 형식으로 4연을 자연스럽게 작성해보아라.",
        ],
        "references": ["있다.", None],
    },
    {
        "id": 34,
        "category": "이해(Understanding)",
        "questions": [
            "너는 '-요체'(예시 - 안녕하세요. 저는 AI에요.)를 사용해서만 답변할 수 있어.\n질문: 물리학이 뭐야?",
            "방금 대답한 내용을 요약해줘.",
        ],
        "references": [None, None],
    },
    {
        "id": 35,
        "category": "이해(Understanding)",
        "questions": [
            "좋아하는 요리를 소개하는 글을 다섯 문단으로 작성해줘. 단, 요리의 특징, 맛, 레시피를 꼭 포함해야 해.",
            "작성한 글의 핵심 정보를 추출해서 JSON 형식으로 출력해봐.",
        ],
        "references": [None, None],
    },
    {
        "id": 36,
        "category": "문법(Grammar)",
        "questions": [
            "한글 맞춤법을 고려할 때 다음 중 사이시옷 표기가 잘못된 것은? (정답, 두 개)\n\n숫자, 툇간, 횟수, 갯수, 촛점",
            "위에서 제시한 다섯 개 단어를 가나다순으로 정렬해봐.",
        ],
        "references": ["개수, 초점", "갯수, 숫자, 촛점, 툇간, 횟수"],
    },
    {
        "id": 37,
        "category": "문법(Grammar)",
        "questions": [
            "한국어에서 사과라는 단어는 여러 의미를 지니고 있어. 대표적인 의미 두 가지를 말해봐.",
            "동음이의어와 다의어의 차이가 뭐야?",
        ],
        "references": [
            "아래 뜻 중 두 가지 이상 언급했으면 정답으로 인정.\n1. 자기의 잘못을 인정하고 용서를 빎. 2. (타인의) 잘못을 용서함. 3. 사과나무의 열매.",
            "다의어는 하나의 단어가 여러 가지 의미가 있는 것이고, 동음이의어는 표기가 같으면서 소리가 같고 뜻이 다른 단어를 뜻한다.",
        ],
    },
    {
        "id": 38,
        "category": "문법(Grammar)",
        "questions": [
            "두 문장 사이에 가장 알맞은 접속어를 고르고, 이유를 설명하시오.\n\n이 세상에는 아름다운 것들이 참으로 많다고들 한다. 나는 그런 아름다움을 모른다.\n\n1. 그래서\n2. 그러나\n3. 그리고\n4. 따라서",
            "'그러나'와 비슷한 역할을 하는 접속어는 어떤 게 있는지 설명하시오.",
        ],
        "references": ["2. 그러나", None],
    },
    {
        "id": 39,
        "category": "문법(Grammar)",
        "questions": [
            "문장성분의 호응을 고려할 때 적절한 조사로 바꿔봐.\n\n나는 빠른 자동차을 보았다.",
            "위 문장을 안은문장과 안긴문장으로 나눠봐.",
        ],
        "references": ["나는 빠른 자동차를 보았다.", "안은문장 - 나는 보았다.\n안긴문장 - 자동차"],
    },
    {
        "id": 40,
        "category": "문법(Grammar)",
        "questions": [
            "제26항 한자어에서, 'ㄹ' 받침 뒤에 연결되는 'ㄷ, ㅅ, ㅈ'은 된소리로 발음한다.\n제28항 표기상으로는 사이시옷이 없더라도, 관형격 기능을 지니는 사이시옷이 있어야 할(휴지가 성립되는) 합성어의 경우에는, 뒤 단어의 첫소리 'ㄱ, ㄷ, ㅂ, ㅅ, ㅈ'을 된소리로 발음한다.\n위을 참고할 때 다음 문장 중 넷과 다른 하나는?\n- [길가]에 개나리가 만개했다.에서 '길가'\n- 너희들이 그 모양이니 [발전]이 없는 거야. 에서 '발전'\n- [발바닥]에 땀이 나도록 뛰었다. 에서 '발바닥'\n- [초승달]이 뜬 저녁, 매화가 흐트러졌다.  에서 '초승달'\n- 민수는 [손재주]가 좋아 무엇이든 잘 만든다. 에서 '손재주'",
            "한국어 어휘에서 한자어, 고유어 중에서 무엇이 더 많이 존재하나?",
        ],
        "references": ["너희들이 그 모양이니 [발전]이 없는 거야. 에서 '발전'", "한자어"],
    },
    {
        "id": 41,
        "category": "문법(Grammar)",
        "questions": [
            "한국어에서는 비음이 아닌 ㄱ, ㄷ, ㅂ이 비음의 영향으로 ㅇ, ㄴ, ㅁ 으로 발음되는 현상이 있다. 이때 '쫓는'의 발음은 무엇인가?",
            "이러한 현상을 비음화라고 한다. 그렇다면 칼날이 [칼랄]로 발음되는 현상의 이름은 무엇일까?",
        ],
        "references": ["쫀는", "유음화"],
    },
    {
        "id": 42,
        "category": "문법(Grammar)",
        "questions": [
            "나는어제친구와김치찌개를먹었다.\n\n이 문장을 올바르게 띄어 써보아라.",
            "아래 문장의 높임 표현을 올바르게 수정보아라.\n\n할머니가 밥을 먹는다.",
        ],
        "references": ["나는 어제 친구와 김치찌개를 먹었다.", "할머니께서 진지를 잡수신다."],
    },
]
df_questions = pd.DataFrame(df_questions)

PROMPT_STRATEGY = {
    "cot-1-shot": [
        {
            "role": "system",
            "content": "문제를 해결하는 과정을 단계별로 명확하게 설명한 다음, 최종 답변을 제시해 주세요.",
        },
        {
            "role": "user",
            "content": "한 도시에서 매년 나무를 심고 있습니다. 첫해에 100그루를 심었고, 매년 전년도에 심은 나무의 수보다 20%씩 더 많은 나무를 심고 있습니다. 2년 후에는 총 몇 그루의 나무를 심게 될까요?",
        },
        {
            "role": "assistant",
            "content": """이 문제를 해결하기 위해 단계별로 접근해 보겠습니다.

1단계: 주어진 정보 정리
- 첫해에 100그루를 심음
- 매년 전년도보다 20% 더 많은 나무를 심음
- 2년 후의 총 나무 수를 구해야 함

2단계: 두 번째 해에 심은 나무 수 계산
- 첫해보다 20% 증가
- 100 + (100의 20%) = 100 + 20 = 120그루

3단계: 세 번째 해에 심은 나무 수 계산
- 두 번째 해보다 20% 증가
- 120 + (120의 20%) = 120 + 24 = 144그루

4단계: 3년 동안 심은 총 나무 수 계산
- 첫해: 100그루
- 둘째 해: 120그루
- 셋째 해: 144그루
- 총합: 100 + 120 + 144 = 364그루

따라서, 2년 후(즉, 3년째)에는 총 364그루의 나무를 심게 됩니다.""",
        },
    ],
    "1-shot": [
        {
            "role": "user",
            "content": "Git의 rebase와 merge의 차이점을 설명하고, 각각 어떤 상황에서 사용하는 것이 좋은지 예를 들어 설명해주세요.",
        },
        {
            "role": "assistant",
            "content": """Git에서 `rebase`와 `merge`는 두 개의 브랜치를 통합하는 데 사용되는 두 가지 주요 명령어입니다. 이들의 차이점과 사용 상황을 설명하겠습니다.

### Rebase

#### 개념
- `rebase`는 하나의 브랜치의 커밋을 다른 브랜치의 끝에 다시 적용하는 것입니다. 즉, 커밋 히스토리를 새로 만든다고 할 수 있습니다.
- 결과적으로 커밋 히스토리가 깔끔하게 정리됩니다.

#### 사용 상황
- **깔끔한 히스토리가 필요할 때**: 브랜치 히스토리가 병합 없이 일직선으로 이어지도록 하고 싶을 때 유용합니다.
- **작업 중인 브랜치를 최신 상태로 유지하고 싶을 때**: 예를 들어, `feature` 브랜치에서 작업 중인데 `main` 브랜치에서 업데이트가 있을 경우, `feature` 브랜치를 `rebase`하여 최신 상태로 유지할 수 있습니다.

#### 예제
1. `feature` 브랜치에서 작업 중입니다.
2. `main` 브랜치에 새로운 커밋이 추가되었습니다.
3. `feature` 브랜치에서 `git rebase main`을 실행합니다.
4. `feature` 브랜치의 커밋들이 `main` 브랜치의 최신 커밋들 뒤에 재배치됩니다.

```bash
git checkout feature
git rebase main
```

### Merge

#### 개념
- `merge`는 두 개의 브랜치를 합치는 방법으로, 두 브랜치의 히스토리를 유지하며 새로운 병합 커밋을 생성합니다.
- `merge`는 기존의 히스토리를 그대로 보존합니다.

#### 사용 상황
- **히스토리를 유지하고 싶을 때**: 각 브랜치의 작업 기록을 그대로 보존하면서 병합할 때 사용합니다.
- **협업 중 충돌을 명확히 해결하고 기록을 남기고 싶을 때**: 여러 사람이 같은 저장소에서 작업할 때, `merge`는 충돌 해결 과정과 그 기록을 명확히 남길 수 있습니다.

#### 예제
1. `feature` 브랜치에서 작업을 완료했습니다.
2. `main` 브랜치에 병합하고 싶습니다.
3. `main` 브랜치로 체크아웃한 후 `feature` 브랜치를 병합합니다.

```bash
git checkout main
git merge feature
```

### 요약

- **Rebase**는 히스토리를 깔끔하게 정리하는 데 유리하며, 주로 개인 작업이나 `feature` 브랜치를 최신 상태로 유지할 때 사용됩니다.
- **Merge**는 두 브랜치의 작업 히스토리를 보존하면서 병합하며, 협업 과정에서 충돌 해결과 기록을 명확히 남기는 데 유리합니다.

각 방법의 장단점과 사용 상황을 고려하여 적절히 선택하는 것이 중요합니다.""",
        },
    ],
    "default": [],
}

DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if not include_last_assistant is defined %}{% set include_last_assistant = true %}{% endif %}{% for message in messages %}{{ sot_token }}{% if message.role == 'user' %}{{ '### User:\\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\\n\\n' }}{% elif message.role == 'system' %}{{ '### System:\\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\\n\\n' }}{% elif message.role == 'assistant' %}{{ '### Assistant:\\n' }}{% if not loop.last or include_last_assistant %}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{% endif %}{% else %}{# Do nothing #}{% endif %}{{ eot_token }}{% endfor %}{% if not include_last_assistant %}{# Do nothing #}{% elif not add_generation_prompt %}{{ eos_token }}{% elif add_generation_prompt %}{{ '### Assistant:\\n' }}{% else %}{# Do nothing #}{% endif %}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_devices", help=" : CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument(
        "-m",
        "--model",
        help=" : Model to evaluate",
        default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0",
    )
    parser.add_argument("-ml", "--model_len", help=" : Maximum Model Length", default=4096, type=int)
    parser.add_argument("-eot", "--eot_token", type=str, help="End of turn token", default="")
    parser.add_argument("-sot", "--sot_token", type=str, help="Start of turn token", default="")
    parser.add_argument("-gpu_utils", "--gpu_memory_utilization", type=float, help="", default=0.8)
    args = parser.parse_args()

    print(f"Args - {args}")

    # vllm                              0.6.4.post1
    # vllm-flash-attn                   2.6.1
    # torch                             2.5.1+cu121
    # HACK: 환경에서 tp_size를 2 이상으로 설정하면, torch단에서 애러가 발생하면서 프로세스가 죽는다.
    #       관련해 강제로 1로 설정하게 해 놨으나, 계속 지켜봐야할 문제.
    if args.gpu_devices != "0":
        raise ValueError("Only one GPU is allowed for evaluation.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    gpu_counts = len(args.gpu_devices.split(","))

    args.model = Path(args.model)

    return args, gpu_counts


def main(args, gpu_counts) -> None:
    model_dir_ls = [path for path in args.model.glob("*") if path.is_dir()]
    model_dir_ls = [args.model] if not model_dir_ls else model_dir_ls

    # NOTE: model eval시 평가 결과는, 각 checkpoint 폴더 아래쪽에 generated 폴더를 만들어 저장된다.
    #       checkpoint폴더 들이 담겨져 있는 폴더의 경로를 건내도 되고, 아님 checkpoint 폴더의 경로를 건내도 된다.
    #       단, checkpoint 폴더 안쪽에, generated 폴더가 있음, eval을 pass함.
    # output_dir
    #  ┣ checkpoint-1
    #  ┃ ┣ generated
    #  ┃ ┃ ┣ 1-shot.jsonl
    #  ┃ ┃ ┣ cot-1-shot.jsonl
    #  ┃ ┃ ┗ default.jsonl
    #  ┃ ┗ model_files...
    #  ┣ checkpoint-2
    #  ┃ ┣ ...
    #  ┣ checkpoint-3
    #  ┃ ┣ ...

    for model_dir_path in model_dir_ls:
        save_dir = model_dir_path.joinpath("generated")
        if save_dir.exists() and list(save_dir.glob("*.jsonl")):
            print(f"Skipping {model_dir_path.as_posix()}")
            continue

        # 실재, 모델이 저장되어 있는 폴더인지 아닌지 확인.
        try:
            # AutoConfig.from_pretrained(model_dir_path.as_posix())
            AutoTokenizer.from_pretrained(model_dir_path.as_posix())
        except BaseException as e:
            print(f"Skipping {model_dir_path.as_posix()}")
            continue

        save_dir.mkdir(exist_ok=True)

        # HACK: GPU device 확인해 볼 것
        llm = LLM(
            model=model_dir_path.as_posix(),
            tensor_parallel_size=1,  # gpu_counts,
            max_model_len=args.model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,  # !
        )

        chat_template = llm.llm_engine.tokenizer.tokenizer.chat_template
        bos_token = llm.llm_engine.tokenizer.tokenizer.bos_token
        print("-" * 100)
        print(f"현재 사용하고 있는 chat_template: {chat_template}")
        print(f"현재 사용하고 있는 bos_token: {bos_token}")
        print(f"현재 사용하고 있는 eot_token: {args.eot_token}")
        print(f"현재 사용하고 있는 sot_token: {args.sot_token}")

        if "add_generation_prompt" not in chat_template:
            raise ValueError("chat_template에 add_generation_prompt가 없습니다.")
        if "sot_token" not in chat_template:
            warnings.warn(
                "chat_template에 sot_token이 없습니다. apply_chat_template시 예상대로 동작하지 않을 수 있습니다."
            )
        elif "eot_token" in chat_template and args.eot_token == "":
            warnings.warn(
                "chat_template에 eot_token이 있으나, args.eot_token이 비어있습니다. apply_chat_template시 예상대로 동작하지 않을 수 있습니다."
            )
        if "eot_token" not in chat_template:
            warnings.warn(
                "chat_template에 eot_token이 없습니다. apply_chat_template시 예상대로 동작하지 않을 수 있습니다."
            )
        elif "sot_token" in chat_template and args.sot_token == "":
            warnings.warn(
                "chat_template에 sot_token이 있으나, args.sot_token이 비어있습니다. apply_chat_template시 예상대로 동작하지 않을 수 있습니다."
            )

        sampling_params = SamplingParams(
            temperature=0,
            skip_special_tokens=True,
            max_tokens=args.model_len,
            stop=[
                "<|endoftext|>",
                "[INST]",
                "[/INST]",
                "<|im_end|>",
                "<|end|>",
                "<|eot_id|>",
                "<end_of_turn>",
                "<eos>",
            ],
        )

        for strategy_name, prompts in PROMPT_STRATEGY.items():

            def format_single_turn_question(question):
                text = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                    prompts
                    + [
                        {"role": "user", "content": question[0]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                if not text.startswith(bos_token):
                    text = bos_token + text

                return text

            def format_double_turn_question(question, single_turn_output):
                text = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                    prompts
                    + [
                        {"role": "user", "content": question[0]},
                        {"role": "assistant", "content": single_turn_output},
                        {"role": "user", "content": question[1]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not text.startswith(bos_token):
                    text = bos_token + text

                return text

            single_turn_questions = df_questions["questions"].map(format_single_turn_question)
            single_turn_outputs = [
                output.outputs[0].text.strip() for output in llm.generate(single_turn_questions, sampling_params)
            ]

            multi_turn_questions = df_questions[["questions", "id"]].apply(
                lambda x: format_double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]),
                axis=1,
            )
            multi_turn_outputs = [
                output.outputs[0].text.strip() for output in llm.generate(multi_turn_questions, sampling_params)
            ]

            df_output = pd.DataFrame(
                {
                    "id": df_questions["id"],
                    "category": df_questions["category"],
                    "questions": df_questions["questions"],
                    "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
                    "references": df_questions["references"],
                }
            )

            df_output.to_json(
                save_dir.joinpath(f"{strategy_name}.jsonl").as_posix(),
                orient="records",
                lines=True,
                force_ascii=False,
            )

        destroy_model_parallel()
        release_memory(llm)
        # TODO: 나중에 tp_size가 2 이상 가능할 때, 이 부분이 예상치 못한 애러를 발생시킬 수 있음. 테스트 해 봐야 함.
        destroy_distributed_environment()

        del llm


if "__main__" in __name__:
    args, gpu_counts = get_args()
    main(args, gpu_counts)
