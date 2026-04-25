# crew_blog.py
import os
from crewai import Agent, Task, Crew, Process
# from langchain_community.llms import Ollama
from crewai import LLM

print("로컬 AI 모델을 로드하는 중입니다. 잠시만 기다려주세요...")

# 1. Ollama를 통해 로컬에 설치된 LLM 인스턴스 초기화
# ollama_gemma와 ollama_mistral은 각각 gemma와 mistral 모델을 사용하는 AI 인스턴스입니다.
# 이 모델들은 텍스트 생성과 분석에 독립적으로 사용됩니다.
llm_writer = LLM(
    model="ollama/gemma4:e4b"
)

llm_researcher = LLM(
    model="ollama/mistral:7b"
)

# 2. AI 에이전트 정의 (페르소나 부여)
# [에이전트 1: 수석 기술 연구원]
researcher = Agent(
    role='수석 기술 산업 연구원 (Senior Tech Researcher)',
    goal='주어진 기술 주제에 대해 최신 동향을 파악하고 핵심 인사이트를 3가지로 요약한다.',
    backstory='당신은 실리콘밸리 최고 수준의 기술 분석가입니다. 복잡한 데이터를 분석하여 트렌드를 읽어내는 데 탁월한 능력을 갖추고 있으며, 항상 객관적이고 정확한 정보만을 다룹니다.',
    verbose=True,            # 작업 진행 상황을 콘솔에 상세히 출력
    allow_delegation=False,  # 다른 에이전트에게 일을 떠넘기지 않음
    llm=llm_researcher       # 논리 분석에 강한 Mistral 모델 할당
)

# [에이전트 2: 테크 콘텐츠 작가]
writer = Agent(
    role='전문 테크 블로거 및 카피라이터 (Tech Content Strategist)',
    goal='연구원이 제공한 팩트를 바탕으로 대중이 읽기 쉽고 흥미로운 블로그 포스트를 작성한다.',
    backstory='당신은 어려운 기술 용어를 일반 대중도 이해하기 쉽게 풀어쓰는 마법 같은 필력을 가진 유명 블로거입니다. 독자의 흥미를 유발하는 제목과 서론을 작성하는 데 유능합니다.',
    verbose=True,
    allow_delegation=False,
    llm=llm_writer           # 창의적 글쓰기에 강한 Gemma 모델 할당
)

# 3. 작업(Task) 정의
topic = "2026년 자율형 AI(Agentic AI)가 기업 업무 환경에 미치는 영향"

# [작업 1: 데이터 조사 및 요약]
task1 = Task(
    description=f'다음 주제에 대해 심층 분석을 진행하세요: "{topic}". 2026년 최신 동향을 반영하여 가장 중요한 기술적 발전과 비즈니스적 영향 3가지를 명확히 요약하세요.',
    expected_output='주제에 대한 3가지 핵심 인사이트가 담긴 상세한 불릿 포인트(Bullet point) 리스트',
    agent=researcher
)

# [작업 2: 블로그 기사 작성]
task2 = Task(
    description='연구원이 제공한 3가지 핵심 인사이트를 활용하여 매력적인 블로그 기사를 작성하세요. 기사는 매력적인 제목, 서론, 본론(3가지 인사이트 확장), 결론으로 구성되어야 하며 한국어로 작성해야 합니다. 분량은 500자 이상이어야 합니다.',
    expected_output='Markdown 형식으로 작성된 완성된 한국어 블로그 포스트',
    agent=writer
)

# 4. Crew(에이전트 팀) 구성 및 프로세스 실행
print(f"\n[{topic}]에 대한 블로그 작성 파이프라인을 가동합니다...\n")
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    process=Process.sequential  # 순차적 실행 (연구원이 끝난 후 작가가 작업을 시작함)
)

# 파이프라인 실행(Kickoff)
result = crew.kickoff()

print("\n==================================================")
print("✨ 완성된 블로그 글 ✨")
print("==================================================")
print(result)
 