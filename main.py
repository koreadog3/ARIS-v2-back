import os
import re
import json
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

# 1. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Koyeb/UptimeRobot 헬스체크용 엔드포인트
@app.get("/")
def health_check():
    return {"status": "ARIS Backend is Running"}

# 2. OpenAI 클라이언트 세팅 (환경변수: OPENAI_API_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# 3. 구글 뉴스 RSS 수집 함수 (공통 레이더)
def fetch_google_news(query: str, limit: int = 5):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        
        news_list = []
        for item in items[:limit]:
            news_list.append({
                "title": item.title.text if item.title else "",
                "link": item.link.text if item.link else "",
                "description": item.description.text if item.description else ""
            })
        return news_list
    except Exception as e:
        print(f"📡 크롤링 에러: {e}")
        return []

# ==========================================
# 🚨 EWS API: GPT가 직접 판단하고 요약하는 지능형 모드
# ==========================================
@app.get("/api/ews")
def get_ews_data():
    ews_news = fetch_google_news("주한 대사관 철수 OR 대피 OR 영사관 폐쇄 OR 여행경보", limit=3)
    
    alert_message = "[대기 중] 현재 감지된 국내 공관 철수 및 특이 동향 없음."
    status_level = "low"
    ai_summary = ""
    ai_source = "gpt-4o-mini" if client else "disabled(no OPENAI_API_KEY)"

    for news in ews_news:
        if client is None:
            break

        # 1차 필터: 키워드가 걸리면 GPT에게 심층 분석 지시
        if re.search(r'(철수|대피|폐쇄|중단|경보)', news['title']):
            prompt = f"""
            당신은 전술 지휘 통제소(ARIS)의 분석관입니다.
            아래 뉴스가 '실제 주한 외국 공관 철수/대피/폐쇄' 또는 '한국에 대한 여행경보 상향'과 관련된 '실제 사건'인지 판단하세요.
            영화, 드라마, 단순 과거 회고, 훈련, 단순 의견이면 "거짓(false)"으로 판단하세요.

            [뉴스 정보]
            제목: {news['title']}
            내용: {news['description']}

            반드시 아래 JSON 형식으로만 대답하세요.
            {{
                "is_real_threat": true/false,
                "summary": "한국어로 2줄 이내의 상황 요약 (진짜 위협일 경우에만 작성, 아니면 빈칸)"
            }}
            """

            try:
                # GPT-4o-mini 출동
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" }, # JSON 형태로만 답변하게 강제
                    temperature=0.0 # 창의성 0 (가장 사실적이고 냉정하게 판단)
                )

                result = json.loads(response.choices[0].message.content)

                # GPT가 "이거 진짜 위협이다!" 라고 판단했을 때만 사이렌 울림
                if result.get("is_real_threat"):
                    alert_message = f"🚨 [EWS 발령] {news['title']}"
                    status_level = "high"
                    ai_summary = result.get("summary", "")
                    break # 가장 먼저 찾은 진짜 위협 1개만 띄우고 종료
                    
            except Exception as e:
                print(f"🧠 GPT API 에러: {e}")

    return {
        "status": status_level,
        "message": alert_message,
        "ai_source": ai_source,
        "ai_summary": ai_summary # 프론트엔드에 요약본도 같이 던져줌
    }


# ==========================================
# 🌍 RISK API: 중복 국가 병합 및 텍스트 정제
# ==========================================
@app.get("/api/risk")
def get_risk_data():
    risk_news = fetch_google_news("전쟁 OR 무력충돌 OR 미사일 OR 공습", limit=8)

    TARGET_COUNTRIES = [
        "미국", "중국", "러시아", "우크라이나", "이스라엘", "이란", "대만", "북한", "한국",
        "일본", "영국", "프랑스", "독일", "시리아", "레바논", "팔레스타인", "예멘", "수단",
        "미얀마", "인도", "파키스탄", "필리핀"
    ]

    grouped_conflicts = {}
    global_score_sum = 0
    total_items = 0

    for news in risk_news:
        title = news["title"]
        link = news["link"]

        # 기사 제목에서 [속보] 접두어와 끝의 언론사 표기를 제거합니다.
        clean_title = re.sub(r"\[.*?\]\s*", "", title)
        clean_title = re.sub(
            r"\s*[-|ㆍ]\s*[가-힣a-zA-Z0-9]+(일보|신문|방송|뉴스|통신|미디어|KBS|SBS|MBC|YTN|JTBC|MBN).*$",
            "",
            clean_title
        ).strip()
        if not clean_title:
            clean_title = title

        mentioned_countries = [country for country in TARGET_COUNTRIES if country in title]

        if len(mentioned_countries) >= 2:
            country_pair = f"{mentioned_countries[0]} | {mentioned_countries[1]}"
        elif len(mentioned_countries) == 1:
            country_pair = f"{mentioned_countries[0]} | 국제 분쟁"
        else:
            country_pair = "국제 분쟁 지역"

        signal_type = "군사"
        if re.search(r"(제재|외교|협상|대사)", title):
            signal_type = "외교"
        elif re.search(r"(경제|유가|환율)", title):
            signal_type = "경제"

        base_score = 7.5 if "미사일" in title or "공습" in title else 6.0

        if country_pair in grouped_conflicts:
            grouped_conflicts[country_pair]["score"] = min(10.0, grouped_conflicts[country_pair]["score"] + 0.5)
            if link and link not in grouped_conflicts[country_pair]["links"]:
                grouped_conflicts[country_pair]["links"].append(link)
        else:
            grouped_conflicts[country_pair] = {
                "countries": country_pair,
                "score": base_score,
                "signal": signal_type,
                "detail": clean_title,
                "links": [link] if link else []
            }

        global_score_sum += base_score
        total_items += 1

    conflicts = list(grouped_conflicts.values())
    conflicts.sort(key=lambda x: x["score"], reverse=True)

    avg_score = round(global_score_sum / total_items, 1) if total_items > 0 else 1.0

    return {
        "global_threat_level": avg_score,
        "conflicts": conflicts
    }
