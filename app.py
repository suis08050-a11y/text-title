# app.py

import streamlit as st
from transformers import pipeline # AI 모델 파이프라인 사용

# --- 설정 및 모델 로딩 ---
st.title("🤖 AI 텍스트 제목 생성기 (AI Title Generator)")
st.write("긴 텍스트를 입력하면, AI가 핵심 내용을 요약하여 적절한 제목을 추천해 줍니다.")

# 텍스트 요약 파이프라인 로딩
# bart-large-cnn 모델은 요약(summarization)에 널리 사용됩니다.
# 이 모델은 비교적 크지만 Streamlit 환경에서 사용 가능합니다.
@st.cache_resource
def load_summarizer():
    # 'sshleifer/distilbart-cnn-12-6' 모델은 좀 더 작고 빠릅니다 (선택 가능)
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# --- 사용자 인터페이스 ---
input_text = st.text_area("여기에 분석할 텍스트를 입력하세요:", 
                          height=300, 
                          value="새로운 기술인 AI는 우리의 삶의 다양한 부분에서 혁신을 가져오고 있습니다. 특히 자연어 처리 분야에서는 챗봇, 자동 번역, 그리고 텍스트 생성에 이르기까지 그 영향력이 매우 큽니다. 이러한 기술의 발전은 교육, 금융, 의료 등 여러 산업에서 효율성을 극대화하며 새로운 비즈니스 모델을 창출하고 있습니다. 하지만 데이터 프라이버시 문제와 일자리 감소에 대한 우려 또한 존재하여, 기술 발전과 함께 사회적 논의가 필요합니다.")

if st.button("제목 생성 실행"):
    if len(input_text) < 50:
        st.warning("텍스트가 너무 짧습니다. 50자 이상 입력해주세요.")
    else:
        with st.spinner("AI가 텍스트를 분석하고 제목을 생성 중입니다..."):
            # 요약 파이프라인 실행
            # min_length와 max_length를 조정하여 제목 길이 제한
            summary = summarizer(input_text, 
                                 max_length=30, 
                                 min_length=10, 
                                 do_sample=False)
            
            # 요약 결과를 제목처럼 포맷
            suggested_title = summary[0]['summary_text'].strip()

        st.success("✅ 제목 생성이 완료되었습니다!")
        st.subheader("추천 제목")
        st.markdown(f"**\" {suggested_title} \"**")
        
        st.markdown("---")
        st.write("*(참고: 이 제목은 AI 요약 모델이 생성한 것입니다.)*")
