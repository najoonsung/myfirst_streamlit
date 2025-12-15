import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import koreanize_matplotlib
from konlpy.tag import Okt
import re
from collections import Counter
import plotly.express as px
import networkx as nx
from itertools import combinations
from wordcloud import WordCloud


st.set_page_config(
    page_title="K팝 데몬 헌터스 분석",      # 페이지 Tab의 타이틀
    page_icon="📊",                      # 페이지 Tab의 아이콘
    layout="wide",                       # 페이지 레이아웃: centered, wide
    # 사이드바 초기 상태: auto, collapsed, expanded
    initial_sidebar_state="expanded",
)
# 사이드바 설정
st.write("C035318_이준성_데이터 시각화 3차")
st.sidebar.header("데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    # 명사 전처리
    # description 컬럼의 데이터를 리스트로 변환
    descriptions = df["description"].fillna("").astype(str).tolist()

    # Okt 객체 생성
    okt = Okt()

    # 불용어 사전 불러오기
    with open(
        "/Users/ijunseong/Downloads/40eea22ef4a89f629abd87eed535ac6a-4f7a635040442a995568270ac8156448f2d1f0cb/stopwords-ko.txt",
        "r",
        encoding="utf-8",
    ) as f:
        stopwords = f.read().splitlines()

    # 불용어 사전 추가
    stopwords.extend(
        ["케이팝","K팝","아이돌","가수","그룹","멤버", "넷플릭스", "애니메이션"
    "음악","앨범","공연","무대","활동","컴백", "케이팝데몬헌터스","케이팝 데몬 헌터스","KPopDemonHunters", "케데헌","데몬","헌터스", "데헌", "애니메이션", "흥행", "글로벌","세계","영화","스케","팬덤","문화","최근","확장","미국","대한","민국","대한민국","콘텐츠",
    "가장", "역시", "한편", "가치","감독","개발","개설","개척","경제","인기","중심","공식","작품","운영","기반","열풍","관계자","올해","형성","시작","예술","시청"]
    )

    # 1) 명사 추출 + 불용어 제거
    all_nouns = []
    for i, text in enumerate(descriptions):
        # 정제: 한글과 공백을 제외한 모든 문자 제거
        text_cleaned = re.sub(r"[^가-힣\s]", "", text)

        # 형태소 분리 후 명사만 추출
        nouns = okt.nouns(text_cleaned)

        # 한 글자 단어 및 불용어 제거
        nouns = [word for word in set(nouns) if (len(word) > 1) and (word not in stopwords)]

        # 전처리된 단어목록을 all_nouns에 추가
        all_nouns.append(nouns)

    # 공통 슬라이더 위젯 추가
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        TOPN = st.slider("Top N", 10, 60, 20, 5)
    with c2:
        window_days = st.slider("최근 기간(일)", 3, 30, 7, 1)
    with c3:
        min_doc = st.slider("최소 등장 문서수(희귀 제거)", 1, 20, 3, 1)

    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
    # 최근 떠오르는 키워드 확인을 위해 기간 나누기 -> 지피티 질문
    cut = df["pubDate"].max() - pd.Timedelta(days=window_days)
    recent_mask = df["pubDate"] >= cut
    title = []
    for t in df["title"].fillna("").astype(str).tolist():
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"[^가-힣\s]", " ", t)
        nouns = okt.nouns(t)
        nouns = [w for w in set(nouns) if len(w) > 1 and w not in stopwords]
        title.append(nouns) # 제목별 명사 추출
    desc = []
    for t in df["description"].fillna("").astype(str).tolist():
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"[^가-힣\s]", " ", t)
        nouns = okt.nouns(t)
        nouns = [w for w in set(nouns) if len(w) > 1 and w not in stopwords]
        desc.append(nouns) # 내용별 명사 추출
    # A) Seaborn 제목 Top 키워드
    st.markdown("### Seaborn : 제목에서 가장 많이 반복되는 키워드")
    title_doc_counter = Counter()
    for toks in title:
        for w in toks:
            title_doc_counter[w] += 1

    title_top = (
        pd.DataFrame(title_doc_counter.items(), columns=["keyword", "docs"])
        .query("docs >= @min_doc")
        .sort_values("docs", ascending=False)
        .head(TOPN)
    )

    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    sns.barplot(data=title_top, x="docs", y="keyword")
    plt.title("제목에서 가장 많이 나오는 키워드")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("최근 일주일간 팝업이 가장 제목에서 많이 나오는 키워드로 선정되었다.")


    # B) Altair  최근 vs 이전 키워드 상승
  
    st.markdown("### Altair : 최근 vs 이전: 급상승 키워드")

    recent_counter = Counter()
    past_counter = Counter() # 카운터 사용 -> 사용법 지피티 질문

    for i, toks in enumerate(desc):
        if recent_mask.iloc[i]:
            for w in toks:
                recent_counter[w] += 1
        else:
            for w in toks:
                past_counter[w] += 1

    # 상승점수: 최근비율 - 과거비율(문서수로 정규화) -> 지피티 질문
    n_recent = max(int(recent_mask.sum()), 1)
    n_past   = max(int((~recent_mask).sum()), 1)

    rows = []
    all_words = set(recent_counter.keys()) | set(past_counter.keys())
    for w in all_words:
        r = recent_counter.get(w, 0)
        p = past_counter.get(w, 0)
        if (r + p) < min_doc:
            continue
        score = (r / n_recent) - (p / n_past)
        rows.append((w, r, p, score))

    rise = pd.DataFrame(rows, columns=["keyword", "recent_docs", "past_docs", "rise_score"])
    rise = rise.sort_values("rise_score", ascending=False).head(TOPN)

    chart = alt.Chart(rise).mark_bar().encode(
        y=alt.Y("keyword:N", sort="-x", title=None),
        x=alt.X("rise_score:Q", title="Rise score (recent rate - past rate)"),
        tooltip=["keyword", "recent_docs", "past_docs", "rise_score"]
    ).properties(height=360)

    st.altair_chart(chart, use_container_width=True)

    st.write("해석 : 일주일동안에 급증한 키워드는 ‘서현, 빌리브로 이전 기간 대비 약 30프로 증가했다.")

    # C) Plotly 1개: 제목 vs 본문 편향 키워드

    st.markdown("### Plotly : 제목 & 본문 편향 키워드")

    desc_doc_counter = Counter() # 본문에 얼마나 자주 나왔는지 
    for toks in desc:
        for w in toks:
            desc_doc_counter[w] += 1

    N = len(df)
    bias_rows = []
    common = set(title_doc_counter.keys()) | set(desc_doc_counter.keys()) # 지피티 질문 ->제목에 나오거나 또는 본문에 나온 적이 있는 모든 키워드의 합집합

    for w in common:
        t = title_doc_counter.get(w, 0) # 제목에 많이 나온 단어
        d = desc_doc_counter.get(w, 0) # 본문에 많이 나온 단어
        if (t + d) < min_doc:
            continue
        # 제목비율 - 본문비율 (양수면 제목에서 과다하게 나옴)
        bias = (t / max(N, 1)) - (d / max(N, 1))
        bias_rows.append((w, t, d, bias, t + d))

    bias_df = pd.DataFrame(bias_rows, columns=["keyword", "title_docs", "desc_docs", "bias", "total"])
    bias_df = bias_df.sort_values("bias", ascending=False).head(40)

    fig = px.scatter(
        bias_df,
        x="desc_docs",
        y="title_docs",
        size="total",
        color="bias",
        hover_name="keyword",
        title="Headline Bias: 제목에 더 자주(+) / 본문에 더 자주(-) 등장",
        labels={"desc_docs": "Docs with keyword in description", "title_docs": "Docs with keyword in title"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("해석 : 제목 편향이 큰 키워드는 '팝업'과 '그래미'로 기사 제목에서 이슈화/프레이밍하기 위해 사용하는 단어의 후보로 볼 수 있다.")

    st.divider()

    from matplotlib import font_manager

    han_font_path = font_manager.findfont('AppleGothic')
    st.subheader("워드클라우드 (키워드 빈도 기반)")

    wc_source = st.radio("워드클라우드 기준", ["본문(description) 명사", "제목(title) 명사"], horizontal=True)

    if wc_source == "본문(description) 명사":
        tokens_for_wc = [w for doc in desc for w in doc]
    else:
        tokens_for_wc = [w for doc in title for w in doc]

    freq = Counter(tokens_for_wc)

    if len(freq) == 0:
        st.warning("워드클라우드를 만들 키워드가 없습니다. 전처리/불용어를 확인하세요.")
    else:
        top_wc = st.slider("워드클라우드에 포함할 상위 키워드 수", 50, 500, 200, 50)
        wc = WordCloud(
            font_path=han_font_path,
            width=1200,
            height=600,
            background_color="white",
            max_words=top_wc
        ).generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("WordCloud")
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

    st.divider()
    
    st.subheader("네트워크 분석")   
    min_count_net = 10 # 최소 노드 설정


    # edge 만들기
    edge_list = []
    for nouns in desc:
        if nouns and len(nouns) > 1:
            edge_list.extend(combinations(sorted(set(nouns)), 2))

    edge_counts = Counter(edge_list)
    filtered_edges = {edge: w for edge, w in edge_counts.items() if w >= min_count_net}

    if len(filtered_edges) == 0:
        st.warning("min_count=10에서 남는 엣지가 없습니다. (데이터가 적으면 5로 낮추세요.)")
        st.stop()

    # 그래프 생성
    G = nx.Graph()
    G.add_weighted_edges_from([(a, b, w) for (a, b), w in filtered_edges.items()])

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        st.warning("그래프가 비어있습니다.")
        st.stop()

    # 키워드 선택
    keyword = st.selectbox("부분 네트워크 중심 키워드 선택", sorted(G.nodes()))

    nbrs = list(G.neighbors(keyword))
    if len(nbrs) == 0:
        st.info(f"'{keyword}'는 연결(이웃)이 없습니다. 다른 키워드를 선택하세요.")
        st.stop() 

    # 부분 그래프
    H = G.subgraph([keyword] + nbrs).copy() # 연결 이웃 betweenness 기준 -> 지피티 질문

    # 시각화
    pos = nx.spring_layout(H, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    nx.draw_networkx(
        H, pos,
        ax=ax,
        with_labels=True,
        node_size=[2200 if n == keyword else 700 for n in H.nodes()],
        node_color=["orange" if n == keyword else "skyblue" for n in H.nodes()],
        width=[max(1, H[u][v].get("weight", 1) * 0.03) for u, v in H.edges()],
        edge_color="gray",
        font_family=plt.rcParams["font.family"],
        font_size=10,
        alpha=0.9,
    )

    ax.set_title(f"부분 네트워크: '{keyword}' 중심 (min_count=10)")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
