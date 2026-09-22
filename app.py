import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import random
import hashlib

st.set_page_config(
    page_title="AI 가위바위보 게임",
    page_icon="✊"
)

st.title("✌️ ✊ 🖐️ AI 가위바위보 게임")
st.write("카메라에 가위, 바위, 보 중 하나를 보여주세요!")
st.write("AI가 손 모양을 인식하고 컴퓨터와 승부합니다.")


# =========================
# 모델 불러오기
# =========================
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(
        "keras_model.h5",
        compile=False
    )


model = load_ai_model()

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = f.readlines()


# =========================
# 점수 저장
# =========================
if "user_score" not in st.session_state:
    st.session_state.user_score = 0

if "computer_score" not in st.session_state:
    st.session_state.computer_score = 0

if "draw_score" not in st.session_state:
    st.session_state.draw_score = 0

if "last_photo" not in st.session_state:
    st.session_state.last_photo = None

if "computer_choice" not in st.session_state:
    st.session_state.computer_choice = None

if "game_result" not in st.session_state:
    st.session_state.game_result = None


# =========================
# 이름 통일
# =========================
def convert_name(name):
    name = name.strip()

    # labels.txt 앞의 0, 1, 2 제거
    if " " in name:
        name = name.split(" ", 1)[1]

    name = name.lower()

    # 가위
    if "가위" in name or "찌" in name or "scissors" in name:
        return "가위"

    # 바위
    elif "바위" in name or "묵" in name or "rock" in name:
        return "바위"

    # 보
    elif "보" in name or "빠" in name or "paper" in name:
        return "보"

    return name


# =========================
# 승패 판정
# =========================
def check_winner(user, computer):

    if user == computer:
        return "무승부"

    if (
        (user == "가위" and computer == "보")
        or (user == "바위" and computer == "가위")
        or (user == "보" and computer == "바위")
    ):
        return "승리"

    return "패배"


# =========================
# 화면에 점수 표시
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("🙂 나", st.session_state.user_score)
col2.metric("🤖 컴퓨터", st.session_state.computer_score)
col3.metric("🤝 무승부", st.session_state.draw_score)

st.divider()


# =========================
# 카메라
# =========================
camera = st.camera_input("📷 손 모양을 촬영하세요")


if camera is not None:

    # 같은 사진이 계속 처리되는 것을 방지
    photo_bytes = camera.getvalue()
    photo_hash = hashlib.md5(photo_bytes).hexdigest()

    image = Image.open(camera).convert("RGB")

    # Teachable Machine 이미지 크기
    image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array

    # AI 예측
    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    detected_name = class_names[index]

    user_choice = convert_name(detected_name)

    # 새로운 사진일 때만 컴퓨터 선택 및 점수 계산
    if st.session_state.last_photo != photo_hash:

        computer_choice = random.choice(
            ["가위", "바위", "보"]
        )

        result = check_winner(
            user_choice,
            computer_choice
        )

        st.session_state.computer_choice = computer_choice
        st.session_state.game_result = result
        st.session_state.last_photo = photo_hash

        if result == "승리":
            st.session_state.user_score += 1

        elif result == "패배":
            st.session_state.computer_score += 1

        else:
            st.session_state.draw_score += 1


    # =========================
    # 결과 출력
    # =========================

    icons = {
        "가위": "✌️",
        "바위": "✊",
        "보": "🖐️"
    }

    st.subheader("🎮 게임 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🙂 나")
        st.write(
            f"# {icons.get(user_choice, '')} {user_choice}"
        )

    with col2:
        st.write("### 🤖 컴퓨터")
        st.write(
            f"# {icons.get(st.session_state.computer_choice, '')} "
            f"{st.session_state.computer_choice}"
        )

    st.write(
        f"AI 인식 정확도: {confidence * 100:.1f}%"
    )

    st.divider()

    if st.session_state.game_result == "승리":
        st.success("🎉 이겼습니다!")

    elif st.session_state.game_result == "패배":
        st.error("😭 졌습니다!")

    else:
        st.warning("🤝 비겼습니다!")


# =========================
# 점수 초기화
# =========================
st.divider()

if st.button("🔄 점수 초기화"):

    st.session_state.user_score = 0
    st.session_state.computer_score = 0
    st.session_state.draw_score = 0
    st.session_state.last_photo = None
    st.session_state.computer_choice = None
    st.session_state.game_result = None

    st.rerun()
