import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import random
import hashlib


# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="AI 가위바위보 게임",
    page_icon="✊"
)

st.title("✌️ ✊ 🖐️ AI 가위바위보 게임")
st.write("카메라에 가위, 바위, 보 중 하나를 보여주세요!")
st.write("AI가 손 모양을 인식한 뒤 컴퓨터와 가위바위보 대결을 합니다.")


# =========================
# AI 모델 불러오기
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "keras_model.h5",
        compile=False
    )


model = load_model()


# =========================
# labels.txt 불러오기
# =========================
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = f.readlines()


# =========================
# 점수 초기 설정
# =========================
if "win" not in st.session_state:
    st.session_state.win = 0

if "lose" not in st.session_state:
    st.session_state.lose = 0

if "draw" not in st.session_state:
    st.session_state.draw = 0

if "last_photo" not in st.session_state:
    st.session_state.last_photo = None

if "user_choice" not in st.session_state:
    st.session_state.user_choice = None

if "computer_choice" not in st.session_state:
    st.session_state.computer_choice = None

if "result" not in st.session_state:
    st.session_state.result = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0


# =========================
# 라벨 이름 변환
# =========================
def convert_name(name):

    name = name.strip()

    # labels.txt가
    # 0 가위
    # 1 바위
    # 2 보
    # 형태일 경우 앞 숫자 제거
    parts = name.split(" ", 1)

    if len(parts) == 2 and parts[0].isdigit():
        name = parts[1]

    name = name.lower()

    if "가위" in name or "찌" in name or "scissors" in name:
        return "가위"

    elif "바위" in name or "묵" in name or "rock" in name:
        return "바위"

    elif "보" in name or "빠" in name or "paper" in name:
        return "보"

    return name


# =========================
# 승패 판단
# =========================
def check_result(user, computer):

    if user == computer:
        return "무승부"

    if user == "가위" and computer == "보":
        return "승리"

    if user == "바위" and computer == "가위":
        return "승리"

    if user == "보" and computer == "바위":
        return "승리"

    return "패배"


# =========================
# 현재 점수
# =========================
st.subheader("🏆 현재 전적")

col1, col2, col3 = st.columns(3)

col1.metric(
    "🎉 승리",
    st.session_state.win
)

col2.metric(
    "😭 패배",
    st.session_state.lose
)

col3.metric(
    "🤝 무승부",
    st.session_state.draw
)

st.divider()


# =========================
# 카메라
# =========================
camera = st.camera_input(
    "📷 가위, 바위, 보 중 하나를 촬영하세요"
)


if camera is not None:

    # 촬영한 이미지 데이터
    photo_bytes = camera.getvalue()

    # 같은 사진으로 점수가 계속 올라가는 것 방지
    photo_hash = hashlib.md5(
        photo_bytes
    ).hexdigest()

    # 새로운 사진일 때만 게임 실행
    if st.session_state.last_photo != photo_hash:

        image = Image.open(camera).convert("RGB")

        image = ImageOps.fit(
            image,
            (224, 224),
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(image)

        normalized_image = (
            image_array.astype(np.float32) / 127.5
        ) - 1

        data = np.ndarray(
            shape=(1, 224, 224, 3),
            dtype=np.float32
        )

        data[0] = normalized_image

        # =========================
        # AI 예측
        # =========================
        prediction = model.predict(
            data,
            verbose=0
        )

        index = np.argmax(prediction)

        confidence = float(
            prediction[0][index]
        )

        detected_name = class_names[index]

        user_choice = convert_name(
            detected_name
        )

        # 컴퓨터 랜덤 선택
        computer_choice = random.choice(
            ["가위", "바위", "보"]
        )

        # 승패 확인
        result = check_result(
            user_choice,
            computer_choice
        )

        # 결과 저장
        st.session_state.user_choice = user_choice
        st.session_state.computer_choice = computer_choice
        st.session_state.result = result
        st.session_state.confidence = confidence
        st.session_state.last_photo = photo_hash

        # =========================
        # 점수 증가
        # =========================
        if result == "승리":
            st.session_state.win += 1

        elif result == "패배":
            st.session_state.lose += 1

        elif result == "무승부":
            st.session_state.draw += 1

        # 점수 화면 즉시 갱신
        st.rerun()


# =========================
# 게임 결과 표시
# =========================
if st.session_state.result is not None:

    st.divider()

    st.subheader("🎮 게임 결과")

    icons = {
        "가위": "✌️",
        "바위": "✊",
        "보": "🖐️"
    }

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🙂 나")

        user_icon = icons.get(
            st.session_state.user_choice,
            ""
        )

        st.markdown(
            f"# {user_icon} "
            f"{st.session_state.user_choice}"
        )

    with col2:

        st.markdown("### 🤖 컴퓨터")

        computer_icon = icons.get(
            st.session_state.computer_choice,
            ""
        )

        st.markdown(
            f"# {computer_icon} "
            f"{st.session_state.computer_choice}"
        )


    st.write(
        f"AI 인식 정확도: "
        f"{st.session_state.confidence * 100:.1f}%"
    )

    st.divider()


    # =========================
    # 결과 메시지
    # =========================
    if st.session_state.result == "승리":

        st.success(
            "🎉 이겼습니다!"
        )

    elif st.session_state.result == "패배":

        st.error(
            "😭 졌습니다!"
        )

    else:

        st.warning(
            "🤝 비겼습니다!"
        )


# =========================
# 점수 초기화
# =========================
st.divider()

if st.button(
    "🔄 전적 초기화",
    use_container_width=True
):

    st.session_state.win = 0
    st.session_state.lose = 0
    st.session_state.draw = 0

    st.session_state.last_photo = None
    st.session_state.user_choice = None
    st.session_state.computer_choice = None
    st.session_state.result = None
    st.session_state.confidence = 0

    st.rerun()
