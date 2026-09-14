import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="가위바위보 AI",
    page_icon="✊"
)

st.title("✌️ ✊ 🖐️ 가위바위보 AI")
st.write("카메라로 가위, 바위, 보를 보여주세요!")

# 모델 불러오기
model = tf.keras.models.load_model(
    "keras_model.h5",
    compile=False
)

# 라벨 불러오기
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = f.readlines()

camera = st.camera_input("사진 촬영")

if camera is not None:

    image = Image.open(camera).convert("RGB")

    size = (224, 224)

    image = ImageOps.fit(
        image,
        size,
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

    prediction = model.predict(data)

    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    # Teachable Machine labels.txt의 숫자 제거
    class_name = class_names[index].strip()

    if " " in class_name:
        class_name = class_name.split(" ", 1)[1]

    st.subheader("AI 판정")

    if "가위" in class_name:
        st.success("✌️ 가위!")
    elif "바위" in class_name:
        st.success("✊ 바위!")
    elif "보" in class_name:
        st.success("🖐️ 보!")
    else:
        st.success(class_name)

    st.write(
        f"정확도: {confidence_score * 100:.1f}%"
    )