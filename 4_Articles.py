
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import PIL
import tensorflow as tf


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_model():
    try:
        model = tf.keras.applications.InceptionV3(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def predict_skin_cancer(image, model):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0][0][1]
    return predicted_class

st.set_page_config(
    page_title="Skin Cancer",
    page_icon="♋",
    layout="wide",
    initial_sidebar_state="expanded",
)

lottie_health = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json"
)
lottie_welcome = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_puciaact.json"
)
lottie_healthy = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json"
)

st.title("Welcome to team Diagnose!")
st_lottie(lottie_welcome, height=300, key="welcome")
st.text("this is article")

# ... (rest of your introduction code)

# Skin Cancer Prediction Section
# st.header("Skin Cancer Detection")
# pic = st.file_uploader(
#     label="Upload a picture",
#     type=["png", "jpg", "jpeg"],
#     accept_multiple_files=False,
#     help="Upload a picture of your skin to get a diagnosis",
# )

# if st.button("Predict"):
#     if pic:
#         st.header("Results")

#         cols = st.columns([1, 2])
#         with cols[0]:
#             st.image(pic, caption=pic.name, use_column_width=True)

#         with cols[1]:
#             model = load_model()

#             if model:
#                 with st.spinner("Predicting..."):
#                     img = PIL.Image.open(pic)
#                     predicted_class = predict_skin_cancer(img, model)

#                     st.write(f"**Prediction:** `{predicted_class}`")
#             else:
#                 st.error("Model could not be loaded. Please check the model file.")

#         st.warning(
#             ":warning: This is not a medical diagnosis. Please consult a doctor for a professional diagnosis."
#         )
#     else:
#         st.error("Please upload an image")
