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
        model = tf.keras.models.load_model("./model/InceptionV3().h5")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

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

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("##")
        st.write(
            """
            Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to
            reduce a lot of manual effort needed in diagnosis.

            Our application detects the following diseases:
            "basal cell carcinoma",
              "benign",
                "melanoma"
            """
        )
        st.write("##")
        st.write(
            "[Learn More >](https://www.researchgate.net/publication/356093241_Characteristics_of_publicly_available_skin_cancer_image_datasets_a_systematic_review)"
        )
    with right_column:
        st_lottie(lottie_health, height=500, key="check")

with st.container():
    st.write("---")
    cols = st.columns(2)
    with cols[0]:
        st.header("How it works?")
        """
        Our application utilizes machine learning to predict what skin disease you may have, from just your skin images!
        We then recommend you specialized doctors based on your type of disease, if our model predicts you're healthy we'll suggest you a general doctor.
        ##
        [Learn More >](https://youtu.be/sFIXmJn3vGk)
        """
    with cols[1]:
        st_lottie(lottie_healthy, height=300, key="healthy")

# Skin Cancer Prediction Section
st.header("Skin Cancer Detection")
pic = st.file_uploader(
    label="Upload a picture",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="Upload a picture of your skin to get a diagnosis",
)

if st.button("Predict"):
    if pic:
        st.header("Results")

        cols = st.columns([1, 2])
        with cols[0]:
            st.image(pic, caption=pic.name, use_column_width=True)

        with cols[1]:
            model = load_model()

            if model:
                with st.spinner("Predicting..."):
                    img = PIL.Image.open(pic)
                    img = img.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, axis=0)

                    prediction = model.predict(img_array)
                    prediction = tf.nn.softmax(prediction)

                    score = tf.reduce_max(prediction)
                    score = tf.round(score * 100, 2)

                    prediction = tf.argmax(prediction, axis=1)
                    prediction = prediction.numpy()
                    prediction = prediction[0]

                    labels = ["Melanoma", "Benign", "Basal Cell Carcinoma"]
                    disease = labels[prediction].title()
                    st.write(f"**Prediction:** `{disease}`")
                    #st.write(f"**Confidence:** `{score:.2f}%`")
            else:
                st.error("Model could not be loaded. Please check the model file.")

        st.warning(
            ":warning: This is not a medical diagnosis. Please consult a doctor for a professional diagnosis."
        )
    else:
        st.error("Please upload an image")
