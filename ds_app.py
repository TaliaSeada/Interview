import streamlit as st

import inference_app
import validation_app


def main():
    st.set_page_config(page_title="Titanic Model Apps", layout="wide")

    app = st.sidebar.radio(
        "App",
        [
            "Validation report",
            "Inference",
        ],
    )

    if app == "Validation report":
        validation_app.main()
    else:
        inference_app.main()


if __name__ == "__main__":
    main()
