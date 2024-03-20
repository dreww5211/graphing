import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
#streamlit run d:/coding/graphing/Home.py

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.title("Graphing Calculator")
    
    with col3:
        st.write('')
st.divider()

with st.container():
    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        st.write(' ')

    with col2:    
        img1 = Image.open("normal graphing img.png")
        st.image(img1)    
        button_clicked = st.button("Normal/ Differentiation graphing")
        if button_clicked:
            switch_page("Normal graphing")
    
    with col3:
        st.write('')
    st.divider()

with st.container():
    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        st.write(' ')
    
    with col2:
        img2 = Image.open("integration graphing img.png")
        st.image(img2)
        button_clicked = st.button("Integration graphing")
        if button_clicked:
            switch_page("Integration graphing")

    with col3:
        st.write(' ')
    st.divider()

with st.container():
    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        st.write('')
    
    with col2:
        img3 = Image.open("graph generator with specific points.jpg")
        st.image(img3)
        button_clicked = st.button("Graph generator with specific points")
        if button_clicked:
            switch_page("Graph generator with specific points")
    
    with col3:
        st.write('')
    st.divider()

with st.container():
    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        st.write('')

    with col2:
        img4 = Image.open("matrix intersections.jpg")
        st.image(img4)
        button_clicked = st.button("Matrix intersections")
        if button_clicked:
            switch_page("Matrix intersections")
    
    with col3:
        st.write('')
    st.divider()