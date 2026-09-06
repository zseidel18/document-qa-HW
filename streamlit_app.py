import streamlit as st

hw1_page = st.Page('HW/HW1.py', title = 'Homework 1', default=True)
hw2_page = st.Page('HW/HW2.py', title = 'Homework 2', default=False)

pg = st.navigation([hw1_page, hw2_page])
st.set_page_config(page_title= 'HW Manager')
pg.run()