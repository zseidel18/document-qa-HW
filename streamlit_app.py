import streamlit as st

hw1_page = st.Page('HW/HW1.py', title = 'Homework 1', default=False)
hw2_page = st.Page('HW/HW2.py', title = 'Homework 2', default=False)
hw3_page = st.Page('HW/HW3.py', title = 'Homework 3', default=False)
hw4_page = st.Page('HW/HW4.py', title = 'Homework 4', default=True)



pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page])
st.set_page_config(page_title= 'HW Manager')
pg.run()