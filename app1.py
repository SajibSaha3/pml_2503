import streamlit as st
import pickle

model = pickle.load(open("iris_model.pkl","rb"))

st.title("flower prediction system")

sep_len= st.number_input("enter the sepal length")
sep_wid= st.number_input("enter the sepal width")
pe_len= st.number_input("enter the petal length")
pe_wid= st.number_input("enter the petal width")

feature = [sep_len,sep_wid,pe_len,pe_wid]
pred = model.predict([feature])

if st.button("Submit"):
    st.write("predicted flower ", pred)