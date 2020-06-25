#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


import streamlit as st
import pandas as pd
import numpy as np

model = pickle.load(open('model.pkl', 'rb')) 

#def predict(arg):
 #   int_features = [int(x) for x in request.form.values()] 
  #  final_features = [np.array(int_features)]
   # prediction = model.predict(final_features)
    #return (prediction) 

def predict(input_df):
    predictions_df = model.predict(input_df)
    predictions = predictions_df[0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('attrition_image.png')
    image_attrition = Image.open('attrition_narrow.jpg')

    st.image(image,use_column_width=True)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict attrition rate of an organization')

    st.sidebar.image(image_attrition)

    st.title("Attrition Rate Prediction App")

    if add_selectbox == 'Online':
        
        growth_rate = st.number_input('Employee rating of growth', min_value=1, max_value=100, value=25)
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        Time_of_service = st.number_input('years of working in this org', min_value=0, max_value=50, value=25)
        Education_Level_2 = st.selectbox('Are you of Education_level_2', [1,0])
        Unit_Sales = st.selectbox("Are you from dept. sales ", [1,0])
        VAR2 = st.number_input('select your var2 value', min_value=-2, max_value=2)
        Compensation_and_Benefits_type4 = st.selectbox('Are you getting type4 compensation & benefits', [1,0])
        VAR7_2 = st.selectbox('is your var7 type 2?', [1,0])
        Pay_Scale_4 = st.selectbox('is your pay scale type 4?', [1,0])
        Unit_Security = st.selectbox('Are you from dept. security', [1,0])
        Unit_Human_Resource_Management = st.selectbox('Are you from dept. HR', [1,0])
        VAR6_6 = st.selectbox('is your var6 type 6', [1,0])
        VAR1_5 = st.selectbox('is your var1 type 5', [1,0])
        Pay_Scale_9= st.selectbox('is your pay scale type 9?', [1,0])
        Unit_RandD = st.selectbox('Are you from dept. R&D', [1,0])
        Post_Level_2 = st.selectbox('is your post level -2?', [1,0])
        Unit_Purchasing = st.selectbox('Are you from dept. purchasing', [1,0])
        Work_Life_balance_2 = st.selectbox('will you rate ur work life balance as 2/10', [1,0])
        VAR3_07 = st.selectbox('is your var3 type 0.7075?', [1,0])
        Time_since_promotion_1= st.selectbox('is it 1 year since ur promotion', [1,0])
        "are you a male ? select box if yes, dont select if not"
        if st.checkbox('Male'):
            Gender_M = 1
        else:
            Gender_M = 0   
        
            
        Compensation_and_Benefits_type1= st.selectbox('Are you getting type1 compensation & benefits', [1,0])
        Pay_Scale_2= st.selectbox('is your pay scale type 2?', [1,0])
    
        output=""
        
        input_dict = { 'growth_rate': growth_rate,'Age' : age,'Time_of_service' : Time_of_service,'Education_Level_2 ': Education_Level_2,
                        'Unit_Sales' : Unit_Sales,'VAR2 ': VAR2,'Compensation_and_Benefits_type4' : Compensation_and_Benefits_type4,'VAR7_2' : VAR7_2,
                        'Pay_Scale_4.0 ': Pay_Scale_4,'Unit_Security' : Unit_Security,'Unit_Human Resource Management' : Unit_Human_Resource_Management,
                        'VAR6_6' : VAR6_6,'VAR1_5' : VAR1_5,'Pay_Scale_9.0 ': Pay_Scale_9,'Unit_RandD' : Unit_RandD,'Post_Level_2' : Post_Level_2,
                        'Unit_Purchasing' : Unit_Purchasing,'Work_Life_balance_2 ': Work_Life_balance_2,'VAR3_0.7075' : VAR3_07,
                        'Time_since_promotion_1' : Time_since_promotion_1,'Gender_M ': Gender_M,
                        'Compensation_and_Benefits_type1 ': Compensation_and_Benefits_type1, 'Pay_Scale_2' : Pay_Scale_2 }
        
   
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(input_df)
            #output = predict(model=model, input_df=input_df)
            

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            prediction = predict(data)
            #prediction = predict_model(estimator=model,data=data)
            st.write(prediction)

if __name__ == '__main__':
    
    run() 
    


# In[ ]:





# In[ ]:





# In[ ]:




