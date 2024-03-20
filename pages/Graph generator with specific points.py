import numpy as np
from bokeh.plotting import *
from bokeh.models import *
import streamlit as st
from sympy import *
from streamlit_extras.switch_page_button import switch_page
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

session_state = st.session_state

def generateGraph(coorList):
    x, y, a, b, c, d = symbols('x y a b c d')
    if len(coorList) == 2:
        equations = [Eq(a*x + b, y) for x, y in coorList]
        solution = linsolve(equations, a, b)
        solList = list(solution)
        try:
            expr = (solList[0][0])*x + solList[0][1]
        except:
            return "No graph can cross all these points."
    if len(coorList) == 3:
        equations = [Eq(a*x**2 + b*x + c, y) for x, y in coorList]
        solution = linsolve(equations, a, b, c)
        solList = list(solution)
        try:
            expr = (solList[0][0])*x**2 + (solList[0][1])*x + solList[0][2]
        except:
            return "No graph can cross all these points."
    if len(coorList) == 4:
        equations = [Eq(a*x**3 + b*x**2 + c*x + d, y) for x, y in coorList]
        solution = linsolve(equations, a, b, c, d)
        solList = list(solution)
        try:
            expr = (solList[0][0])*x**3 + (solList[0][1])*x**2 + (solList[0][2])*x + solList[0][3]
        except:
            return "No graph can cross all these points."
    if len(coorList) == 1:
        return "There is infinite solutions to pass only one point."

    
    return expr

def evaluate_expression(expr, x_vals):
    x = symbols('x')
    # Use lambdify to convert the symbolic expression to a NumPy-compatible function
    expr_func = lambdify(x, expr, 'numpy')
    y_vals = expr_func(x_vals)
    return y_vals


with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("## Graph generator with specific points")
    
    with col2:
        st.write('')
    
    with col3:
        st.write("Enter coordinates of the points you wanna choose to generate a graph that go through all of them!")
st.divider()

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.df = pd.DataFrame(
            [
                {"x-value": '', "y-value": ''}
            ]
        )
        
        for i in range(3):
            st.session_state.df.loc[len(st.session_state.df)] = ['','']
        st.session_state.df = st.session_state.df.style.hide()
        st.session_state.df = st.data_editor(st.session_state.df, num_rows='fixed', hide_index='True')
        

    with col2:
        st.write('')


if st.button("Generate"):
    st.divider()

    with st.container():
        col1, col2, col3 = st.columns([1,8,1])

        with col1:
            st.write(' ')

        with col2:
            x_vals = np.linspace(-100, 100, 5000)
            p = figure(x_axis_label="x", y_axis_label="y", match_aspect=True)
            p.axis[0].fixed_location = 0
            p.axis[1].fixed_location = 0
            p.y_range = Range1d(-10, 10)
            p.x_range = Range1d(-10, 10)
            p.plot_height=800
            p.plot_width=800
            coorList = []
            
            for i, row in st.session_state.df.iterrows():
                try:
                    if row[0] == '' and row[1] == '':
                        pass
                    else:
                        try:
                            coor = (int(row[0]), int(row[1]))
                        except:
                            coor = (float(row[0]), float(row[1]))
                        coorList.append(coor)
                except:
                    raise ValueError
            print(generateGraph(coorList))
            expr = generateGraph(coorList)
            if expr == "No graph can cross all these points.":
                st.write("# No graph can cross all these points.")
            else:
                expr = sympify(expr)
                y_vals = evaluate_expression(expr, x_vals)
                p.line(x_vals, y_vals, line_width=2)

                for i in coorList:
                    p.dot(x=[i[0]], y=[i[1]], size=30, color='black')
                
                
                
                
                tooltips = [
                ('x,y','@x, @y')
            ]
                p.add_tools(HoverTool(tooltips=tooltips))

                st.write(expr)
                st.bokeh_chart(p, use_container_width=True)
        
        with col3:
            st.write('')
