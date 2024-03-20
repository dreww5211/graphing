import numpy as np
from bokeh.plotting import *
from bokeh.models import *
import streamlit as st
from sympy import *
from streamlit_extras.switch_page_button import switch_page
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

session_state = st.session_state

def evaluate_expression(expr, x_vals):
    x = symbols('x')
    # Use lambdify to convert the symbolic expression to a NumPy-compatible function
    expr_func = lambdify(x, expr, 'numpy')
    y_vals = expr_func(x_vals)
    return y_vals

def sympyParser(expre):
    
    exprList = list(expre)
    
    for i,expr in enumerate(exprList):
        if i == 0:
            if expr.isdecimal() == True and exprList[i+1].isalpha() == True:
                exprList[i] = exprList[i] + "*" 
            else:
                pass
        elif exprList[i] == exprList[-1]:
            if exprList[i-1].isdecimal() == True and expr.isalpha() == True:
                exprList[i-1] = exprList[i-1] + "*"
        else:
            if exprList[i-1].isdecimal() == True and expr.isalpha() == True:
                exprList[i-1] = exprList[i-1] + "*"
            elif expr.isdecimal() == True and exprList[i+1].isalpha() == True:
                exprList[i] = exprList[i] + "*"

    for i,expr in enumerate(exprList):
        if i != 0 and expr != exprList[-1]:            
            if expr == "(":
                if exprList[i-1] == 'x' or exprList[i-1].isdecimal() == True or exprList[i-1] == ")":
                    exprList[i-1] = exprList[i-1] + "*"

    for i,expr in enumerate(exprList):
        if i != 0 and i != len(exprList)-1:
            if expr == ")":
                if exprList[i+1] == 'x' or exprList[i+1].isdecimal() == True or exprList[i+1] == "(":
                    exprList[i] = exprList[i] + "*"        
    
    expre = ''.join(str(x) for x in exprList)

    return expre

def numpyParser(expre):
    if 'log' in expre:
        exprList = list(expre)
        for i,char in enumerate(exprList):
            if char == 'g':
                for k in range(i+2, len(exprList)):
                    if exprList[k] == ')':
                        exprList[k] = ", 10" + exprList[k]
                        break
        expre = ''.join(str(x) for x in exprList)
            

    if 'ln' in expre:
        expre = expre.replace('ln', 'log')
    

    return expre

def find_bracket_pairs(expr):
    stack = []
    bracket_pairs = []

    for i, char in enumerate(expr):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                end = i
                bracket_pairs.append([start, end])

    return bracket_pairs

def intercept(expression):
    x, y, z = symbols('x y z')
    expr = sympify(expression)
    yIntercept = (0, expr.subs(x, 0).evalf(3))
    if not yIntercept[1].is_real:
        yIntercept = (0, 0)

    # if 's' in str(expr) or 't' in str(expr) or 'l' in str(expr):
    x_intercepts_list = []
    for i in range(-100, 101):
        try:
            x_intercept = nsolve(Eq(expr, 0), x, i)
            try:
                imaginary_part = x_intercept.as_real_imag()[1]
                if Abs(imaginary_part) < 1e-10:
                    realX_intercept = x_intercept.evalf(3).as_real_imag()[0]
                    x_intercepts_list.append(realX_intercept)
            except:
                realX_intercept = x_intercept.evalf()
                x_intercepts_list.append(realX_intercept)
        except:
            pass
    
    return yIntercept, x_intercepts_list

def graphsIntersection(graphList):
  x = symbols('x')
  eqList = []
  for i in graphList:
    expr = sympify(i)
    eqList.append(expr)
  coorList = []

  for eq1 in eqList:
      for eq2 in eqList:
        if eq2 != eq1:
            eq1_simplified = simplify(eq1)
            eq2_simplified = simplify(eq2)            
            
            try:      
                if 's' in str(eq1) or 't' in str(eq1) or 's' in str(eq2) or 't' in str(eq2) or 'l' in str(eq1) or 'l' in str(eq2):
                    raise ValueError
                solutions_set = solve(Eq(eq2_simplified, eq1_simplified), x, domain=S.Reals - {0})
                try:
                    for sol in solutions_set:
                        imaginary_part = sol.as_real_imag()[1]
                        if Abs(imaginary_part) < 1e-10:
                        # Extract the real part and append it to the list
                            real_part = sol.evalf(3).as_real_imag()[0]
                            coordinate = (real_part, eq2.subs(x, real_part))
                            coorList.append(coordinate)
                except:
                    for sol in solutions_set:
                        numerical_value = sol.evalf()
                        coordinate = (numerical_value, eq2.subs(x, numerical_value))
                        coorList.append(coordinate)
            except ValueError:
                for i in range(-100, 101):
                    try:
                        solution = nsolve(Eq(eq2_simplified, eq1_simplified), x, i)
                        try:
                            imaginary_part = solution.as_real_imag()[1]
                            if Abs(imaginary_part) < 1e-10:
                                numerical_value = solution.evalf(3).as_real_imag()[0]
                                coordinate = (numerical_value, eq2.subs(x, numerical_value))
                                coorList.append(coordinate)
                        except:
                            numerical_value = solution.evalf()
                            coordinate = (numerical_value, eq2.subs(x, numerical_value))
                            coorList.append(coordinate)
                    except:
                        pass
                

  length = len(coorList)

  for i in range(length):
      for j in range(length):      
          if j != i:
              if str(coorList[j]) == str(coorList[i]):
                  coorList[j] = ""

  coorList = [x for x in coorList if x != ""]
  return coorList

def checkAsymptote(expr):
    x = symbols('x')
    bracketList = find_bracket_pairs(expr)
    backFound = False    
    denominators = []

    if 'log' in expr:
        solList = []
        outputList = []
        for i,char in enumerate(expr):
            if char == 'l':
                for j in range(i+4, len(expr)):
                    if expr[j] == ')':
                        solvePt = expr[i+4:j]
                        break
        
        solution = solveset(Eq(sympify(sympyParser(solvePt)), 0), x)
        solList.extend(solution)
        for sol in solList:
            if sol.is_real:
                expr = sympyParser(expr)
                vertical_asymptote = limit(sympify(expr), x, sol, dir='-')
                if vertical_asymptote in [-oo, oo]:
                    outputList.append(sol)
        
        if len(outputList) > 0:
            return sorted(outputList)
        else:
            return None    

        
        


    for i,char in enumerate(expr):
        if char == '/':
            if expr[i+1] == "(":
                for j in bracketList:
                    if j[0] == i+1:
                        denominators.append(expr[(j[0]+1):j[1]])
            elif expr[i+1].isalpha() == True or expr[i+1].isdecimal() == True:
                for o in range(i+2,len(expr)):                    
                    if expr[o].isalpha() != True and expr[o].isdecimal() != True:
                        denominators.append(expr[(i+1):(o)])
                        backFound = True
                        break
                if backFound == False:
                    denominators.append(expr[(i+1):])
    
    solList = []
    outputList = []
    for denominator in denominators:
        denominator = sympyParser(denominator.strip("{}"))
        denominator = sympify(denominator)
        solution = solveset(Eq(denominator, 0), x)
        solList.extend(solution)
    
    for sol in solList:
        if sol.is_real:
            expr = sympyParser(expr)
            vertical_asymptote = limit(sympify(expr), x, sol, dir='-')
            if vertical_asymptote in [-oo, oo]:
                outputList.append(sol) 
            else:
                # If no asymptote is found
                pass

    if len(outputList) > 0:
        return sorted(outputList)
    else:
        return None

def area_between_functions(exprList, interval):
    x = symbols('x')

    if len(exprList) == 2:
        # Find the absolute difference
        absolute_difference = abs(exprList[0] - exprList[1])

        # Integrate the absolute difference over the specified interval
        integral = integrate(exprList[0] - exprList[1], x)
        area = integrate(absolute_difference, (x, interval[0], interval[1]))
    
    if len(exprList) == 1:
        integral = integrate(exprList[0], x)
        area = integrate(abs(exprList[0]), (x, interval[0], interval[1]))

    return integral, area.evalf(5)



with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("## Integration graphing")
    
    with col2:
        st.write('')
    
    with col3:
        st.write('')
st.divider()

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.df = pd.DataFrame(
            [
                {"Colour": '',  """Function
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                """: ''}
            ]
        )
        for i in range(1):
            st.session_state.df.loc[len(st.session_state.df)] = ['','']
        st.session_state.df = st.data_editor(st.session_state.df, num_rows='fixed', hide_index='True')
        
        st.session_state.ef = pd.DataFrame(
            [
                {"Lower interval": '', "Upper interval": ''}
            ]
        )
        st.session_state.ef = st.data_editor(st.session_state.ef, num_rows='fixed', hide_index='True')
    with col2:
        if len(st.session_state.df.iloc[:,1]) != 0:
            with st.container():
                col1,col2 = st.columns([3,1])

                with col2:
                    st.write("Interpreted as")
                    for func in st.session_state.df.iloc[:,1]:
                        if func != None and func != '':
                            st.write(sympify(sympyParser(func)))

if st.button("Generate"):
    st.divider()

    col1, col2 = st.columns([2,8])
    with col1:
        st.write('')
    
    with col2:
        x_vals = np.linspace(-100, 100, 5000)
        p = figure(x_axis_label="x", y_axis_label="y", match_aspect=True)
        p.axis[0].fixed_location = 0
        p.axis[1].fixed_location = 0
        p.y_range = Range1d(-10, 10)
        p.x_range = Range1d(-10, 10)
        p.plot_height=800
        p.plot_width=800
        funcList = []

        for i, row in st.session_state.df.iterrows():
            if row[1] == '':
                pass
            else:
                expr = numpyParser(row[1])
                expr = sympify(sympyParser(expr))

                y_vals = evaluate_expression(expr, x_vals)
                asymptoteVal = checkAsymptote(row[1])
                if asymptoteVal != None:
                    if len(asymptoteVal) == 1: 
                        x1 = np.linspace(-100, (float(asymptoteVal[0])-0.01), 5000)
                        y1 = evaluate_expression(expr, x1)
                        x2 = np.linspace((float(asymptoteVal[0])+0.01), 100, 5000)
                        y2 = evaluate_expression(expr, x2)
                        p.line(x1, y1, line_width=2, color=row[0])
                        p.line(x2, y2, line_width=2, color=row[0])
                    elif len(asymptoteVal) == 2:
                        x1 = np.linspace(-100, (float(asymptoteVal[0])-0.01), 5000)
                        y1 = evaluate_expression(expr, x1)
                        x2 = np.linspace((float(asymptoteVal[0])+0.01), (float(asymptoteVal[1])-0.01), 5000)
                        y2 = evaluate_expression(expr, x2)
                        x3 = np.linspace((float(asymptoteVal[1])+0.01), 100, 5000)
                        y3 = evaluate_expression(expr, x3)
                        p.line(x1, y1, line_width=2, color=row[0])
                        p.line(x2, y2, line_width=2, color=row[0])
                        p.line(x3, y3, line_width=2, color=row[0]) 
                else:
                    p.line(x_vals, y_vals, line_width=2, color=row[0])
                
                yIntercept, x_intercepts_list = intercept(expr)
                if yIntercept != (0,0):
                    p.dot(x=[float(yIntercept[0])], y=[float(yIntercept[1])], size=30, color='black')
                if len(x_intercepts_list)>0:
                    for i in x_intercepts_list:
                        xIntercept = (i.evalf(3),0)
                        if xIntercept != (0,0):
                            p.dot(x=[float(xIntercept[0])], y=[float(xIntercept[1])], size=30, color='black')
                funcList.append(expr)

        if len(funcList) > 1:
            coorList = graphsIntersection(funcList)
            for i in coorList:
                p.dot(x=[float(i[0])], y=[float(i[1])], size=30, color='black')
        
        interval = [int(st.session_state.ef.iat[0,0]), int(st.session_state.ef.iat[0,1])]
            
        integral, area = area_between_functions(funcList, interval)

        if len(funcList) == 2:
            x_vals = np.linspace(interval[0], interval[1], (interval[1]-interval[0])*7)
            for x_val in x_vals:
                y_val1 = evaluate_expression(funcList[0], np.array([x_val]))[0]
                y_val2 = evaluate_expression(funcList[1], np.array([x_val]))[0]
                
                # Draw upward vertical line for negative y-values
                if y_val1 < 0:
                    p.line(x=[x_val, x_val], y=[y_val1, y_val2], line_width=2, color='blue')
                # Draw downward vertical line for positive y-values
                elif y_val1 > 0:
                    p.line(x=[x_val, x_val], y=[y_val1, y_val2], line_width=2, color='blue')
        else:
            x_vals = np.linspace(interval[0], interval[1], (interval[1]-interval[0])*5)
            for x_val in x_vals:
                y_val = evaluate_expression(expr, np.array([x_val]))[0]
                
                # Draw upward vertical line for negative y-values
                if y_val < 0:
                    p.line(x=[x_val, x_val], y=[0, y_val], line_width=2, color='blue')
                # Draw downward vertical line for positive y-values
                elif y_val > 0:
                    p.line(x=[x_val, x_val], y=[0, y_val], line_width=2, color='blue')


        tooltips = [
        ('x,y','@x, @y')
        ]                            
        p.add_tools(HoverTool(tooltips=tooltips))
        st.bokeh_chart(p, use_container_width=True)
    
    with col1:
        st.write("The integral for above:")
        st.write(integral)
        st.write("The shaded area:")
        st.write(nsimplify(area))
        st.write("Or")
        st.write(area)
