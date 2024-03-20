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
                for k in range(i+2, len(exprList)): # loop from the operand of log until the end of list
                    if exprList[k] == ')':
                        exprList[k] = ", 10" + exprList[k] # modify to make the expression valid for Numpy
                        break
        expre = ''.join(str(x) for x in exprList)
            

    if 'ln' in expre:
        expre = expre.replace('ln', 'log') # modify to make the expression valid for Numpy
    

    return expre


# def latexParser(expre):
#     exprList = list(expre)
#     bracketList = find_bracket_pairs(expre)
  
    

#     for i,expr in enumerate(exprList):
#         if expr == "^" and exprList[i+1] == "(":
#             exprList[i+1] = "{"
#             for k in range(i+2, len(exprList)):
#                 if exprList[k] == ")":
#                     exprList[k] = "}"
#                     break
    
#     for i,expr in enumerate(exprList):
#         frontFound = False
#         backFound = False
        
#         if expr == "/": # detect '\'
#             exprList[i] = '' # remove '\'  
#             # front part of the fraction
#             if exprList[i-1] == ")" or exprList[i-1] == "}": # detect the letter before '/'
#                 exprList[i-1] = "}" # change it to '}'

#                 for j in bracketList:
#                     if j[1] == i-1: # match the another bracket of the bracket pair
#                         if exprList[j[0]-1].isalpha() == True and exprList[j[0]-1] != 'x':
#                             for k in range(i-2, -1, -1):
#                                 if exprList[k] == 'l':
#                                     exprList[k] = "\\frac{" + exprList[k]
#                                     exprList[j[0]] = ''
#                                     break
#                         else:    
#                             exprList[j[0]] = "\\frac{" # change it to '\\frac{'
            
            
#             elif exprList[i-1].isalpha() == True or exprList[i-1].isdecimal() == True:
#                 exprList[i-1] = exprList[i-1] + "}"
#                 for k in range(i-2, -1, -1):
#                     if exprList[k].isalpha() != True and exprList[k].isdecimal() != True:
#                         exprList[k+1] = "\\frac{" + exprList[k+1]
#                         frontFound = True
#                         break
#                 if frontFound == False:
#                     exprList[0] = "\\frac{" + exprList[0]
            
#             # back part of the fraction
#             if exprList[i+1] == "(":
#                 exprList[i+1] = "{"
#                 for l in bracketList:
#                     if l[0] == i+1:
#                         exprList[l[1]] = "}"
#             elif exprList[i+1].isalpha() == True or exprList[i+1].isdecimal() == True:
#                 exprList[i+1] = "{" + exprList[i+1]
#                 for o in range(i+2,len(exprList)):                    
#                     if exprList[o].isalpha() != True and exprList[o].isdecimal() != True:
#                         exprList[o-1] = exprList[o-1] + "}"
#                         backFound = True
#                         break
#                 if backFound == False:
#                     exprList[-1] = exprList[-1] + "}"
            
#             joinedStr = ''.join(str(x) for x in exprList)
#             newList = list(joinedStr)
#             counter = 0
#             for j,expr in enumerate(newList):
#                 if j != 0:
#                     if expr == "\\" and newList[j-1] == "}":
#                         newList[j] = "{" + expr
#                         for i in range(j+6, len(newList)):
#                             if newList[i] == "}":
#                                 counter = counter + 1
#                             if counter == 2:
#                                 newList[i] = newList[i] + "}"
#                                 break

#     joinedStr = ''.join(str(x) for x in exprList)
#     newList = list(joinedStr)
#     return ''.join(str(x) for x in newList)


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

def diffCalc(expression, tanX):
    x, y, z = symbols('x y z')
    val = float(tanX)
    expr = sympify(expression) #make expr eligible for sympy to differentiate
    derivative = diff(expr, x)
    m = derivative.subs(x,val) #sub values into derivative to get gradient 
    y = expr.subs(x, val) #sub values into og expr to get y
    const = solveset(Eq(m*val + z, y), z) #solve y=mx+c to get c
    solList = list(const) #convert the solution of c from dictionary to list 
    c = solList[0]
    g = np.linspace(val - 1, val + 1, 5000)
    tanFunction = m*x + c
    gradient = m.evalf(3)
    return g, tanFunction, gradient, derivative

def intercept(expression, x_range=(-100, 100)):
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
    # else:             
    #     # Use a numerical solver to find x-intercepts within the specified range
    #     x_intercepts_set = solveset(Eq(expr, 0), x, domain=Interval(x_range[0], x_range[1]))

    #     # Filter x-intercepts to ensure they are real
    #     x_intercepts_list = [x_intercept.evalf() for x_intercept in x_intercepts_set if x_intercept.is_real]

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

def turningPoint(expr):
    x,y,z = symbols('x y z')
    stationary = solve(diff(expr, x), x)

    coorList = []
    try:
        for sol in stationary:
            imaginary_part = sol.as_real_imag()[1]
            if Abs(imaginary_part) < 1e-10:
            # Extract the real part and append it to the list
                real_part = sol.evalf(3).as_real_imag()[0]
                coordinate = (real_part, expr.subs(x, real_part))
                coorList.append(coordinate)
    except:
        for sol in stationary:
            numerical_value = sol.evalf()
            coordinate = (numerical_value, expr.subs(x, numerical_value))
            coorList.append(coordinate)
    return coorList

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("## Normal/Differentiation Graphing")
    
    with col2:
        st.write('')
    
    with col3:
        st.write("Enter a function without $y = $ and its colour to graph it, you can choose to plot a tangent if not just leave it blank. Use brackets for best results as it is easier for the program to parse. Quartic functions aren't stable, so there may be bugs in the display. Exponential functions like $e^x$, $2^x$ are not supported.")
st.divider()


col1, col2 = st.columns(2)
with col1:
    st.session_state.df = pd.DataFrame(
        [
            {"Colour": '', "Tangent (x-value)": '', """Function
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             """: ''}
        ]
    )
    st.session_state.df = st.data_editor(st.session_state.df, num_rows="dynamic")

with col2:
    if len(st.session_state.df.iloc[:,2]) != 0:
        with st.container():
            col1,col2 = st.columns([3,1])

            with col2:
                st.write("Interpreted as")
                for func in st.session_state.df.iloc[:,2]:
                    if func != None and func != '':
                        if 'ln' in func:
                            st.write(str(sympify(sympyParser(func))).replace('log', 'ln'))
                        else:
                            st.write(sympify(sympyParser(func)))


if st.button("Update"):
    st.divider()
    
    coll1, coll2 = st.columns([2,8])
    with coll1:
        try:
            for i, row in st.session_state.df.iterrows():
                st.write(f":{row[0]}[{row[0].upper()}:]")
                if 'ln' in func:
                    st.write(str(sympify(sympyParser(func))).replace('log', 'ln'))
                else:
                    st.write(sympify(sympyParser(row[2])))
                if row[1] != None and row[1] != '':
                    expr = sympify(sympyParser(row[2]))
                    x, tanFunction, gradient, derivative = diffCalc(expr, row[1])
                    st.write(f":{row[0]}[The gradient at $x$={row[1]}: ${gradient}$]")
        except:
            pass

    with coll2: 
        try:
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
                expr = numpyParser(row[2])
                expr = sympify(sympyParser(expr))
                if row[1] != '' and row[1] != None:
                    x, tanFunction, gradient, derivative = diffCalc(expr, row[1])
                    y_vals = evaluate_expression(expr, x_vals)
                    asymptoteVal = checkAsymptote(row[2])
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
                    tan_vals = evaluate_expression(tanFunction, x)
                    p.line(x, tan_vals, line_width=2, color=row[0], line_dash='dashed')
                    
                else:
                    y_vals = evaluate_expression(expr, x_vals)
                    asymptoteVal = checkAsymptote(row[2])
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
                
                turningPtList = turningPoint(expr)
                for pt in turningPtList:
                    if pt != (0,0):
                        p.dot(x=[float(pt[0])], y=[float(pt[1])], size=30, color='black')
                
                
                funcList.append(expr)
            
            
            if len(funcList) > 1:
                coorList = graphsIntersection(funcList)
                for i in coorList:
                    p.dot(x=[float(i[0])], y=[float(i[1])], size=30, color='black')
            tooltips = [
            ('x,y','@x, @y')
           ]                            
            p.add_tools(HoverTool(tooltips=tooltips))
            st.bokeh_chart(p, use_container_width=True)
        except:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(' ')
            
            with col2:
                st.markdown("""
# Invalid input
#### Please check if you have left a function or a colour cell empty and make sure your input is valid.                            
                            """)
            
            with col3:
                st.write('')
