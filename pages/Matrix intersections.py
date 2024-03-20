from sympy import *
import numpy as np
import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
from PIL import Image

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

session_state = st.session_state

def processInput(eqList):
    x, y, z, k = symbols('x y z k')
    equations = []
    
    for eq in eqList:
        a, b, c, d = eq  # Unpack the values as usual
        
        equations.append(Eq(a*x + b*y + c*z, d))
    
    return equations

def notSingular(matrix, rhsMatrix):
    x, y, z = symbols('x y z')
    det_value = matrix.det()
    if det_value != 0:
        inverse_matrix = matrix.inv()
        solution_matrix = inverse_matrix * rhsMatrix
        x, y, z = solution_matrix[0], solution_matrix[1], solution_matrix[2]
        return True, det_value, (x,y,z)
    else:
        return False, det_value, ''

def intersectionType(eqList):
    x, y, z = symbols('x y z')
    parallelCount = 0
    for eq1 in eqList:
        for eq2 in eqList:
            if eq1 != eq2:
                if simplify(eq1.lhs / eq2.lhs) == eq1.rhs / eq2.rhs:
                    if parallelCount == 0:
                        parallelCount += 2
                    else:
                        parallelCount += 1
    if parallelCount > 3:
        return 'planar', eqList, '', '', ''
    # Choose the variable to eliminate based on coefficients
    coefficients = [eqList[0].lhs.coeff(z), eqList[1].lhs.coeff(z), eqList[2].lhs.coeff(z)]
    elimEqList = []
    eq1Multiple = []
    eq2Multiple = []

    for i in range(1,3):
        lcm_coeff = lcm(abs(coefficients[0]), abs(coefficients[i]))
        eq1Times = lcm_coeff/abs(coefficients[0])
        eq2Times = lcm_coeff/abs(coefficients[i])
        
        eq1Multiple.append(eq1Times)
        eq2Multiple.append(eq2Times)

        if (coefficients[0] > 0 and coefficients[i] > 0) or (coefficients[0] < 0 and coefficients[i] < 0):
            eliminated_eq = Eq((eq1Times*(eqList[0].lhs) - eq2Times*(eqList[i].lhs)), (eq1Times*(eqList[0].rhs) - eq2Times*(eqList[i].rhs)))
            minus = True
        else:
            eliminated_eq = Eq((eq1Times*(eqList[0].lhs) + eq2Times*(eqList[i].lhs)), (eq1Times*(eqList[0].rhs) + eq2Times*(eqList[i].rhs)))
            minus = False
        elimEqList.append(eliminated_eq)
    
    
    
    try:
        if simplify(elimEqList[0].lhs / elimEqList[1].lhs) == elimEqList[0].rhs / elimEqList[1].rhs:
            return 'sheaf', elimEqList, eq1Multiple, eq2Multiple, minus
        else:
            parallelCount = 0
            for i in range(1,3):
                if simplify(eqList[0].lhs / eqList[i].lhs) == eqList[0].rhs / eqList[i].rhs:
                    if parallelCount == 0:
                        parallelCount += 2
                    else:
                        parallelCount += 1
                if simplify(eqList[1].lhs / eqList[2].lhs) == eqList[1].rhs / eqList[2].rhs:
                    if parallelCount == 0:
                        parallelCount += 2
                    else:
                        parallelCount += 1

            if parallelCount == 0:
                return 'prism', elimEqList, eq1Multiple, eq2Multiple, minus
            if parallelCount == 2:
                return '2 parallel', eqList, '', '', ''
            # if parallelCount == 3:
            #     return 'all parallel', eqList, '', ''
    except:
        parallelCount = 0
        # for eq1 in eqList:
        #     for eq2 in eqList:
        #         if eq1 != eq2:
        #             if simplify(eq1.lhs / eq2.lhs) == eq1.rhs / eq2.rhs:
        #                 if parallelCount == 0:
        #                     parallelCount += 2
        #                 else:
        #                     parallelCount += 1
        for i in range(1,3):
            if simplify(eqList[0].lhs / eqList[i].lhs) == eqList[0].rhs / eqList[i].rhs:
                if parallelCount == 0:
                    parallelCount += 2
                else:
                    parallelCount += 1
            if simplify(eqList[1].lhs / eqList[2].lhs) == eqList[1].rhs / eqList[2].rhs:
                if parallelCount == 0:
                    parallelCount += 2
                else:
                    parallelCount += 1

        if parallelCount == 0:
            return 'prism', elimEqList, eq1Multiple, eq2Multiple, minus
        if parallelCount == 2:
            return '2 parallel', eqList, '', '', ''
        # if parallelCount == 3:
        #     return 'all parallel', eqList, '', ''

def proveAllParallel(eqList):
    x, y, z = symbols('x y z')
    coefficients = [eqList[0].lhs.coeff(z), eqList[1].lhs.coeff(z), eqList[2].lhs.coeff(z)]
        
    eq1Times = abs(coefficients[1])/abs(coefficients[0])
    eq2Times = abs(coefficients[2])/abs(coefficients[0])

    if (coefficients[0] > 0 and coefficients[1] > 0) or (coefficients[0] < 0 and coefficients[1] < 0):
        eq1 = f"${str(eq1Times)}({str(eqList[0].lhs).replace('*','')}) = {str(eq1Times)}({str(eqList[0].rhs)})$"
    else:
        eq1 = f"$-{str(eq1Times)}({str(eqList[0].lhs).replace('*','')}) = -{str(eq1Times)}({str(eqList[0].rhs)})$"
    if (coefficients[0] > 0 and coefficients[2] > 0) or (coefficients[0] < 0 and coefficients[2] < 0):
        eq2 = f"${str(eq2Times)}({str(eqList[0].lhs).replace('*','')}) = {str(eq2Times)}({str(eqList[0].rhs)})$"
    else:
        eq2 = f"$-{str(eq2Times)}({str(eqList[0].lhs).replace('*','')}) = -{str(eq2Times)}({str(eqList[0].rhs)})$"
    parallelList = [eqParser(eqList[0]), eq1, eq2]
         
    return parallelList

def provePara(eqList):
    x, y, z = symbols('x y z')
    coefficients = [eqList[0].lhs.coeff(z), eqList[1].lhs.coeff(z), eqList[2].lhs.coeff(z)]
    parallelEq = []
    parallelCount = 0

    for i in range(1,3):
        if simplify(eqList[0].lhs / eqList[i].lhs) == eqList[0].rhs / eqList[i].rhs:
            if parallelCount == 0:
                parallelCount += 2
                parallelEq.append(eqList[0])
                parallelEq.append(eqList[i])
            else:
                parallelCount += 1
                parallelEq.append(eqList[0])
                parallelEq.append(eqList[i])
        if simplify(eqList[1].lhs / eqList[2].lhs) == eqList[1].rhs / eqList[2].rhs:
            if parallelCount == 0:
                parallelCount += 2
                parallelEq.append(eqList[1])
                parallelEq.append(eqList[2])
            else:
                parallelCount += 1
                parallelEq.append(eqList[1])
                parallelEq.append(eqList[2])
    for i,eq in enumerate(parallelEq):
        for j,eq1 in enumerate(parallelEq):
            if i != j:
                if eq == eq1:
                    parallelEq[j] = ''
    paraEqNum = []
    for i in range(3):
        if eqList[i] in parallelEq:
            paraEqNum.append(i)
    return paraEqNum


def eqParser(eq):
    try:
        latexEq = f"${str(eq.lhs).replace('*','')} = {str(eq.rhs).replace('*','')}$"
    except:
        return ''
    return latexEq

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("## Matrix intersections")
    
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
                {"Ax": '', "By": '', "Cz": '', "= D": ''}
            ]
        )

        for i in range(2):
            st.session_state.df.loc[len(st.session_state.df)] = ['','','','']
        st.session_state.df = st.session_state.df.style.hide()
        st.session_state.df = st.data_editor(st.session_state.df, num_rows='fixed', hide_index='True', width=600)

if st.button('Determine'):
    st.divider()
    eqList = []
    matrixList = []
    rhsList = []

    for i, row in st.session_state.df.iterrows():
        try:
            if row[0] == '' and row[1] == '' and row[2] == '' and row[3] == '':
                pass
            else:
                eqList.append([int(row[0]),int(row[1]),int(row[2]),int(row[3])])
                matrixList.append([int(row[0]),int(row[1]),int(row[2])])
                rhsList.append(int(row[3]))
        except:
            raise ValueError
            
    
    processed = processInput(eqList)
    matrix = Matrix([list(map(int, eq)) for eq in matrixList])
    rhsMatrix = Matrix([val for val in rhsList])
    bool,detVal, coorVal = notSingular(matrix, rhsMatrix)
    
    col1,col2,col3 = st.columns(3)

    with col2:
        st.write('### 1st Step:')
        with st.container():
            col1,col2 = st.columns([1,4])
            with col1:
                st.write('##### Matrix = ')
            with col2:
                for i,row in enumerate(matrixList):
                    st.write(str(row).replace(',', ''))
                st.write(f"##### Determinant value = {detVal}")
                
                if bool:
                    for i,val in enumerate(coorVal):
                        if i == 0:
                            st.write('$x = $')
                        if i == 1:
                            st.write('$y = $')
                        if i == 2:
                            st.write('$z = $')
                        st.write(val)
                    aPoint = Image.open("1Point.jpg")
                    st.image(aPoint)
                    st.write("The matrix is non-singular so the system is consistent and has an unique solution. The planes meet at a single point.")
                else:
                    st.write("The matrix is singular so we have to carry on.")

        if not bool:
            st.write('### 2nd Step:')
            with st.container():
                col1,col2 = st.columns([1,4])
                with col1:
                    st.write('##### Equations = ')
                with col2:
                    result, elimEqList, eq1Multiple, eq2Multiple, minus = intersectionType(processed)
                    if result == 'sheaf':
                        for i,eq in enumerate(processed):
                            st.write(eqParser(eq), '...(', str(i), ')')
                        st.write('##### Eliminate z')
                        for i,eq in enumerate(elimEqList):
                            if minus:
                                st.write(f"{eq2Multiple[i]}( {i+1} ) - {eq1Multiple[i]}( 0 ): {eqParser(eq)}")
                            else:
                                st.write(f"{eq2Multiple[i]}( {i+1} ) + {eq1Multiple[i]}( 0 ): {eqParser(eq)}")
                        sheaf = Image.open("sheaf.jpg")
                        st.image(sheaf)
                        st.write('The planes form a sheaf. The system of equations is consistent and has infinitely many solutions represented by the line of intersection of the three planes.')
                    if result == 'prism':
                        for i,eq in enumerate(processed):
                            st.write(eqParser(eq), '...(', str(i), ')')
                        st.write('##### Eliminate z')
                        for i,eq in enumerate(elimEqList):
                            if minus:
                                st.write(f"{eq2Multiple[i]}( {i+1} ) - {eq1Multiple[i]}( 0 ): {eqParser(eq)}")
                            else:
                                st.write(f"{eq2Multiple[i]}( {i+1} ) + {eq1Multiple[i]}( 0 ): {eqParser(eq)}")
                        prism = Image.open("prism.jpg")
                        st.image(prism)
                        st.write('The planes form a prism. The system of equations is inconsistent and has no solutions.')
                    if result == '2 parallel':
                        for i,eq in enumerate(processed):
                            st.write(eqParser(eq), '...(', str(i), ')')
                        parallelEq = provePara(processed)
                        
                        twoPara = Image.open("2Para.jpg")
                        st.image(twoPara)
                        st.write(f"( {parallelEq[0]} ) and ( {parallelEq[1]} ) are parallel.")   
                        st.write('Two of the planes are parallel and non-identical. The system of equations is inconsistent and has no solutions.')
                    # if result == 'all parallel':
                    #     for i,eq in enumerate(processed):
                    #         st.write(eqParser(eq), '...(', str(i), ')')                        
                    #     st.write('All of the planes are parallel and non-identical. The system of equations is inconsistent and has no solutions.')
                    if result == 'planar':
                        parallelList = proveAllParallel(processed)

                        for eq in parallelList:
                            st.write(eq)
                        allPara = Image.open("allPara.jpg")
                        st.image(allPara) 
                        st.write('All three equations represent the same plane. The system of equations is consistent and has infinitely many solutions.')
                    
