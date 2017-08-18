# -*- coding: utf-8 -*-

#    Copyright (C) 2017
#    by Klemens Fritzsche, 2e63a67d46@leckstrom.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sympy as sp
from sympy.solvers.solveset import linsolve
import numpy as np
import symbtools as st
import math as mm
from IPython import embed

#~ dbg = True
dbg = False

# use prime number in subs_random_numbers()?
srn_prime = True
# specify how to optimize calculation of pseudo inverse / orthocomplement
# options: "free_symbols", "count_ops", "none"
#~ pinv_optimization = "free_symbols"
pinv_optimization = "count_ops"


def debug(*args, obj="None"):
    if dbg:
        for i in range(len(args)):
            if obj=="sympy":
                sp.pprint(args[i])
            else:
                print(args[i])

def not_simplify(expr, **kwargs):
    return expr

custom_simplify = sp.simplify
custom_simplify = not_simplify

def is_square_matrix(matrix):
    debug("is_square_matrix()")
    m, n = matrix.shape
    return True if (m==n) else False

def is_zero_matrix(matrix):
    m_rand = st.subs_random_numbers(matrix, prime=srn_prime)

    for i in range(len(m_rand)):
        if not np.allclose(float(m_rand[i]), 0):
            return False
    return True

def is_unit_matrix(matrix):
    debug("is_unit_matrix()")
    assert is_square_matrix(matrix), "Matrix is not a square matrix."
    m, n = matrix.shape
    m_rand = st.subs_random_numbers(matrix, prime=srn_prime)

    for i in range(len(m_rand)):
        if i%(m+1)==0:
            if not np.allclose(float(m_rand[i]), 1):
                return False
        else:
            if not np.allclose(float(m_rand[i]), 0):
                return False
    return True

def remove_zero_columns(matrix):
    """ this function removes zero columns of a matrix
    """
    m, n = matrix.shape
    M = sp.Matrix([])

    for i in range(n):
        if not is_zero_matrix(matrix.col(i)):
            M = st.concat_cols(M, matrix.col(i))
    return M

def matrix_to_vectorlist(matrix):
    m, n = matrix.shape
    vectors = []
    for i in range(n):
        vectors.append(matrix.col(i))
    return vectors

def count_zero_entries(vector):
    """ returns an integer with number of zeros
    """
    number = 0
    # check if zeros exist, if so: count them
    atoms = vector.atoms()
    if 0 in atoms:
        for i in range(len(vector)):
            if vector[i]==0:
                number+=1
    return number

def nr_of_ops(vector):
    m = vector.shape[0]
    ops = 0
    for i in range(m):
        ops += vector[i].count_ops()
    return ops

def is_linearly_independent(matrix, column_vector):
    m, n = matrix.shape

    rank1 = st.rnd_number_rank(matrix)
    tmp = st.concat_cols(matrix, column_vector)
    rank2 = st.rnd_number_rank(tmp)
    
    assert rank2 >= rank1
        
    return rank2 > rank1

def get_column_index_of_matrix(matrix, column):
    """ if V is a column vector of a matrix A, than this function returns
        the column index.
    """
    m1, n1 = matrix.shape
    for index in range(n1):
        if matrix.col(index)==column:
            return index
    return None

def reshape_matrix_columns(P):
    debug("reshape_matrix_columns()")
    m0, n0 = P.shape

    # pick m0 of the simplest lin. independent columns of P:
    P_new = remove_zero_columns(P)
    m1, n1 = P_new.shape

    list_of_cols = matrix_to_vectorlist(P_new)

    # sort by "complexity"
    if pinv_optimization=="free_symbols":
        cols_sorted_by_atoms = sorted( list_of_cols, \
                            key=lambda x: x.free_symbols, reverse=False)
    elif pinv_optimization=="count_ops":
        cols_sorted_by_atoms = sorted( list_of_cols, \
                              key=lambda x: nr_of_ops(x), reverse=False)
    elif pinv_optimization=="none":
        cols_sorted_by_atoms = list_of_cols

    # sort by number of zero entries
    colvecs_sorted = sorted( cols_sorted_by_atoms, \
                      key=lambda x: count_zero_entries(x), reverse=True)

    # pick m suitable column vectors and add to new matrix A: ----------
    A = colvecs_sorted[0]
    for j in range(len(colvecs_sorted)-1):
        column = colvecs_sorted[j+1]
        if is_linearly_independent(A,column):
            A = st.concat_cols(A,column)

        if st.rnd_number_rank(A)==m1:
            break

    assert A.is_square
            
    # calculate transformation matrix R: -------------------------------
    #R_tilde = sp.Matrix([])
    used_cols = []
    R = sp.Matrix([])
    for k in range(m1):
        new_column_index = k
        old_column_index = get_column_index_of_matrix(P,A.col(k))
        used_cols.append(old_column_index)
        tmp = sp.zeros(n0,1)
        tmp[old_column_index] = 1

        #R_tilde = st.concat_cols(R_tilde,tmp)
        R = st.concat_cols(R,tmp)
    #R=R_tilde

    # remainder columns of R matrix
    #m2,n2 = R_tilde.shape
    m2,n2 = R.shape
    for l in range(n0):
        if l not in used_cols:
            R_col = sp.zeros(n0,1)
            R_col[l] = 1
            R = st.concat_cols(R,R_col)

    m3, n3 = A.shape
    r2 = st.rnd_number_rank(A)
    assert m3==r2, "A problem occured in reshaping the matrix."

    # calculate B matrix: ----------------------------------------------
    B = sp.Matrix([])
    tmp = P*R
    for i in range(n0-m0):
        B = st.concat_cols(B,tmp.col(m0+i))

    
    assert is_zero_matrix( (P*R) - st.concat_cols(A,B)), \
                            "A problem occured in reshaping the matrix."

    debug("returning from reshape_matrix_columns()")
    return A, B, R


def right_pseudo_inverse(P, inv="adj"):
    debug("right_pseudo_inverse()")
    """ Calculates a right pseudo inverse with as many zero entries as
        possible. Given a [m x n] matrix P, the algorithm picks m
        linearly independent column vectors of P to form a regular
        block matrix A such that Q = P*R = (A B) with the permuation
        matrix P. The right pseudo inverse of Q is
            Q_rpinv = ( A^(-1) )
                      (   0    )
        and the right pseudo inverse of P
            P_rpinv = R*( A^(-1) )
                        (   0    ).
        There is a degree of freedom in choosing the column vectors for
        A. At the moment this can be done with respect to "count_ops"
        (minimizes number of operations) or "free_symbols" (minimizes
        the number of symbols).
    """
    m0, n0 = P.shape
    r = st.rnd_number_rank(P)
    assert r==m0, "not implemented"

    A, B, R = reshape_matrix_columns(P)

    #apparently this is quicker than A_inv = A.inv() #[m x m]:
    # TODO: check why inv="adj" leads to problems with example ER for
    # phi=0!
    if (inv=="normal"):
        debug("Calculating normal inverse...")
        A_inv = A.inv()
        debug("normal inverse, done")
    else:
        debug("Calculating inverse using berkowitz det...")
        A_det = A.berkowitz_det()
        debug("done berkowith det. calculating inverse")
        A_inv = A.adjugate()/A_det
        debug("inverse, done")

    p = n0-m0
    zero = sp.zeros(p,m0)
    Q = custom_simplify(A_inv)
    P_pinv = custom_simplify( R*st.concat_rows(Q,zero) )

    PP_pinv = P*P_pinv
    debug("PP_pinv ", PP_pinv.shape, " = ")
    #embed()
    #debug(PP_pinv, obj="sympy")
    #assert is_unit_matrix(PP_pinv),"Rightpseudoinverse is not correct."
    debug("returning from right_pseudo_inverse()")
    return P_pinv

def has_right_ortho_complement(matrix):
    r = st.rnd_number_rank(matrix)
    m, n = matrix.shape
    if(m>=n):
        return True if (r<n) else False
    return True

def right_ortho_complement(P):
    assert has_right_ortho_complement(P), "there is no ortho complement!"
    
    m0, n0 = P.shape
    r = st.rnd_number_rank(P)
    if r==m0:
        A, B, R = reshape_matrix_columns(P)

        #apparently this is quicker than A_inv = A.inv() #[m x m]:
        A_det = A.berkowitz_det()
        A_inv = A.adjugate()/A_det

        minusAinvB = custom_simplify(-A_inv*B)

        p = n0-m0
        unit_matrix = sp.eye(p)
        Q = custom_simplify(A_inv)
        P_roc = custom_simplify( R*st.concat_rows(minusAinvB, unit_matrix) )
    else:
        P_roc = st.nullspaceMatrix(P)

    assert is_zero_matrix(P*P_roc), "Right orthocomplement is not correct."
    return P_roc

def has_left_ortho_complement(matrix):
    r = st.rnd_number_rank(matrix)
    m, n = matrix.shape
    if(n>=m):
        return True if (r<m) else False
    return True

def left_ortho_complement(P):
	P_loc = right_ortho_complement(P.T).T
	assert is_zero_matrix(P_loc*P), "Left orthocomplement is not correct"
	return P_loc


s = sp.Symbol('lambda', commutative=False)
st.make_global([s], upcount=2)

# for convenience
def gen_state(n, alpha=1, name="x", enum=1):
    """ n: number of state variables
        alpha: number of time derivatives
    """
    global xx
    xx = sp.Matrix([])
    xx_tmp = sp.Matrix([])
    
    for i in range(enum,enum+n):
        xx = st.row_stack(xx, sp.Matrix([sp.var(name+str(i))]))

    for i in range(alpha+1):
        x_ndot = st.time_deriv(xx, xx, order=i)
        xx_tmp = st.row_stack(xx_tmp, x_ndot)
    
    xx_atoms = list(xx_tmp.atoms())
    st.make_global(xx_atoms, upcount=2)
    
    return xx_tmp


def bin_coeff(a,b):
    """ returns binomial coefficient of a over b
    """
    assert (a-b)>=0, "a is supposed to be greater or equal b!"
    return (mm.factorial(a))/(mm.factorial(b)*mm.factorial(a-b))

def Top(*args,**kwargs):
    # Assuming *args to be a list of the form [A0,A1,A2,...]
    # Also assuming s is shifted to the right, such that
    #    A(s) = A0 + A1*s + A2*s**2 + ... + An*s**n
    #    (even though s is a commutative symbol).
    # TODO: make this more generic
    coeff = list(args)
    alpha = len(coeff)-1
    m,n = coeff[0].shape
    beta = kwargs.get("beta", alpha)

    TopA = sp.zeros(m*(beta+1),n*(alpha+beta+1))

    # calculate T_beta(Ai):
    for i in range(alpha+1):
        left_zero_block = sp.zeros(m*(beta+1),n*i)
        right_zero_block = sp.zeros(m*(beta+1),n*(alpha-i))

        Ti_mid = sp.Matrix([])
        for j in range(beta+1):
            # jth column of the middle block of T_beta(Ai)
            Ti_mid_hyperrowj = sp.Matrix([])
            for k in range(beta+1):
                if k<=j:
                    Ti_mid_hyperrowj = st.col_stack(Ti_mid_hyperrowj,\
                                bin_coeff(j,k)*\
                                st.time_deriv(coeff[i],xx,order=(j-k)))
                else:
                    Ti_mid_hyperrowj = st.col_stack(Ti_mid_hyperrowj,\
                                sp.zeros(m,n))
            Ti_mid = st.row_stack(Ti_mid, Ti_mid_hyperrowj)
        Ti = st.col_stack(left_zero_block, Ti_mid, right_zero_block)

        TopA = TopA + Ti
    return TopA
    
def Top_aug(*args, **kwargs):
    """ Returns augmented matrix row(Top(*args,**kwargs),(I,0))
    """
    coeff = list(args)
    alpha = len(coeff)-1
    m1, n1 = coeff[0].shape
    beta = kwargs.get("beta", alpha)

    TopA = Top(*args, beta=beta)
    m2, n2 = TopA.shape
    matrix_aug = st.col_stack(sp.eye(m1), sp.zeros(m1,n2-m1))
    return st.row_stack(TopA, matrix_aug)

def Hright(*args,**kwargs):
    coeff = list(args)
    alpha = len(coeff)-1
    m,n = coeff[0].shape
    beta = kwargs.get("beta", alpha)

    HrightA = sp.zeros(m*(alpha+beta+1),n*(beta+1))
    # calculate T_beta(Ai):
    for i in range(alpha+1):
        hi_top = sp.Matrix([])
        for j in range(beta+1):
            # jth col of T_beta(Ai)
            hi_hypercolj = sp.Matrix([])
            # k is the row of T_beta(Ai)
            for k in range(beta+1+i):
                if (i+j-k)>=0:
                    hi_hypercolj = st.row_stack(hi_hypercolj,\
                                (-1)**(i+j-k)*bin_coeff(i+j,i+j-k)*\
                                st.time_deriv(coeff[i],xx,order=(i+j-k)))
                else:
                    hi_hypercolj = st.row_stack(hi_hypercolj,sp.zeros(m,n))
            hi_top = st.col_stack(hi_top,hi_hypercolj)

        zero_block = sp.zeros(m*(alpha-i),n*(beta+1))
        hi = st.row_stack(hi_top, zero_block)
        HrightA = HrightA + hi
    return HrightA

def Hright_aug(*args,**kwargs):
    """ Returns augmented matrix col(Hright(*args,**kwargs),(I,0))
    """
    coeff = list(args)
    alpha = len(coeff)-1
    m1, n1 = coeff[0].shape
    beta = kwargs.get("beta", alpha)

    HrightA = Hright(*args, beta=beta)
    m2, n2 = HrightA.shape
    matrix_aug = st.row_stack(sp.eye(m1), sp.zeros(m2-m1,m1))
    return st.col_stack(HrightA, matrix_aug)

def Hleft(*args,**kwargs):
    coeff = list(args)
    alpha = len(coeff)-1
    m,n = coeff[0].shape
    beta = kwargs.get("beta", alpha) # if beta is not given, use beta=alpha

    HleftA = sp.zeros(m*(beta+1),n*(alpha+beta+1))
    # calculate h_beta(Ai):
    for i in range(alpha+1):
        hi = sp.Matrix([])
        for j in range(beta+1):
            # jth col of h_beta(Ai)
            zero = sp.zeros(m,n)
            hi_hyperrowj = sp.Matrix([])
            # k is the row of h_beta(Ai)
            for k in range(beta+alpha+1):
                if (k<=i+j):
                    hi_hyperrowj=st.col_stack(hi_hyperrowj,\
                                bin_coeff(i+j,k)*st.time_deriv(coeff[i],xx,order=(i+j-k)))
                else:
                    hi_hyperrowj=st.col_stack(hi_hyperrowj,zero)
            hi = st.row_stack(hi,hi_hyperrowj)
        HleftA = HleftA + hi
    return HleftA

def Hleft_aug(*args,**kwargs):
    """ Returns augmented matrix col(Hright(*args,**kwargs),(I,0))
    """
    coeff = list(args)
    alpha = len(coeff)-1
    m1, n1 = coeff[0].shape
    beta = kwargs.get("beta", alpha) # if beta is not given, use beta=alpha

    HleftA = Hleft(*args, beta=beta)
    m2, n2 = HleftA.shape
    matrix_aug = st.col_stack(sp.eye(n1), sp.zeros(n1,n2-n1))

    return st.row_stack(HleftA, matrix_aug)


def is_unimodular(*args):
    """ Testing unimodularity of matrices in ddt. Returns -1 if
        matrix is not unimodular. Returns the order of its inverse,
        if matrix is unimodular.
    """
    coeff = list(args)
    alpha = len(coeff)-1
    m1, n1 = coeff[0].shape
    assert m1==n1, "Unimodular matrices are square matrices!"
    # highest possible order of the inverse
    beta = alpha*(m1-1)

    debug("order of matrix "+str(alpha))
    debug("maximum order of inverse "+str(beta))
    for order in range(beta+1):
        debug("testing inverse order="+str(order))
        TopA = Top(*args, beta=order)
        TopA_aug = Top_aug(*args, beta=order)
        m, n = TopA.shape
        TopA_rank = st.rnd_number_rank(TopA)
        unimodular = \
               ( TopA_rank == st.rnd_number_rank(TopA_aug) ) \
                                               and ( TopA_rank == m )
        if unimodular:
            debug("Matrix is unimodular. Inverse polynomial matrix has degree " \
                                                        +str(order)+".")
            return order
        else:
            debug("failed. increasing order")
    return -1


def unimodular_inverse(*args, **kwargs):
    coeff = list(args)
    alpha = len(coeff)-1
    m1, n1 = coeff[0].shape
    assert m1==n1, "Unimodular matrices are square matrices!"

    if "beta" in kwargs:
        TopA = Top(*args, beta=kwargs.get("beta"))
        TopA_rpinv = right_pseudo_inverse(TopA)
        m2,n2 = TopA_rpinv.shape
        Io = st.col_stack(sp.eye(m1), sp.zeros(m1,m2-m1))
        A_inv = sp.simplify(Io*TopA_rpinv)
        return A_inv
    else:
        order = is_unimodular(*args)
        if order!=-1:
            TopA = Top(*args, beta=order)
            TopA_rpinv = right_pseudo_inverse(TopA,inv="normal")
            #~ TopA_rpinv = right_pseudo_inverse(TopA)
            m2,n2 = TopA_rpinv.shape
            Io = st.col_stack(sp.eye(m1), sp.zeros(m1,m2-m1))
            A_inv = sp.simplify(Io*TopA_rpinv)
            return A_inv
        else:
            print("Matrix is not unimodular!")
            return None

def is_lefttinvertible(*args):
    coeff = list(args)
    n, m = coeff[0].shape
    assert n>=m, "Wrong dimension!"
    alpha = len(coeff)

    if m==n:
        return is_unimodular(*args)

    # assumption: same upper bound for beta as for unimodular matrices
    # (remains to be proven)
    beta = alpha*(n-1)

    for order in range(1,beta+1):
        HleftA = Hleft(*args, beta=order)
        HleftA_aug = Hleft_aug(*args, beta=order)

        if st.rnd_number_rank(HleftA)==st.rnd_number_rank(HleftA_aug):
            return order
    return -1

def is_rightinvertible(*args):
    coeff = list(args)
    m, n = coeff[0].shape
    assert m<=n, "Wrong dimension!"
    alpha = len(coeff)

    if m==n:
        return is_unimodular(*args)

    # assumption: same upper bound for beta as for unimodular matrices
    # (remains to be proven)
    beta = alpha*(n-1)

    for order in range(1,beta+1):
        dA = Hright(*args, beta=order)
        dA_aug = Hright_aug(*args, beta=order)

        if st.rnd_number_rank(dA)==st.rnd_number_rank(dA_aug):
            return order
    return -1

def right(*args,**kwargs):
    """ input:  A_0+s*A_1+...+s**alpha*A_alpha
        output: B_0+B_1*s+...+B_alpha*s**beta
        where s=d/dt
    """
    coeff = list(args)
    m1, n1 = coeff[0].shape
    alpha = len(coeff)
    
    B_list = []
    for k in range(alpha):
        Bk = sp.zeros(m1,n1)
        for i in range(k,alpha):
            Bk = Bk + bin_coeff(i,k)*st.time_deriv(coeff[i],xx,order=(i-k))
        B_list.append(Bk) 

    return tuple(B_list)

def left(*args,**kwargs):
    """ input: A_0+A_1*s+...+A_alpha*s**alpha
        output:  B_0+s*B_1+...+s**alpha*B_beta
        where s=d/dt
    """
    coeff = list(args)
    m1, n1 = coeff[0].shape
    alpha = len(coeff)

    B_list = []
    for k in range(alpha):
        Bk = sp.zeros(m1,n1)
        for i in range(k,alpha):
            Bk = Bk + (-1)**(i+k)*bin_coeff(i,k)*st.time_deriv(coeff[i],xx,order=(i-k))
        B_list.append(Bk) 
 
    return tuple(B_list)


sp.init_printing(use_latex='mathjax')
#~ xx = gen_state(4,2,enum=0)
#~ sp.pprint(xx.T)
