import numpy as np  # Para el cálculo numérico
import sympy as sym  # Para cálculo simbolico y detalles esteticos
from scipy.special import factorial  # Versión vectorizada de n! = n×(n-1)!
import matplotlib.pyplot as plt  # Para el gráfico de las funciones
from matplotlib.colors import Normalize, Colormap  # Para el coloreo dinamico
from matplotlib import ticker  # Detalles para ejes/barras de color
from matplotlib import rcParams  # Para aumentar la resolución de los gráficos
from tkinter import *
from tkinter import ttk,Entry,Button,Label,Tk,Text,StringVar
from tkinter import messagebox
import tkinter as tk

root = tk.Tk()# ventana raiz 
root.title('Dominio colorido de funciones complejas ')
root.geometry('900x900') # tamaño de la ventana 
root.config(bg='white')


def forma_binomica(z, show=False):
    '''
    Expresa en forma binomica al número complejo z devolviendo por separado la 
    parte real y la parte imaginaria. Si __show=True__ el resultado además se
    imprime en latex.
    '''
    a = np.real(z)
    b = np.imag(z)
    if show:
        display(sym.Eq(sym.symbols('z'), a + 1j*b))
    return a, b


def forma_exponencial(z, show=False):
    '''
    Expresa en forma exponencial al número complejo z devolviendo por separado
    su magnitud y su argumento o fase. Si __show=True__ el resultado además se
    imprime en latex.
    '''
    r = np.abs(z)
    theta = np.angle(z)
    if show:
        display(sym.Eq(sym.symbols('z'), r*sym.exp(1j*theta/np.pi*sym.pi)))
    return r, theta


def raices_enesimas(z0, n, j=None):
    '''
    Calcula las __n__ raíces enésimas del número complejo __z0__.
    Devuelve la jotaésima si el argumento __j__ es un número entero
    menor que __n__ y mayor o igual que 0. En cualquier otro caso,
    devuelve un array con las __n__ raíces.
    Obs: Si se quiere vectorizar esta función (i.e. ingresar un array
    __z0__ en vez de un número complejo) debe especificarse un __j__
    válido.
    '''
    r0, theta0 = forma_exponencial(z0)
    if j in set(range(n)):
        zj = (r0**(1/n))*np.exp(1j*(theta0 + 2*j*np.pi)/n)
        return zj
    elif j is not None:
        raise ValueError('j debería ser un entero en el rango [0, n-1]')
    else:
        k = np.arange(0, n)
        zk = (r0**(1/n))*np.exp(1j*(theta0 + 2*k*np.pi)/n)
        return zk


def positive_angle(z):
    '''
    Dado un array __z__, devuelve el argumento complejo para cada
    valor de __z__ en el rango [0, 2π).
    '''
    theta = np.angle(z)
    mask = theta < 0
    theta[mask] += 2*np.pi
    return theta


def veces_pi(val, pos):
    '''
    Función auxiliar para ponerle los ejes a la barra de color de la función
    coloreado_de_dominio.
    '''
    frac = val/np.pi
    epsilon = 1e-10
    if np.abs(frac) < epsilon:
        return '0'
    elif np.abs(frac - 1/4) < epsilon:
        return 'π/4'
    elif np.abs(frac - 1/2) < epsilon:
        return 'π/2'
    elif np.abs(frac - 3/4) < epsilon:
        return '3π/4'
    elif np.abs(frac - 1) < epsilon:
        return 'π'
    elif np.abs(frac - 5/4) < epsilon:
        return '5π/4'
    elif np.abs(frac - 3/2) < epsilon:
        return '3π/2'
    elif np.abs(frac - 7/4) < epsilon:
        return '7π/4'
    elif np.abs(frac - 2) < epsilon:
        return '2π'
    return f'{frac:.2f}π'


def coloreado_de_dominio(T, dom=(-5, 5), N=72, modulo='lineal', ax=None):
    '''
    Grafica los efectos de aplicar la transformación __T__ a una región cuadrada
    del espacio delimitada por los valores de __dom__.
    Parametros
    ----------
        T : str
            Expresión en función de un solo parametro complejo z.
            Ej : T='(z-I)/(z+I)'
        dom : tuple de 2×2 or list of two 2×2 tuples, default : (-5, 5)
            Limites de la región cuadrada cuya transformación se quiere
            visualizar.
            Ej : dom=(-1500, 1500)
        N : int, default : 72
            Cantidad de valores que se quieren graficar por eje. Es decir,
            se graficaran N² puntos en total.
        ax : two matplotlib.axes._subplots.AxesSubplot, default : None
            Eje de matplotlib donde graficar el dominio coloreado.

    Devuelve
    --------
        ax : eje de los gráficos generados.
    '''
    # Si no se brindó un eje para graficar, se lo crea.
    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=150, constrained_layout=True)
        fig.set_size_inches(6, 6)
        fig.set_facecolor('w')
    # Se define un dominio numérico cuadrado
    X, Y = np.meshgrid(np.linspace(
        dom[0], dom[1], N), np.linspace(dom[0], dom[1], N))
    Z = X + 1j*Y  # Se transforma el dominio de R² a C
    # Transformo el string de la expresión de T a una función simbolica T_sym.
    # Defino temporalmente un dominio simbólico.
    z = sym.symbols('z', complex=True)
    T_sym = sym.sympify(T, evaluate=False).subs(
        {'z': z, 'i': sym.I, 'e': sym.E, 'pi': sym.pi})
    # Obtengo una versión numérica de la función T_sym para evaluarla luego.
    T_num = sym.lambdify(
        z, T_sym, ['numpy', {'factorial': factorial, 'sen': np.sin, 'Sign': np.sign, }])
    # Evaluo la versión numérica en el dominio cuadrado de C y obtengo la imagen.
    W = T_num(Z)
    R = np.abs(W)  # Calculo su magnitud.
    Theta = positive_angle(W)  # Calculo su argumento entre 0 y 2π.
    # Defino el mapa de colores utilizado para colorear el dominio según el argumento.
    cmap = 'hsv'
    color_norm = Normalize(0, 2*np.pi)
    colormap = plt.cm.ScalarMappable(color_norm, cmap)
    colors = colormap.get_cmap()
    # Utilizo la función imshow para graficar una imagen coloreada sobre el dominio.
    phase = ax.imshow(Theta, cmap=colors, norm=color_norm,
                      aspect='equal', origin='lower',
                      interpolation='nearest',
                      extent=(dom[0], dom[1], dom[0], dom[1]))
    # Utilizo la función contourf para modificar la claridad de la imagen según la magmnitud de la imagen.
    if modulo == 'log':
        # Si se especificó "log" uso escala logaritmica.
        mod = ax.contourf(Z.real, Z.imag, R,
                          locator=ticker.LogLocator(), cmap='Greys_r',
                          alpha=0.45)
        ax.contour(Z.real, Z.imag, R,
                   locator=ticker.LogLocator(), cmap='Greys_r',
                   alpha=0.45, linewidths=1.)
        plt.colorbar(mod, ax=ax, orientation='horizontal',
                     label='$|T(z)|$')
        plt.savefig("Correlaciones Pearson.jpg")
    elif modulo == 'lineal':
        # Si se especificó "lineal" uso escala lineal.
        mod = ax.contourf(Z.real, Z.imag, R, cmap='Greys_r',
                          alpha=0.45)
        ax.contour(Z.real, Z.imag, R, cmap='Greys_r',
                   alpha=0.45, linewidths=1.)
        plt.colorbar(mod, ax=ax, orientation='horizontal',
                     label='$|T(z)|$')
    # Coloco la barra de color que indica la relación entre colores y argumentos
    plt.colorbar(phase, ax=ax, orientation='vertical',
                 label='$arg(T(z))$',
                 ticks=np.linspace(0, 2*np.pi, 5),
                 format=ticker.FuncFormatter(veces_pi))
    # Coloco los titulos de los ejes
    ax.set_ylabel(r'$\Im(z)$')
    ax.set_xlabel(r'$\Re(z)$')
    # Obtengo una expresión en latex de la función dada por el usuario.
    ltx_expr = sym.latex(sym.Eq(sym.sympify('T(z)'), T_sym))
    ax.set_title(r'$T:\mathbb{C} \to \mathbb{C}, \qquad$' +
                 f'${ltx_expr}$')
    return ax

def GraficaDominio():
    T = texto.get()
    modulo = texto1.get()
    
    # @param ["(z-i)/(z+i)", "exp(z)", "log(z)"] {allow-input: true}
    
    #T = mensaje
    # @markdown Función de un solo parametro complejo.
    #modulo = "log"  # @param ["lineal", "log", "None"]
    # @markdown Define si se colorearán curvas de nivel en escala lineal o logaritmica.

    # @param ["(-3, 3)", "(-30, 30)", "(0, 5)", "(-5, 0)"] {type:"raw", allow-input: true}
    dominio = (-15, 15)
    # @markdown tupla de 2×2.
    # @markdown Limites de la región cuadrada donde se coloreará el dominio.
    N = 3001  # @param {type:"integer"}
    # @markdown Cantidad de puntos que se quieren graficar por eje. Es decir,
    # @markdown se graficaran N² puntos en total (y luego el color se interpola entre ellos).
    ax = coloreado_de_dominio(T, dom=dominio, N=N, modulo=modulo, )
    plt.savefig("C:/Users/52221/Desktop/interfaz/img.png")
    imagengraficada = PhotoImage(file="img.png")
    imagen_sub=imagengraficada.subsample(2)
    fondo = Label(root,image=imagen_sub).place(x=10,y=100)
    root.mainloop()
    





#imagengraficada = PhotoImage(file="img.png")
#fondo = Label(root,image=imagengraficada).place(x=0,y=0)
texto = StringVar()
texto1 = StringVar()
label=Label(root,text="Ingrese su función",fg='#c311b1',font=('Verdana',12)).place(x=500,y=150)
label1=Label(root,text="Cruvas de nivel \n log o  lineal",fg='#c311b1',font=('Verdana',12)).place(x=500,y=200)
mensajeTxt=Entry(root,textvariable=texto).place(x=680,y=152)
mensajeTxt1=Entry(root,textvariable=texto1).place(x=680,y=200)

s = ttk.Style()
s.configure(
    "MyButton.TButton",
    foreground="#3c8995",
    background="#3c8995",
    padding=10,
    font=("Times", 12),
    
)

Graficar = ttk.Button( style="MyButton.TButton",text='Colorear Domino',command=GraficaDominio)
Graficar.place(x=600,y=300)

label1 = Label(root, text="INTRUCCIONES DEL PROGRAMA \n\n 1.- Para potencias utiliza '**'  Ejm: z**2\n 2.- Para exponencial usar exp() \n 3.- Para logaritmo log() \n 4.- Para el conjugado 'conjugate()'\n 5.- Para agregar la unidad imaginaria utiliza '1j' O 'I' \n 6.- Para funciones seno usa 'sin()', \n 7.- Para funciones coseno usa 'cos()' \n 8.- Para funciones tangente() usa 'tan()' \n\n" , fg='#c311b1',font=('Verdana',12)).place(x=10,y=600)
imagenpre = PhotoImage(file="img.png")
imagen_sub1=imagenpre.subsample(2)
fondo = Label(root,image=imagen_sub1).place(x=10,y=100)

root.mainloop()






