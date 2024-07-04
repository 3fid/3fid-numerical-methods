import numpy as np
import matplotlib.pyplot as plt

# Automaticky preklad z matlabu 
# asi stoji dost zahovno

def lichobeznik(funkce, odkud, kam, krok):
        xArr = np.arange(odkud, kam + krok, krok)
        integral = 0
        for x in xArr:
            integral += funkce(x)
            integral -= 0.5 * funkce(xArr[0]) + 0.5 * funkce(xArr[-1])
            integral *= krok
        return integral

def romberg():
    f = lambda x: np.sin(x) * np.exp(np.cos(x))
       
    integ = np.zeros(4)
    h0 = 0.1
    
    for i in range(4):
        h = h0 / (2 ** i)
        integ[i] = lichobeznik(f, 0, np.pi, h)
    
    v = np.exp(1) - np.exp(-1)
    
    r1 = 4/3 * integ[1] - 1/3 * integ[0]
    r2 = 64/45 * integ[2] - 20/45 * integ[1] + 1/45 * integ[0]
    
    plt.figure()
    plt.plot(range(1, 5), integ, 'b*-', label='slozeny lichobeznik')
    plt.plot([1, 4], [v, v], 'r', label='presne reseni')
    plt.plot([1, 2], [r1, r1], 'g', label='romb1')
    plt.plot([1, 3], [r2, r2], 'y', label='romb2')
    plt.legend('southeast')
    plt.show()

romberg()
