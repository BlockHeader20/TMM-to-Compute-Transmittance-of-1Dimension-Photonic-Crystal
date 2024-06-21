import numpy as np
import matplotlib.pyplot as plt

class PhotonicCrystal1d():
    Eps0 = 8.854e-12
    Mu0 = 4 * np.pi * 1e-7

    def __init__(self, N, d1, d2, epsr1, epsr2, mur1=1, mur2=1):
        """
        :params
        N: number of layers (NOT periods. If N==3, arranged like d1, d2, d1)
        d1: thickness of layer 1 (in "meter")
        d2: thickness of layer 2 (in "meter")
        epsr1: relative permittivity of layer 1
        epsr2: relative permittivity of layer 2
        mur1: relative permeability of layer 1
        mur2: relative permeability of layer 2
        """
        self.N = N
        self.d1 = d1
        self.d2 = d2
        self.epsr1 = epsr1
        self.epsr2 = epsr2
        self.mur1 = mur1
        self.mur2 = mur2
        self.Z1 = np.sqrt((self.Mu0*mur1)/(self.Eps0*epsr1))
        self.Z2 = np.sqrt((self.Mu0*mur2)/(self.Eps0*epsr2))

        self.last = None # indicate the type of the last layer (1 or 2)

        self.periods = None
        if N % 2 == 0:
            self.periods = N // 2

        self.transferMatrices = {}
        self.propagationMatrices = {}

        self.envEpsr = None
        self.envMur = None
        self.omega = None

    def calculateTransferMatrices(self, envEpsr, envMur=1):
        """
        :params
        envEpsr: environment permittivity
        envMu: environment permeability
        """
        self.envEpsr = envEpsr
        self.envMur = envMur

        envZ = np.sqrt((self.Mu0*envMur)/(self.Eps0*envEpsr))
        
        delta = self.Z1 / envZ
        mat = .5 * np.array([[1 + delta, 1 - delta], [1 - delta, 1 + delta]])
        self.transferMatrices["env2one"] = mat

        delta = self.Z2 / self.Z1
        mat = .5 * np.array([[1 + delta, 1 - delta], [1 - delta, 1 + delta]])
        self.transferMatrices["one2two"] = mat

        delta = self.Z1 / self.Z2
        mat = .5 * np.array([[1 + delta, 1 - delta], [1 - delta, 1 + delta]])
        self.transferMatrices["two2one"] = mat

        if self.N % 2 == 0:
            self.last = 2
            delta = envZ / self.Z2
            mat = .5 * np.array([[1 + delta, 1 - delta], [1 - delta, 1 + delta]])
            self.transferMatrices["last2env"] = mat
        else:
            self.last = 1
            delta = envZ / self.Z1
            mat = .5 * np.array([[1 + delta, 1 - delta], [1 - delta, 1 + delta]])
            self.transferMatrices["last2env"] = mat
    
    def calculatePropagationMatrices(self, omega):
        """
        :params
        omega: angular frequency of the wave (in "Hz")
        """
        self.omega = omega

        k1 = omega * np.sqrt(self.Eps0 * self.Mu0 * self.epsr1 * self.mur1)
        k2 = omega * np.sqrt(self.Eps0 * self.Mu0 * self.epsr2 * self.mur2)

        self.propagationMatrices["one"] = np.array([[np.exp(1j*k1*self.d1), 0], [0, np.exp(-1j*k1*self.d1)]])
        self.propagationMatrices["two"] = np.array([[np.exp(1j*k2*self.d2), 0], [0, np.exp(-1j*k2*self.d2)]])

    def showParas(self):
        print("Current Parameters:")
        print(f"Environment permittivity(EpsilonRelative): {self.envEpsr}")
        print(f"Environment permeability(MuRelative): {self.envMur}")
        print(f"Angular frequency: {self.omega:.2e}Hz")
        # print("\n", end="")

    def simulate_rt(self, RT=False):
        """
        :param
        RT: default False, return r and t
            if True, return R and T
        """
        if self.envEpsr is None or self.omega is None:
            raise ValueError("Params not initialized!")
        Q = self.transferMatrices["last2env"]
        flag = self.last-1
        for layer in range(self.N-1):
            if flag == 0:   # The layer is type 1
                Q = Q @ self.propagationMatrices["one"]
                Q = Q @ self.transferMatrices["two2one"]
            else:           # The layer is type 2
                Q = Q @ self.propagationMatrices["two"]
                Q = Q @ self.transferMatrices["one2two"]
            flag = (flag+1)%2
        assert flag == 0
        Q = Q @ self.propagationMatrices["one"]
        Q = Q @ self.transferMatrices["env2one"]
        r = -Q[1, 0]/Q[1, 1]
        t = Q[0, 0] - (Q[0, 1]*Q[1, 0])/(Q[1, 1])
        if RT:
            R = abs(r) ** 2
            T = 1 - R
            return R, T
        return r, t
    
if __name__ == "__main__":
    crystal = PhotonicCrystal1d(N=30, d1=15e-2, d2=5e-2, epsr1=2, epsr2=4)
    # crystal = PhotonicCrystal1d(N=30, d1=15e-2, d2=5e-2, epsr1=4, epsr2=8)
    crystal.calculateTransferMatrices(envEpsr=1, envMur=1) # put our crystal in vacuum
    omegas = 2 * np.pi * np.array([3.1e9, 3.38e9, 3.6e9])
    for omega in omegas:
        crystal.calculatePropagationMatrices(omega)
        crystal.showParas()
        R, T = crystal.simulate_rt(RT=True)
        print("Result:")
        print(f"R = {R:.2f}")
        print(f"T = {T:.2f}")
        # print(f"r = ({r.real:.2e}) + ({r.imag:.2e}j)")
        # print(f"t = ({t.real:.2e}) + ({t.imag:.2e}j)")
        print("\n", end="")
    
    # 绘制过程中R和T的变化图像
    freq = np.linspace(1.5e9, 4e9, 300)
    omegas = 2 * np.pi * freq
    Ts = []
    Rs = []
    for omega in omegas:
        crystal.calculatePropagationMatrices(omega)
        R, T = crystal.simulate_rt(RT=True)
        Ts.append(T)
        Rs.append(R)
    fig, ax = plt.subplots()
    ax.plot(freq, Rs, label='R', color='#ff7f0e')
    ax.plot(freq, Ts, label='T', color='#1f77b4')
    ax.legend()
    ax.set_title('Reflectivity and Transmittance for Different Frequency')
    ax.set_xlabel('f / Hz')
    ax.set_ylabel('R and T')
    ax.grid(True)
    plt.show()









    
    
