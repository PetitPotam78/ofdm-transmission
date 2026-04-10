#!/usr/bin/env python3.8

import numpy as np
import matplotlib.pyplot as plt
from math import floor
from scapy.all import Ether, IP, ICMP
import binascii
import scipy
import itertools

class Ofdm : 
    """
        Classe permettant d'implémenter un modem OFDM à N modulateurs complexes ou à iFFT
    """
    @staticmethod   
    def filtre_rec(symboles_mod, NbEchParSymb):
        signal_bb = np.repeat(symboles_mod,NbEchParSymb)
        return(signal_bb)
    
    @staticmethod
    def exp_comp(fp,te,N):
        #Création de l'exponentielle complexe
        t=np.arange(0,N*te,te)
        exp=np.exp(2j*np.pi*fp*t)
        return(exp)
    
    @staticmethod
    def exp_para(nb_sp, f0, df,te,N):
        sp_para = Ofdm.exp_comp(f0,te,N)
        for i in range(1,nb_sp):
            sp = Ofdm.exp_comp(f0+i*df,te,N) 
            sp_para = np.vstack((sp_para,sp))
        return(sp_para)
    
    @staticmethod
    def mapping(symbs_num, mapping_table):
        nb_symb_ofdm = symbs_num.shape[0]
        nb_sp = symbs_num.shape[1]
        symbs_para=np.zeros((nb_symb_ofdm, nb_sp), dtype = complex)
        for i in range(nb_symb_ofdm):
            symbs_para[i,] = np.array([mapping_table[tuple(b)] for b in symbs_num[i,:,:]])
        return(symbs_para)
    
    @staticmethod
    def zero_pad(symbs_para, nb_zero):
        nb_symb_ofdm = symbs_para.shape[0]
        nb_sp = symbs_para.shape[1]
        symbs_para_pad = np.zeros((nb_symb_ofdm, nb_sp+nb_zero), dtype = complex)
        for i in range(nb_symb_ofdm):
            for j in range(nb_sp):
                #print(j)
                if j < int(nb_sp/2) :
                    symbs_para_pad[i,j] = symbs_para[i,j]
                else :
                    symbs_para_pad[i,j+nb_zero] = symbs_para[i,j]
        return(symbs_para_pad)

    @staticmethod
    def rem_zero_pad(symbs_rcv, nb_sp):
        nb_symb_ofdm = symbs_rcv.shape[0]
        nb_symb_sp = symbs_rcv.shape[1]
        nb_zero = nb_symb_sp - nb_sp
        symbs_rcv_unpad = np.zeros((nb_symb_ofdm, nb_sp), dtype = complex)
        for i in range(nb_symb_ofdm):
            for j in range(nb_symb_sp):
                if j < nb_sp/2 :
                    symbs_rcv_unpad[i,j] =  symbs_rcv[i, j]
                elif j >= (nb_sp/2 + nb_zero) :
                    symbs_rcv_unpad[i, j-nb_zero] = symbs_rcv[i, j]
        return(symbs_rcv_unpad)
    
    @staticmethod
    def symbs_ofdm_to_sp(symbs_ofdm) :
        nb_symb_ofdm = symbs_ofdm.shape[0]
        nb_sp = symbs_ofdm.shape[1]
        symbs_sp = np.zeros((nb_sp, nb_symb_ofdm), dtype = complex)
        for i in range(nb_symb_ofdm):
            for j in range(nb_sp):
                symbs_sp[j,i] = symbs_ofdm[i,j]
        return(symbs_sp)

    @staticmethod
    def plot_constel_sp(symbs_sp, figsize = (12,35)) :
        nb_sp = symbs_sp.shape[0]
        fig, ax = plt.subplots(int(nb_sp/2), 2, figsize=figsize)
        ligne =0
        colonne = 0
        for i in range(nb_sp):
            ax[ligne,colonne].plot(np.real(symbs_sp[i]), np.imag(symbs_sp[i]), 'o', mew=2)
            ax[ligne,colonne].grid()
            ax[ligne,colonne].set_ylabel(' Partie imaginaire des \n symboles de modulation', fontsize=16)
            ax[ligne,colonne].set_xlabel('Partie réelle des symboles de modulation', fontsize=16)
            ax[ligne,colonne].set_title('Diagramme de constellation de la porteuse '+str(i), fontsize=16)
            ax[ligne,colonne].xaxis.set_tick_params(labelsize=14)
            ax[ligne,colonne].yaxis.set_tick_params(labelsize=14)
            #ax[ligne,colonne].set_xlim([-1.5,1.5])
            #ax[ligne,colonne].set_ylim([-1.5,1.5])
            colonne+=1
            if i%2 == 1 : 
                ligne += 1
                colonne = 0
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def upconv(signal, fp, te):
        #Création de l'exponentielle complexe
        t=np.arange(0,len(signal)*te,te)
        exp=np.exp(2j*np.pi*fp*t)
        #Translation de fréquence upconversion
        sig_analytique=signal*exp
        sig_module=np.real(sig_analytique)
        return(sig_module)
    
    @staticmethod
    def downconv_filt(signal, fp, te, fc, ordre=3):
        #Création de l'exponentielle complexe
        t=np.arange(0,len(signal)*te,te)
        exp=np.exp(-2j*np.pi*fp*t)
        #Translation de fréquence complexe downconversion
        signal_down = exp*signal
        ordre=3
        fe = int(1/te)
        fcn=fc/(fe/2)
        b, a = scipy.signal.butter(ordre, fcn, btype='low')
        signal_filt=scipy.signal.filtfilt(b, a, signal_down) 
        return(signal_filt)

    @staticmethod
    def moy_glissante(signal, nb_ech_par_symb):
        #Si les échantillons sont complexes mettre dtype = complexe sinon la partie imaginaire
        #sera supprimée lors des assignations : signal_moy[0] = signal[0], signal_moy[i] = np.sum(signal[0:i])/i
        signal_moy = np.zeros(len(signal), dtype = complex)
        signal_moy[0] = signal[0]
        for i in range(1,nb_ech_par_symb) :
            signal_moy[i] = np.sum(signal[0:i])/i
        for i in range(nb_ech_par_symb+1, len(signal)+1) :
            signal_moy[i-1] = np.sum(signal[i-nb_ech_par_symb:i]) / nb_ech_par_symb
        return(signal_moy)
    
    @staticmethod
    def downsample(signal, downsampling, offset = 0):
        signal_down=np.array([], dtype=complex)
        for i in range(offset, len(signal), downsampling):
            signal_down = np.append(signal_down,signal[i])
        return(signal_down)
    
    @staticmethod
    def detection(symbs_rcv, mapping_table):
        constellation = np.array([val for val in mapping_table.values()])
        symbs_detect=[min(constellation, key=lambda symb_mod:abs(np.square(np.real(symbr)-np.real(symb_mod))\
        +np.square(np.imag(symbr)-np.imag(symb_mod)))) for symbr in symbs_rcv]
        return(np.array(symbs_detect))
    
    @staticmethod
    def demapping(symbs_rcv, mapping_table) : 
        demapping_table = {v : k for k, v in mapping_table.items()}
        symbs_num=np.array([demapping_table[symb] for symb in symbs_rcv])
        bits_rcv=np.ravel(symbs_num)
        return (bits_rcv)

    @staticmethod
    def PS(bits_rcv_para, nb_sp, nb_symb_ofdm, bits_par_symb_sp):   
        bits_rcv = np.empty((nb_symb_ofdm*nb_sp, bits_par_symb_sp),dtype=int)
        for i in range(nb_symb_ofdm):
            for j in range(nb_sp):
                bits_rcv[i*nb_sp+j,] = bits_rcv_para[j,i*bits_par_symb_sp:(i+1)*bits_par_symb_sp]
        return(bits_rcv.ravel())

class Mesure :
    @staticmethod
    def dsp(signal, fe, mono_bi = "bi", unit = "dBm", affichage = 'oui'):
        N=len(signal)
        
        if mono_bi == "bi":
            S = 1/N*np.fft.fftshift(np.fft.fft(signal))
            f = np.arange(-fe/2, fe/2, fe/N)
        else :
            S1 = 1/N*np.fft.fft(signal)
            S=np.concatenate((S1[0:1], 2*S1[1:int(N/2)]))
            f = np.arange(0, fe/2, fe/N)
        
        S_aff= np.abs(S)
        if unit == "Veff" :
            S_aff = S_aff/np.sqrt(2)
        elif unit == "dBm" :
            S2 = S_aff/np.sqrt(2)
            S_aff=10*np.log10(np.square(S2)/50*1000)
        
        if affichage == 'oui':
            fig, ax = plt.subplots(figsize = (15, 6))
            ax.plot(f, S_aff)
            ax.grid()
            plt.tight_layout()

        return(f, S_aff)
    
    @staticmethod
    def dsp_moy(signal, fe, nval_FFT, affichage = 'no'):
        N=len(signal)
        S_mag = 0
        f = np.arange(-fe/2, fe/2, fe/nval_FFT)
        for i in range(0, floor(N/nval_FFT)*nval_FFT, nval_FFT):
            S_mag = np.abs(1/nval_FFT*np.fft.fftshift(np.fft.fft(signal[i:i+nval_FFT])))
            if i == 0 :
                S_mag_moy = S_mag
            else : 
                S_mag_moy=i/(i+1)*S_mag_moy+1/(i+1)*S_mag
        S_dBm_moy=10*np.log10(np.square(S_mag_moy/np.sqrt(2))/50*1000)
        if affichage == 'yes' :
            fig, ax = plt.subplots(figsize = (15, 6))
            ax.plot(f, S_dBm_moy)
            ax.grid()
            plt.show()
        return(f, S_dBm_moy)
    
    @staticmethod
    def constellation(signal, taille = 6, titre = "Diagramme de Constellation"):
        fig, ax = plt.subplots(figsize = (taille, taille))
        plt.plot(np.real(signal), np.imag(signal), 'o', mew=2)
        ax.grid()
        ax.set_ylabel(' Partie imaginaire des \n symboles de modulation', fontsize=16)
        ax.set_xlabel('Partie réelle des symboles de modulation', fontsize=16)
        ax.set_title(titre, fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        plt.show()
    
    @staticmethod
    def eye(signal, nsymb, nech_par_symb, titre = 'Diagramme de l\'oeil'):
        len_sig = len(signal)
        nech = nsymb * nech_par_symb
        fig, ax = plt.subplots(figsize = (15, 6))
        for i in range(0, floor(len_sig/nech)*nech, nech):
            sig_to_plot = signal[i:i+nech]
            ax.plot(sig_to_plot)
        ax.grid()
        ax.set_title(titre, fontsize =14)
        ax.set_xlabel('Numéro d\'échantillon', fontsize=14)
        ax.set_ylabel('Valeur de l\'échantillon', fontsize=14)

class Modem :
    """
        Classe permettant d'implémenter un MODulateur/dEModulateur PAM ou ASK (2,4,8), QPSK,
        et 16QAM.
    """
    
    def __init__(self, ModType, NbSymboles, bits):
        """
        Contructeur de la classe

        Parametres :
            ModType : type de modulation, PAM, ASK, PSK ou QAM
            NbSymboles : nombre de symboles de la modulation. 2, 4 ou 8 pour PAM ou ASK, 4 pour PSK et 16 pour QAM
            bits : tableau de bits numpy       
        """
        self.modtype = ModType
        self.nsymb = NbSymboles
        self.mod = (ModType, NbSymboles)
        if ModType == 'PAM' or ModType == 'ASK' :
            self.symb_type = 'reel'
        else : 
            self.symb_type = 'complexe'
        self.bits = bits
        self.bits_par_symb = int(np.log2(self.nsymb))
        self.symbs_num = bits.reshape(int(len(bits)/self.bits_par_symb), self.bits_par_symb)
        if (self.nsymb & (self.nsymb-1) == 0) and self.nsymb != 0 :
            self.bit_par_symb=int(np.log2(self.nsymb))
        else :
            raise ValueError('La deuxième valeur qui correspond au nombre de symboles \
            doit être une puissance de 2 : 2, 4, 8, 16, 32, 64, ...')
            
    def create_MP(self, amplitude, phase_ori = 0):
        """
        Fonction en charge de créer la table de mapping de chaque modulation
    
        Parametres
        ----------
        amplitude : amplitude maximale des sybmole de modulaiton pour une modulation PAM ou ASK,
                    amplitude max de la sinusoide pour une modulation PSK et amplitude max de I
                    et Q pour une modulation QAM
        phase_ori : utilisé seulement pour la modulation QPSK, phase à l'origine du premier
                    symbole (par déffaut = 0
   
        Retourne
        -------
        mapping_table : la table de mapping sous forme d'un dictionnaire
        """
        match self.mod:
            case ('PAM',2)| ('ASK',2) :
                mapping_table = {(0,) : -amplitude,
                                (1,) : amplitude}
            case ('PAM',4)| ('ASK',4) :
                mapping_table = {(0,0) : -3,
                                (0,1) : -1,
                                (1,0) : 1,
                                (1,1) : 3}
                for key in mapping_table.keys() :
                    mapping_table[key] = mapping_table[key] * amplitude / (self.nsymb-1) 
            case ('PAM',8)| ('ASK',8) :
                mapping_table = {(0,0,0) : -7,
                                (0,0,1) : -5,
                                (0,1,0) : -3,
                                (0,1,1) : -1,
                                (1,0,0) : 1,
                                (1,0,1) : 3,
                                (1,1,0) : 5,
                                (1,1,1) : 7}
                for key in mapping_table.keys() :
                    mapping_table[key] = mapping_table[key] * amplitude / (self.nsymb-1)    
            case ('PSK',4) :
                mapping_table = {(0,0) : 1*amplitude*np.exp(1j*phase_ori),
                                (0,1) : 1j*amplitude*np.exp(1j*phase_ori),
                                (1,0) : -1*amplitude*np.exp(1j*phase_ori),
                                (1,1) : -1j*amplitude*np.exp(1j*phase_ori)}
            case ('QAM',16) :
                mapping_table = {
                    (0,0,0,0) : -3-3j,
                    (0,0,0,1) : -3-1j,
                    (0,0,1,0) : -3+3j,
                    (0,0,1,1) : -3+1j,
                    (0,1,0,0) : -1-3j,
                    (0,1,0,1) : -1-1j,
                    (0,1,1,0) : -1+3j,
                    (0,1,1,1) : -1+1j,
                    (1,0,0,0) :  3-3j,
                    (1,0,0,1) :  3-1j,
                    (1,0,1,0) :  3+3j,
                    (1,0,1,1) :  3+1j,
                    (1,1,0,0) :  1-3j,
                    (1,1,0,1) :  1-1j,
                    (1,1,1,0) :  1+3j,
                    (1,1,1,1) :  1+1j}
                for key in mapping_table.keys() :
                    mapping_table[key] = mapping_table[key] * amplitude / 3 
            case _:
                mapping_table = None
                print(f'La modulation {self.nsymb}{self.modtype} n\'est pas implémentée')
        self.mapping_table = mapping_table
        return(mapping_table)

    def mapping(self, amplitude, phase_ori=0):
        """
        Effectue l'association entre les symboles numériques et les symboles de modulation  

        Parametres
        ----------
        amplitude : amplitude maximale des symboles de modulation pour une modulation PAM ou ASK, amplitude max de la sinusoide pour une modulation PSK et amplitude max de Iet Q pour une modulation QAM
        phase_ori : utilisé seulement pour la modulation QPSK, phase à l'origine du premier symbole (par déffaut = 0
                    
        Retourne
        -------
        symbs_mod : le vecteur avec les symboles de modulation
        """
        self.mapping_table = self.create_MP(amplitude, phase_ori)
        symbs_mod=np.array([self.mapping_table[tuple(symb)] for symb in self.symbs_num])
        return(symbs_mod)
    
    def ri_cosur(self, nsymb_aff, beta, ech_par_symb):
        """
        Génère la réponse impulsionnelle d'un filtre en cosinus surélevé

        Parametres :
        ----------
        - nsymb_aff : durée sur laquelle s'étend la réponse impulsionnelle en nombre de symboles. Doit être impair pour avoir une réponse symétrique. Typiquement 3, 5, 7 ou 9
        - beta : facteur de roll-off compris entre 0 et 1. Fixe la largeur de bande BW du signal modulé : BW = (1 + beta) x R
        - ech_par_symb : fixe le facteur de upsampling N du fitre c'est à dire le nombre d'échantillons par symboles. Le nombre d'échantillons de la réponse est nsymb_aff x ech_par_symb et le retard introduit par le filtre la moitié            
        
        Retourne :
        -------
        - symbs_mod : le vecteur avec la réponse impulsionnelle du filtre
        """  
        t = np.arange(-nsymb_aff/2*ech_par_symb, nsymb_aff/2*ech_par_symb)
        return np.where(np.abs(2*t) == ech_par_symb / beta, np.pi / 4 * np.sinc(t/ech_par_symb),
        np.sinc(t/ech_par_symb) * np.cos(np.pi*beta*t/ech_par_symb) / (1 - (2*beta*t/ech_par_symb) ** 2))

    def upsampling(self, symb,upsampling_factor):
        if self.symb_type == 'complexe':
            symb_up=np.array([], dtype=complex)
            for val in symb:
                pulse = np.zeros(upsampling_factor, dtype=complex)
                pulse[0] = val
                symb_up=np.concatenate((symb_up, pulse))
        else :
            symb_up=np.array([])
            for val in symb:
                pulse = np.zeros(upsampling_factor)
                pulse[0] = val
                symb_up=np.concatenate((symb_up, pulse))
        return(symb_up)

    def filtre_MF(self, symboles_mod, NbEchParSymb, type= 'rectangular', nsymb_aff= 7 , beta= 0.4):
        self.nech = NbEchParSymb
        if type == 'rectangular':
            signal_bb = np.repeat(symboles_mod,NbEchParSymb)
        elif type == 'cosur':
            symbs_mod_impuls = self.upsampling(symboles_mod, NbEchParSymb)
            unit_resp = self.ri_cosur(nsymb_aff, beta, NbEchParSymb)
            signal_bb = np.convolve(symbs_mod_impuls, unit_resp)
        return(signal_bb)
    
    def delay_sig(self, signal,samples):
        retard=np.zeros(samples)
        signal_ret=np.concatenate((retard, signal))
        return(signal_ret)

    def upconv(self, signal, fp, te):
        #Création de l'exponentielle complexe
        t=np.arange(0,len(signal)*te,te)
        reel=np.cos(2*np.pi*fp*t)
        im=np.sin(2*np.pi*fp*t)
        exp=reel+im*1j
        #Translation de fréquence upconversion
        sig_analytique=signal*exp
        sig_module=np.real(sig_analytique)
        return(sig_module)

    def downconv(self, signal, fp, te):
        #Création de l'exponentielle complexe
        t=np.arange(0,len(signal)*te,te)
        reel=np.cos(2*np.pi*fp*t)
        im=np.sin(2*np.pi*fp*t)
        if self.symb_type == 'complexe' :
            exp=reel-im*1j
            #Translation de fréquence complexe downconversion
            signal_down=exp*signal
        else : 
            #Translation de fréquence réel downconversion
            signal_down=reel*signal      
        return(signal_down)

    #Filtrage à fp

    def filtre_rcv(self, signal, type = 'butter', fc = 10 , fe = 100, ordre = 3):
        if type == 'butter':
            fcn=fc/(fe/2)
            b, a = scipy.signal.butter(ordre, fcn, btype='low')
            sig_filtre = 2*scipy.signal.filtfilt(b, a, signal)
            return(sig_filtre)
        
        if type == 'moy_glissante':
            #Si les échantillons sont complexes mettre dtype = complexe sinon la partie imaginaire
            #sera supprimée lors des assignations : signal_moy[0] = signal[0], signal_moy[i] = np.sum(signal[0:i])/i
            if self.symb_type == 'complexe':
                signal_moy = np.zeros(len(signal), dtype = complex)
            else : 
                signal_moy = np.zeros(len(signal))
            signal_moy[0] = signal[0]
            for i in range(1,self.nech) :
                signal_moy[i] = np.sum(signal[0:i])/i
            for i in range(self.nech+1, len(signal)+1) :
                signal_moy[i-1] = np.sum(signal[i-self.nech:i]) / self.nech
            return(signal_moy)

    def downsample(self, signal, downsampling, offset = 0):
        if self.symb_type == 'complexe':
            signal_down=np.array([], dtype=complex)
        else :
            signal_down=np.array([])
        for i in range(offset, len(signal), downsampling):
            signal_down = np.append(signal_down,signal[i])
        return(signal_down)

    def detection(self, symbs_rcv):
        constellation = np.array([val for val in self.mapping_table.values()])
        if self.symb_type == 'complexe':
            symbs_detect=[min(constellation, key=lambda symb_mod:abs(np.square(np.real(symbr)-np.real(symb_mod))\
            +np.square(np.imag(symbr)-np.imag(symb_mod)))) for symbr in symbs_rcv]
        else :
            symbs_detect=[min(constellation, key=lambda symb_mod:abs(symbr-symb_mod)) for symbr in symbs_rcv]
        return(np.array(symbs_detect))

    def demapping(self, symbs_rcv) : 
        demapping_table = {v : k for k, v in self.mapping_table.items()}
        symbs_num=np.array([demapping_table[symb] for symb in symbs_rcv])
        bits_rcv=np.ravel(symbs_num)
        return (bits_rcv)

class Canal :
    
    @staticmethod
    def awgn(signal, mean, std) :
        num_samples = len(signal)
        noise = np.random.normal(mean, std, size=num_samples)
        signal_bruite=signal+noise
        return(signal_bruite)

class Source :

    @staticmethod
    def random(Nombre_bits):
        bits = np.random.binomial(1,0.5,Nombre_bits)
        return(bits)

    def icmp(self, IPd, IPs = '192.168.1.1', MACs= '00:01:02:03:04:05', MACd = '06:07:08:09:0A:0B', req_rep = 'echo-request'):
        frame_tr=Ether(src=MACs, dst=MACd)/IP(src=IPs, dst=IPd)/ICMP(type = req_rep)
        bits=self.frame_to_bits(frame_tr)
        return(bits)
    
    def frame_to_bits(self, frame):
        frame_dec =list(bytes(frame))
        frame_bin=[]
        for val in frame_dec:
            z=format(val, "08b")
            frame_bin += list(z)
        bits=np.array([int(x) for x in frame_bin])
        return(bits)

    def bits_to_frame(self, bits) :
        frame_bits8 = bits.reshape(int(len(bits)/8), 8)
        frame_dec = np.packbits(frame_bits8)
        frame_bytes = frame_dec.tobytes()
        frame_object = Ether(frame_bytes)
        return(frame_object)

if __name__ == '__main__':
    print('test')
      
