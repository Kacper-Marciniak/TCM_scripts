import numpy as np
from scipy.optimize import fsolve
import math

class Optimization:
    def __init__(self):
        self.liczba_rzedow_zebow = 85
        self.obrobka_wejscie = 10 #[mm]
        self.obrobka_wyjscie = 10 #[mm]
        self.dl_obrobki = 220 #[mm]
        self.czas_pz = 50 #[min]
        print('Optimazation module activated')

    def znajdz_ilosc_przejsc_przylozenia(self,skrawanie): # z gory
        wyiskrzanie = 0.005
        wykanczajaco = 0.035
        zgrubnie = 0.150
        ilosc_przejsc = 0
        if skrawanie>0:
            ilosc_przejsc += 1
            skrawanie -= wyiskrzanie
            if skrawanie>0: 
                ilosc_przejsc+=1
                skrawanie -= wykanczajaco
                while skrawanie>0:
                    skrawanie -= zgrubnie
                    ilosc_przejsc += 1
        return ilosc_przejsc

    def znajdz_ilosc_przejsc_natarcia(self,skrawanie): # z przodu
        przejscie = 0.06
        ilosc_przejsc = 0
        while skrawanie>0:
            skrawanie -= przejscie
            ilosc_przejsc += 1
        return ilosc_przejsc

    def znajdz_czas_obrobki(self, ilosc_przejsc, v_skrawania):
        dlugosc_calkowita = self.dl_obrobki + self.obrobka_wejscie + self.obrobka_wyjscie
        czas_na_przejscie = dlugosc_calkowita/v_skrawania #[min]
        czas_calkowity_obrobki = czas_na_przejscie*ilosc_przejsc*self.liczba_rzedow_zebow    
        czas_calkowity = czas_calkowity_obrobki + self.czas_pz
        return czas_calkowity

    def regeneration_I_calculate_outputs(self, stepienie):
        '''
        Calculate outputs for regeneration type I
        '''
        
        ilosc_przejsc = self.znajdz_ilosc_przejsc_przylozenia(stepienie)
        czas_calkowity = self.znajdz_czas_obrobki(ilosc_przejsc,700)
        koszt = 61.67 + 4.5 * czas_calkowity
        return czas_calkowity, koszt, ilosc_przejsc

    def regeneration_II_solution_options(self, stepienie_natarcia, stepienie_przylozenia):
        '''
        Used to find soultions for particular P point
        '''  
        ilosc_przejsc_przylozenia = self.znajdz_ilosc_przejsc_natarcia(stepienie_przylozenia)
        ilosc_przejsc_natarcia = self.znajdz_ilosc_przejsc_przylozenia (stepienie_natarcia)
        czas_calkowity_przylozenia  = self.znajdz_czas_obrobki(ilosc_przejsc_przylozenia ,700)
        czas_calkowity_natarcia = self.znajdz_czas_obrobki(ilosc_przejsc_natarcia,500)
        czas_calkowity = czas_calkowity_przylozenia  + czas_calkowity_natarcia
        koszt_przylozenia  = 61.67 + 4.5 * czas_calkowity_przylozenia 
        koszt_natarcia = 61.67 + 4.5 * czas_calkowity_natarcia
        koszt = koszt_przylozenia + koszt_natarcia
        return round(czas_calkowity,0), round(koszt,2), ilosc_przejsc_natarcia, round(stepienie_natarcia,3)*1000,  ilosc_przejsc_przylozenia, round(stepienie_przylozenia,3)*1000 

    def regeneration_II_calculate_outputs(self,stepienie):
        '''
        Used to generate set of possible solutions  (P(x,y) coordinates) for regeneration II.
        Coordinates are pased to the solver to finds outputs parameters for them.
        '''
        # Calculate E & D points coordinates. There is a lot unused parameters with can be later parametrized
        xD = stepienie
        xE = 0
        yD = 0
        gamma = 10 # Angle with come from blunt model
        gamma_rad = math.radians(gamma)
        yE = math.tan(gamma_rad) * xD

        # Find set of P points with lay between E and D, then calcuate outputs for them
        number_of_points = 10     
        ED = math.sqrt((xD-xE)**2 + (yD-yE)**2)
        def equation(input,i):
            xP,yP = input
            return (xE-xD)*(yE-yP) - (yE-yD)*(xE-xP), (xE-xP)**2 + (yE-yP)**2 -(ED**2)*(i/number_of_points)**2
        solutions = []
        for i in range(1,number_of_points):
            xP,yP = fsolve(equation,(1,1),i)
            output = self.regeneration_II_solution_options(xP,yP)
            solutions.append(output)
        return list(solutions)

    def regeneration_z_gory_calculate_outputs(self, stepienie):
        '''
        Calculate outputs for regeneration type "z góry"
        '''
        gamma = 10 # angle with come from blunt model
        gamma_rad = math.radians(gamma)
        stepienie = math.tan(gamma_rad) * stepienie

        ilosc_przejsc = self.znajdz_ilosc_przejsc_natarcia(stepienie)
        czas_calkowity = self.znajdz_czas_obrobki(ilosc_przejsc,500)
        koszt = 61.67 + 4.5 * czas_calkowity
        return czas_calkowity, koszt, ilosc_przejsc, round(stepienie,3)
