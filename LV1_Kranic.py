'''
Zadatak1:
def total_euro(radni_sati, satnica):
   return radni_sati*satnica

radni_sati=float(input('Radni sati:'))
satnica=float(input('eura/h:'))

ukupna_zarada=total_euro(radni_sati, satnica)
print('Ukupno:', ukupna_zarada, ' eura')
'''


'''
Zadatak 2
def odredi_ocjenu(broj):
    if 0.9<=broj<=1:
        return 'A'
    elif 0.8<=broj<0.9:
        return 'B'
    elif 0.7<=broj<0.8:
        return 'C'
    elif 0.6<=broj<0.7:
        return 'D'
    elif 0.0<=broj<0.6:
        return 'F'
    else:
        return 'Greška: Uneseni broj nije u parametrima.'
    
try:
    ocjena=float(input('Unesite broj:'))
    print('Ocjena: ', odredi_ocjenu(ocjena))
except ValueError:
    print('Greška.Unos mora biti broj!')
'''


'''
Zadatak 3:
brojevi = []

while True:
    unos=input('Unesite broj(Done za kraj petlje): ')

    if unos.lower()=="done":
        break

    try:
        broj=float(unos)
        brojevi.append(broj)
    except ValueError:
        print('Greška: Unesite ispravan broj.')

if brojevi:
    min_broj=min(brojevi)
    max_broj=max(brojevi)
    srednja_vrijednost=sum(brojevi)/len(brojevi)
    brojevi.sort()
    print('Uneseni brojevi: ', brojevi)
    print('Najmanji broj: ', min_broj)
    print('Najveci broj: ', max_broj)
    print('Srednja vrijednost: ', srednja_vrijednost)
else:
    print('Niste unjeli niti jedan broj.')
'''



