# 🤖 Tehisintellekti rakendamise projektiplaani mall (CRISP-DM)

<br>
<br>


## 🔴 1. Äritegevuse mõistmine
*Fookus: mis on probleem ja milline on hea tulemus?*


### 🔴 1.1 Kasutaja kirjeldus ja eesmärgid
Kellel on probleem ja miks see lahendamist vajab? Mis on lahenduse oodatud kasu? Milline on hetkel eksisteeriv lahendus?

> Probleem on tudengitel, kes tahavad endale huvi pakkuvat ainet. Hetkene ÕIS2 otsing vajab liiga spetsiifilist märksõna ehk ei ole võimalik otsida aine sisu järgi, aga arendatav lahendus aitaks otsida ained üldsõnaliselt kirjeldades.

### 🔴 1.2 Edukuse mõõdikud
Kuidas mõõdame rakenduse edukust? Mida peab rakendus teha suutma?

> Hindame rakenduse edukust kasutajate tagasiside põhjal, mis vastaksid antud kriteeriumitele: rakendus peab suutama soovitada ained kasutajale kasutades päringut ja päringute ajalugu (kontekst) ning soovitused peavad olema relevantsed ja täpsed. Vastuse ooteaeg ei tohi olla pikem kui 10s.

### 🔴 1.3 Ressursid ja piirangud
Millised on ressursipiirangud (nt aeg, eelarve, tööjõud, arvutusvõimsus)? Millised on tehnilised ja juriidilised piirangud (GDPR, turvanõuded, platvorm)? Millised on piirangud tasuliste tehisintellekti mudelite kasutamisele?

> <b>Ressursipiirangud:</b> eelarve peaks olema võimalikult soodne, ajapiirang 3 kuud, 2 arendajat ja testijad (võib-olla tudengid või ülikooli töötajad) ning arvutusvõimsuse vähendamiseks saab ligipääsu ainult ÕIS2 (aktiivsed) kasutajad.<br> <b>Tehnilised ja juriidilised piirangud:</b> kas kasutajate vestlusi salvestatakse, küsimus peab olema ÕIS2 ainetega seotud (ülejäänutele ei vasta), piirata päringute arvu.
<br> <b>Piirangud tasuliste tehisintellekti mudelite kasutamisele:</b> rakendus jookseb ülikooli serveris (ei kasuta väliseid teenuseid).

<br>
<br>


## 🟠 2. Andmete mõistmine
*Fookus: millised on meie andmed?*

### 🟠 2.1 Andmevajadus ja andmeallikad
Milliseid andmeid (ning kui palju) on lahenduse toimimiseks vaja? Kust andmed pärinevad ja kas on tagatud andmetele ligipääs?

> Andmeid on vaja RAG süsteemi toimimiseks ning andmed on kõikide registreeritavate Tartu Ülikooli ainete kohta. Andmed on pärit ülikoolilt endalt, mis on veebist avalikult kättesaadavad (2 aasta andmed).

### 🟠 2.2 Andmete kasutuspiirangud
Kas andmete kasutamine (sh ärilisel eesmärgil) on lubatud? Kas andmestik sisaldab tundlikku informatsiooni?

> Seda teavad Tartu Ülikooli IT inimesed (saab isikuandmed vajadusel kustutada).

### 🟠 2.3 Andmete kvaliteet ja maht
Millises formaadis andmeid hoiustatakse? Mis on andmete maht ja andmestiku suurus? Kas andmete kvaliteet on piisav (struktureeritus, puhtus, andmete kogus) või on vaja märkimisväärset eeltööd)?

> .csv fail, milles on 3301 rida ja 223 tunnust. Andmete suurus on 45,3 MB ja on vaja teha eeltööd filtreerimisel ja puhastamisel.

### 🟠 2.4 Andmete kirjeldamise vajadus
Milliseid samme on vaja teha, et kirjeldada olemasolevaid andmeid ja nende kvaliteeti.

> Vaja on analüüsida 223 veeru tähendused ning välja valida olulised veerud. Seejärel on vaja valida õige veerg info leidmiseks, puhastada json väljad, panna kokku vabatekstilised kirjeldavad tunnused keelemudelile või RAG süsteemile analüüsiks. Vaja on üle vaadata puuduvate tunnuste hulk ning otsustada, mida nendega ette võtta.

<br>
<br>


## 🟡 3. Andmete ettevalmistamine
Fookus: Toordokumentide viimine tehisintellekti jaoks sobivasse formaati.

### 🟡 3.1 Puhastamise strateegia
Milliseid samme on vaja teha andmete puhastamiseks ja standardiseerimiseks? Kui suur on ettevalmistusele kuluv aja- või rahaline ressurss?

> Andmed on vaja puhastada natukene sarnasel viisil nagu 2.4 andmete kirjelduses mainitud. Võimalik, et oleks vaja imputeerida puuduvaid andmeid või neid otsida mõnest teisest ÕIS2 APIst või järeldada muudest andmetest. Andmete puhastamisele võiks kuluda umbes 1 nädal.

### 🟡 3.2 Tehisintellektispetsiifiline ettevalmistus
Kuidas andmed tehisintellekti mudelile sobivaks tehakse (nt tükeldamine, vektoriseerimine, metaandmete lisamine)?

> Olenevalt erinevatest meetoditest saame anda tehisintellektile kirjelduse andmetest ning ligipääsu puhastatud andmetele, et neid vajadusel filtreerida jne. RAG süsteemi jaoks on vaja välja valida aineid kirjeldavad veerud ning teha iga aine jaoks üks kirjeldav tekst. Valitud andmed tuleb vektoresituse kujule viimise mudeliga teisendada vektoriteks. Selle abil saab RAG süsteem semantiliselt otsingu järgi valida otsingule vastavad ained.

<br>
<br>

## 🟢 4. Tehisintellekti rakendamine
Fookus: Tehisintellekti rakendamise süsteemi komponentide ja disaini kirjeldamine.

### 🟢 4.1 Komponentide valik ja koostöö
Millist tüüpi tehisintellekti komponente on vaja rakenduses kasutada? Kas on vaja ka komponente, mis ei sisalda tehisintellekti? Kas komponendid on eraldiseisvad või sõltuvad üksteisest (keerulisem agentsem disan)?

> ...

### 🟢 4.2 Tehisintellekti lahenduste valik
Milliseid mudeleid on plaanis kasutada? Kas kasutada valmis teenust (API) või arendada/majutada mudelid ise?

> ...

### 🟢 4.3 Kuidas hinnata rakenduse headust?
Kuidas rakenduse arenduse käigus hinnata rakenduse headust?

> ...

### 🟢 4.4 Rakenduse arendus
Milliste sammude abil on plaanis/on võimalik rakendust järk-järgult parandada (viibadisain, erinevte mudelite testimine jne)?

> ...


### 🟢 4.5 Riskijuhtimine
Kuidas maandatakse tehisintellektispetsiifilisi riske (hallutsinatsioonid, kallutatus, turvalisus)?

> ...

<br>
<br>

## 🔵 5. Tulemuste hindamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🔵 5.1 Vastavus eesmärkidele
Kuidas hinnata, kas rakendus vastab seatud eesmärkidele?

> ...

<br>
<br>

## 🟣 6. Juurutamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🟣 6.1 Integratsioon
Kuidas ja millise liidese kaudu lõppkasutaja rakendust kasutab? Kuidas rakendus olemasolevasse töövoogu integreeritakse (juhul kui see on vajalik)?

> ...

### 🟣 6.2 Rakenduse elutsükkel ja hooldus
Kes vastutab süsteemi tööshoidmise ja jooksvate kulude eest? Kuidas toimub rakenduse uuendamine tulevikus?

> ...