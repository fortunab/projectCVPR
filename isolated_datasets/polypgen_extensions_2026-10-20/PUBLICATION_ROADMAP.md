# Plan pentru CVPR main și o variantă medicală pentru CBMS

Actualizat: 20 octombrie 2026. Recomandări de cercetare; acest plan nu autorizează și nu pornește experimente suplimentare față de cele două deja aprobate. Rulările curente și concluziile lor sunt documentate separat în RESULTS.md.

## Evaluarea proiectului

Avem un protocol util: audit al provenienței, separare pe secvențe/pacienți, selecție exclusiv pe surse, clasificatori comparați pe aceleași caracteristici, ablații și salvarea predicțiilor. În experimentele PolypGen încheiate anterior, CBMP nu a depășit consecvent controalele puternice. O reprezentare DINOv2 mai bună ar demonstra un efect al encoderului; beneficiul specific CBMP trebuie evaluat prin contraste între clasificatori care folosesc același encoder.

Prioritatea pentru CVPR este o contribuție de cunoaștere clară: un mecanism nou și justificat sau o analiză nouă, bine delimitată și cu implicații generale pentru recunoașterea vizuală. Combinarea unui encoder existent cu clustering ponderat și prototipuri multiple necesită o diferențiere atentă față de literatura existentă. Un nume nou pentru combinație și un număr mare de experimente nu demonstrează această diferențiere.

[CVPR 2027](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) cere explicit cercetare originală de calitate și include imagistica medicală și transferul între domenii. Ghidul de evaluare al [ediției 2026](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines) subliniază contribuția, corectitudinea tehnică și impactul; depășirea SOTA nu este singurul criteriu. Acest ghid al ediției anterioare oferă context, fără să înlocuiască regulile finale pentru 2027.

## Ordinea recomandată

1. **Încheierea și interpretarea experimentelor 4–5.** Comparăm CBMP cu linear/cosine și multiprototipuri, separat pentru fiecare encoder și centru. Raportăm diferențe pereche, incertitudine la nivelul unității de grupare disponibile, cu limitele ei, și variație între semințele de optimizare. Pentru PolypGen, secvențele nu stabilesc independența pacienților. Verificăm și ce configurații aleg sursele: dacă metoda revine frecvent la K=1 sau beta=0, acest comportament limitează afirmația că multiplicarea sau echilibrarea centrelor este mecanismul util.

2. **Audit de noutate.** Construim un tabel cu lucrările cele mai apropiate: obiectiv, tip de prototipuri, utilizarea domeniilor sursă, utilizarea datelor țintă, encoder înghețat/adaptat, mecanism nou și protocol de selecție. Diferența propusă trebuie formulată matematic și legată de o eroare observată. Termeni precum „domain-robust” nu înlocuiesc demonstrarea robusteții.

3. **O singură ipoteză centrală.** O direcție precisă de investigat: când amplifică ponderarea egală a centrelor estimările nesigure, construite din prea puține grupuri independente, și poate o regularizare bazată pe suportul efectiv să reducă această eroare? Stabilitatea modurilor clasei între pacienți poate oferi o măsură verificabilă. Trebuie păstrate și aparițiile rare relevante, inclusiv cele care lipsesc din unele centre. Aceasta este o ipoteză candidată, nu o afirmație de noutate sau eficacitate. Verificăm inclusiv dacă o regulă propusă este doar o reformulare a beta sau a unei interpolări de centroizi deja cunoscute. Mecanismul concret se alege după auditul literaturii și analiza erorilor; secvența pozitivă unică de la C5 nu demonstrează singură cauza eșecului antrenării pe surse.

4. **Protocol pentru dezvoltare și confirmare.** Rezultatele externe inspectate acum sunt valide pentru metoda fixată înaintea lor. Dacă dezvoltăm ulterior o metodă pe baza acestor rezultate, evaluarea ei pe aceleași ținte devine exploratorie. Rezervăm o cohortă/instituție externă nefolosită sau un benchmark independent înainte de următoarele iterații. Mai multe imagini de la aceiași pacienți nu creează automat un test independent.

5. **Comparații care pot respinge ipoteza.** Păstrăm clasificatorii logistici regularizați și variantele ponderate existente. Controalele prioritare suplimentare sunt o bancă simplă de prototipuri centru–clasă, cu capacitate și agregare comparabile, și un cap liniar GroupDRO regularizat pe aceleași caracteristici înghețate, cu grupuri și selecție fixate pe surse. [Lucrarea GroupDRO](https://arxiv.org/abs/1911.08731) arată de ce regularizarea este importantă pentru generalizarea pe grupurile dificile. Un clasificator neliniar simplu poate verifica dacă eventualul câștig provine doar din capacitatea frontierei. Algoritmii care adaptează encoderul aparțin unui braț separat, cu optimizare și buget comparabile. Nu comparăm o metodă cu encoder adaptat cu controale înghețate pentru a atribui tot câștigul prototipurilor.

6. **Ablație mecanistică.** Pentru orice componentă nouă, testăm adăugarea și eliminarea ei, separat de numărul de prototipuri, echilibrarea centrelor și encoder. Măsurăm stabilitatea la re-eșantionarea pacienților, utilizarea prototipurilor, concentrarea pe centre, clasele rare și costul. O măsură de „invarianță” are valoare numai dacă se leagă de discriminarea clasei și de rezultatele țintă, fără a le folosi la selecție.

7. **Redactare din rezultate.** Actualizăm introducerea, obiectivele și contribuțiile după alegerea afirmației finale. Secțiunea experimentală trebuie să explice exact ce testează fiecare comparație. Raportăm cazurile nefavorabile, limitele provenienței PolypGen și caracterul de subset CAMELYON. Închiderea proiectului include scripturi de reproducere, dependențe, configurații, manifestele permise de licențe, tabele și figuri generate automat.

## Lucrări apropiate care stabilesc cerința de noutate

Acestea sunt suprapuneri conceptuale verificate în surse primare, nu afirmații că metodele sunt identice. Ele trebuie analizate în detaliu înainte de alegerea contribuției.

| Lucrare | Suprapunere relevantă | Diferență de păstrat în comparație |
|---|---|---|
| [COMEN, CVPR 2022](https://arxiv.org/abs/2203.13006) | Folosește centroizi de clasă și relații între prototipuri pentru generalizare între domenii. | Descoperă domenii latente și învață reprezentarea; CBMP folosește identități de centre și caracteristici înghețate. |
| [HPDR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Rethinking_Generalizable_Face_Anti-spoofing_via_Hierarchical_Prototype-guided_Distribution_Refinement_in_CVPR_2024_paper.html) | Prototipuri multiple și clasificare ghidată de prototipuri pentru recunoaștere binară în domenii nevăzute. | Face anti-spoofing, reprezentare hiperbolică învățată și relații ierarhice; nu același encoder/protocol. |
| [InfProto, IJCV 2024](https://link.springer.com/article/10.1007/s11263-024-02158-9) | Modelează distribuții de prototipuri condiționate de clasă pentru generalizare în imagini medicale, inclusiv polipi. | Segmentare dintr-un singur domeniu sursă, învățare a reprezentării și clasificator adaptiv; folosește și statistici ale instanței la inferență, spre deosebire de banca noastră statică. DOI: 10.1007/s11263-024-02158-9. |
| [PIKACHU, MIDL 2026](https://proceedings.mlr.press/v315/kumar26b.html) | Encoder înghețat utilizat în sarcini medicale, prototipuri din medii normalizate și similaritate cosine cu temperatură. | Adaptare few-shot din suport etichetat pentru sarcină; nu protocolul nostru fără adaptare la centrul țintă. Testează atât backbones medicale, cât și generice. |

Consecință: „prototipuri multiple”, „encoder înghețat”, „clasificare cosine” sau „prototipuri medicale” nu pot constitui singure revendicarea de noutate.

## Poziționarea pe conferințe

| Conferință | Potrivire și contribuție necesară |
|---|---|
| CVPR main | Potrivire tematică pentru generalizare și imagistică medicală. Avem nevoie de o contribuție originală cu interes pentru comunitatea de viziune și de dovezi care o izolează de encoder, selecție și compoziția datelor. Un studiu negativ poate avea valoare dacă descoperirea este nouă, explicată și generală; simpla absență a unui câștig nu este suficientă. |
| CBMS | Alternativă firească pentru un studiu medical riguros: robustețe între instituții, limitări, cost, calibrare și tipare de eroare relevante aplicației. Contribuția și limitele trebuie să rămână clare. Rezultatele retrospective de clasificare nu stabilesc automat utilitate clinică prospectivă. |
| ITiCSE | Necesită o contribuție despre predarea sau învățarea informaticii. Ar presupune o întrebare educațională și dovezi educaționale proprii. Experimentul actual de clasificare medicală nu îndeplinește singur acest scop. |

Potrivirea CBMS este susținută de [apelul oficial 2026](https://2026.cbms-conference.org/call-for-papers/), care include procesarea imaginilor medicale, machine vision și AI în sănătate. Cerințele ediției viitoare trebuie verificate când sunt publicate. Pentru ITiCSE, [apelul 2026](https://iticse.acm.org/2026/call-for-papers/) definește explicit educația în informatică, iar [pagina 2027](https://iticse.acm.org/2027/) confirmă identitatea conferinței. CBMS și ITiCSE nu sunt doar niveluri de dificultate diferite; urmăresc contribuții diferite.

## Calendar practic până la CVPR 2027

Conform [apelului oficial CVPR 2027](https://cvpr.thecvf.com/Conferences/2027/CallForPapers), înregistrarea lucrării este pe **10 noiembrie 2026**, depunerea pe **16 noiembrie**, iar suplimentul pe **23 noiembrie**. De la 20 octombrie sunt 27 de zile până la depunere.

Verificăm acum și existența profilurilor OpenReview actualizate pentru toți coautorii. Apelul avertizează că profilurile noi fără email instituțional pot necesita până la două săptămâni de moderare; acest pas nu trebuie lăsat pentru săptămâna înregistrării.

| Interval orientativ | Rezultat de obținut |
|---|---|
| 20–23 octombrie | Experimentele curente închise, interpretare și audit al lucrărilor apropiate. |
| 24–30 octombrie | O ipoteză precisă; dacă este justificată, o implementare minimă și ablații pe surse/date de dezvoltare. |
| 31 octombrie–5 noiembrie | Evaluare de confirmare fixată înainte de scoruri, controale și verificarea limitelor. Accesul la date și resursele de calcul pot impune un calendar mai lung. |
| 6–9 noiembrie | Decizie realistă pentru CVPR main, redactare completă, verificare critică și înregistrare înaintea termenului. |
| 10–16 noiembrie | Corecturi și verificări ale versiunii finale, reproducibilitate, depunere. |

Acesta este un calendar de prioritizare, nu o promisiune că o contribuție nouă poate fi demonstrată în acest interval. Dacă noutatea sau confirmarea independentă rămân insuficiente, construim o lucrare medicală solidă pentru o ediție potrivită CBMS și continuăm direcția CVPR cu timpul necesar. Refolosirea lucrării la altă conferință trebuie planificată în acord cu politicile privind depunerile simultane.
