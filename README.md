# Unsicherheit mitliefern: eine Schranke für die Kosten der gewählten Entscheidung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-entscheidungs-schranke-demo.streamlit.app/)**

Stück 9 der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel ist das **5 × 5 Gitter** aus Stück 3 ([spo-kuerzester-weg-demo](https://github.com/sebastian-hanisch/spo-kuerzester-weg-demo): Wege von links oben nach rechts unten, 40 Bögen, 70 Wege): Ein lineares Modell (Kleinste Quadrate, KQ) sagt die Bogenkosten voraus, ein exakter Löser wählt daraus den kürzesten Weg. Die Demo trainiert dabei nichts Neues. Sie fragt, was man zu der gewählten Entscheidung **mitliefern** kann: eine obere **Schranke für die tatsächlichen Kosten dieses Weges**, die ihr Niveau (hier 90 %) im Mittel über Kalibrierungen hält.

## Warum dieses Problem

Wer eine Entscheidung aus einer Prognose ableitet, will wissen, wie teuer sie höchstens wird, nicht wie falsch die Prognose höchstens ist. Der naheliegende Weg ist, je Bogen eine Fehlerschranke zu bestimmen und sie über die Bögen des gewählten Weges zu addieren. Der Weg der Demo legt die **konforme Schranke direkt auf die Kosten des gewählten Weges** (der Score ist die tatsächlichen minus die vorhergesagten Wegkosten). Die Kernfrage: **Überdeckt die Schranke auf den Wegkosten das Ziel, und wie breit ist sie im Vergleich zur Summe der Kantenschranken?** Dazu eine zweite Beobachtung: Die Vorhersage genau des Weges, den der Löser wählt, ist **optimistisch**.

Alle Zahlen dieses Abschnitts stehen mit Zelle, Statistik und Zahl der Läufe in „Befunde“ (Rauschstufe 0,25 / 0,5 / 0,75, Grad der Wahrheit 1 / 4 / 6, je Zelle 30 Zufalls-Seeds, 1000 Trainings-, 1000 Kalibrier- und 5000 Testfälle, Ziel-Überdeckung 90 %):

- **Die Schranke auf den Wegkosten trifft das Ziel, und die Summe der Kantenschranken überdeckt mehr als nötig.** Im Median über die 30 Seeds liegt die Überdeckung der Wegkosten-Schranke in allen 9 Zellen zwischen 90,0 und 90,2 %; die der Summe der Kantenschranken zwischen 93,4 und 100,0 %.
- **Dafür ist die Summe breiter.** Im Median der Zellen ist die Wegkosten-Schranke in allen 9 Zellen schmaler, um 2,6 bis 42,0 Prozentpunkte der Vorhersagekosten; der kleinste Abstand steht bei Grad 6 mit Rauschstufe 0,25 (51,8 gegen 54,3 %), und nur dort ist die Wegkosten-Schranke in einzelnen Läufen nicht schmaler (6 von 30 Seeds). Mit Bonferroni wird die Summe in allen 270 Läufen noch breiter und überdeckt 99,7 bis 100,0 %.
- **Die Vorhersage des gewählten Weges ist im Median in allen 9 Zellen zu optimistisch** (0,1 bis 7,4 % der vorhergesagten Kosten) und der Optimismus wächst mit dem Grad der Wahrheit (der das lineare Modell zunehmend fehlspezifiziert; ob die Fehlspezifikation allein die Ursache ist, ist nicht isoliert). Bei einem zufällig gewählten Weg derselben Vorhersage liegt der Median im Betrag bei höchstens 0,13 %. Die naheliegende Deutung ist der **Gewinnerfluch**: Der Löser wählt, was in der Prognose billig aussieht.
- **Die Garantie ist marginal.** Mit 20 Kalibrierfällen liegt die Überdeckung von 90 % der Kalibrierungen zwischen 77,9 und 98,1 % (5. bis 95. Perzentil bei Grad 4, Rauschstufe 0,5; das Minimum liegt bei 57,0 %), im Mittel über alle Kalibrierungen bei 90,5 %. Der Median der Breite sinkt von 20 auf 100 Kalibrierfälle um 0,9 bis 6,6 Punkte und ändert sich von 100 auf 1000 um höchstens 0,7 Punkte (bei Grad 6 steigt er leicht); die Streuung der Breite zwischen den Kalibrierungen schrumpft durchgehend.

**Dopplungsgrenze zu den Zeitreihen-Demos.** Die Idee „Summenfehler statt Einzelfehler“ ist nicht neu: In [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) (Sicherheitsbestand über ein Schutzintervall) verfehlt die Faustregel, die die Streuung der Ein-Tages-Fehler hochskaliert, das Ziel (α = 88,5 % bei Ziel 95 %, Mittelwert der vier Prognosequellen), während die Regeln auf den Summenfehlern es treffen (konform 95,0 %); dort steht sie als Herkunft dieser Idee. Konforme Prognoseintervalle und ihre Rangregel stehen in [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo) (dort: bei einem zweiseitigen 95-%-Intervall mit 30 Werten im Fenster verlangt die obere Stufe 97,5 % den 31. von 30 Werten). **Neu in dieser Demo** sind die **Auswahl durch den Löser** (der Optimismus des gewählten Weges und die fehlende Garantie der Summe für den gewählten Weg) und der **Score auf dem Wert der gewählten Entscheidung**. Die Konzeptquelle, Cortes-Gomez et al., *Decision-Focused Uncertainty Quantification* (arXiv 2410.01767; in der zweiten Fassung umbenannt in *Utility-Directed Conformal Prediction*), ist ein einziges Papier, das ich **nur auf der Ebene des Abstracts** gelesen habe: konforme Mengen unter Berücksichtigung des Entscheidungsverlusts bei erhaltener Abdeckung. Die Demo bildet es nicht nach; was dort im Einzelnen bewiesen oder gemessen wird, ist hier nicht wiedergegeben.

## Modell

**Welt** (aus Stück 3 übernommen; dort aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft): Merkmale $x \sim \mathcal N(0, I_5)$; eine Gewichtsmatrix $B \in \{0,1\}^{40 \times 5}$ mit fairen Münzwürfen; mittlere Kosten $m_j(x) = ((Bx)_j/\sqrt{5} + 3)^{\deg} / 3{,}5^{\deg} + 1$; beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$, **unabhängig je Bogen und Fall**. Grad 1 ist linear in $x$ (ein lineares Modell kann die Wahrheit exakt abbilden), Grad 4 und 6 weichen zunehmend davon ab. $e$ ist die Rauschstufe (0,25, 0,5 oder 0,75 in der App).

**Modell und Löser.** Lineares Modell mit Achsenabschnitt (6 × 40 = 240 Parameter), Kleinste Quadrate über `numpy.linalg.lstsq`, angepasst auf 1000 Trainingsfällen. Der Löser (Rückwärtsrechnung über die 25 Knoten, Gleichstand: rechts vor unten) wählt den Weg $w(\hat c)$ mit den vorhergesagten Kosten $\hat c^\top w$ und den tatsächlichen Kosten $c^\top w$.

**Score und Schranke (Verfahren C).** Auf $n$ Kalibrierfällen, die weder im Training noch im Test sind, ist der Score $s = c^\top w(\hat c) - \hat c^\top w(\hat c)$. Mit dem Rang $k = \lceil (n+1)(1-\alpha) \rceil$ ist $s_{(k)}$ der $k$-kleinste Score und $U = \hat c^\top w + s_{(k)}$ die Schranke für einen neuen Fall; bei $k > n$ ist sie unendlich (keine Garantie). Sind Kalibrier- und Testfälle austauschbar, gilt $P(s_{n+1} \le s_{(k)}) \ge k/(n+1) \ge 1 - \alpha$, **im Mittel über Fälle und Kalibrierungen**, nicht für eine einzelne Kalibrierung und nicht je Merkmalswert. Das Niveau ist $1 - \alpha = 0{,}9$.

**Gegenproben.** *Summe der Kantenschranken (A):* je Bogen $e$ das Konformquantil $q_e$ der Bogenfehler $c_e - \hat c_e$ über alle Kalibrierfälle zum Niveau 0,9, addiert über die 8 Bögen des gewählten Weges: $U_A = \hat c^\top w + \sum_{e:\,w_e = 1} q_e$. *Mit Bonferroni (B):* dasselbe mit dem Niveau $1 - 0{,}1/8$ je Bogen. Für A und B folgt aus den Bogenquantilen keine Garantie für den gewählten Weg, denn die Quantile gelten für feste Bögen, der Weg aber ist vom Löser aus der Vorhersage ausgewählt; ihre Überdeckung ist hier gemessen, nicht zugesichert.

**Messgrößen** (auf den 5000 Testfällen): **Überdeckung** = Anteil der Fälle mit tatsächlichen Wegkosten $\le U$; **Breite** = mittleres $U$ minus mittlere vorhergesagte Wegkosten, in Prozent der mittleren vorhergesagten Wegkosten; **Optimismus** = (mittlere tatsächliche minus mittlere vorhergesagte Wegkosten) in Prozent der mittleren vorhergesagten.

Die Herleitung steht im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

- **Handbeispiel** (Experiment 1, von Hand gerechnet): fünf Kalibrier-Scores $-0{,}5;\ 0{,}2;\ 0{,}9;\ 1{,}4;\ 2{,}0$, $n = 5$. Niveau 0,5: $k = \lceil 6 \cdot 0{,}5 \rceil = 3$, Schranke 0,9. Niveau 0,8: $k = 5$, Schranke 2,0. Niveau 0,9: $k = 6 > 5$, Schranke **unendlich** (mit 5 Werten ist höchstens $5/6$ erreichbar). Für einen neuen Fall mit vorhergesagten Kosten 10,0 heißt das Obergrenze 10,9 beziehungsweise 12,0.
- **Messung.** Jede Zelle (Grad 1 / 4 / 6 × Rauschstufe 0,25 / 0,5 / 0,75) läuft mit den festen Zufalls-Seeds 0 bis 29; je Seed 1000 Trainings-, 1000 Kalibrier- und 5000 Testfälle aus getrennten Teilströmen. Angegeben werden **Median und Spanne (Minimum bis Maximum)** über die Seeds, nie ein einzelner Seed. Die Bänder stehen vorgerechnet in `esc_bands.json` und sind mit `python tools/measure_bands.py` reproduzierbar (ein Prozess, feste Seeds). Zufall nur über den ganzzahligen SplitMix64 aus `esc_rng.py`, nie Pythons `hash()`. Die App rechnet die eine Welt der Seitenleiste live (Modell, Wege, Quantile) und zeigt daneben die Bänder.
- **Kontrolle des Optimismus (Zusatzmessung dieser Demo, nicht in der Vormessung).** Dieselbe Vorhersage wird zusätzlich an einem **zufällig gewählten Weg** bewertet (gleichverteilt über die 70 Wege, fester Zufallsstrom): So zeigt sich, ob der Optimismus an der Auswahl durch den Löser hängt oder schon an der Prognose.
- **Kalibrierfälle (Experiment 4).** Die 1000 Kalibrierfälle jedes Laufs werden in getrennte Blöcke von 20, 100 oder 1000 Fällen zerlegt (50, 10 und 1 Block je Seed, bei 30 Seeds 1500, 300 und 30 Kalibrierungen je Zelle); jeder Block ergibt eine Schranke, deren Überdeckung auf den 5000 Testfällen gemessen wird. Zum Vergleich dienen die Rang-Formel $k/(n+1)$ und die Beta-Verteilung der Überdeckung einer festen Kalibrierung bei Scores ohne Gleichstand (Parameter $k$ und $n + 1 - k$).
- **Stabilitätsprobe (Rauschstufe 0,5).** Dieselben drei Zellen mit 100 weiteren Zufalls-Seeds 30 bis 129 (nur Zusammenfassungen), um die Zahlen der 30 Seeds zu prüfen.
- **Herkunft des Codes.** `esc_rng.py`, `esc_grid.py` und `esc_world.py` sind aus Stück 3 kopiert und umbenannt (die Herkunft steht im Modulkopf; `esc_world.py` ist um den Kalibrier-Teilstrom erweitert), `esc_training.py` enthält nur dessen KQ-Teil. Nichts wird aus einer anderen Demo importiert.
- **Zufallsquelle und Plattform.** Die Ausgabe des SplitMix64 hängt nur von Seed und Position ab. Plattformabhängig ist der Schritt zu Normalverteilten (`log` und `cos` der C-Bibliothek, letzte Stelle); die Tests prüfen gemessene Zahlen deshalb als Bänder und Ungleichungen, exakt nur Rechnungen ohne Messung (Handbeispiel, Rangformel, Hand-Instanz).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus `esc_bands.json` (eigene Messung: KQ-Modell, 1000 Trainings-, 1000 Kalibrier- und 5000 Testfälle, Ziel-Überdeckung 0,9, Zufalls-Seeds 0 bis 29; Überdeckung, Breite und Optimismus in Prozent als Median [Minimum; Maximum] über die 30 Seeds), wenn nichts anderes dabeisteht. Die Tabellen werden von `tests/readme_tables.py` aus der JSON-Datei erzeugt, und `tests/test_claims.py` prüft, dass jede Zeile wörtlich hier steht; die Aussagen darunter stehen dort als Bänder, Zählungen und Ungleichungen.

### Überdeckung der drei Schranken (Ziel 90 %)

| Grad | Rauschstufe | Summe der Kantenschranken | mit Bonferroni | konform auf den Wegkosten |
|---|---|---|---|---|
| 1 | 0,25 | 100,0 [100,0; 100,0] | 100,0 [100,0; 100,0] | 90,2 [87,4; 92,2] |
| 1 | 0,50 | 100,0 [100,0; 100,0] | 100,0 [100,0; 100,0] | 90,2 [87,7; 91,6] |
| 1 | 0,75 | 100,0 [100,0; 100,0] | 100,0 [100,0; 100,0] | 90,2 [87,9; 91,5] |
| 4 | 0,25 | 95,6 [94,0; 96,8] | 100,0 [99,8; 100,0] | 90,2 [88,0; 92,1] |
| 4 | 0,50 | 98,3 [96,9; 99,2] | 100,0 [100,0; 100,0] | 90,1 [88,2; 92,4] |
| 4 | 0,75 | 99,4 [98,4; 99,9] | 100,0 [100,0; 100,0] | 90,0 [88,4; 91,9] |
| 6 | 0,25 | 93,4 [91,4; 94,6] | 99,9 [99,7; 100,0] | 90,0 [87,5; 92,1] |
| 6 | 0,50 | 95,1 [92,8; 96,4] | 99,9 [99,8; 100,0] | 90,0 [87,6; 92,2] |
| 6 | 0,75 | 96,6 [94,6; 97,8] | 100,0 [99,8; 100,0] | 90,0 [88,1; 91,9] |

### Breite der drei Schranken (in Prozent der vorhergesagten Wegkosten)

| Grad | Rauschstufe | Summe der Kantenschranken | mit Bonferroni | konform auf den Wegkosten |
|---|---|---|---|---|
| 1 | 0,25 | 20,9 [20,6; 21,3] | 27,5 [27,1; 28,0] | 6,8 [5,9; 7,4] |
| 1 | 0,50 | 41,8 [41,3; 42,9] | 55,0 [54,2; 56,2] | 13,8 [12,2; 15,0] |
| 1 | 0,75 | 63,0 [62,1; 64,6] | 82,8 [81,4; 84,7] | 21,0 [18,7; 22,5] |
| 4 | 0,25 | 29,9 [26,2; 33,4] | 82,4 [69,3; 95,2] | 21,2 [15,4; 26,0] |
| 4 | 0,50 | 46,4 [44,6; 49,0] | 113,7 [99,5; 130,6] | 24,2 [19,7; 29,4] |
| 4 | 0,75 | 66,8 [64,2; 69,3] | 152,0 [135,7; 169,9] | 29,0 [24,9; 34,5] |
| 6 | 0,25 | 54,3 [42,1; 67,3] | 195,2 [148,7; 239,3] | 51,8 [38,8; 68,0] |
| 6 | 0,50 | 65,3 [57,0; 76,1] | 225,6 [181,9; 271,2] | 52,9 [38,6; 68,9] |
| 6 | 0,75 | 80,3 [75,1; 90,5] | 267,9 [223,2; 327,5] | 56,1 [43,1; 73,6] |

### Optimismus der Vorhersage des gewählten Weges (in Prozent der vorhergesagten Wegkosten)

| Grad | Rauschstufe | gewählter Weg | zufällig gewählter Weg (Kontrolle) |
|---|---|---|---|
| 1 | 0,25 | 0,1 [-0,1; 0,5] | 0,0 [-0,2; 0,2] |
| 1 | 0,50 | 0,4 [-0,1; 1,2] | 0,0 [-0,5; 0,5] |
| 1 | 0,75 | 0,9 [0,0; 1,9] | -0,1 [-0,7; 0,7] |
| 4 | 0,25 | 2,1 [0,7; 3,6] | 0,0 [-0,6; 0,7] |
| 4 | 0,50 | 2,3 [0,8; 3,7] | 0,0 [-0,7; 0,8] |
| 4 | 0,75 | 2,6 [0,9; 4,0] | 0,0 [-0,8; 1,1] |
| 6 | 0,25 | 6,8 [3,6; 12,4] | 0,1 [-1,2; 1,5] |
| 6 | 0,50 | 7,0 [3,5; 12,4] | 0,0 [-1,4; 1,7] |
| 6 | 0,75 | 7,4 [3,7; 12,8] | -0,1 [-1,5; 2,0] |

| Frage | Befund |
|---|---|
| **Trifft die Wegkosten-Schranke das Ziel?** | Im Median in allen 9 Zellen zwischen 90,0 und 90,2 %. Die einzelnen Läufe (9 Zellen mal 30 Seeds, je eine Kalibrierung von 1000 Fällen) liegen zwischen 87,4 und 92,4 %. Eine einzelne Kalibrierung trifft das Ziel also nicht genau (Experiment 4). |
| **Überdeckt die Summe der Kantenschranken mehr als nötig?** | Im Median in allen 9 Zellen mindestens 93,4 %; bei Grad 1 liegt sie in allen 90 Läufen bei mindestens 99,98 % (in 80 bei 100,0 %), bei Grad 4 und 6 in keinem der 180 Läufe unter 91,4 %. Das Ziel von 90 % verfehlt sie in keinem der 270 Läufe; eine Garantie dafür gibt es nicht (siehe Modell). |
| **Wie viel schmaler ist die Wegkosten-Schranke?** | Im Median der Zellen schmaler in allen 9 Zellen, um 2,6 (Grad 6, Rauschstufe 0,25: 51,8 gegen 54,3 %) bis 42,0 Prozentpunkte (Grad 1, Rauschstufe 0,75: 21,0 gegen 63,0 %); bei Rauschstufe 0,5: 28,1 Punkte bei Grad 1, 22,2 bei Grad 4 und 12,4 bei Grad 6 (Differenzen der ungerundeten Mediane, daher um bis zu 0,1 von den Differenzen der gerundeten Tabellenwerte verschieden). Je Lauf schmaler in 264 von 270 Läufen; die 6 Ausnahmen liegen alle bei Grad 6 mit Rauschstufe 0,25. |
| **Hilft Bonferroni?** | Nein, für die Breite: in allen 270 Läufen breiter als die einfache Summe (bei Grad 6 und Rauschstufe 0,5: 225,6 gegen 65,3 %) und mit 99,7 bis 100,0 % in jedem Lauf weit über dem Ziel. |
| **Ist die Vorhersage des gewählten Weges optimistisch?** | Im Median in allen 9 Zellen ja: 0,1 bis 7,4 %. Der Optimismus wächst in allen drei Rauschstufen mit dem Grad (bei Rauschstufe 0,5: 0,4 / 2,3 / 7,0 % bei Grad 1 / 4 / 6) und in allen drei Graden mit der Rauschstufe (bei Grad 4: 2,1 / 2,3 / 2,6 %). Je Lauf positiv in 258 von 270 Läufen; die 12 anderen liegen bei Grad 1 (9 bei Rauschstufe 0,25, 3 bei 0,5). |
| **Entsteht der Optimismus durch die Auswahl?** | Die Kontrolle stützt es: bei einem zufällig gewählten Weg derselben Vorhersage liegt der Median in allen 9 Zellen im Betrag bei höchstens 0,13 %, die Läufe zwischen −1,5 und 2,0 %. Das ist eine Deutung (Gewinnerfluch), kein Versuch, der die Auswahl allein variiert. |

### Kalibrierfälle (Rauschstufe 0,5; Überdeckung der Wegkosten-Schranke über alle Kalibrierungen der 30 Zufalls-Seeds)

| Grad | Kalibrierfälle n | Kalibrierungen | Überdeckung Mittel | Rang-Formel k / (n + 1) | 5. bis 95. Perzentil | Anteil unter 90 % | Anteil unter 90 % (Beta-Formel) | Breite Median [Min; Max] |
|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 1500 | 90,59 % | 90,48 % | 79,3 bis 98,3 % | 39 % | 39 % | 14,8 [2,9; 29,6] % |
| 1 | 100 | 300 | 90,17 % | 90,10 % | 84,9 bis 94,6 % | 45 % | 45 % | 13,9 [8,7; 20,0] % |
| 1 | 1000 | 30 | 90,04 % | 90,01 % | 88,2 bis 91,4 % | 43 % | 48 % | 13,8 [12,2; 15,0] % |
| 4 | 20 | 1500 | 90,49 % | 90,48 % | 77,9 bis 98,1 % | 37 % | 39 % | 26,9 [1,6; 75,2] % |
| 4 | 100 | 300 | 90,11 % | 90,10 % | 84,8 bis 94,3 % | 44 % | 45 % | 24,2 [13,3; 41,0] % |
| 4 | 1000 | 30 | 90,05 % | 90,01 % | 88,4 bis 91,8 % | 47 % | 48 % | 24,2 [19,7; 29,4] % |
| 6 | 20 | 1500 | 90,55 % | 90,48 % | 78,6 bis 98,1 % | 38 % | 39 % | 58,8 [4,6; 201,8] % |
| 6 | 100 | 300 | 90,17 % | 90,10 % | 84,9 bis 94,5 % | 45 % | 45 % | 52,2 [25,4; 96,0] % |
| 6 | 1000 | 30 | 89,99 % | 90,01 % | 87,9 bis 91,7 % | 53 % | 48 % | 52,9 [38,6; 68,9] % |

| Frage | Befund |
|---|---|
| **Gilt die Garantie im Mittel?** | Ja, im Rahmen der Messung: Der Mittelwert der Überdeckung über die Kalibrierungen liegt in allen 9 Zeilen höchstens 0,11 Punkte neben der Rang-Formel $k/(n+1)$ (bei 1000 Kalibrierfällen nur 30 Kalibrierungen: 89,99 bis 90,05 %). |
| **Gilt sie für die einzelne Kalibrierung?** | Nein. Mit 20 Kalibrierfällen liegt die Überdeckung in 37 bis 39 % der Kalibrierungen unter 90 % (Beta-Formel: 39 %) und im 5. bis 95. Perzentil zwischen 77,9 und 98,3 %; mit 1000 Kalibrierfällen noch in 43 bis 53 % der 30 Kalibrierungen unter 90 %, aber nur knapp (5. Perzentil 87,9 bis 88,4 %). |
| **Sinkt die Breite mit mehr Kalibrierfällen?** | Der Median sinkt von 20 auf 100 Fälle um 0,9 (Grad 1), 2,7 (Grad 4) und 6,6 Punkte (Grad 6) und ändert sich von 100 auf 1000 Fälle um höchstens 0,7 Punkte (Grad 6: Anstieg von 52,2 auf 52,9 %). Die Spanne über die Kalibrierungen schrumpft durchgehend: Grad 4: 1,6 bis 75,2 % bei n = 20, 13,3 bis 41,0 % bei n = 100, 19,7 bis 29,4 % bei n = 1000. Siehe „Befunde und Korrekturen“. |

### Stabilitätsprobe (Rauschstufe 0,5, 100 weitere Zufalls-Seeds 30 bis 129)

| Grad | Überdeckung Summe | Überdeckung Wegkosten | Breite Summe | Breite Wegkosten | Optimismus |
|---|---|---|---|---|---|
| 1 | 100,0 [100,0; 100,0] | 90,1 [87,1; 92,3] | 41,8 [40,9; 42,6] | 13,8 [12,6; 15,1] | 0,4 [-0,5; 1,0] |
| 4 | 98,3 [96,9; 99,4] | 90,1 [87,6; 92,6] | 46,5 [42,3; 49,2] | 23,8 [18,4; 33,4] | 2,1 [0,6; 4,0] |
| 6 | 95,0 [92,7; 97,4] | 90,2 [87,5; 93,0] | 65,1 [51,5; 77,4] | 51,5 [34,0; 75,7] | 7,3 [2,1; 14,4] |

Die Medianzahlen der 30 Seeds bestätigen sich mit 100 anderen Seeds: Die Überdeckung der Wegkosten-Schranke weicht im Median um höchstens 0,2 Punkte ab, ihre Breite um höchstens 1,5 Punkte (Grad 6: 52,9 gegen 51,5 %), der Optimismus um höchstens 0,3 Punkte.

### Prüfungen des Lösers und der Rechenwege

- **Löser gegen Aufzählung der 70 Wege:** Vorbefund aus Stück 3 (300 Instanzen, größte Abweichung im Wert 7·10⁻¹⁵); im Repo läuft die kleine Fassung (`tests/test_oracle_entscheidungs_schranke.py`).
- **Rang $k$ gegen exakte Bruchrechnung:** $n = 0$ bis 600, alle Niveaus in Schritten von 0,001, 600399 Prüfungen, keine Abweichung (Scratch-Lauf, nicht im Repo).
- **Konformquantil gegen die Aufzählung der Rangstatistik:** 5000 Zufallsfolgen mit gerundeten Werten (viele Gleichstände) und 8 Niveaus, 40000 Prüfungen, keine Abweichung (Scratch-Lauf).
- **Die drei Schranken gegen reine Python-Schleifen mit eigener Wegaufzählung:** 200 Zufallsinstanzen auf Gittern von 2 × 2 bis 4 × 4, größte Abweichung 7·10⁻¹⁵, unendliche Schranken stimmen überein (Scratch-Lauf). Dabei fiel ein Fehler der Referenz auf, nicht der Demo: das Bonferroni-Niveau $1 - 0{,}2/6$ als Gleitkommazahl 0,9666666666666667 verschiebt den Rang von 29 auf 30 und macht die Schranke bei $n = 29$ fälschlich unendlich, wenn die Referenz die Zahl als Dezimalbruch liest. Die Demo rundet $(n+1) \cdot \text{Niveau}$ vor dem Aufrunden auf 9 Stellen; die Referenz rechnet das Niveau als exakten Bruch.

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vormessung reproduzieren sich nicht auf die Stelle.** Die Vormessung (`entscheidungsorientiert-planung/vormessung/eo_v10_konform.py`, Rohausgabe `out_eo_v10_konform.txt`) benutzte eine andere Zufallsquelle (den Generator `default_rng` von numpy statt SplitMix64); hier gelten die neuen Zahlen. Rauschstufe 0,5, Median über 30 Seeds, Vormessung gegen diese Messung:

| Grad | Verfahren | Vormessung: Überdeckung / Breite | diese Messung: Überdeckung / Breite |
|---|---|---|---|
| 1 | Summe der Kantenschranken | 100,0 % / 41,6 % | 100,0 % / 41,8 % |
| 1 | mit Bonferroni | 100,0 % / 54,9 % | 100,0 % / 55,0 % |
| 1 | konform auf den Wegkosten | 90,0 % / 13,9 % | 90,2 % / 13,8 % |
| 1 | Optimismus des gewählten Weges | 0,40 % | 0,36 % |
| 4 | Summe der Kantenschranken | 98,4 % / 46,0 % | 98,3 % / 46,4 % |
| 4 | mit Bonferroni | 100,0 % / 111,0 % | 100,0 % / 113,7 % |
| 4 | konform auf den Wegkosten | 90,1 % / 23,0 % | 90,1 % / 24,2 % |
| 4 | Optimismus des gewählten Weges | 2,13 % | 2,27 % |
| 6 | Summe der Kantenschranken | 95,1 % / 64,1 % | 95,1 % / 65,3 % |
| 6 | mit Bonferroni | 99,9 % / 212,8 % | 99,9 % / 225,6 % |
| 6 | konform auf den Wegkosten | 90,1 % / 51,2 % | 90,0 % / 52,9 % |
| 6 | Optimismus des gewählten Weges | 7,15 % | 6,99 % |

  Die Überdeckungen weichen um höchstens 0,2 Punkte ab, die Breiten um höchstens 12,8 Punkte (Bonferroni bei Grad 6: 212,8 gegen 225,6 %), bei der Wegkosten-Schranke um höchstens 1,7 Punkte. Die Rangfolge der Verfahren und die Aussagen sind in allen drei Graden dieselben.
- **„Mit mehr Kalibrierfällen sinkt die Breite“ (Plan, Experiment 4) trägt nur zum Teil.** Der Median der Breite sinkt von 20 auf 100 Fälle (Grad 1: 14,8 auf 13,9 %; Grad 4: 26,9 auf 24,2 %), danach nicht mehr (13,8 und 24,2 % bei 1000 Fällen; Grad 6 steigt von 52,2 auf 52,9 %). Bei 20 Fällen ist der Median am größten, weil der Rang $k = 19$ von 20 ein hohes Quantil wählt (Deutung, nicht isoliert). Was bis 1000 Fälle durchgehend schrumpft, ist die Streuung der Breite zwischen den Kalibrierungen (Grad 4: 1,6 bis 75,2 % bei 20 Fällen, 19,7 bis 29,4 % bei 1000) und die der Überdeckung. Der Text der App nennt beides.
- **Rauschstufe 0 wird nicht angeboten.** Der Plan ließ „Deg und Rauschen wählbar“. Bei Grad 1 ohne Rauschen ist KQ bis auf Rundung exakt, alle Scores liegen bei 10⁻¹⁴ oder darunter, die Breite ist null und die Überdeckung bedeutungslos. Die App bietet 0,25 / 0,5 / 0,75; der Randfall „Rauschen null: Schranke = Vorhersage plus 0“ ist im Test mit exakten Scores abgedeckt (`tests/test_conformal.py`, `tests/test_runs.py`).
- **Der Plan nannte „Summe der Kantenschranken ≥ Schranke auf den Wegkosten“ ausdrücklich nicht als Satz.** Gemessen ist: die Summe ist im Median in allen 9 Zellen breiter und in 264 von 270 Läufen; die 6 Ausnahmen stehen bei Grad 6 mit Rauschstufe 0,25. Der Test prüft die Zählung, nicht einen Satz.
- **Die Kontrolle mit zufälligem Weg ist neu** (Plan: keine Auswahl unter mehreren Entscheidungsregeln, kein Vergleichsweg). Sie stützt die Deutung „Gewinnerfluch“, belegt sie aber nicht als Ursache.
- **Nur als Hinweis gesagt:** Dass die Summe bei Grad 6 und Rauschstufe 0,25 fast so breit ist wie die Wegkosten-Schranke (54,3 gegen 51,8 %), erklärt die Demo nicht; vermutlich überwiegt dort der Modellfehler, der sich in beiden Schranken in voller Höhe zeigt (nicht isoliert).

## Ehrliche Grenzen

- **Marginal, nicht je Fall.** Die Garantie gilt im Mittel über Fälle und Kalibrierungen. Die Schranke hat dieselbe Breite für alle Fälle; eine Abhängigkeit der Breite von den Merkmalen (lokale Schranken) ist nicht gemessen. Eine einzelne Kalibrierung kann unter dem Ziel liegen (Experiment 4).
- **Austauschbarkeit ist vorausgesetzt und nicht verletzt worden.** Kalibrier- und Testfälle stammen aus derselben Verteilung. Ein Verteilungswechsel ist in dieser Demo nicht gemessen; die Zeitreihen-Demo [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo) misst ihn für Prognoseintervalle.
- **Rauschen unabhängig je Bogen.** So ist es gemessen, und die Summe der Kantenschranken wird dadurch konservativ (die Bogenfehler heben sich teilweise auf; sie überdeckt mehr als nötig und ist breiter). Bei korreliertem Rauschen ist nichts gemessen.
- **Ein Modell, ein Gitter, eine Kostenformel.** Nur ein lineares KQ-Modell, nur das 5 × 5 Gitter, nur Grad 1 / 4 / 6 und Rauschstufe 0,25 / 0,5 / 0,75. Ein auf den Regret trainiertes Modell (SPO+, Stück 3) ist nicht gemessen; ob der Optimismus dort kleiner oder größer ist, ist offen.
- **Eine Entscheidungsregel.** Der Löser wählt den kürzesten Weg unter der Vorhersage. Eine Auswahl unter mehreren Entscheidungsregeln, robuste Entscheidungen oder Nebenbedingungen mit Wahrscheinlichkeit (Chance-Constraints) sind nicht Gegenstand; die Schranke ist eine Aussage **nach** der Entscheidung, keine robuste Entscheidung.
- **Nur die obere Schranke.** Keine untere Schranke und keine Aussage über den Regret gegen den besten Weg.
- **Die Summe der Kantenschranken ist nicht widerlegt, sondern ohne Garantie.** Ihre Überdeckung liegt in den 270 Läufen zwischen 91,4 und 100,0 %; dass sie das Ziel in diesen Zellen hält, folgt nicht aus einem Satz.
- **Ursachen sind Deutungen:** „Gewinnerfluch“ als Ursache des Optimismus (Kontrolle mit zufälligem Weg, kein isolierender Versuch); die Erklärung der engen Breiten bei Grad 6 und Rauschstufe 0,25 (siehe oben).
- **Zahlen sind Messungen unter Windows.** Unter Linux (CI und Streamlit Cloud) können einzelne Werte in der letzten Stelle abweichen (Normalverteilung aus `log` und `cos`); die Tests prüfen Bänder.
- **Quelle nur auf Abstract-Ebene.** Decision-Focused Uncertainty Quantification (arXiv 2410.01767, in der zweiten Fassung Utility-Directed Conformal Prediction; ein einziges Papier) ist nicht nachgebaut und nicht im Volltext gelesen; die Welt (Gitter, Kostenformel) ist aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut (Stück 3), nicht gegen sie geprüft.
- **Keine Aussage über Rechenzeit.**

## Verifikation

Die Suite hat **185 Tests**.

- **Kern gegen Handrechnung:** Handbeispiel (Ränge 3 / 5 / 6, Schranken 0,9 / 2,0 / unendlich, garantierte Überdeckung 1/2 und 5/6), n = 1 und n = 0, Rangrand ohne Gleitkommarest ($n = 9$ und 19, 99); die drei Schranken an einer Hand-Instanz mit zwei Wegen und an einer mit zwei Bögen je Weg; Löser auf 2 × 2, 3 × 3 und 5 × 5 (kopiert aus Stück 3: `tests/test_grid.py`); Welt (`tests/test_world.py`).
- **Orakel** (`tests/test_oracle_entscheidungs_schranke.py`, unabhängige Rechenwege): Löser gegen die Aufzählung der 70 Wege (rekursiv erzeugt, auch mit Gleichständen); Rang gegen exakte Bruchrechnung für $n = 0$ bis 300 und Niveaus in Hundertsteln; Konformquantil gegen die Aufzählung der Rangstatistik durch Zählen (300 Folgen, Gleichstände); Überdeckung gegen die **geschlossene Formel** im einfachen Modell mit zwei Wegen und Normalrauschen (Mittel über 3000 Kalibrierungen gleich $k/(n+1)$ innerhalb des Bandes; Überdeckung einer festen Kalibrierung Beta-verteilt, Anteil unter 0,9 und Streuung gegen die Beta-Verteilungsfunktion); die drei Schranken gegen reine Python-Schleifen auf 3 × 3 und 4 × 4; Überdeckung, Breite und Optimismus gegen Schleifen; KQ gegen die Normalengleichungen; SplitMix64 gegen eine reine Python-Ganzzahlrechnung (`tests/test_rng.py`).
- **Eigenschaften:** Schranke monoton im Niveau; Verschieben aller Scores um $c$ verschiebt die Schranke um $c$, Skalieren skaliert sie; die Reihenfolge der Scores ist gleichgültig; Gleichstände ändern die Rangregel nicht; Bonferroni nie schmaler als die einfache Summe; Kalibrier- und Testfälle der Welt sind gleich verteilt, das Rauschen je Bogen unkorreliert.
- **Randfälle:** n = 0, 1, 2 Kalibrierfälle, alle Scores gleich, Niveau am Rand des Reglers (0,5 und 0,95, jede Stufe von 3 bis 6), Niveau außerhalb von (0, 1), Rauschen null mit exakten Scores, negative Scores, Block größer als die Kalibrierung, Block zu klein für das Niveau, kleinstes Gitter (`tests/test_edge_cases.py`, `tests/test_conformal.py`, `tests/test_runs.py`).
- **Claims:** jede Zahl dieser README steht in `tests/test_claims.py` als wörtliche Tabellenzeile, Band, Zählung oder Ungleichung; welche Aussage in welchem Test steht, zeigt die Tabelle dort im Kopf. Messungen, die nicht im Test stehen, sind oben als „Scratch-Lauf“ gekennzeichnet.
- **Verdikte:** reine Funktionen, Schwellen und jede Zahl im Text (`tests/test_verdicts.py`); jedes Preset zeigt die erwarteten Verdikte (`tests/test_app.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler an den Grenzen, Permalink mit Einrasten, PDF), Layout (Schnellstart und Kostenwelt in der Seitenleiste, Regler der Experimente im Hauptbereich, fester Kopf, aktives Preset hervorgehoben), Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Seitenleiste (Schnellstart, Kostenwelt), Ergebnis, Kernfrage, vier Experimente, Grenzen, Formeln |
| `esc_rng.py`, `esc_grid.py` | SplitMix64 und Gitter mit Löser (kopiert aus Stück 3) |
| `esc_world.py`, `esc_training.py` | Kostenwelt mit Kalibrierdaten (nach Stück 3), KQ-Modell (KQ-Teil aus Stück 3) |
| `esc_conformal.py` | Konformquantil, Rang, die drei Schranken, Überdeckung, Breite, Optimismus (Kern, nur numpy) |
| `esc_runs.py` | Lauf einer Welt, Läufe über Seeds, Kalibrierblöcke, Kontrolle mit zufälligem Weg |
| `esc_bands.py`, `esc_bands.json` | lädt die vorgerechneten Bänder; Messung (`tools/measure_bands.py`) |
| `esc_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `esc_evaluation.py` | Handbeispiel, Zahlen der Experimente, Verdikte (reine Funktionen), Tabellen aus den Bändern |
| `esc_layout.py` | Fester Kopf (Stylesheet) |
| `esc_presets.py` | Permalink (Begrenzen und Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `esc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `esc_pdf_export.py` | PDF-Export der aktuellen Welt (fpdf2) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenprobe im Test) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly, pandas, fpdf2 |
| `tools/measure_bands.py` | erzeugt `esc_bands.json` |
| `tests/` | Tests (siehe „Verifikation“); `tests/readme_tables.py` erzeugt die Tabellen dieser README |

## Bewusst nicht umgesetzt

- **Nachbau von Decision-Focused Uncertainty Quantification (arXiv 2410.01767, in der zweiten Fassung Utility-Directed Conformal Prediction):** nur Einordnung auf Abstract-Ebene.
- **Merkmalsabhängige Schranken** (lokale oder normalisierte Scores): die Breite ist konstant.
- **Verteilungswechsel und nachgeführte Kalibrierung** (adaptive konforme Verfahren): siehe forecast-interval-demo für Prognoseintervalle.
- **Korreliertes Rauschen, andere Gitter, andere Kostenformeln, ein auf SPO+ trainiertes oder flexibles Modell.**
- **Robuste Optimierung und Chance-Constraints:** die Schranke ist eine Aussage nach der Entscheidung, keine robuste Entscheidung.
- **Eine Wahl unter mehreren Entscheidungsregeln.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
