# Workshop-Aufgabe: Reinforcement Learning mit CarRacing

In dieser Aufgabe arbeitest du mit einer vorgegebenen Reinforcement-Learning-Basis auf der Gymnasium-Umgebung CarRacing.
Dein Ziel ist nicht, sofort einen perfekten Agenten zu bauen. Dein Ziel ist, den gegebenen Code zu verstehen, gezielt zu erweitern und durch eigene Experimente zu begruenden, welche Aenderungen sinnvoll sind.

Verwendete Umgebung:
- Gymnasium Box2D CarRacing
- Environment-ID: `CarRacing-v3`
- Beobachtung: RGB-Bild mit Form `96 x 96 x 3`
- Aktionen: In dieser Aufgabe verwendest du zunaechst eine kleine diskrete Aktionsmenge

## Deine Lernziele
Nach der Bearbeitung solltest du in der Lage sein:

1. Zustand, Aktion und Reward in einem RL-Problem sauber zu benennen.
2. die Grundidee eines Policy-Gradient-Verfahrens zu erklaeren.
3. die Rolle von Bildvorverarbeitung fuer RL zu beschreiben.
4. den gegebenen Code gezielt zu erweitern.
5. eigene Experimente zu planen, durchzufuehren und zu interpretieren.

## Deine Materialien
Du arbeitest mit den folgenden Dateien:

- [README.md](README.md): dieses Aufgabenblatt
- [requirements.txt](requirements.txt): benoetigte Pakete
- [src/workshop_rl/train_reinforce.py](src/workshop_rl/train_reinforce.py): der gegebene Starter-Code

Die Datei [src/workshop_rl/train_reinforce.py](src/workshop_rl/train_reinforce.py) ist bewusst stark kommentiert. Lies sie nicht nur oberflaechlich, sondern nutze sie wie einen roten Faden.
## Vorbereitung

Aktiviere zuerst die virtuelle Umgebung und installiere die Pakete:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

Hinweis:
- Die Versionen in [requirements.txt](requirements.txt) sind bewusst auf eine typische Python-3.10-Umgebung abgestimmt.
- Unter macOS kann die Installation von Box2D und pygame etwas dauern.

## Erste Ausfuehrung
Starte zunaechst einen sehr kurzen Trainingslauf:

```bash
python -m src.workshop_rl.train_reinforce --episodes 5 --render none
```

Dabei entstehen standardmaessig zwei Dateien:
- `car_racing_policy.pt` mit den Modellgewichten
- `training_rewards.png` mit einer Reward-Kurve

Wenn du spaeter einen Evaluationslauf mit Darstellung sehen willst, kannst du zum Beispiel Folgendes ausfuehren:
```bash
python -m src.workshop_rl.train_reinforce --mode evaluate --render human --episodes 3
```

## Aufgabe 1: Verstehe das gegebene RL-Setup
Lies den Code in [src/workshop_rl/train_reinforce.py](src/workshop_rl/train_reinforce.py) aufmerksam durch.

Beantworte danach fuer dich oder in deiner Gruppe die folgenden Fragen:
1. Was ist in diesem Projekt der Zustand?
2. Was ist die Aktion?
3. Was ist der Reward?
4. Warum ist CarRacing schwieriger als einfache Umgebungen wie CartPole?
5. Warum ist Lernen direkt aus Pixeln anspruchsvoll?

## Aufgabe 2: Erklaere den Datenfluss im Code
Arbeite den Ablauf im Starter-Code nach und notiere dir zu jeder Station kurz, was dort passiert.

Diese Teile solltest du dabei gezielt untersuchen:
1. `WorkshopConfig`
2. `DiscreteActionMapper`
3. `preprocess_observation`
4. `PolicyNetwork`
5. `run_episode`
6. `update_policy_from_episode`

Leitfragen:
1. Wo wird die Beobachtung vorbereitet?
2. Wo wird die Aktion ausgewaehlt?
3. Wo werden Rewards gesammelt?
4. Wo findet das eigentliche Lernen statt?
5. Welche Stellen im Code sind bewusst fuer spaetere Erweiterungen offen gelassen?

## Aufgabe 3: Fuehre einen ersten Basisversuch durch
Bevor du den Code veraenderst, sollst du ein Ausgangsexperiment durchfuehren.

Fuehre zum Beispiel diesen Befehl aus:
```bash
python -m src.workshop_rl.train_reinforce --episodes 10 --render none
```

Beobachte dabei:
1. Wie sehen die episodischen Rewards aus?
2. Schwanken die Ergebnisse stark?
3. Faehrt der Agent erkennbar sinnvoll oder eher zufaellig?
4. Wie plausibel ist das Verhalten fuer ein frisch gestartetes REINFORCE-Training?

Halte deinen Eindruck kurz fest.
## Kurzerklaerung zur Methode

Der gegebene Code trainiert eine Policy $\pi(a|s)$. Das ist eine Wahrscheinlichkeitsverteilung ueber Aktionen gegeben einen Zustand.

Hier ist der Zustand ein Bild der Rennstrecke. Das Netzwerk entscheidet also anhand von Pixeln, welche diskrete Aktion als naechstes ausgefuehrt wird.
Die Policy wird mit REINFORCE aktualisiert. Die Grundidee ist:

$$
\nabla J(\theta) \approx \sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t
$$

Intuition:
- Aktionen, die spaeter zu gutem Return fuehren, sollen wahrscheinlicher werden.
- Aktionen, die spaeter zu schlechtem Return fuehren, sollen unwahrscheinlicher werden.

Du musst diese Formel nicht auswendig lernen, aber du solltest die Richtung verstehen: gute Entscheidungen werden verstaerkt, schlechte abgeschwaecht.
## Warum der Aktionsraum hier diskret ist

CarRacing unterstuetzt eigentlich kontinuierliche Steuerung mit Lenken, Gas und Bremse. Fuer diese Workshop-Aufgabe wird das absichtlich vereinfacht.

Du arbeitest zunaechst mit einer kleinen Menge diskreter Aktionen wie:
1. nichts tun
2. rechts lenken
3. links lenken
4. Gas geben
5. bremsen

Diese Vereinfachung ist didaktisch sinnvoll, weil du dadurch den Policy-Gradienten, das Sampling und den Loss leichter nachvollziehen kannst.
## Deine Hauptaufgabe: Erweitere den gegebenen Code

Der Starter-Code ist absichtlich nicht fertig optimiert. An mehreren Stellen gibt es Erweiterungspotenzial. Deine Aufgabe ist es, gezielt Aenderungen umzusetzen und deren Wirkung zu untersuchen.

Bearbeite mindestens drei der folgenden Aufgaben.
## Erweiterungsaufgabe A: Beobachtungen besser aufbereiten

Arbeite in `preprocess_observation`.

Dein Auftrag:

1. Reduziere das Bild auf Graustufen.
2. Schneide testweise den unteren Bereich mit den Anzeigen weg.
3. Vergleiche das Verhalten mit der urspruenglichen Version.

Fragen zur Auswertung:
1. Hilft die einfachere Eingabe?
2. Geht dabei wichtige Information verloren?
3. Wirkt das Training stabiler oder schneller?

## Erweiterungsaufgabe B: Return-Normalisierung aktivieren

Arbeite in `update_policy_from_episode`.

Im Code findest du bereits einen TODO-Hinweis fuer die Return-Normalisierung.

Dein Auftrag:

1. Normalisiere die Returns auf Mittelwert `0` und Standardabweichung `1`.
2. Fuehre einen neuen Trainingslauf durch.
3. Vergleiche das Verhalten mit dem Basisversuch.

Fragen zur Auswertung:
1. Warum kann die Normalisierung die Gradienten stabilisieren?
2. Siehst du einen klaren Unterschied oder nur eine leichte Tendenz?

## Erweiterungsaufgabe C: Entropy-Bonus einbauen

Arbeite ebenfalls in `update_policy_from_episode`.

Dein Auftrag:

1. Nutze die bereits gespeicherte Entropie der Aktionsverteilung.
2. Erweitere den Loss um einen kleinen Entropy-Bonus.
3. Teste verschiedene Koeffizienten, zum Beispiel `0.001`, `0.01`, `0.02`.

Fragen zur Auswertung:
1. Was bedeutet hohe Entropie in diesem Kontext?
2. Erkundet der Agent dadurch mehr?
3. Wird das Verhalten hilfreicher oder nur chaotischer?

## Erweiterungsaufgabe D: Action Repeat untersuchen

Arbeite mit dem CLI-Parameter `--action-repeat`.

Dein Auftrag:

1. Teste mindestens drei verschiedene Werte, zum Beispiel `1`, `2`, `4`, `6`.
2. Vergleiche Reward und Fahrverhalten.

Fragen zur Auswertung:
1. Ist haeufigeres Entscheiden automatisch besser?
2. Warum kann ein zu grosses `action_repeat` problematisch sein?

## Erweiterungsaufgabe E: Den Aktionsraum verbessern

Arbeite in `DiscreteActionMapper` und im Netzwerk-Setup.

Dein Auftrag:

1. Erweitere die Aktionsmenge um Kombinationen wie "links + Gas" und "rechts + Gas".
2. Passe die Anzahl der Netzwerkausgaenge an.
3. Teste, ob Kurven dadurch besser gefahren werden koennen.

Fragen zur Auswertung:
1. Warum sind kombinierte Aktionen in CarRacing oft wichtig?
2. Hilft der groessere Aktionsraum oder macht er das Lernen schwerer?

## Erweiterungsaufgabe F: Domain Randomization testen

Arbeite mit dem Schalter `--domain-randomize`.

Dein Auftrag:

1. Aktiviere die visuelle Variation der Umgebung.
2. Vergleiche Training und Verhalten mit der Standardkonfiguration.

Fragen zur Auswertung:
1. Lernt das Modell robustere visuelle Merkmale?
2. Wird das Training anfangs schwieriger?

## Erweiterungsaufgabe G: Von REINFORCE zu Actor-Critic

Diese Aufgabe ist anspruchsvoller und eignet sich gut fuer schnelle Gruppen oder eine Fortsetzung.

Dein Auftrag:

1. Erweitere das Modell um einen zusaetzlichen Value-Kopf.
2. Nutze einen Advantage-Term statt nur des reinen Returns.
3. Fuege einen Value-Loss hinzu.

Fragen zur Auswertung:
1. Warum kann das die Varianz reduzieren?
2. Welche zusaetzliche Lernaufgabe kommt dadurch ins System?

## Dokumentiere deine Experimente

Zu jeder bearbeiteten Aenderung solltest du kurz notieren:

1. Welche Aenderung hast du gemacht?
2. Was war deine Hypothese vor dem Lauf?
3. Mit welchen Parametern hast du getestet?
4. Was ist im Training passiert?
5. Wie interpretierst du das Ergebnis?

Wichtig:
Ein gutes Ergebnis ist in dieser Aufgabe nicht automatisch ein hoher Score. Ein gutes Ergebnis ist vor allem eine saubere Beobachtung mit plausibler Begruendung.

## Reflexionsfragen

Beantworte zum Abschluss einige der folgenden Fragen:
1. Warum ist CarRacing fuer tabellarisches RL ungeeignet?
2. Warum hat REINFORCE oft eine hohe Varianz?
3. Welche Rolle spielt die Beobachtungsvorverarbeitung?
4. Welche Rolle spielt die Wahl des Aktionsraums?
5. Welche deiner Aenderungen war aus deiner Sicht am wirksamsten?
6. Welche Information im Bild scheint fuer den Agenten besonders wichtig zu sein?

## Erwartung an deine Bearbeitung
Du musst aus diesem Starter keinen perfekten Rennfahrer machen.

Wichtig ist, dass du zeigen kannst:
1. dass du den Code verstanden hast,
2. dass du gezielte Aenderungen vornehmen kannst,
3. dass du Experimente sinnvoll vergleichst,
4. dass du deine Beobachtungen fachlich begruenden kannst.

## Optionale Zusatzaufgaben

Wenn du frueher fertig bist, kannst du zusaetzlich an einem dieser Punkte arbeiten:
1. Frame-Stacking einbauen
2. Reward-Clipping ausprobieren
3. weitere Plot-Ausgaben speichern
4. Modellgewichte gezielt laden und vergleichen
5. eine sauberere Evaluationsroutine schreiben

## Abschluss

Bearbeite diese Workshop-Aufgabe nicht als reines Coding-Problem, sondern als Experimentieraufgabe.

Die zentrale Frage lautet nicht nur: "Funktioniert es?"

Die wichtigere Frage lautet: "Warum verhaelt sich der Agent genau so, und was sagt mir das ueber Reinforcement Learning?"
# Reinforcement Learning Workshop: CarRacing mit Gymnasium

Dieses Mini-Projekt ist als Workshop-Aufgabe gedacht. Es soll nicht einfach nur "laufen", sondern als gut lesbarer Ausgangspunkt dienen, an dem man Reinforcement Learning Schritt fuer Schritt erklaeren, diskutieren und erweitern kann.

Die verwendete Umgebung ist CarRacing aus Gymnasium:

- Umgebung: Gymnasium Box2D CarRacing
- ID: `CarRacing-v3`
- Beobachtung: RGB-Bild mit Form `96 x 96 x 3`
- Aktionen: Wir verwenden in diesem Workshop die diskrete Variante mit 5 Aktionen

Warum gerade diese Umgebung?

- Sie ist visuell anschaulich.
- Der Agent arbeitet direkt mit Bilddaten.
- Man kann sehr gut ueber Beobachtungen, Aktionen, Rewards und Exploration sprechen.
- Es ist anspruchsvoll genug, damit echte Designentscheidungen in der RL-Pipeline wichtig werden.

## Lernziele

Nach diesem Workshop sollen die Teilnehmenden:

1. die Grundidee eines RL-Setups erklaeren koennen,
2. ein einfaches Policy-Gradient-Verfahren verstehen,
3. sehen, warum Beobachtungsvorverarbeitung wichtig ist,
4. einfache RL-Experimente planen, durchfuehren und interpretieren,
5. bestehenden Code gezielt erweitern koennen.

## Projektstruktur

- [README.md](README.md): Workshop-Handout, Erklaerungen und Aufgaben
- [requirements.txt](requirements.txt): benoetigte Python-Pakete
- [src/workshop_rl/train_reinforce.py](src/workshop_rl/train_reinforce.py): gut dokumentierter Starter-Code

## Installation

Falls noch nicht geschehen, zuerst die virtuelle Umgebung aktivieren und dann die Pakete installieren.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Hinweis fuer macOS: Box2D, pygame und gymnasium koennen bei manchen Python-Setups etwas laenger fuer die Installation brauchen.

Die Versionen in [requirements.txt](requirements.txt) sind bewusst so gepinnt, dass sie in einer typischen Python-3.10-Umgebung zusammen funktionieren.

## Schnellstart

Ein kurzer Testlauf ohne Rendering:

```bash
python -m src.workshop_rl.train_reinforce --episodes 5
```

Nach dem Training werden standardmaessig zwei Dateien geschrieben:

- `car_racing_policy.pt` mit den Modellgewichten
- `training_rewards.png` mit einer einfachen Reward-Kurve

Ein Evaluationslauf mit Fensterdarstellung:

```bash
python -m src.workshop_rl.train_reinforce --mode evaluate --render human --episodes 3
```

## Die RL-Idee in diesem Projekt

Wir trainieren eine Policy $\pi(a|s)$, also eine Wahrscheinlichkeitsverteilung ueber Aktionen gegeben einen Zustand.

Hier ist der Zustand kein abstrakter Zahlenvektor, sondern ein Bild der Rennstrecke. Das Netzwerk schaut also auf Pixel und entscheidet, welche diskrete Aktion als naechstes ausgefuehrt wird.

Die Policy wird mit REINFORCE aktualisiert. Die zentrale Idee ist:

$$
\nabla J(\theta) \approx \sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t
$$

Dabei ist:

- $\theta$ die Menge der Modellparameter,
- $a_t$ die ausgewaehlte Aktion zum Zeitpunkt $t$,
- $s_t$ die Beobachtung zum Zeitpunkt $t$,
- $G_t$ der diskontierte Return ab Schritt $t$.

Die Intuition ist einfach:

- Fuehrt eine Aktion spaeter zu gutem Return, dann soll sie wahrscheinlicher werden.
- Fuehrt eine Aktion zu schlechtem Return, dann soll sie unwahrscheinlicher werden.

## Warum diskrete Aktionen?

CarRacing erlaubt kontinuierliche Aktionen fuer Lenken, Gas und Bremse. Fuer einen ersten Workshop ist das aber oft unnoetig komplex. Deshalb arbeiten wir mit einer kleinen festen Aktionsmenge:

1. nichts tun
2. nach rechts lenken
3. nach links lenken
4. Gas geben
5. bremsen

Das ist nicht optimal, aber didaktisch sehr praktisch:

- Die Policy kann direkt als kategoriale Verteilung modelliert werden.
- Sampling und Loss werden leichter nachvollziehbar.
- Die Teilnehmenden koennen sich auf RL-Konzepte konzentrieren.

## Wie der Starter-Code aufgebaut ist

Die Datei [src/workshop_rl/train_reinforce.py](src/workshop_rl/train_reinforce.py) ist absichtlich linear und "buchartig" aufgebaut.

### 1. Konfiguration

In `WorkshopConfig` stehen die wichtigsten Hyperparameter zentral an einer Stelle:

- Anzahl Episoden
- Lernrate
- Discount-Faktor `gamma`
- Action Repeat
- maximale Episodenlaenge
- Geraet `cpu` oder `mps`

Didaktische Idee: Erst alle Stellschrauben sichtbar machen, dann einzeln veraendern.

### 2. Diskrete Aktionen

Die Klasse `DiscreteActionMapper` uebersetzt eine diskrete Aktions-ID in ein konkretes Aktionsarray fuer die Umgebung.

Beispiel:

- `0 -> [0.0, 0.0, 0.0]`
- `1 -> [0.6, 0.0, 0.0]`
- `3 -> [0.0, 0.7, 0.0]`

Didaktische Idee: Die Policy arbeitet auf einer einfachen Menge `{0,1,2,3,4}`, aber die Umgebung bekommt trotzdem das erwartete Aktionsformat.

### 3. Beobachtungsvorverarbeitung

Die Funktion `preprocess_observation` konvertiert ein RGB-Bild in einen PyTorch-Tensor im Format `C x H x W` und skaliert die Werte auf `[0,1]`.

Hier steckt absichtlich Erweiterungspotenzial:

- Graustufen statt RGB
- Zuschneiden des unteren Dashboard-Bereichs
- Downsampling
- Frame-Stacking fuer Bewegungsinformation

Genau dort koennen die Workshop-Mitglieder spaeter experimentieren.

### 4. Policy-Netzwerk

Das Netzwerk ist ein kleines CNN mit drei Faltungsbloeken und einem MLP-Kopf.

Die Ausgaben sind `logits`, also noch keine normalisierten Wahrscheinlichkeiten. Erst beim Sampeln wird daraus eine kategoriale Verteilung.

Didaktische Idee: Das Modell ist klein genug, um lesbar zu bleiben, aber realistisch genug, um Bilddaten zu verarbeiten.

### 5. Episode sammeln

`run_episode` fuehrt eine komplette Episode aus:

- Beobachtung vorverarbeiten
- Aktion samplen
- `action_repeat` mal ausfuehren
- Reward aufsummieren
- Log-Wahrscheinlichkeiten und Rewards speichern

Damit ist sauber getrennt zwischen:

- Datensammlung in der Umgebung
- Lernen aus der gesammelten Trajektorie

### 6. Policy-Update

`update_policy_from_episode` berechnet:

1. diskontierte Returns,
2. daraus den REINFORCE-Loss,
3. einen Gradientenschritt.

Im Code sind bewusst Stellen markiert, an denen man spaeter Verbesserungen einbauen kann, zum Beispiel:

- Return-Normalisierung
- Entropy-Bonus
- Baseline oder Value Head

## Workshop-Ablaufvorschlag

### Teil A: Verstehen

Lest gemeinsam den Code und besprecht:

1. Was ist hier Zustand, Aktion und Reward?
2. Warum ist CarRacing schwieriger als CartPole?
3. Warum ist Lernen aus Pixeln aufwaendiger?
4. Warum kann REINFORCE hohe Varianz haben?

### Teil B: Erste Ausfuehrung

Fuehrt einen kurzen Lauf aus:

```bash
python -m src.workshop_rl.train_reinforce --episodes 10
```

Beobachtet:

- episodische Rewards
- ob der Agent ueberhaupt Strecke trifft
- wie stark die Resultate zwischen Laeufen schwanken

Diskussion:

- Warum ist das Training instabil?
- Welche Rolle spielt Exploration?
- Welche Rolle spielt die Vorverarbeitung?

## Konkrete Workshop-Aufgaben

Die folgenden Aufgaben sind so formuliert, dass die Teilnehmenden direkt am gegebenen Code arbeiten.

### Aufgabe 1: Beobachtungen besser aufbereiten

Ziel: Die Eingabe fuer das Netzwerk einfacher und informativer machen.

Arbeitsschritte:

1. Veraendert `preprocess_observation` so, dass das Bild auf Graustufen reduziert wird.
2. Schneidet den unteren Bereich mit den Anzeigen testweise weg.
3. Vergleicht das Training mit und ohne diese Aenderung.

Leitfragen:

- Hilft weniger visuelle Komplexitaet?
- Geht dabei wichtige Information verloren?
- Wird das Training stabiler oder schneller?

### Aufgabe 2: Return-Normalisierung aktivieren

Ziel: Die Varianz des Policy-Gradienten reduzieren.

Arbeitsschritte:

1. Sucht in `update_policy_from_episode` die TODO-Stelle zur Return-Normalisierung.
2. Normalisiert die Returns auf Mittelwert `0` und Standardabweichung `1`.
3. Vergleicht Lernkurven vor und nach der Aenderung.

Leitfragen:

- Warum kann das die Gradienten stabilisieren?
- Verbessert sich die Lernbarkeit sofort oder nur leicht?

### Aufgabe 3: Entropy-Bonus hinzufuegen

Ziel: Mehr Exploration erzwingen.

Arbeitsschritte:

1. Speichert zusaetzlich zur Log-Wahrscheinlichkeit auch die Entropie der Aktionsverteilung.
2. Erweitert den Loss um einen kleinen Entropy-Term.
3. Testet verschiedene Koeffizienten, zum Beispiel `0.001`, `0.01`, `0.02`.

Leitfragen:

- Was bedeutet hohe oder niedrige Entropie in diesem Kontext?
- Wird der Agent mutiger oder chaotischer?

### Aufgabe 4: Action Repeat untersuchen

Ziel: Verstehen, wie fein oder grob der Agent steuert.

Arbeitsschritte:

1. Testet `action_repeat = 1`, `2`, `4`, `6`.
2. Vergleicht Reward und Fahrverhalten.

Leitfragen:

- Ist haeufigeres Entscheiden immer besser?
- Warum kann zu grosses `action_repeat` problematisch sein?

### Aufgabe 5: Kleine Aktionsmenge erweitern

Ziel: Bessere Steuerbarkeit erreichen.

Arbeitsschritte:

1. Erweitert `DiscreteActionMapper` um kombinierte Aktionen wie "links + Gas" und "rechts + Gas".
2. Passt die Anzahl der Netzwerkausgaenge an.
3. Testet, ob der Agent Kurven besser nehmen kann.

Leitfragen:

- Warum sind kombinierte Aktionen in CarRacing oft wichtig?
- Wird die Policy dadurch hilfreicher oder schwerer zu lernen?

### Aufgabe 6: Domain Randomization ausprobieren

Ziel: Robustheit gegen visuelle Veraenderungen testen.

Arbeitsschritte:

1. Aktiviert `domain_randomize=True` in der Umgebung.
2. Vergleicht das Verhalten mit der Standardumgebung.

Leitfragen:

- Lernt das Modell robustere visuelle Merkmale?
- Wird das Training anfangs schwieriger?

### Aufgabe 7: Aus REINFORCE ein Actor-Critic machen

Ziel: Das Projekt in Richtung modernerer RL-Verfahren erweitern.

Arbeitsschritte:

1. Ergaenzt einen zweiten Netzwerkkopf fuer einen Value-Output.
2. Ersetzt `G_t` im Policy-Loss durch einen Advantage-Term.
3. Fuegt einen Value-Loss hinzu.

Leitfragen:

- Warum sollte das die Varianz senken?
- Welche neue Lernaufgabe kommt dadurch hinzu?

## Experimentvorlage fuer die Teilnehmenden

Fuer jede Aenderung sollte kurz protokolliert werden:

1. Welche Aenderung wurde gemacht?
2. Welche Hypothese gab es vorher?
3. Welche Hyperparameter wurden benutzt?
4. Was ist im Training passiert?
5. Was ist die plausibelste Erklaerung?

Ein gutes Ergebnis ist im Workshop nicht zwingend ein hoher Score. Ein gutes Ergebnis kann auch sein:

- eine saubere Hypothese,
- ein klarer Vergleich,
- eine nachvollziehbare Interpretation.

## Mögliche Diskussionsfragen

1. Warum ist CarRacing fuer tabellarisches RL ungeeignet?
2. Warum reicht eine einzige Episode selten fuer stabile Updates?
3. Welche Nachteile hat REINFORCE gegenueber PPO oder A2C?
4. Welche Rolle spielt die Wahl der Aktionen fuer den Lernerfolg?
5. Welche Information im Bild ist wirklich wichtig?

## Hinweise zur Erwartungshaltung

Dieses Projekt ist ein didaktischer Starter und kein stark optimierter SOTA-Trainer. Das ist Absicht.

Die Teilnehmenden sollen:

- den Datenfluss verstehen,
- die mathematische Idee wiedererkennen,
- konkrete Verbesserungen selbst implementieren,
- beobachten, dass RL stark von Designentscheidungen abhaengt.

## Naechste sinnvolle Erweiterungen

Wenn der Workshop weitergefuehrt werden soll, bieten sich diese Schritte an:

1. Frame-Stacking
2. Reward-Clipping
3. Baseline oder Actor-Critic
4. Replay fuer Off-Policy-Verfahren
5. Wechsel zu PPO

Viel wichtiger als eine perfekte Schlussperformance ist, dass man jede Aenderung fachlich begruenden kann.