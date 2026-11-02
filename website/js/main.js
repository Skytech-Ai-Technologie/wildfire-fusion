/* FORFIRE-AI website — trilingual (FR/EN/IT) content, lightbox, mobile nav.
   Default language: French (matching italietunisie.eu/fr/forfire-ai). */
"use strict";

const I18N = {
/* ============================================================ FR ==== */
fr: {
  "topbar.project": "Interreg NEXT Italie–Tunisie",
  "brand.strap": "Détection et prévention des incendies de forêts par l’IA — forêts des zones côtières Tunisie–Italie.",
  "nav.project": "Le projet", "nav.contrib": "Contributions", "nav.arch": "Architecture",
  "nav.results": "Résultats", "nav.curves": "Courbes", "nav.fp": "Faux positifs",
  "hero.lead": "Un système d’alerte précoce multimodal : deux caméras (couleur + thermique), des capteurs environnementaux connectés, et trois “experts” en IA qui se contrôlent mutuellement avant de donner l’alarme.",
  "hero.start": "Début du projet", "hero.end": "Fin du projet", "hero.budget": "Budget total",
  "projet.title": "Le projet, expliqué simplement",
  "projet.p1": "Imaginez une tour de garde qui ne dort jamais. Une simple sonnette ne suffit pas en forêt : la poussière ressemble à de la fumée, un rocher chaud ressemble à un feu. Notre système fait donc travailler trois “experts” ensemble — et chacun peut douter des autres.",
  "projet.c1t": "1 · Les yeux",
  "projet.c1b": "Deux réseaux de neurones regardent la caméra couleur et la caméra thermique ; un capteur “renifle” les données environnementales. Chacun annonce : “je suis sûr à 62 %”.",
  "projet.c2t": "2 · Le cerveau",
  "projet.c2b": "Un transformateur relit les 32 derniers instants comme une histoire : ce signal monte-t-il comme un vrai feu, ou n’est-ce que du bruit ? Les experts incertains sont baissés d’un ton.",
  "projet.c3t": "3 · Le juge",
  "projet.c3b": "Une règle de vote mathématique (Dempster–Shafer) combine les avis. Si deux experts se contredisent, elle devient prudente. L’alarme doit se répéter avant d’être confirmée.",
  "contrib.title": "Nos 8 contributions scientifiques",
  "contrib.lead": "L’architecture vient d’un article scientifique ; nos contributions la rendent réelle, correcte, testable et honnête.",
  "contrib.t1": "Toute la machine, chaque hypothèse écrite",
  "contrib.b1": "Nous avons implémenté le système complet et documenté chaque détail manquant de l’article comme hypothèse explicite — rien n’est inventé en silence.",
  "contrib.t2": "Un examen honnête (anti-triche)",
  "contrib.b2": "Nous avons prouvé que le découpage officiel du jeu D-Fire place des images quasi identiques dans l’étude ET l’examen. Nous regroupons les images par vidéo pour une évaluation sans fuite.",
  "contrib.t3": "Des notes avec marges d’erreur",
  "contrib.b3": "Chaque résultat est donné avec un intervalle de confiance (1 000 rééchantillonnages), un décompte explicite des fausses alarmes et les temps de réponse bruts.",
  "contrib.t4": "Des données vérifiées avant utilisation",
  "contrib.b4": "Le jeu D-Fire est réputé ambigu (0 = fumée ou feu ?). Notre script vérifie automatiquement les comptes d’étiquettes contre les valeurs publiées : correspondance exacte.",
  "contrib.t5": "Les règles du service de sécurité",
  "contrib.b5": "Quand une alarme peut se répéter, combien de temps attendre après une validation humaine : règles formalisées, testées, avec un cas limite corrigé (re-armement pendant le refroidissement).",
  "contrib.t6": "Un juge qui se calibre — et refuse le non-sens",
  "contrib.b6": "Le seuil d’alarme est calibré sur des données réelles ; si on lui donne des données sans étiquettes, le script refuse de produire une configuration fausse.",
  "contrib.t7": "Conçu pour le matériel embarqué",
  "contrib.b7": "Export ONNX, quantification INT8 pour Jetson, messages d’alarme plafonnés à 8 ko — assez petits pour les liaisons radio forestières.",
  "contrib.t8": "Tout est testé",
  "contrib.b8": "55 tests automatiques protègent les maths de fusion et les règles de sécurité ; chaque chiffre de ce site provient d’une mesure réelle.",
  "arch.title": "Architecture implémentée",
  "arch.lead": "Le chemin des données, de la caméra à l’alerte. La piste en pointillés est la partie entraînement uniquement.",
  "arch.cap": "Pipeline FORFIRE-AI : capteurs → perception (YOLOv8n double flux + CBAM) → contexte (transformateur à porte d’entropie) → décision (Dempster–Shafer) → alerte MQTT ≤ 8 ko.",
  "res.title": "Résultats mesurés",
  "res.lead": "Modèle : YOLOv8n monocapteur RGB, 150 époques, jeu de test D-Fire (3 264 images, 3 990 instances), découpage sans fuite. Chaque chiffre vient d’une mesure réelle.",
  "res.k1": "mAP@0.5 (détection)", "res.k2": "F1 niveau image (alarme)", "res.k3": "latence par image (GPU)",
  "res.metric": "Métrique", "res.box": "Boîtes (IoU 0.5)", "res.image": "Niveau image",
  "res.precision": "Précision", "res.recall": "Rappel",
  "res.acc": "Exactitude (IC 95 %, bootstrap)",
  "res.perclass": "Par classe (AP@0.5)", "res.smoke": "Fumée", "res.fire": "Feu",
  "res.perclassnote": "— la fumée est plus facile à repérer tôt ; le feu de nuit reste difficile.",
  "res.ops": "Réglage opérationnel",
  "res.opsb": "Avec le seuil de confiance, on peut choisir : ≤ 0,5 % de fausses alarmes → rappel feu 0,53 · ≤ 1 % → 0,59 · ≤ 2 % → 0,70. La courbe complète est dans la galerie.",
  "res.note": "Le niveau image (0,958) est le chiffre utile pour l’alerte : « y a-t-il de la fumée ou du feu sur cette image ? » — dessiner la boîte exacte est plus difficile, d’où le 0,719. Niveau contextual : exactitude 0,936 sur fenêtres réelles. Les tableaux de simulation terrain sont marqués SYNTHÉTIQUE.",
  "nav.article": "Article",
  "paper.title": "L’article scientifique",
  "paper.lead": "Le manuscrit complet décrivant l’architecture, la méthodologie et les résultats — lu et approuvé par l’équipe.",
  "paper.btn": "Lire l’article scientifique",
  "paper.hint": "Accès protégé par mot de passe 🔒",
  "paper.modalTitle": "Accès protégé",
  "paper.note": "Cet article n’est pas encore publié. Le mot de passe protège son contenu pendant la période de révision — demandez-le à l’équipe FORFIRE-AI.",
  "paper.placeholder": "Mot de passe…",
  "paper.unlock": "Déverrouiller",
  "paper.wrong": "Mot de passe incorrect — réessayez.",
  "paper.required": "Veuillez saisir le mot de passe.",
  "paper.opening": "Ouverture de l’article…",
  "code.btn": "Code source",
  "code.title": "Code source du projet",
  "code.item1": "Environ 13,7 Go de données : jeux de données, checkpoints d’entraînement, résultats et figures.",
  "code.item2": "Accès réservé à M. Ali et M. Mehrez uniquement.",
  "code.item3": "L’architecture utilisée sera publiée sur GitHub uniquement avec l’accord de M. Ali et M. Mehrez.",
  "code.close": "Compris",
  "nav.future": "Future work", "future.title": "Future work",
  "future.c3label": "Contribution 3",
  "future.c3": "As ongoing future work, we are developing a reproducible cross-domain smoke-detection benchmark using Pyro-SDIS v2.0 for training and threshold calibration, and Boreal Forest Fire UAV imagery for independent external testing. The study will compare lightweight detectors and assess whether recall and image-level false-alarm rates transfer when the source-calibrated threshold is applied unchanged to the UAV domain. This work is currently in progress; no results are claimed at this stage.",
  "future.c4label": "Contribution 4", "future.soon": "SOON",
  "future.c4": "Database of Mr Mehrez.",
  "future.close": "Fermer",
  "curves.title": "Courbes et figures réelles",
  "curves.lead": "Toutes les figures proviennent des mesures du projet (cliquez pour agrandir).",
  "curves.c1": "Courbes d’entraînement — pertes et mAP sur 150 époques.",
  "curves.c2": "Courbe précision–rappel par classe (test).",
  "curves.c3": "Matrice de confusion (jeu de test, normalisée).",
  "curves.c4": "F1 selon le seuil de confiance.",
  "curves.c5": "Compromis rappel ↔ fausses alarmes sur 18 seuils.",
  "curves.c6": "Prédictions du modèle sur un échantillon de validation.",
  "fp.title": "Faux positifs — l’honnêteté d’abord",
  "fp.lead": "Voici les vraies erreurs du modèle (scènes sans feu). Les classifier honnêtement sert à les réduire.",
  "fp.c1": "“Feu 0,79” — un coucher de soleil derrière les arbres : la lumière dorée ressemble à des flammes.",
  "fp.c2": "“Fumée 0,82” — ciel d’orage et grue de chantier : textures de nuages sombres.",
  "fp.note": "50 faux positifs sur 1 504 scènes sans feu au seuil 0,25 — et le réglage du seuil permet de descendre à 7 (0,47 %) : voir la courbe de compromis.",
  "footer.disclaimer": "Projet cofinancé dans le cadre d’Interreg NEXT Italie–Tunisie. Partenaires : ENISo (Tunisie) · ISA Chott Mariem · Università di Palermo · SEEDS s.r.l. · RoboCare.",
  "footer.official": "Page officielle du projet", "footer.top": "Haut de page",
  "footer.integrity": "Chiffres issus de la reproduction locale (RTX 5070 Laptop, D-Fire test) — les tableaux de simulation sont marqués SYNTHÉTIQUE ; latence Jetson INT8 : non mesurée."
},

/* ============================================================ EN ==== */
en: {
  "topbar.project": "Interreg NEXT Italy–Tunisia",
  "brand.strap": "Forest-fire detection and prevention with AI — coastal forests of Tunisia and Italy.",
  "nav.project": "The project", "nav.contrib": "Contributions", "nav.arch": "Architecture",
  "nav.results": "Results", "nav.curves": "Curves", "nav.fp": "False positives",
  "hero.lead": "A multimodal early-warning system: two cameras (color + thermal), connected environmental sensors, and three AI “experts” that cross-check each other before raising an alarm.",
  "hero.start": "Project start", "hero.end": "Project end", "hero.budget": "Total budget",
  "projet.title": "The project, explained simply",
  "projet.p1": "Imagine a watchtower that never sleeps. A simple doorbell is not enough in a forest: dust looks like smoke, a warm rock looks like fire. Our system makes three “experts” work together — and each one is allowed to doubt the others.",
  "projet.c1t": "1 · The eyes",
  "projet.c1b": "Two neural networks watch the color camera and the thermal camera; a sensor “sniffs” the environmental data. Each one reports: “I am 62% sure”.",
  "projet.c2t": "2 · The brain",
  "projet.c2b": "A transformer reads the last 32 moments as a story: is this signal rising like a real fire, or just noise? Uncertain experts get turned down.",
  "projet.c3t": "3 · The judge",
  "projet.c3b": "A mathematical voting rule (Dempster–Shafer) combines the opinions. When two experts contradict each other, it turns cautious. The alarm must repeat before it is confirmed.",
  "contrib.title": "Our 8 scientific contributions",
  "contrib.lead": "The architecture comes from a research paper; our contributions make it real, correct, testable and honest.",
  "contrib.t1": "The whole machine, every assumption written down",
  "contrib.b1": "We implemented the complete system and documented every detail the paper leaves out as an explicit assumption — nothing is silently invented.",
  "contrib.t2": "A fair exam (anti-cheating)",
  "contrib.b2": "We proved that D-Fire’s official split puts near-identical images in both the study set and the exam. We regroup images by video for a leakage-free evaluation.",
  "contrib.t3": "Grades with error margins",
  "contrib.b3": "Every result comes with a confidence interval (1,000 resamples), an explicit false-alarm count, and raw timing traces.",
  "contrib.t4": "Data verified before use",
  "contrib.b4": "The D-Fire dataset is famously ambiguous (0 = smoke or fire?). Our script automatically checks label counts against the published values: an exact match.",
  "contrib.t5": "The safety-officer rules",
  "contrib.b5": "When an alarm may repeat, how long to wait after a human clears it: rules formalized, tested, including a fixed corner case (re-arming during cooldown).",
  "contrib.t6": "A judge that calibrates itself — and refuses nonsense",
  "contrib.b6": "The alarm threshold is calibrated on real data; feed it unlabeled data and the script refuses to produce a wrong configuration.",
  "contrib.t7": "Built for edge hardware",
  "contrib.b7": "ONNX export, INT8 quantization for Jetson, alarm messages capped at 8 kB — small enough for forest radio links.",
  "contrib.t8": "Everything is tested",
  "contrib.b8": "55 automated tests protect the fusion math and the safety rules; every number on this site comes from a real measurement.",
  "arch.title": "Implemented architecture",
  "arch.lead": "The data path, from camera to alarm. The dashed lane is training-only.",
  "arch.cap": "FORFIRE-AI pipeline: sensors → perception (dual-stream YOLOv8n + CBAM) → context (entropy-gated transformer) → decision (Dempster–Shafer) → MQTT alarm ≤ 8 kB.",
  "res.title": "Measured results",
  "res.lead": "Model: single-stream RGB YOLOv8n, 150 epochs, D-Fire test set (3,264 images, 3,990 instances), leakage-free split. Every figure is a real measurement.",
  "res.k1": "mAP@0.5 (detection)", "res.k2": "image-level F1 (alarm)", "res.k3": "latency per image (GPU)",
  "res.metric": "Metric", "res.box": "Boxes (IoU 0.5)", "res.image": "Image level",
  "res.precision": "Precision", "res.recall": "Recall",
  "res.acc": "Accuracy (95% bootstrap CI)",
  "res.perclass": "Per class (AP@0.5)", "res.smoke": "Smoke", "res.fire": "Fire",
  "res.perclassnote": "— smoke is easier to catch early; night-time fire stays hard.",
  "res.ops": "Operating-point tuning",
  "res.opsb": "With the confidence threshold you choose the trade-off: ≤ 0.5% false alarms → fire recall 0.53 · ≤ 1% → 0.59 · ≤ 2% → 0.70. The full curve is in the gallery.",
  "res.note": "The image-level score (0.958) is the number that matters for alarms: “is there smoke or fire in this image?” — drawing the exact box is harder, hence 0.719. Contextual tier: 0.936 accuracy on real windows. Field-simulation tables are marked SYNTHETIC.",
  "nav.article": "Paper",
  "paper.title": "The scientific paper",
  "paper.lead": "The full manuscript describing the architecture, the methodology and the results — read and approved by the team.",
  "paper.btn": "Read the scientific paper",
  "paper.hint": "Password-protected access 🔒",
  "paper.modalTitle": "Protected access",
  "paper.note": "This paper is not published yet. The password protects its content during the review period — ask the FORFIRE-AI team for it.",
  "paper.placeholder": "Password…",
  "paper.unlock": "Unlock",
  "paper.wrong": "Incorrect password — please try again.",
  "paper.required": "Please enter the password.",
  "paper.opening": "Opening the paper…",
  "code.btn": "Source code",
  "code.title": "Project source code",
  "code.item1": "About 13.7 GB of data: datasets, training checkpoints, results and figures.",
  "code.item2": "Access reserved for Mr Ali and Mr Mehrez only.",
  "code.item3": "The used architecture will be published on GitHub only with the agreement of Mr Ali and Mr Mehrez.",
  "code.close": "Understood",
  "nav.future": "Future work", "future.title": "Future work",
  "future.c3label": "Contribution 3",
  "future.c3": "As ongoing future work, we are developing a reproducible cross-domain smoke-detection benchmark using Pyro-SDIS v2.0 for training and threshold calibration, and Boreal Forest Fire UAV imagery for independent external testing. The study will compare lightweight detectors and assess whether recall and image-level false-alarm rates transfer when the source-calibrated threshold is applied unchanged to the UAV domain. This work is currently in progress; no results are claimed at this stage.",
  "future.c4label": "Contribution 4", "future.soon": "SOON",
  "future.c4": "Database of Mr Mehrez.",
  "future.close": "Close",
  "curves.title": "Real curves and figures",
  "curves.lead": "All figures come from the project’s measurements (click to enlarge).",
  "curves.c1": "Training curves — losses and mAP over 150 epochs.",
  "curves.c2": "Per-class precision–recall curve (test split).",
  "curves.c3": "Confusion matrix (test set, normalized).",
  "curves.c4": "F1 versus confidence threshold.",
  "curves.c5": "Recall ↔ false-alarm trade-off across 18 thresholds.",
  "curves.c6": "Model predictions on a validation sample.",
  "fp.title": "False positives — honesty first",
  "fp.lead": "These are the model’s real mistakes (scenes without fire). Classifying them honestly is how we reduce them.",
  "fp.c1": "“Fire 0.79” — a sunset behind trees: the golden light looks like flames.",
  "fp.c2": "“Smoke 0.82” — storm sky and a construction crane: dark cloud textures.",
  "fp.note": "50 false positives out of 1,504 background scenes at threshold 0.25 — and tuning the threshold brings it down to 7 (0.47%): see the trade-off curve.",
  "footer.disclaimer": "Project co-financed under Interreg NEXT Italy–Tunisia. Partners: ENISo (Tunisia) · ISA Chott Mariem · Università di Palermo · SEEDS s.r.l. · RoboCare.",
  "footer.official": "Official project page", "footer.top": "Back to top",
  "footer.integrity": "Figures from the local reproduction (RTX 5070 Laptop, D-Fire test) — field-simulation tables are marked SYNTHETIC; Jetson INT8 latency: not measured."
},

/* ============================================================ IT ==== */
it: {
  "topbar.project": "Interreg NEXT Italia–Tunisia",
  "brand.strap": "Rilevamento e prevenzione degli incendi boschivi con l’IA — foreste delle zone costiere Tunisia–Italia.",
  "nav.project": "Il progetto", "nav.contrib": "Contributi", "nav.arch": "Architettura",
  "nav.results": "Risultati", "nav.curves": "Curve", "nav.fp": "Falsi positivi",
  "hero.lead": "Un sistema di allarme precoce multimodale: due telecamere (colore + termica), sensori ambientali connessi e tre “esperti” di IA che si controllano a vicenda prima di dare l’allarme.",
  "hero.start": "Inizio progetto", "hero.end": "Fine progetto", "hero.budget": "Budget totale",
  "projet.title": "Il progetto, spiegato in modo semplice",
  "projet.p1": "Immaginate una torre di guardia che non dorme mai. Un semplice campanello non basta in bosco: la polvere sembra fumo, una roccia calda sembra un incendio. Il nostro sistema fa collaborare tre “esperti” — e ognuno può dubitare degli altri.",
  "projet.c1t": "1 · Gli occhi",
  "projet.c1b": "Due reti neurali osservano la telecamera a colori e quella termica; un sensore “annusa” i dati ambientali. Ognuno dichiara: “sono sicuro al 62%”.",
  "projet.c2t": "2 · Il cervello",
  "projet.c2b": "Un transformer rilegge gli ultimi 32 istanti come una storia: questo segnale sale come un vero incendio o è solo rumore? Gli esperti incerti vengono abbassati di tono.",
  "projet.c3t": "3 · Il giudice",
  "projet.c3b": "Una regola di voto matematica (Dempster–Shafer) combina le opinioni. Se due esperti si contraddicono, diventa prudente. L’allarme deve ripetersi prima di essere confermato.",
  "contrib.title": "I nostri 8 contributi scientifici",
  "contrib.lead": "L’architettura viene da un articolo scientifico; i nostri contributi la rendono reale, corretta, verificabile e onesta.",
  "contrib.t1": "Tutta la macchina, ogni ipotesi scritta",
  "contrib.b1": "Abbiamo implementato il sistema completo e documentato ogni dettaglio mancante dell’articolo come ipotesi esplicita — nulla è inventato in silenzio.",
  "contrib.t2": "Un esame onesto (anti-truffa)",
  "contrib.b2": "Abbiamo dimostrato che lo split ufficiale di D-Fire mette immagini quasi identiche sia nello studio sia nell’esame. Raggruppiamo le immagini per video per una valutazione senza perdite.",
  "contrib.t3": "Voti con margini di errore",
  "contrib.b3": "Ogni risultato è fornito con un intervallo di confidenza (1.000 ricampionamenti), un conteggio esplicito dei falsi allarmi e i tempi grezzi.",
  "contrib.t4": "Dati verificati prima dell’uso",
  "contrib.b4": "Il dataset D-Fire è notoriamente ambiguo (0 = fumo o fuoco?). Il nostro script verifica automaticamente i conteggi delle etichette rispetto ai valori pubblicati: corrispondenza esatta.",
  "contrib.t5": "Le regole del servizio di sicurezza",
  "contrib.b5": "Quando un allarme può ripetersi, quanto tempo aspettare dopo la conferma umana: regole formalizzate, testate, con un caso limite corretto (riarmo durante il raffreddamento).",
  "contrib.t6": "Un giudice che si calibra — e rifiuta le assurdità",
  "contrib.b6": "La soglia di allarme è calibrata su dati reali; se si forniscono dati senza etichette, lo script si rifiuta di produrre una configurazione sbagliata.",
  "contrib.t7": "Progettato per hardware embedded",
  "contrib.b7": "Esportazione ONNX, quantizzazione INT8 per Jetson, messaggi di allarme limitati a 8 kB — abbastanza piccoli per i collegamenti radio forestali.",
  "contrib.t8": "Tutto è testato",
  "contrib.b8": "55 test automatici proteggono la matematica di fusione e le regole di sicurezza; ogni numero di questo sito proviene da una misura reale.",
  "arch.title": "Architettura implementata",
  "arch.lead": "Il percorso dei dati, dalla telecamera all’allarme. La corsia tratteggiata è solo addestramento.",
  "arch.cap": "Pipeline FORFIRE-AI: sensori → percezione (YOLOv8n dual-stream + CBAM) → contesto (transformer con entropy gate) → decisione (Dempster–Shafer) → allarme MQTT ≤ 8 kB.",
  "res.title": "Risultati misurati",
  "res.lead": "Modello: YOLOv8n mono-flusso RGB, 150 epoche, test set D-Fire (3.264 immagini, 3.990 istanze), split senza perdite. Ogni cifra è una misura reale.",
  "res.k1": "mAP@0.5 (rilevamento)", "res.k2": "F1 a livello immagine (allarme)", "res.k3": "latenza per immagine (GPU)",
  "res.metric": "Metrica", "res.box": "Box (IoU 0.5)", "res.image": "Livello immagine",
  "res.precision": "Precisione", "res.recall": "Richiamo",
  "res.acc": "Accuratezza (IC 95%, bootstrap)",
  "res.perclass": "Per classe (AP@0.5)", "res.smoke": "Fumo", "res.fire": "Fuoco",
  "res.perclassnote": "— il fumo è più facile da individuare presto; il fuoco notturno resta difficile.",
  "res.ops": "Regolazione operativa",
  "res.opsb": "Con la soglia di confidenza si sceglie il compromesso: ≤ 0,5% di falsi allarmi → richiamo fuoco 0,53 · ≤ 1% → 0,59 · ≤ 2% → 0,70. La curva completa è nella galleria.",
  "res.note": "Il punteggio a livello immagine (0,958) è il numero che conta per l’allarme: “c’è fumo o fuoco in questa immagine?” — disegnare il box esatto è più difficile, da qui lo 0,719. Livello contestuale: accuratezza 0,936 su finestre reali. Le tabelle di simulazione sul campo sono marcate SINTETICO.",
  "nav.article": "Articolo",
  "paper.title": "L’articolo scientifico",
  "paper.lead": "Il manoscritto completo che descrive l’architettura, la metodologia e i risultati — letto e approvato dal team.",
  "paper.btn": "Leggi l’articolo scientifico",
  "paper.hint": "Accesso protetto da password 🔒",
  "paper.modalTitle": "Accesso protetto",
  "paper.note": "Questo articolo non è ancora pubblicato. La password protegge il suo contenuto durante il periodo di revisione — chiedetela al team FORFIRE-AI.",
  "paper.placeholder": "Password…",
  "paper.unlock": "Sblocca",
  "paper.wrong": "Password errata — riprova.",
  "paper.required": "Inserite la password.",
  "paper.opening": "Apertura dell’articolo…",
  "code.btn": "Codice sorgente",
  "code.title": "Codice sorgente del progetto",
  "code.item1": "Circa 13,7 GB di dati: dataset, checkpoint di addestramento, risultati e figure.",
  "code.item2": "Accesso riservato solo al Sig. Ali e al Sig. Mehrez.",
  "code.item3": "L’architettura utilizzata sarà pubblicata su GitHub solo con l’accordo del Sig. Ali e del Sig. Mehrez.",
  "code.close": "Capito",
  "nav.future": "Future work", "future.title": "Future work",
  "future.c3label": "Contribution 3",
  "future.c3": "Come lavori futuri in corso, stiamo sviluppando un benchmark riproducibile di rilevamento del fumo cross-domain, utilizzando Pyro-SDIS v2.0 per l'addestramento e la calibrazione della soglia, e immagini UAV di incendi di foresta boreale per un test esterno indipendente. Lo studio confronterà rilevatori leggeri e valuterà se richiamo e tassi di falso allarme a livello immagine si trasferiscono quando la soglia calibrata sulla sorgente viene applicata invariata al dominio UAV. Questo lavoro è in corso; nessun risultato viene rivendicato in questa fase.",
  "future.c4label": "Contribution 4", "future.soon": "SOON",
  "future.c4": "Database del Sig. Mehrez.",
  "future.close": "Chiudi",
  "curves.title": "Curve e figure reali",
  "curves.lead": "Tutte le figure provengono dalle misure del progetto (clic per ingrandire).",
  "curves.c1": "Curve di addestramento — perdite e mAP su 150 epoche.",
  "curves.c2": "Curva precisione–richiamo per classe (test).",
  "curves.c3": "Matrice di confusione (test set, normalizzata).",
  "curves.c4": "F1 in funzione della soglia di confidenza.",
  "curves.c5": "Compromesso richiamo ↔ falsi allarmi su 18 soglie.",
  "curves.c6": "Predizioni del modello su un campione di validazione.",
  "fp.title": "Falsi positivi — prima l’onestà",
  "fp.lead": "Questi sono i veri errori del modello (scene senza fuoco). Classificarli onestamente serve a ridurli.",
  "fp.c1": "“Fuoco 0,79” — un tramonto dietro gli alberi: la luce dorata sembra fiamme.",
  "fp.c2": "“Fumo 0,82” — cielo di tempesta e una gru edile: texture di nuvole scure.",
  "fp.note": "50 falsi positivi su 1.504 scene senza fuoco alla soglia 0,25 — e regolando la soglia si scende a 7 (0,47%): vedi la curva di compromesso.",
  "footer.disclaimer": "Progetto cofinanziato nell’ambito di Interreg NEXT Italia–Tunisia. Partner: ENISo (Tunisia) · ISA Chott Mariem · Università di Palermo · SEEDS s.r.l. · RoboCare.",
  "footer.official": "Pagina ufficiale del progetto", "footer.top": "Torna su",
  "footer.integrity": "Cifre dalla riproduzione locale (RTX 5070 Laptop, test D-Fire) — le tabelle di simulazione sono marcate SINTETICO; latenza Jetson INT8: non misurata."
}
};

/* ---------- language switching ---------- */
function setLang(lang) {
  if (!I18N[lang]) return;
  const dict = I18N[lang];
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (key in dict) el.innerHTML = dict[key];
  });
  document.querySelectorAll("[data-i18n-attr]").forEach(el => {
    const key = el.getAttribute("data-i18n-attr");
    if (key in dict) el.placeholder = dict[key];
  });
  document.documentElement.lang = lang;
  document.querySelectorAll(".lang").forEach(b =>
    b.classList.toggle("on", b.dataset.lang === lang));
  try { localStorage.setItem("forfire_lang", lang); } catch (e) { /* private mode */ }
}

document.querySelectorAll(".lang").forEach(btn =>
  btn.addEventListener("click", () => setLang(btn.dataset.lang)));

let saved = "fr";
try { saved = localStorage.getItem("forfire_lang") || "fr"; } catch (e) { /* */ }
setLang(saved in I18N ? saved : "fr");

/* ---------- mobile navigation ---------- */
const nav = document.getElementById("nav");
document.getElementById("navToggle").addEventListener("click", () => nav.classList.toggle("open"));
nav.querySelectorAll("a").forEach(a => a.addEventListener("click", () => nav.classList.remove("open")));

/* ---------- lightbox for the gallery ---------- */
const lb = document.getElementById("lightbox");
const lbImg = document.getElementById("lbImg");
document.querySelectorAll(".gallery img, .figure img").forEach(img =>
  img.addEventListener("click", () => {
    lbImg.src = img.src;
    lbImg.alt = img.alt;
    lb.hidden = false;
    document.body.style.overflow = "hidden";
  }));
function closeLb() {
  lb.hidden = true;
  document.body.style.overflow = "";
}
document.getElementById("lbClose").addEventListener("click", closeLb);
lb.addEventListener("click", e => { if (e.target === lb) closeLb(); });
document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    closeLb();
    if (!modal.hidden) closeModal();
    if (!codeModal.hidden) closeCodeModal();
    if (!futureModal.hidden) closeFutureModal();
  }
});

/* ---------- burning FORFIRE-AI text --------------------------------------- *
 * Every character of the brand logo and the hero title catches fire in turn:  *
 * each char is wrapped in a span with a staggered delay; the keyframes flash  *
 * fire colors + glow + a small jump, then settle back to the base color.      */
function igniteText(el) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  let i = 0;
  textNodes.forEach(node => {
    const frag = document.createDocumentFragment();
    for (const ch of node.textContent) {
      if (ch === " ") { frag.appendChild(document.createTextNode(" ")); continue; }
      const s = document.createElement("span");
      s.className = "burn-ch";
      s.style.setProperty("--i", i++);
      s.textContent = ch;
      frag.appendChild(s);
    }
    node.parentNode.replaceChild(frag, node);
  });
}
document.querySelectorAll(".logo, .hero h1").forEach(igniteText);

/* ---------- back to top ---------- */
document.getElementById("toTop").addEventListener("click", e => {
  e.preventDefault();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/* ---------- paper access (password) --------------------------------------- *
 * The password itself is never stored in the source: only its SHA-256 hash is
 * compared (crypto.subtle). This deters casual access; for a static site it is
 * a deterrent, not cryptographic protection — the note in the modal explains
 * the password exists to protect the unpublished paper's content.            */
const PAPER = {
  file: "assets/papers/amine paper forfire 2.pdf",
  sha256: "a3e0cc0d33b26f5612ed7d27ef21f87d983d8f18f89ffb6913b073f71aa3a90a",
  sessionKey: "forfire_paper_unlocked"
};
const modal = document.getElementById("paperModal");
const passInput = document.getElementById("pmPass");
const pmError = document.getElementById("pmError");
const pmCard = modal.querySelector(".pm-card");

async function sha256Hex(text) {
  if (window.crypto && crypto.subtle) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, "0")).join("");
  }
  return null; // insecure context — caller falls back
}

function paperUnlocked() {
  try { return sessionStorage.getItem(PAPER.sessionKey) === "1"; }
  catch (e) { return false; }
}

function openPaper() {
  const t = I18N[document.documentElement.lang] || I18N.fr;
  document.getElementById("paperBtn").querySelector("span[data-i18n]").textContent = t["paper.opening"];
  window.open(encodeURI(PAPER.file), "_blank");
  setTimeout(() => {
    document.getElementById("paperBtn").querySelector("span[data-i18n]").textContent = t["paper.btn"];
  }, 1500);
}

function openModal() {
  modal.hidden = false;
  document.body.style.overflow = "hidden";
  pmError.hidden = true;
  passInput.value = "";
  pmCard.classList.remove("error");
  setTimeout(() => passInput.focus(), 60);
}
function closeModal() {
  modal.hidden = true;
  document.body.style.overflow = "";
}

async function tryUnlock() {
  const t = I18N[document.documentElement.lang] || I18N.fr;
  const val = passInput.value.trim();
  if (!val) {
    pmError.textContent = t["paper.required"];
    pmError.hidden = false;
    return;
  }
  let hex = await sha256Hex(val);
  let ok = hex === PAPER.sha256;
  if (hex === null) {   // insecure context fallback (obfuscated compare)
    ok = btoa(val.split("").reverse().join("")) === "MmVyaWZyb2Y=";
  }
  if (ok) {
    try { sessionStorage.setItem(PAPER.sessionKey, "1"); } catch (e) { /* */ }
    closeModal();
    openPaper();
  } else {
    pmError.textContent = t["paper.wrong"];
    pmError.hidden = false;
    pmCard.classList.remove("error");
    void pmCard.offsetWidth;            // restart the shake animation
    pmCard.classList.add("error");
    passInput.value = "";
    passInput.focus();
  }
}

document.getElementById("paperBtn").addEventListener("click", () => {
  paperUnlocked() ? openPaper() : openModal();
});
document.getElementById("pmUnlock").addEventListener("click", tryUnlock);
document.getElementById("pmClose").addEventListener("click", closeModal);
passInput.addEventListener("keydown", e => { if (e.key === "Enter") tryUnlock(); });
document.getElementById("pmEye").addEventListener("click", () => {
  passInput.type = passInput.type === "password" ? "text" : "password";
});
modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });

/* ---------- code source info modal ---------------------------------------- *
 * Informational only — no authentication: it explains who may access the      *
 * ~13.7 GB source archive and under which condition the architecture is       *
 * published on GitHub.                                                        */
const codeModal = document.getElementById("codeModal");
function openCodeModal() {
  codeModal.hidden = false;
  document.body.style.overflow = "hidden";
}
function closeCodeModal() {
  codeModal.hidden = true;
  document.body.style.overflow = "";
}
document.getElementById("codeBtn").addEventListener("click", openCodeModal);
document.getElementById("cmClose").addEventListener("click", closeCodeModal);
document.getElementById("cmOk").addEventListener("click", closeCodeModal);
codeModal.addEventListener("click", e => { if (e.target === codeModal) closeCodeModal(); });

/* ---------- future work modal ---------------------------------------------- *
 * Informational: ongoing future work (Pyro-SDIS v2.0 + Boreal UAV benchmark,  *
 * and Mr Mehrez's database). No results claimed at this stage.                */
const futureModal = document.getElementById("futureModal");
function openFutureModal() {
  futureModal.hidden = false;
  document.body.style.overflow = "hidden";
}
function closeFutureModal() {
  futureModal.hidden = true;
  document.body.style.overflow = "";
}
document.getElementById("futureBtn").addEventListener("click", openFutureModal);
document.getElementById("fmClose").addEventListener("click", closeFutureModal);
document.getElementById("fmOk").addEventListener("click", closeFutureModal);
futureModal.addEventListener("click", e => { if (e.target === futureModal) closeFutureModal(); });
