In main_graph.py c'è un parametro per scelgiere quale pezzo della pipeline attivare.
Per ora ho testato la prima parte che prende in input i pdf, crea chunk gerarchici e li carica su neo4j.

La seconda parte dovrebbe prendere i titoli dei vari chunk ed estrarre le parole chiave, qui entrava in gioco la chiamata a openAI, e sarà una parte da adattare per usare un modello su Leonardo. (In futuro l'idea sarebbe di estrarre le parole chiave da tutto il chunk e magari creare relazioni più complesse).

Lo stesso vale per il terzo blocco, che invece serve per caricare gli embeddings, questo è meno rilevante, e lo possiamo fare in un'altro momento.

Tutti i file realtivi al knowlege graph sono stati spostati in src/knowlege_graph
e li ho separati per funzionalità, sperando di aver reso il tutto più modulare.

Per quanto riguarda l'env, dovrei aver aggiunto al template le altre variabili necessarie al momento

TODO: modify to make this part clear

