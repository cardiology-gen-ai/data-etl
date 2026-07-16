# UMLS Normalization e UMLS Connections

Questi due moduli arricchiscono i nodi `Concept` già estratti dai documenti clinici.

```text
Section --MENTIONS--> Concept
                         |
                         +-- normalizzazione UMLS opzionale
                         |
                         +-- relazioni SNOMED locali opzionali
```

Non eseguono entity extraction, non modificano lo schema delle entità e non richiedono che ogni concetto ottenga un CUI.

## `umls_normalization.py`

### Scopo

Associa, quando possibile, un concetto locale a un concetto UMLS tramite un **CUI** (`Concept Unique Identifier`).

Esempio:

```text
"dilated cardiomyopathy"
→ UMLS CUI: C0007193
```

### Come funziona

Per ogni nodo `Concept`, il modulo:

1. costruisce una lista di nomi e alias, includendo eventuali espansioni di acronimi validate;
2. cerca candidati tramite l’API UMLS;
3. elimina i candidati incompatibili con il tipo locale del concetto;
4. ordina i candidati in base al metodo di ricerca, all’alias utilizzato e alla similarità;
5. accetta automaticamente solo i match sufficientemente affidabili.

I risultati possono avere tre stati principali:

* `umls_matched`: match accettato e scritto sul nodo;
* `umls_low_confidence`: candidato trovato, ma da revisionare;
* `umls_no_match`: nessun candidato adeguato.

Per i match accettati vengono salvati, tra gli altri:

* `umls_cui`;
* nome canonico UMLS;
* score e metodo di normalizzazione;
* semantic types UMLS;
* timestamp e stato della normalizzazione.

La copertura è intenzionalmente parziale: non trovare un CUI non rende il concetto locale inutilizzabile.

### Opzioni importanti

* `dry_run=true`: valuta ed esporta i risultati senza modificare Neo4j;
* `force=false`: conserva i mapping già accettati;
* `force=true`: rivaluta anche i mapping esistenti e può sostituirli o rimuoverli.

La configurazione normale consigliata usa `force=false`.

## `umls_connections.py`

### Scopo

Trova relazioni ontologiche tra concetti locali già normalizzati e, opzionalmente, le materializza in Neo4j.

Il modulo usa le relazioni SNOMED CT rese disponibili tramite UMLS. Un CUI UMLS non è necessariamente un concetto SNOMED: solo i CUI con una rappresentazione `SNOMEDCT_US` possono contribuire a questa fase.

### Come funziona

Il modulo considera solo concetti che:

* hanno `normalization_status = "umls_matched"`;
* possiedono un CUI valido;
* non hanno un tipo ambiguo;
* non sono marcati per revisione del tipo.

Successivamente:

1. raggruppa i concetti locali per CUI;
2. sceglie un rappresentante locale per ogni CUI;
3. recupera le relazioni SNOMED associate;
4. filtra relazioni, sorgenti e target non ammessi;
5. conserva solo relazioni il cui target CUI esiste anche nel grafo locale;
6. collassa evidenze duplicate;
7. produce file di audit e statistiche;
8. opzionalmente scrive relazioni dirette tra nodi `Concept`.

Non vengono creati nodi UMLS o SNOMED esterni:

```text
Concept locale --UMLS_RELATION--> Concept locale
```

### Profili di relazione

* `core`: relazioni principali, come gerarchie, finding site, morphology e procedure site;
* `expanded`: `core` più una prima estensione controllata;
* `audit_all`: include anche relazioni sperimentali o da revisione ed è utilizzabile solo per l’export.

### Materializzazione

Le modalità disponibili sono:

* `none`: sola analisi ed esportazione;
* `safe_only`: scrive soltanto relazioni considerate sicure.

Con `safe_only`, una relazione viene scritta solo se:

* è presente nel catalogo;
* è compatibile con i tipi locali;
* non richiede revisione;
* ha policy `safe` o `hierarchy`;
* è marcata come materializzabile;
* entrambi i concetti rappresentanti esistono.

La write può usare `replace_existing_connections=true` per sostituire in modo riproducibile le precedenti relazioni UMLS dello stesso documento.

## Output principali

### Normalizzazione

* metadati UMLS sui nodi `Concept`;
* file JSONL per la revisione dei match;
* statistiche su match, low-confidence e no-match.

### Connessioni

* CSV dei candidati;
* JSON delle connessioni collassate;
* report statistico JSON;
* riepilogo Markdown;
* report della materializzazione Neo4j, quando abilitata.

## Esecuzione

Normalizzazione:

```bash
CONFIG_PATH=config.json \
KG_PIPELINE_PHASE=normalization \
python3 src/main_graph.py
```

Connessioni UMLS/SNOMED:

```bash
CONFIG_PATH=config.json \
KG_PIPELINE_PHASE=umls_connections \
python3 src/main_graph.py
```

Entrambe le fasi leggono i nodi già presenti in Neo4j. L’accesso all’API richiede `UMLS_API_KEY`.

## In sintesi

`umls_normalization.py` risponde alla domanda:

> A quale concetto UMLS corrisponde questo `Concept` locale?

`umls_connections.py` risponde alla domanda:

> Quali relazioni SNOMED esistono tra i concetti UMLS già presenti nel nostro grafo locale?
