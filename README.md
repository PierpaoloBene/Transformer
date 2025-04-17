
# Transformer

Questa repository nasce con l'obiettivo di ricostruire un modello Transformer da zero, seguendo passo dopo passo il  paper _"Attention Is All You Need"_ (Vaswani et al., 2017). Il codice si ispira al walkthrough di [Alladin Person](https://www.youtube.com/watch?v=U0s0f995w14), con spiegazioni concettuali ispirate a [Peter Bloem](https://peterbloem.nl/blog/transformers).

---

## Cos'è un Transformer?

Il Transformer è un’architettura che ha rivoluzionato il modo in cui i modelli neurali gestiscono le sequenze. Diversamente dai modelli RNN o LSTM, che elaborano le parole una alla volta, il Transformer è completamente basato sull'attenzione. Ogni parola in input può "guardare" tutte le altre contemporaneamente, permettendo al modello di cogliere contesti globali in modo efficiente e parallelo.

Il cuore dell'architettura è costituito da due componenti principali:

- L'**encoder** prende in ingresso una sequenza (ad esempio una frase in inglese) e la trasforma in una serie di vettori che rappresentano il significato di ciascun token tenendo conto del contesto in cui si trova. Ogni token, infatti, non è più trattato isolatamente, ma viene arricchito con informazioni su tutte le altre parole della sequenza grazie al meccanismo di self-attention. L'encoder è costituito da una pila di blocchi identici (tipicamente 6), ciascuno composto da un modulo di multi-head self-attention seguito da una rete neurale feed-forward. Alla fine, l'encoder produce una rappresentazione contestuale per ogni parola della sequenza in input.

- Il **decoder** è responsabile della generazione della sequenza in output (ad esempio la traduzione in un'altra lingua). Anch'esso è formato da una pila di blocchi, simili a quelli dell’encoder, ma con una struttura leggermente più complessa. Ogni blocco del decoder contiene:
  1. un modulo di self-attention mascherato (masked self-attention) che impedisce al modello di vedere parole future, mantenendo la generazione autoregressiva;
  2. un modulo di encoder-decoder attention, che consente a ogni token in generazione di "guardare" tutta la sequenza di input elaborata dall’encoder;
  3. una rete feed-forward.

Questo design consente al decoder di generare l’output un token alla volta, usando sia i token già generati, sia l’intera rappresentazione dell’input prodotta dall’encoder. A ogni passaggio, il decoder aggiorna la sua previsione fino a quando non viene generato un token di fine sequenza.

---

## Self-Attention

Il modulo `SelfAttention` implementa uno dei concetti centrali del Transformer: l’attenzione su sé stessi, o **self-attention**. Questo meccanismo permette a ogni parola di valutare quanto siano importanti le altre parole della frase per comprendere il proprio significato.

Nel dettaglio:
- Ogni input viene trasformato in tre vettori: **query**, **key** e **value**.
- Si calcola una misura di compatibilità tra le query e le key (chiamata **energia**).
- Si applica una **softmax** per ottenere dei pesi di attenzione.
- Questi pesi vengono usati per combinare i value in un vettore d’uscita.

L’operazione viene fatta su più “teste” contemporaneamente, permettendo al modello di guardare il contesto da prospettive diverse: questo è ciò che chiamiamo **multi-head attention**.

Ecco un’illustrazione presa dal paper:

<img width="240" alt="Forward" src="https://github.com/user-attachments/assets/ab4c00a0-b2f3-45b1-9302-f40064e44d34" />


---

## Riassunto del codice

Il modulo `SelfAttention` definito in PyTorch segue questi passaggi fondamentali:

1. **Proiezione lineare** degli input in `query`, `key`, `value`.
2. **Reshape** per suddividere i vettori in diverse teste di attenzione.
3. **Calcolo dell’energia** tramite prodotto scalare tra `query` e `key`.
4. **Applicazione della maschera**, se necessaria, per evitare attenzione su token non rilevanti.
5. **Normalizzazione con softmax** e **scaling** (divisione per radice quadrata della dimensione).
6. **Pesatura dei `value`** tramite i punteggi di attenzione.
7. **Concatenazione e proiezione finale** per riportare i risultati nella dimensione originale.

L'uso di `torch.einsum` permette di esprimere il prodotto scalare e l'applicazione dei pesi in maniera efficiente e leggibile.

---


## Perché è così rivoluzionario

La chiave del successo del Transformer è la sua **capacità di elaborare tutta la sequenza in parallelo**, grazie al meccanismo di self-attention.

Nei modelli RNN, ogni parola deve aspettare quella precedente, creando un collo di bottiglia. Il Transformer invece può calcolare le rappresentazioni di **tutti i token contemporaneamente**, sfruttando appieno la potenza delle GPU.

Questo ha permesso di addestrare modelli molto grandi in tempi ragionevoli e ha aperto la strada ai moderni LLM come BERT, GPT, T5, ecc.

---

## Limitazioni del Transformer

Sebbene il Transformer sia molto potente, non è privo di difetti. Tra i principali:

- La **complessità computazionale** dell’attenzione è **quadratica** rispetto alla lunghezza della sequenza (O(n²)). Questo lo rende costoso per sequenze lunghe.
- Richiede **molta memoria GPU**, specialmente durante l’addestramento.
- Tende a generalizzare male su sequenze molto più lunghe di quelle viste a training.
- Non incorpora strutture sintattiche in modo esplicito, ma solo tramite apprendimento implicito.
- Non ha una memoria persistente tra diverse sequenze: dimentica tutto dopo ogni batch.
- Il processo di **decoding** è ancora **sequenziale**: ogni parola dipende dalla precedente, quindi l'inferenza non è completamente parallela.

Negli anni, sono nate varianti (come Longformer, BigBird, Reformer) per affrontare questi limiti.

---
