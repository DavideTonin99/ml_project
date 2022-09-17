- cnn
    - creare dataset direttamente con pytorch
        - perchè i file python del dataset sono troppi pesanti
    - fare test con cnn base già fatta (quella per mnist) e estrarre i risultati
        - fare n iterazioni come al solito
    - comparare risultati con svm e knn per caso binario
    - mettere tutte le classi e vedere come si comporta

cnn
- fare grafico overfitting
- fare grafico precision, recall over epochs

- transfer learning
    - extract features from big pretrained network
    - run svm/knn
    - **doing**
- run with pretrained net
    - done (resnet18), ma rifare test quando abbiamo intero dataset su cloud

+ pca
    + comparison original and transformed images (n_components = 0.1, 0.5, 0.8, 0.95)
        + done

    - grafico a gomito della varianza

- heatmap

# presentazione

- preprocessing
- elaboration
    - pca
        + grafici
    - svm (vari params)
        + considerations
    - knn (vari params)
        + considerations
    - tabellina results
- cnn
    - test con cnn base (mnist)
        - results
    - transfer learning
        - resnet18
        - resnet50
        - vgg
- confronto risultati machine learning vs cnn
- future works:
    - future improvements:
        - feature extraction and input to svm/knn